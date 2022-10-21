from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

# import numpy as np
# from einops import rearrange

# from time import time
# import itertools as it

import cached_conv as cc
from rave.model import RAVE
from rave.core import search_for_run

from effortless_config import Config

class args(Config):
    RUN = None
    # audio sample rate -- the exported model will convert to/from
    # the RAVE model sample rate
    SR = None
    # should be true for realtime
    CACHED = True
    # number of latents to preserve from RAVE model
    LATENT_SIZE = None
    # predictive context
    CONTEXT = 3
    # number of loops
    LOOPS = 3
    # max memory size
    MEMORY = 1000
    # included in output filename
    NAME = "test"

args.parse_args()
cc.use_cached_conv(args.CACHED)


import logging
from termcolor import colored
logging.basicConfig(
    level=logging.INFO,
    format=colored("[%(relativeCreated).2f] ", "green") +
    "%(message)s")


class LivingLooper(nn.Module):
    loop_index:int
    record_index:int
    n_memory:int
    loop_length:int
    trained_cropped:bool

    has_model:List[bool]

    def __init__(self, 
            rave_model:RAVE, 
            n_loops:int, n_context:int, n_memory:int, 
            n_latent:Optional[int] = None
            ):
        super().__init__()

        self.n_loops = n_loops
        self.n_context = n_context
        self.n_memory = n_memory

        if n_latent is not None:
            rave_model.crop_latent_space(n_latent)

        self.trained_cropped = bool(rave_model.cropped_latent_size)
        self.n_latent = (
            rave_model.cropped_latent_size 
            if self.trained_cropped 
            else rave_model.latent_size)

        self.n_feature = self.n_loops * self.n_context * self.n_latent

        self.pqmf = rave_model.pqmf
        self.encoder = rave_model.encoder
        self.decoder = rave_model.decoder
        self.n_latent_decoder = rave_model.latent_size

        self.register_buffer('weights', 
            torch.empty(self.n_loops, self.n_feature, self.n_latent, 
                requires_grad=False))
        self.register_buffer('center', 
            torch.empty(self.n_loops, self.n_feature, 
                requires_grad=False))
        self.register_buffer('bias', 
            torch.empty(self.n_loops, self.n_latent, 
                requires_grad=False))

        self.register_buffer('memory', 
            torch.empty(n_memory, n_loops, self.n_latent,
                requires_grad=False))

        self.reset()

    @torch.jit.export
    def reset(self):
        self.loop_index = -1
        self.record_index = 0
        self.loop_length = 0

        self.memory.zero_()
        self.weights.zero_()
        self.bias.zero_()
        self.center.zero_()
        self.has_model = [False]*self.n_loops

    def forward(self, i:int, x):
        """
        Args:
            i: loop record index, 0 for no loop, 1-index otherwise
            x: Tensor[1, sample]
        Returns:
            Tensor[loop, sample]
        """
        i = i-1 # convert to zero index loop / negative for no loop
        zs = torch.empty(self.n_loops, self.n_latent)

        if i!=self.loop_index: # change in loop select control
            if self.loop_index >= 0: # previously on a loop
                if self.loop_length > self.n_context: # and it was long enough
                    # finalize loop
                    mem = self.get_frames(self.loop_length)
                    # print(mem.shape)
                    # dataset of features, targets
                    features = mem.unfold(0, self.n_context, 1)[:-1] # time' x loop x latent x ctx
                    features = features.reshape(features.shape[0], -1)
                    targets = mem[self.n_context:,self.loop_index,:] # time' x 1 x latent
                    # assert targets.shape[0]==features.shape[0], f"""
                        # {targets.shape=}, {features.shape=}
                        # """
                    # targets = targets.view(targets.shape[0], -1)
                    self.fit_loop(self.loop_index, features, targets)
            if i>=0: # starting a new loop recording
                self.loop_index = i
                self.loop_length = 0
        if i>=0:
            # print(x.shape)
            # slice on LHS to appease torchscript
            zs[i:i+1] = self.encode(x)[...,0] # remove time dim

        # predictive context (same for all loops right now)
        feature = self.get_frames(self.n_context) # ctx x loop x latent
        # print(f'{feature.shape=}')
        feature = feature.permute(1,2,0).reshape(1,-1) # 1 x feature
        for j in range(self.n_loops):
            if i==j: continue
            # predict from a single feature
            # slice on LHS to appease torchscript
            zs[j:j+1] = self.eval_loop(j, feature)

        # print([f'{z.shape}' for z in zs])

        # zs = torch.cat(zs)

        # update memory
        self.record_frame(zs)
        if self.loop_index >= 0:
            self.loop_length += 1

        y = self.decode(zs[...,None]) # add time dim

        # silence any loops which don't have a model
        mask = torch.cat(
            [torch.ones(1,1,1) if i==j or b else torch.zeros(1,1,1) 
            for j,b in enumerate(self.has_model)])
        y = y * mask
#
        # print(f'{self.loop_length}, {self.record_index}, {self.loop_index}')

        return y


    def record_frame(self, zs):
        """
        zs: Tensor[loop, latent]
        """
        # record_index points to the most recently recorded frame;
        # so increment it first
        self.record_index = (self.record_index+1)%self.n_memory
        self.memory[self.record_index, :, :] = zs

    def get_frames(self, n:int):
        """
        get contiguous tensor out of ring memory
        """
        begin = self.record_index - n + 1
        if begin < 0:
            begin1 = begin % self.n_memory
            block1 = self.memory[begin1:]
            begin2 = max(0, begin)
            block2 = self.memory[begin2:self.record_index+1]
            return torch.cat((block1, block2))
        return self.memory[begin:self.record_index+1]

    def encode(self, x):
        """
        """
        # x = self.resample.from_target_sampling_rate(x)

        if self.pqmf is not None:
            x = self.pqmf(x)

        z = self.encoder(x)[:,:self.n_latent]
        # if self.trained_cropped:
        #     z = p
        # else:
        #     # z, _ = self.encoder(x).chunk(2, 1)
        #     z, _ = p.chunk(2, 1)

        # z, _ = self.encoder(x)
        # z, std = self.post_process_distribution(mean, scale)

        # z = z - self.latent_mean.unsqueeze(-1)
        # z = nn.functional.conv1d(z, self.latent_pca.unsqueeze(-1))

        # z = z[:, :self.n_latent]
        return z

    def decode(self, z):
        # if self.trained_cropped:  # PERFORM PCA BEFORE PADDING
        #     z = nn.functional.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        #     z = z + self.latent_mean.unsqueeze(-1)

        # CAT WITH SAMPLES FROM PRIOR DISTRIBUTION
        pad_size = self.n_latent_decoder - self.n_latent
        pad_latent = torch.randn(
            z.shape[0],
            pad_size,
            z.shape[-1],
            device=z.device,
        )

        z = torch.cat([z, pad_latent], 1)

        # if not self.trained_cropped:  # PERFORM PCA AFTER PADDING
            # z = nn.functional.conv1d(z, self.latent_pca.T.unsqueeze(-1))
            # z = z + self.latent_mean.unsqueeze(-1)

        x = self.decoder(z)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x)

        # x = self.resample.to_target_sampling_rate(x)

        return x

    def feat_process(self, i:int, x):
        return (x - self.center[i]).tanh()

    def fit_loop(self, i:int, x, y):
        """
        Args:
            i: index of loop
            x: Tensor[batch, feature]
            y: Tensor[batch, target]
        Returns:

        """
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        
        # y = y.pow(3)

        c = x.mean(0)
        self.center[i, :] = c

        x = self.feat_process(i, x)

        b = y.mean(0)
        self.bias[i, :] = b

        w = torch.linalg.lstsq(x, y-b).solution
        self.weights[i, :] = w

        # print(x.norm(), y.norm(), w.norm())
        # print(y)

        self.has_model[i] = True

    def eval_loop(self, i:int, x):
        """
        Args:
            i: index of loop
            x: Tensor[batch, feature]
        Returns:
            y: Tensor[batch, target]
        """
        y = self.feat_process(i, x) @ self.weights[i] + self.bias[i]
        # y = y.abs().pow(1/3) * y.sign()
        return y

              


logging.info("loading RAVE model from checkpoint")

RUN = search_for_run(args.RUN)
logging.info(f"using {RUN}")
model = RAVE.load_from_checkpoint(RUN, strict=False).eval()

# model.cropped_latent_size = args.LATENT_SIZE # TODO

logging.info("flattening weights")
for m in model.modules():
    if hasattr(m, "weight_g"):
        nn.utils.remove_weight_norm(m)

logging.info("warmup forward pass")
x = torch.zeros(1, 1, 2**14)
if model.pqmf is not None:
    x = model.pqmf(x)

z, _ = model.reparametrize(*model.split_params(model.encoder(x)))

y = model.decoder(z)

if model.pqmf is not None:
    y = model.pqmf.inverse(y)

model.discriminator = None

sr = model.sr

assert int(args.SR) == sr, f"model sample rate is {sr}"
# if args.SR is not None:
#     target_sr = int(args.SR)
# else:
#     target_sr = sr

# logging.info("build resampling model")
# resample = Resampling(target_sr, sr)
# x = torch.zeros(1, 1, 2**14)
# resample.to_target_sampling_rate(resample.from_target_sampling_rate(x))

logging.info("creating looper")
ls = None if args.LATENT_SIZE is None else int(args.LATENT_SIZE)
looper = LivingLooper(model, args.LOOPS, args.CONTEXT, args.MEMORY, ls)
# smoke test
def feed(i):
    x = torch.randn(1, 1, 2**11)
    looper(i, x)

def smoke_test():
    looper.reset()
    feed(0)
    for _ in range(10):
        feed(1)
    for _ in range(10):
        feed(2)
    for _ in range(10):
        feed(0)
    for _ in range(10):
        feed(3)
# x = torch.zeros(1, 1, 2**11)
# looper(0, x)
# looper(1, x)
# looper(1, x)
# looper(1, x)
# looper(2, x)
# looper(0, x)
logging.info("smoke test with pytorch")
smoke_test()

looper.reset()

logging.info("compiling torchscript")
looper = torch.jit.script(looper)

logging.info("smoke test with torchscript")
smoke_test()

fname = f"ll_{args.NAME}.ts"
logging.info(f"saving '{fname}'")
looper.save(fname)
