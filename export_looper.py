import logging
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
    LATENT_SIZE = 8
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


class LivingLooper(nn.Module):
    loop_index:int
    record_index:int
    n_memory:int

    has_model:List[bool]

    def __init__(self, rave_model:RAVE, n_loops:int, n_context:int, n_memory:int):
        super().__init__()

        self.n_loops = n_loops
        self.n_context = n_context
        self.n_latent = rave_model.cropped_latent_size
        self.n_memory = n_memory

        self.n_feature = self.n_loops * self.n_context * self.n_latent

        self.rave = rave_model

        self.loop_index = 0
        self.record_index = 0

        self.has_model = [False]*n_loops

        self.register_buffer(
            'weights', torch.zeros(self.n_loops, self.n_feature, self.n_latent))

        self.register_buffer(
            'memory', torch.zeros(n_memory, n_loops, self.n_latent))

    def forward(self, x, i:int):
        """
        Args:
            x: Tensor[1, sample]
            i: loop record index, 0 for no loop, 1-index otherwise
        Returns:
            Tensor[loop, sample]
        """
        i = i-1 # convert to zero index loop / negative for no loop
        zs:List[Optional[Tensor]] = [None]*self.n_loops

        if i!=self.loop_index:
            if self.loop_index >= 0:
                # finalize loop
                mem = self.get_frames(self.n_context)
                # dataset of features, targets
                features = mem.unfold(0, self.n_context, 1) # time' x loop x latent x ctx
                # features = features.view(features.shape[0], -1) # time' x feat
                targets = mem[self.n_context:,i,:] # time' x 1 x latent
                assert targets.shape[0]==features.shape[0]
                # targets = targets.view(targets.shape[0], -1)
                self.fit_loop(self.loop_index, features, targets)
            # starting a new loop recording
            self.loop_index = i
            self.record_index = 0
        if i>=0:
            zs[i] = self.encode(x)

        # predictive context (same for all loops right now)
        feature = mem[-self.n_context:] # ctx x loop x latent
        feature = feature.permute(1,2,0).view(1,-1) # 1 x feature
        for j in range(self.n_loops):
            if i==j: continue
            # predict from a single feature
            zs[j] = self.eval_loop(j, features)

        # update memory
        self.record_frame(torch.cat(zs))

        y = self.decode(zs)

        # silence any loops which don't have a model
        mask = torch.cat(
            [torch.ones(1,1) if b else torch.zeros(1,1) for b in self.has_model])
        return y*mask


    def record_frame(self, zs):
        """
        zs: Tensor[loop, latent]
        """
        self.record_index = (self.record_index+1)%self.n_memory
        self.memory[self.record_index, :, :] = zs

    def get_frames(self, n:int):
        """
        get contiguous tensor out of ring memory
        """
        begin = max(0, self.record_index - n)
        block2 = self.memory[begin:self.record_index]
        remain = max(0, n - self.record_index)
        block1 = self.memory[-remain:]
        return torch.cat((block1, block2))

    def encode(self, x):
        """
        """
        # x = self.resample.from_target_sampling_rate(x)

        if self.rave.pqmf is not None:
            x = self.rave.pqmf(x)

        z, _ = self.rave.encoder(x)
        # z, std = self.post_process_distribution(mean, scale)

        # z = z - self.latent_mean.unsqueeze(-1)
        # z = nn.functional.conv1d(z, self.latent_pca.unsqueeze(-1))

        z = z[:, :self.n_latent]
        return z

    def decode(self, z):
        # if self.trained_cropped:  # PERFORM PCA BEFORE PADDING
        #     z = nn.functional.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        #     z = z + self.latent_mean.unsqueeze(-1)

        # CAT WITH SAMPLES FROM PRIOR DISTRIBUTION
        pad_size = self.rave_latent_size - self.n_latent
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

        x = self.rave.decoder(z)

        if self.rave.pqmf is not None:
            x = self.rave.pqmf.inverse(x)

        # x = self.resample.to_target_sampling_rate(x)

        return x

    def fit_loop(self, i:int, x, y):
        """
        Args:
            i: index of loop
            x: Tensor[batch, feature]
            y: Tensor[batch, target]
        Returns:

        """
        x = x.view(x.shape[0], -1)
        y = x.view(y.shape[0], -1)
        w = torch.linalg.lstsq(x, y).solution
        self.weights[i, :] = w

        self.has_model[i] = True

    def eval_loop(self, i:int, x):
        """
        Args:
            i: index of loop
            x: Tensor[batch, feature]
        Returns:
            y: Tensor[batch, target]
        """
        return x @ self.weights[i]
              


logging.info("loading RAVE model from checkpoint")

RUN = search_for_run(args.RUN)
logging.info(f"using {RUN}")
model = RAVE.load_from_checkpoint(RUN, strict=False).eval()

logging.info("flattening weights")
for m in model.modules():
    if hasattr(m, "weight_g"):
        nn.utils.remove_weight_norm(m)

logging.info("warmup forward pass")
x = torch.zeros(1, 1, 2**14)
if model.pqmf is not None:
    x = model.pqmf(x)

z, _ = model.reparametrize(*model.encoder(x))

y = model.decoder(z)

if model.pqmf is not None:
    y = model.pqmf.inverse(y)

model.discriminator = None

sr = model.sr

assert args.SR == sr, f"model sample rate is {sr}"
# if args.SR is not None:
#     target_sr = int(args.SR)
# else:
#     target_sr = sr

# logging.info("build resampling model")
# resample = Resampling(target_sr, sr)
# x = torch.zeros(1, 1, 2**14)
# resample.to_target_sampling_rate(resample.from_target_sampling_rate(x))

logging.info("creating looper")
looper = LivingLooper(model, args.LOOPS, args.CONTEXT, args.MEMORY)
# smoke test
looper(x)

logging.info("compiling torchscript")
looper = torch.jit.script(looper)

fname = f"ll_{args.NAME}.ts"
logging.info(f"saving '{fname}'")
model.save(fname)
