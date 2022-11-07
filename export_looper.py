from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

import numpy as np

# import numpy as np
# from einops import rearrange

# from time import time
# import itertools as it

import cached_conv as cc
from rave.model import RAVE
from rave.core import search_for_run

from rave.weight_norm import remove_weight_norm

from effortless_config import Config

class args(Config):
    # run smoke tests before exporting
    TEST = 1
    # model checkpoint path
    RUN = None
    # audio sample rate -- currently must match RAVE model
    SR = None
    # should be true for realtime
    CACHED = True
    # number of latents to preserve from RAVE model
    LATENT_SIZE = None
    # maximum predictive context size
    CONTEXT = 24
    # maximum number of frames to fit model on
    FIT = 200
    # number of loops
    LOOPS = 3
    # max frames for loop memory
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
    trained_cropped:bool
    sampling_rate:int
    block_size:int
    n_memory:int

    loop_index:int
    record_index:int
    step:int

    latency_correct:int

    loop_length:List[int]
    loop_end_step:List[int]
    loop_context:List[int]

    # has_model:List[bool]

    def n_feature(self, n_context):
         return self.n_loops * n_context * self.n_latent

    def __init__(self, 
            rave_model:RAVE, 
            n_loops:int, 
            n_context:int, # maximum time dimension of model feature
            n_memory:int, # maximum loop memory 
            n_fit:int, # maximum dataset size to fit
            ):
        super().__init__()

        self.n_loops = n_loops
        self.max_n_context = n_context # now a maximum
        self.n_memory = n_memory
        self.n_fit = n_fit

        self.latency_correct = 4
        self.min_loop = 2

        self.loop_end_step = [0]*n_loops
        self.loop_length = [0]*n_loops
        self.loop_context = [self.max_n_context]*n_loops

        self.block_size = model.block_size()
        self.sampling_rate = model.hparams['sr']

        self.trained_cropped = bool(rave_model.cropped_latent_size)
        self.n_latent = (
            rave_model.cropped_latent_size 
            if self.trained_cropped 
            else rave_model.latent_size)

        self.max_n_feature = self.n_feature(self.max_n_context)

        self.pqmf = rave_model.pqmf
        self.encoder = rave_model.encoder
        self.decoder = rave_model.decoder
        self.n_latent_decoder = rave_model.latent_size

        self.register_buffer('weights', 
            torch.empty(self.n_loops, self.max_n_feature, self.n_latent, 
                requires_grad=False))
        self.register_buffer('center', 
            torch.empty(self.n_loops, self.max_n_feature, 
                requires_grad=False))
        self.register_buffer('bias', 
            torch.empty(self.n_loops, self.n_latent, 
                requires_grad=False))

        self.register_buffer('mask', 
            torch.empty(self.n_loops, 
                requires_grad=False))

        # continuously updated last N frames of memory
        self.register_buffer('memory', 
            torch.empty(n_memory, n_loops, self.n_latent,
                requires_grad=False))
        # long-term memory stored at record-end
        self.register_buffer('loop_memory', 
            torch.empty(n_loops, n_memory, n_loops, self.n_latent,
                requires_grad=False))

        self.reset()

    @torch.jit.export
    def reset(self):
        self.step = 0
        self.loop_index = -1
        self.record_index = 0

        for i in range(self.n_loops):
            self.loop_length[i] = 0
            self.loop_end_step[i] = 0

        self.memory.zero_()
        self.loop_memory.zero_()
        self.mask.zero_()
        self.weights.zero_()
        self.bias.zero_()
        self.center.zero_()

    def forward(self, i:int, x):
        """
        Args:
            i: loop record index, 0 for no loop, 1-index otherwise
            x: Tensor[1, sample]
        Returns:
            Tensor[loop, sample]
        """
        self.step += 1
        # return self.decode(self.encode(x)) ### DEBUG

        z = self.encode(x) # always encode for cache, even if result is not used

        if i > self.n_loops:
            i = 0
        i = i-1 # convert to zero index loop / negative for no loop
        zs = torch.empty(self.n_loops, self.n_latent)

        i_prev = self.loop_index
        # print(i, i_prev, self.loop_length)
        if i!=i_prev: # change in loop select control
            if i_prev >= 0: # previously on a loop
                if self.loop_length[i_prev] >= self.min_loop: # and it was long enough
                    # finalize loop
                    ll = self.loop_length[i_prev] = min(
                        self.n_memory, self.loop_length[i_prev])
                    mem = self.get_frames(ll)

                    lc = self.loop_context[i_prev] = min(
                        self.max_n_context, ll//2)
                    # train mem should be 
                    # .... ........
                    # 5678|12345678
                    # .... ........
                    # i.e. wrap the final n_context around
                    train_mem = torch.cat((mem[-lc:], mem),0)[:self.n_fit+lc]

                    # valid_mem = mem[self.n_context:]
                    # print(mem.shape)
                    # dataset of features, targets
                    features = train_mem.unfold(0, lc, 1)[:-1] # time' x loop x latent x ctx
                    features = features.reshape(features.shape[0], -1)
                    targets = train_mem[lc:,self.loop_index,:] # time' x 1 x latent
                    # assert targets.shape[0]==features.shape[0], f"""
                        # {targets.shape=}, {features.shape=}
                        # """
                    # make a copy of memory at the time of loop fitting
                    loop_mem = mem
                    # loop_mem = mem[self.n_context-self.latency_correct:-self.latency_correct]
                    self.loop_memory[i_prev, :loop_mem.shape[0]] = loop_mem
                    # targets = targets.view(targets.shape[0], -1)
                    print(f'fit {i_prev+1}')
                    self.fit_loop(i_prev, features, targets)
                    self.loop_end_step[i_prev] = self.step
            if i>=0: # starting a new loop recording
                self.loop_length[i] = 0
                # self.mask[i] = 0
            self.loop_index = i
                
        if i>=0:
            # print(x.shape)
            # slice on LHS to appease torchscript
            zs[i:i+1] = z[...,0] # remove time dim

        # predictive context (same for all loops right now)
        feature = self.get_frames(self.max_n_context) # ctx x loop x latent
        # print(f'{feature.shape=}')
        for j in range(self.n_loops):
            if i==j: continue
            # predict from a single feature
            # slice on LHS to appease torchscript
            zs[j:j+1] = self.eval_loop(j, feature)

        # print([f'{z.shape}' for z in zs])

        # update memory
        self.record_frame(zs)
        if self.loop_index >= 0:
            self.loop_length[i] += 1

        y = self.decode(zs[...,None]) # add time dim

        # this seems to have been causing some kind of memory fragmentation or leak?
        # mask = torch.cat(
        #     [torch.ones(1,1,1) if i==j or b else torch.zeros(1,1,1) 
        #     for j,b in enumerate(self.has_model)])
        # y = y * mask

        y = y * self.mask[:,None,None]

        # print(f'{self.loop_length}, {self.record_index}, {self.loop_index}')

        # return torch.stack((
        #     (torch.rand(self.block_size)-0.5)/3, 
        #     torch.zeros(self.block_size),
        #     (torch.arange(0,self.block_size)/128*2*np.pi).sin()/3 
        # ))[:,None]

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
        else:
            return self.memory[begin:self.record_index+1]

    def encode(self, x):
        """
        """
        if self.pqmf is not None:
            x = self.pqmf(x)

        z = self.encoder(x)[:,:self.n_latent]
        return z

    def decode(self, z):
        pad_size = self.n_latent_decoder - self.n_latent
        pad_latent = torch.randn(
            z.shape[0],
            pad_size,
            z.shape[-1],
            device=z.device,
        )

        z = torch.cat([z, pad_latent], 1)

        x = self.decoder(z)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x)

        return x

    def feat_process(self, i:int, x):
        return ((x - self.center[i, :x.shape[1]])/2).tanh()

    def fit_loop(self, i:int, feature, z):
        """
        Args:
            i: index of loop
            feature: Tensor[batch, feature]
            z: Tensor[batch, target]
        Returns:

        """
        feature = feature.view(feature.shape[0], -1)
        n_feature = feature.shape[1]

        z = z.view(z.shape[0], -1)
        
        # z = z.abs().pow(2) * z.sign()
        z = torch.where(
            z > 1, ((z+1)/2)**2, torch.where(
                z < -1, -((1-z)/2)**2, z))

        c = feature.mean(0)
        self.center[i, :n_feature] = c

        feature = self.feat_process(i, feature)

        b = z.mean(0)
        self.bias[i, :] = b

        z = z-b

        w = torch.linalg.lstsq(feature, z).solution
        self.weights[i, :n_feature] = w

        print(feature.shape, z.shape, w.shape)
        # print(x.norm(), y.norm(), w.norm())
        # print(y)

        self.mask[i] = 1.

    def eval_loop(self, i:int, feature):
        """
        Args:
            i: index of loop
            feature: Tensor[context, loop, latent]
        Returns:
            Tensor[batch, target]
        """
        ll = self.loop_length[i]
        lc = self.loop_context[i]
        feature = feature[-lc:].permute(1,2,0).reshape(1,-1) # 1 x feature
        n_feature = feature.numel()

        # # print(i, loop_len, self.mask[i])
        # if ll > 0:
        #     j = (self.step - self.loop_end_step[i] + self.latency_correct) % ll
        #     # j = (self.step - self.loop_end_step[i]) % ll
        #     loop_z = self.loop_memory[i, j, i]
        # else:
        #     j = 0
        #     loop_z = torch.zeros(self.n_latent)
        # # if j < self.n_context:
        #     # return loop_z

        z = self.feat_process(i, feature) @ self.weights[i,:n_feature] + self.bias[i]
        # z = z.abs().pow(1/2) * z.sign()
        z = torch.where(
            z > 1, 2*z**0.5 - 1, torch.where(
                z < -1, 1 - 2*(-z)**0.5, z))

        return z
        # # mix = float(j)/(loop_len-1)
        # mix = 1
        # # mix = min(1, (self.step - self.loop_end_step[i])/self.n_context)

        # return z*mix + loop_z*(1-mix)

              

logging.info("loading RAVE model from checkpoint")

RUN = search_for_run(args.RUN)
logging.info(f"using {RUN}")

# debug_kw = {}
debug_kw = {'script':False}#, 'cropped_latent_size':36, 'latent_size':128} ###DEBUG
# debug_kw = {'cropped_latent_size':8, 'latent_size':128} ###DEBUG

model = RAVE.load_from_checkpoint(RUN, **debug_kw, strict=False).eval()

if args.LATENT_SIZE is not None:
    model.crop_latent_space(int(args.LATENT_SIZE))

logging.info("flattening weights")
for m in model.modules():
    if hasattr(m, "weight_g"):
        remove_weight_norm(m)

logging.info("warmup forward pass")
x = torch.zeros(1, 1, 2**14)
if model.pqmf is not None:
    x = model.pqmf(x)

z, _ = model.reparametrize(*model.split_params(model.encoder(x)))
z = model.pad_latent(z)

y = model.decoder(z)

if model.pqmf is not None:
    y = model.pqmf.inverse(y)

model.discriminator = None

sr = model.sr

# print(model.encoder.net[-1])
# print(model.encoder.net[-1].cache)
# print(model.encoder.net[-1].cache.initialized)
# print(model.encoder.net[-1].cache.pad.shape)

assert int(args.SR) == sr, f"model sample rate is {sr}"

logging.info("creating looper")
# ls = None if args.LATENT_SIZE is None else int(args.LATENT_SIZE)
looper = LivingLooper(model, args.LOOPS, args.CONTEXT, args.MEMORY, args.FIT)
looper.eval()

# smoke test
def feed(i):
    x = torch.zeros(1, 1, 2**11)
    looper(i, x)

def smoke_test():
    looper.reset()
    feed(0)
    for _ in range(10):
        feed(1)
    for _ in range(args.CONTEXT+3):
        feed(2)
    for _ in range(10):
        feed(0)

    if args.TEST <= 1: return
    for _ in range(args.MEMORY+3):
        feed(3)
    feed(0)

logging.info("smoke test with pytorch")
if args.TEST > 0:
    smoke_test()

looper.reset()

logging.info("compiling torchscript")
looper = torch.jit.script(looper)

logging.info("smoke test with torchscript")
if args.TEST > 0:
    smoke_test()

fname = f"ll_{args.NAME}.ts"
logging.info(f"saving '{fname}'")
looper.save(fname)
