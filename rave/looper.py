import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from einops import rearrange

from time import time
import itertools as it

import cached_conv as cc

from .model import RAVE

class LivingLooper(nn.Module):#(RAVE):
    def __init__(self, n_feat, n_target):#*a, **kw):
        super().__init__()
        # super().__init__(*a, **kw)
        self.n_feat = n_feat
        self.n_target = n_target
        self.register_buffer('loop_params', torch.zeros(n_feat, n_target))

    def fit_loop(self, x, y):
        """
        Args:
            x: Tensor[batch, feature]
            y: Tensor[batch, target]
        Returns:

        """
        w = torch.linalg.lstsq(x, y).solution
        self.loop_params[:] = w

    def eval_loop(self, x):
        """
        Args:
            x: Tensor[batch, feature]
        Returns:
            y: Tensor[batch, target]
        """
        return x @ self.loop_params