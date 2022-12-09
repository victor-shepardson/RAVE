import torch
import torch.nn as nn
from torch import Tensor
from typing import List
# import torch.nn.utils.weight_norm as wn
# from torch.nn.utils import remove_weight_norm
from .weight_norm import weight_norm as wn
from .weight_norm import remove_weight_norm
import numpy as np
import pytorch_lightning as pl
from .core import multiscale_stft, Loudness, mod_sigmoid
from .core import amp_to_impulse_response, fft_convolve, get_beta_kl_cyclic_annealed, get_beta_kl
from .pqmf import CachedPQMF as PQMF
from sklearn.decomposition import PCA
from einops import rearrange

from time import time
import itertools as it

import cached_conv as cc


class Profiler:
    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep


class Residual(nn.Module):
    def __init__(self, module, cumulative_delay=0):
        super().__init__()
        additional_delay = module.cumulative_delay
        self.aligned = cc.AlignBranches(
            module,
            nn.Identity(),
            delays=[additional_delay, 0],
        )
        self.cumulative_delay = additional_delay + cumulative_delay

    def forward(self, x):
        x_net, x_res = self.aligned(x)
        return x_net + x_res


class ResidualStack(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size,
                 padding_mode,
                 cumulative_delay=0,
                 bias=False,
                 depth=3,
                 boom=2,
                 script=False):
        super().__init__()
        net = []

        maybe_script = torch.jit.script if script else lambda _:_

        res_cum_delay = 0
        # SEQUENTIAL RESIDUALS
        for i in range(depth):
            # RESIDUAL BLOCK
            seq = [nn.LeakyReLU(.2)]
            seq.append(
                wn(
                    cc.Conv1d(
                        dim,
                        dim*boom,
                        kernel_size,
                        padding=cc.get_padding(
                            kernel_size,
                            dilation=3**i,
                            mode=padding_mode,
                        ),
                        dilation=3**i,
                        bias=bias,
                        groups=dim//16,
                    )))

            seq.append(nn.LeakyReLU(.2))
            seq.append(
                wn(
                    cc.Conv1d(
                        dim*boom,
                        dim,
                        1,
                        # padding=cc.get_padding(kernel_size, mode=padding_mode),
                        bias=bias,
                        cumulative_delay=seq[-2].cumulative_delay,
                    )))

            res_net = cc.CachedSequential(*seq)

            net.append(Residual(res_net, cumulative_delay=res_cum_delay))
            res_cum_delay = net[-1].cumulative_delay

        self.net = maybe_script(cc.CachedSequential(*net))
        # self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay

    def forward(self, x):
        return self.net(x)


class UpsampleLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 ratio,
                 padding_mode,
                 cumulative_delay=0,
                 bias=False):
        super().__init__()
        net = []#[nn.LeakyReLU(.2)]
        if ratio > 1:
            net.append(
                wn(
                    cc.ConvTranspose1d(
                        in_dim,
                        out_dim,
                        2 * ratio,
                        stride=ratio,
                        padding=ratio // 2,
                        bias=bias,
                    )))
        else:
            net.append(
                wn(
                    cc.Conv1d(
                        in_dim,
                        out_dim,
                        3,
                        padding=cc.get_padding(3, mode=padding_mode),
                        bias=bias,
                    )))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

    def forward(self, x):
        return self.net(x)


class NoiseGenerator(nn.Module):
    def __init__(self, in_size, data_size, ratios, noise_bands, padding_mode):
        super().__init__()
        net = []
        channels = [in_size] * len(ratios) + [data_size * noise_bands]
        cum_delay = 0
        for i, r in enumerate(ratios):
            net.append(
                cc.Conv1d(
                    channels[i],
                    channels[i + 1],
                    3,
                    padding=cc.get_padding(3, r, mode=padding_mode),
                    stride=r,
                    cumulative_delay=cum_delay,
                ))
            cum_delay = net[-1].cumulative_delay
            if i != len(ratios) - 1:
                net.append(nn.LeakyReLU(.2))

        self.net = cc.CachedSequential(*net)
        self.data_size = data_size
        self.cumulative_delay = self.net.cumulative_delay * int(
            np.prod(ratios))

        self.register_buffer(
            "target_size",
            torch.tensor(np.prod(ratios)).long(),
        )

    def forward(self, x):
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(amp.shape[0], amp.shape[1], self.data_size, -1)

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        return noise


class Generator(nn.Module):
    def __init__(self,
                 latent_size,
                 capacity,
                 boom,
                 data_size,
                 ratios,
                 narrow,
                 loud_stride,
                 noise_ratios,
                 noise_bands,
                 padding_mode,
                 bias=False,
                 script=True
            ):
        super().__init__()

        # maybe_script = torch.jit.script if script else lambda _:_

        out_dim = int(np.prod(ratios) * capacity // np.prod(narrow))

        net = [
            wn(
                cc.Conv1d(
                    latent_size,
                    out_dim,
                    3,
                    padding=cc.get_padding(3, mode=padding_mode),
                    bias=bias,
                ))
        ]

        net.append(nn.LeakyReLU(0.2))

        for i,(r, n) in enumerate(zip(ratios, narrow)):
            in_dim = out_dim
            out_dim = out_dim * n // r

            net.append(
                UpsampleLayer(
                    in_dim,
                    out_dim,
                    r,
                    padding_mode,
                    cumulative_delay=net[-2 if i==0 else -1].cumulative_delay,
                ))
            net.append(
                ResidualStack(
                    out_dim,
                    3,
                    padding_mode,
                    cumulative_delay=net[-1].cumulative_delay,
                    boom=boom,
                    script=script
                ))

        # self.net = maybe_script(cc.CachedSequential(*net))
        self.net =cc.CachedSequential(*net)

        wave_gen = wn(
            cc.Conv1d(
                out_dim,
                data_size,
                7,
                padding=cc.get_padding(7, mode=padding_mode),
                bias=bias,
            ))

        loud_gen = wn(
            cc.Conv1d(
                out_dim,
                1,
                2 * loud_stride + 1,
                stride=loud_stride,
                padding=cc.get_padding(2 * loud_stride + 1,
                                       loud_stride,
                                       mode=padding_mode),
                bias=bias,
            ))

        branches = [wave_gen, loud_gen]

        if noise_bands > 0:
            self.has_noise_branch = True
            noise_gen = NoiseGenerator(
                out_dim,
                data_size,
                noise_ratios,
                noise_bands,
                padding_mode=padding_mode,
            )
            branches.append(noise_gen)
        else:
            self.has_noise_branch = False

        self.synth = cc.AlignBranches(
            *branches,
            cumulative_delay=self.net.cumulative_delay,
        )

        self.loud_stride = loud_stride
        self.cumulative_delay = self.synth.cumulative_delay

    def forward(self, x, add_noise: bool = True):
        x = self.net(x)

        if self.has_noise_branch:
            waveform, loudness, noise = self.synth(x)
        else:
            waveform, loudness = self.synth(x)
            noise = torch.zeros_like(waveform)

        loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform) * mod_sigmoid(loudness)

        if add_noise:
            waveform = waveform + noise

        return waveform

class LayerNorm1d(nn.Module):
    def forward(self, x):
        x = x - x.mean(1, keepdim=True)
        return x / (1e-5 + torch.linalg.vector_norm(x, dim=1, keepdim=True))

class Encoder(nn.Module):
    def __init__(self,
                 data_size,
                 capacity,
                 boom,
                 latent_size,
                 ratios,
                 narrow,
                 padding_mode,
                 norm=None,
                 bias=False,
                 script=True,
                 ):
        super().__init__()
        maybe_wn = (lambda x:x) if norm=='batch' else wn

        # maybe_script = torch.jit.script if script else lambda _:_

        out_dim = capacity

        # try a longer kernel here?
        # and maybe a rectifying activation?
        net = [
            maybe_wn(cc.Conv1d(
                data_size,
                out_dim,
                7,
                padding=cc.get_padding(7, mode=padding_mode),
                bias=bias))
            ]

        if norm=='batch':
            norm = lambda d: nn.BatchNorm1d(d)
        elif norm=='instance':
            norm = lambda d: nn.InstanceNorm1d(d, track_running_stats=True)
        elif norm=='layer':
            norm = lambda d: LayerNorm1d()

        latent_params = 2 # mean, scale

        for i,(r, n) in enumerate(zip(ratios, narrow)):
            in_dim = out_dim
            out_dim = out_dim * r // n

            if norm is not None:
                net.append(norm(in_dim))
            prev_layer_idx = -1
            if norm is not None:
                prev_layer_idx -= 1
            if i>0:
                prev_layer_idx -= 1
            net.append(
                ResidualStack(
                    in_dim,
                    3,
                    padding_mode,
                    # cumulative_delay=net[-2 if norm is not None else -1].cumulative_delay,
                    cumulative_delay=net[prev_layer_idx].cumulative_delay,
                    depth=1,
                    boom=boom,
                    script=script
                ))
            net.append(maybe_wn(
                cc.Conv1d(
                    in_dim,
                    out_dim, 
                    2 * r + 1,
                    padding=cc.get_padding(2 * r + 1, r, mode=padding_mode),
                    # stride=r,
                    bias=bias,
                    cumulative_delay=net[-1].cumulative_delay,
                )))
            # net.append(nn.AvgPool1d(r,r))
            net.append(nn.MaxPool1d(r,r))

            
        net.append(nn.LeakyReLU(0.2)) 

        net.append(maybe_wn(
            cc.Conv1d(
                out_dim,
                latent_size * latent_params,
                3,
                padding=cc.get_padding(3, mode=padding_mode),
                groups=latent_params,
                bias=bias,
                # cumulative_delay=net[-2].cumulative_delay,
                cumulative_delay=net[-3].cumulative_delay,
            )))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay
        
    def forward(self, x, double:bool=False):
        z = self.net(x)
        # duplicate along batch dimension
        if double:
            # z = z.repeat(2, *(1,)*(z.ndim-1))
            z = z.repeat(2, 1, 1)
        # split into mean, scale parameters along channel dimension
        # if use_scale:
            # return torch.split(z, z.shape[1] // 2, 1)
            # return z.chunk(2, 1)
        # else:
            # return z, None
        return z

# class Encoder(nn.Module):
#     def __init__(self,
#                  data_size,
#                  capacity,
#                  latent_size,
#                  ratios,
#                  padding_mode,
#                  use_bn,
#                  bias=False):
#         super().__init__()
#         maybe_wn = (lambda x:x) if use_bn else wn

#         net = [
#             maybe_wn(cc.Conv1d(
#                 data_size,
#                 capacity,
#                 7,
#                 padding=cc.get_padding(7, mode=padding_mode),
#                 bias=bias))
#             ]

#         for i, r in enumerate(ratios):
#             in_dim = 2**i * capacity
#             out_dim = 2**(i + 1) * capacity

#             if use_bn:
#                 net.append(nn.BatchNorm1d(in_dim))
#             net.append(nn.LeakyReLU(.2))
#             net.append(maybe_wn(
#                 cc.Conv1d(
#                     in_dim,
#                     out_dim,
#                     2 * r + 1,
#                     padding=cc.get_padding(2 * r + 1, r, mode=padding_mode),
#                     stride=r,
#                     bias=bias,
#                     cumulative_delay=net[-3 if use_bn else -2].cumulative_delay,
#                 )))

#         net.append(nn.LeakyReLU(.2))
#         net.append(maybe_wn(
#             cc.Conv1d(
#                 out_dim,
#                 2 * latent_size,
#                 5,
#                 padding=cc.get_padding(5, mode=padding_mode),
#                 groups=2,
#                 bias=bias,
#                 cumulative_delay=net[-2].cumulative_delay,
#             )))

#         self.net = cc.CachedSequential(*net)
#         self.cumulative_delay = self.net.cumulative_delay

#     def forward(self, x, double=False):
#         z = self.net(x)
#         # duplicate along batch dimension
#         if double:
#             z = z.repeat(2, *(1,)*(z.ndim-1))
#         # split into mean, scale parameters along channel dimension
#         return torch.split(z, z.shape[1] // 2, 1)


class Discriminator(nn.Module):
    def __init__(self, in_size, capacity, multiplier, n_layers):
        super().__init__()

        out_size = capacity

        net = [
            wn(cc.Conv1d(in_size, out_size, 15, padding=cc.get_padding(15)))
        ]

        for i in range(n_layers):
            in_size = out_size
            out_size = min(1024, in_size*multiplier)

            net.append(nn.Sequential(
                nn.LeakyReLU(.2),
                wn(
                    cc.Conv1d(
                        in_size,
                        out_size,
                        41,
                        stride=multiplier,
                        padding=cc.get_padding(41, multiplier),
                        groups=out_size//capacity
                    ))))

        net.append(nn.Sequential(
            nn.LeakyReLU(.2),
            wn(
                cc.Conv1d(
                    out_size,
                    out_size,
                    5,
                    padding=cc.get_padding(5),
                ))))

        net.append(nn.Sequential(
            nn.LeakyReLU(.2),
            wn(cc.Conv1d(out_size, 1, 1))))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        feature:List[Tensor] = []
        for layer in self.net:
            x = layer(x)
            feature.append(x)
        return feature

# class Discriminator(nn.Module):
#     def __init__(self, in_size, capacity, multiplier, n_layers):
#         super().__init__()

#         out_size = capacity

#         net = [
#             wn(cc.Conv1d(in_size, out_size, 15, padding=cc.get_padding(15)))
#         ]
#         net.append(nn.LeakyReLU(.2))

#         for i in range(n_layers):
#             in_size = out_size
#             out_size = min(1024, in_size*multiplier)

#             net.append(
#                 wn(
#                     cc.Conv1d(
#                         in_size,
#                         out_size,
#                         41,
#                         stride=multiplier,
#                         padding=cc.get_padding(41, multiplier),
#                         groups=out_size//capacity
#                     )))
#             net.append(nn.LeakyReLU(.2))

#         net.append(
#             wn(
#                 cc.Conv1d(
#                     out_size,
#                     out_size,
#                     5,
#                     padding=cc.get_padding(5),
#                 )))
#         net.append(nn.LeakyReLU(.2))
#         net.append(
#             wn(cc.Conv1d(out_size, 1, 1)))
#         self.net = nn.ModuleList(net)

#     def forward(self, x):
#         feature = []
#         for layer in self.net:
#             x = layer(x)
#             if isinstance(layer, nn.Conv1d):
#                 feature.append(x)
#         return feature


class StackDiscriminators(nn.Module):
    def __init__(self, n_dis, *args, factor=2, **kwargs):
        super().__init__()
        self.factor = factor
        self.discriminators = nn.ModuleList(
            [Discriminator(*args, **kwargs) for i in range(n_dis)], )

    def forward(self, x):
        features:List[List[Tensor]] = []
        for layer in self.discriminators:
            features.append(layer(x))
            x = nn.functional.avg_pool1d(x, self.factor)
        return features


class RAVE(pl.LightningModule):
    def __init__(self,
                 data_size,
                 capacity,
                 boom,
                 latent_size,
                 ratios,
                 narrow,
                 bias,
                 loud_stride,
                 use_noise,
                 noise_ratios,
                 noise_bands,
                 d_capacity,
                 d_multiplier,
                 d_n_layers,
                 d_stack_factor,
                 pair_discriminator,
                 ged,
                 adversarial_loss,
                 freeze_encoder,
                 warmup,
                #  kl_cycle,
                 mode,
                 encoder_norm=None,
                 no_latency=False,
                 min_kl=1e-4,
                 max_kl=5e-1,
                 sample_kl=False,
                 path_derivative=False,
                 cropped_latent_size=0,
                 feature_match=True,
                 sr=24000,
                 gen_lr=1e-4,
                 dis_lr=1e-4,
                 gen_adam_betas=(0.5,0.9),
                 dis_adam_betas=(0.5,0.9),
                 grad_clip=None,
                 script=True
                ):
        super().__init__()
        self.save_hyperparameters()

        maybe_script = torch.jit.script if script else lambda _:_

        if data_size == 1:
            self.pqmf = None
        else:
            self.pqmf = PQMF(70 if no_latency else 100, data_size)

        self.loudness = Loudness(sr, 512)

        self.encoder = Encoder(
            data_size,
            capacity,
            boom,
            latent_size,
            ratios,
            narrow,
            "causal" if no_latency else "centered",
            encoder_norm,
            bias,
            script,
        )
        self.decoder = Generator(
            latent_size,
            capacity,
            boom,
            data_size,
            list(reversed(ratios)),
            list(reversed(narrow)),
            loud_stride,
            noise_ratios,
            noise_bands,
            "causal" if no_latency else "centered",
            bias,
            script,
        )

        # print('encoder')
        # for n,p in self.encoder.named_parameters():
        #     print(f'{n}: {p.numel()}')
        # print('generator')
        # for n,p in self.decoder.named_parameters():
        #     print(f'{n}: {p.numel()}')

        if adversarial_loss or feature_match:
            self.discriminator = maybe_script(StackDiscriminators(
                3, factor=d_stack_factor,
                in_size=2 if pair_discriminator else 1,
                capacity=d_capacity,
                multiplier=d_multiplier,
                n_layers=d_n_layers
                ))
        else:
            self.discriminator = None

        self.idx = 0

        # self.register_buffer("latent_pca", torch.eye(latent_size))
        # self.register_buffer("latent_mean", torch.zeros(latent_size))
        # self.register_buffer("fidelity", torch.zeros(latent_size))

        # # this will track the most informative dimensions of latent space
        # # by KLD, in descending order, computed at each validation step
        # self.register_buffer("kld_idxs", 
        #     torch.zeros(latent_size, dtype=torch.long))

        self.latent_size = latent_size


        # tell lightning we are doing manual optimization
        self.automatic_optimization = False 

        self.sr = sr
        self.mode = mode

        self.min_kl = min_kl
        self.max_kl = max_kl
        self.cropped_latent_size = cropped_latent_size

        self.feature_match = feature_match

        self.register_buffer("saved_step", torch.tensor(0))

        self.init_buffers()

        if cropped_latent_size:
            self.crop_latent_space(cropped_latent_size)

    def init_buffers(self):
        self.register_buffer("latent_pca", torch.eye(self.latent_size))
        self.register_buffer("latent_mean", torch.zeros(self.latent_size))
        self.register_buffer("fidelity", torch.zeros(self.latent_size))
        # this will track the most informative dimensions of latent space
        # by KLD, in descending order, computed at each validation step
        self.register_buffer("kld_idxs", 
            torch.zeros(self.latent_size, dtype=torch.long))

    def configure_optimizers(self):
        gen_p = list(self.encoder.parameters())
        gen_p += list(self.decoder.parameters())
        gen_opt = torch.optim.Adam(
            gen_p, self.hparams['gen_lr'], self.hparams['gen_adam_betas'])

        if self.discriminator is None:
            return gen_opt

        dis_p = list(self.discriminator.parameters())
        dis_opt = torch.optim.Adam(
            dis_p, self.hparams['dis_lr'], self.hparams['dis_adam_betas'])

        return gen_opt, dis_opt

    def lin_distance(self, x, y):
        # is the norm across batch items (and bands...) a problem here?
        # return torch.norm(x - y)
        # return torch.linalg.vector_norm(x - y, dim=(-1,-2,-3)).mean()
        return torch.linalg.vector_norm(x - y, dim=tuple(range(1, x.ndim))).mean()

    def norm_lin_distance(self, x, y):
        # return torch.norm(x - y) / torch.norm(x)
        # norm = lambda z: torch.linalg.vector_norm(z, dim=(-1,-2,-3))
        norm = lambda z: torch.linalg.vector_norm(z, dim=tuple(range(1, z.ndim)))
        return (norm(x - y) / norm(x)).mean()

    def log_distance(self, x, y):
        return abs(torch.log(x + 1e-7) - torch.log(y + 1e-7)).mean()

    def distance(self, x, y, y2=None):
        """
        multiscale log + lin spectrogram distance. if y2 is supplied,
        compute the GED: d(x,y) + d(x,y2) - d(y,y2).
        """
        scales = [2048, 1024, 512, 256, 128]
        parts = [x,y]
        if y2 is not None:
            parts.append(y2)
        # batch through the stft
        stfts = multiscale_stft(torch.cat(parts), scales, .75)
        if y2 is None:
            x, y = zip(*(s.chunk(2) for s in stfts))
            lin = sum(map(self.lin_distance, x, y))
            log = sum(map(self.log_distance, x, y))
        else:
            x, y, y2 = zip(*(s.chunk(3) for s in stfts))
            # print([s.shape for s in x])
            lin = (
                sum(map(self.lin_distance, x, y))
                + sum(map(self.lin_distance, x, y2))
                - sum(map(self.lin_distance, y, y2)))
            log = (
                sum(map(self.log_distance, x, y))
                + sum(map(self.log_distance, x, y2))
                - sum(map(self.log_distance, y, y2)))

        return lin + log

    def reparametrize(self, mean, scale):

        if self.cropped_latent_size > 0:
            z = mean
            kl = None
        else:
            if scale is None:
                raise ValueError("""
                    in `reparametrize`:
                    `scale` should not be None while `self.cropped_latent_size` is 0
                """)
            if self.hparams['sample_kl']:
                log_std = scale.clamp(-50, 3)
                u = torch.randn_like(mean)
                z = u * log_std.exp() + mean
                if self.hparams['path_derivative']:
                    log_std = log_std.detach()
                    mean = mean.detach()
                    _u = (z - mean) / log_std.exp()
                    kl = (0.5*(z*z - _u*_u) - log_std).mean((0, 2))
                else:
                    kl = (0.5*(z*z - u*u) - log_std).mean((0, 2))
            else:
                std = nn.functional.softplus(scale) + 1e-4
                var = std * std
                logvar = torch.log(var)
                z = torch.randn_like(mean) * std + mean
                kl = 0.5 * (mean * mean + var - logvar - 1).mean((0, 2))

        return z, kl

    def adversarial_combine(self, score_real, score_fake, mode="hinge"):
        if mode == "hinge":
            loss_dis = torch.relu(1 - score_real).mean() + torch.relu(1 + score_fake).mean()
            loss_gen = -score_fake.mean()
        elif mode == "square":
            loss_dis = (score_real - 1).pow(2).mean() + score_fake.pow(2).mean()
            loss_gen = (score_fake - 1).pow(2).mean()
        elif mode == "nonsaturating":
            score_real = torch.clamp(torch.sigmoid(score_real), 1e-7, 1 - 1e-7)
            score_fake = torch.clamp(torch.sigmoid(score_fake), 1e-7, 1 - 1e-7)
            loss_dis = -(torch.log(score_real).mean() +
                         torch.log(1 - score_fake).mean())
            loss_gen = -torch.log(score_fake).mean()
        else:
            raise NotImplementedError
        return loss_dis, loss_gen    

    def pad_latent(self, z):
        if self.cropped_latent_size:
            # print(f"""
            # {self.latent_size=}, {self.cropped_latent_size=}, {z.shape=}
            # """)
            noise = torch.randn(
                z.shape[0],
                self.latent_size - self.cropped_latent_size,
                z.shape[2],
                device=z.device,
            )
            z = torch.cat([z, noise], 1)    
        return z

    def training_step(self, batch, batch_idx):
        p = Profiler()
        self.saved_step += 1

        x = batch['source'].unsqueeze(1)
        target = batch['target'].unsqueeze(1)

        if self.pqmf is not None:  # MULTIBAND DECOMPOSITION
            x = self.pqmf(x)
            target = self.pqmf(target)
            p.tick("pqmf")

        # GED reconstruction and pair discriminator both require
        # two z samples per input
        use_pairs = self.hparams['pair_discriminator']
        use_ged = self.hparams['ged']
        use_discriminator = self.hparams['adversarial_loss'] or self.hparams['feature_match']
        freeze_encoder = self.hparams['freeze_encoder']
        double_z = use_ged or (use_pairs and use_discriminator)

        # ENCODE INPUT
        if freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                z, kl = self.reparametrize(*self.split_params(self.encoder(x, double=double_z)))
        else:
            z, kl = self.reparametrize(*self.split_params(self.encoder(x, double=double_z)))
        kl = kl.sum() if kl is not None else 0

        z = self.pad_latent(z)

        # DECODE LATENT
        y = self.decoder(z, add_noise=self.hparams['use_noise'])

        if double_z:
            y, y2 = y.chunk(2)
        else:
            y2 = None
        p.tick("decode")

        # DISTANCE BETWEEN INPUT AND OUTPUT
        distance = self.distance(target, y, y2 if use_ged else None)
        p.tick("mb distance")

        if self.pqmf is not None:  # FULL BAND RECOMPOSITION
            # why run inverse pqmf on x instead of
            # saving original audio?
            # some trimming edge case?
            target = self.pqmf.inverse(target)
            y = self.pqmf.inverse(y)
            if y2 is not None:
                y2 = self.pqmf.inverse(y2)
            distance = distance + self.distance(target, y, y2 if use_ged else None)
            p.tick("fb distance")

        if use_ged:
            loud_x, loud_y, loud_y2 = self.loudness(torch.cat((target,y,y2))).chunk(3)
            loud_dist = (
                (loud_x - loud_y).pow(2).mean()
                + (loud_x - loud_y2).pow(2).mean() 
                - (loud_y2 - loud_y).pow(2).mean()) 
        else:
            loud_x, loud_y = self.loudness(torch.cat((target,y))).chunk(2)
            loud_dist = (loud_x - loud_y).pow(2).mean()

        distance = distance + loud_dist
        p.tick("loudness distance")

        feature_matching_distance = 0.
        if use_discriminator:  # DISCRIMINATION
            # note -- could run x and target both through discriminator here
            # shouldn't matter which one is used (?)
            if use_pairs and not use_ged:
                real = torch.cat((target, y), -2)
                fake = torch.cat((y2, y), -2)
                to_disc = torch.cat((real, fake))
            if use_pairs and use_ged:
                real = torch.cat((target, y), -2)
                fake = torch.cat((y2, y), -2)
                fake2 = torch.cat((y, y2), -2)
                to_disc = torch.cat((real, fake, fake2))
            if not use_pairs and use_ged:
                to_disc = torch.cat((target, y, y2))
            if not use_pairs and not use_ged:
                to_disc = torch.cat((target, y))
            discs_features = self.discriminator(to_disc)

            # all but final layer in each parallel discriminator
            # sum is doing list concatenation here
            feature_maps = sum([d[:-1] for d in discs_features], start=[])
            # final layers
            scores = [d[-1] for d in discs_features]

            loss_dis = 0
            loss_adv = 0
            pred_true = 0
            pred_fake = 0

            # loop over parallel discriminators at 3 scales 1, 1/2, 1/4
            for s in scores:
                if use_ged:
                    real, fake = s.split((s.shape[0]//3, s.shape[0]*2//3))
                else:
                    real, fake = s.chunk(2)
                _dis, _adv = self.adversarial_combine(real, fake, mode=self.mode)
                loss_dis = loss_dis + _dis
                loss_adv = loss_adv + _adv
                pred_true = pred_true + real.mean()
                pred_fake = pred_fake + fake.mean()

            if self.feature_match:
                if use_ged:
                    def dist(fm):
                        real, fake, fake2 = fm.chunk(3)
                        return (
                            (real-fake).abs().mean()
                            + (real-fake2).abs().mean()
                            - (fake-fake2).abs().mean()
                        )
                else:
                    def dist(fm):
                        real, fake = fm.chunk(2)
                        return (real-fake).abs().mean()
                feature_matching_distance = 10*sum(
                    map(dist, feature_maps)) / len(feature_maps)

        else:
            pred_true = x.new_zeros(1)
            pred_fake = x.new_zeros(1)
            loss_dis = x.new_zeros(1)
            loss_adv = x.new_zeros(1)

        # COMPOSE GEN LOSS
        # beta = get_beta_kl_cyclic_annealed(
        #     step=self.global_step,
        #     cycle_size=self.hparams['kl_cycle'],
        #     warmup=self.hparams['warmup'],
        #     min_beta=self.min_kl,
        #     max_beta=self.max_kl,
        # )
        beta = get_beta_kl(
            self.global_step, self.hparams['warmup'], self.min_kl, self.max_kl)
        loss_kld = beta * kl

        loss_gen = distance + loss_kld
        if self.hparams['adversarial_loss']:
            loss_gen = loss_gen + loss_adv
        if self.feature_match:
            loss_gen = loss_gen + feature_matching_distance
        p.tick("gen loss compose")

        # OPTIMIZATION
        is_disc_step = self.global_step % 2 and use_discriminator
        grad_clip = self.hparams['grad_clip']

        if use_discriminator:
            gen_opt, dis_opt = self.optimizers()
        else:
            gen_opt = self.optimizers()

        if is_disc_step:
            dis_opt.zero_grad()
            loss_dis.backward()

            if grad_clip is not None:
                dis_grad = nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), grad_clip)
                self.log('grad_norm_discriminator', dis_grad)

            dis_opt.step()
        else:
            gen_opt.zero_grad()
            loss_gen.backward()

            if grad_clip is not None:
                if not freeze_encoder:
                    enc_grad = nn.utils.clip_grad_norm_(
                        self.encoder.parameters(), grad_clip)
                    self.log('grad_norm_encoder', enc_grad)
                dec_grad = nn.utils.clip_grad_norm_(
                    self.decoder.parameters(), grad_clip)
                self.log('grad_norm_generator', dec_grad)

            gen_opt.step()
               
        p.tick("optimization")

        # LOGGING
        # total generator loss
        self.log("loss_generator", loss_gen)

        # KLD loss (KLD in nats per z * beta)
        self.log("loss_kld", loss_kld)
        # spectral + loudness distance loss
        self.log("loss_distance", distance)
        # loudness distance loss
        self.log("loss_loudness", loud_dist)

        # KLD in bits per second
        self.log("kld_bps", self.npz_to_bps(kl))
        # beta-VAE parameter
        self.log("beta", beta)

        if use_discriminator:
            # total discriminator loss
            self.log("loss_discriminator", loss_dis)
            self.log("pred_true", pred_true.mean())
            self.log("pred_fake", pred_fake.mean())
            # adversarial loss
            self.log("loss_adversarial", loss_adv)
            # feature-matching loss
            self.log("loss_feature_matching", feature_matching_distance)

        p.tick("log")

        # print(p)

    def split_params(self, p):
        if self.cropped_latent_size > 0:
            return p, None
        return p.chunk(2,1)

    def encode(self, x):
        if self.pqmf is not None:
            x = self.pqmf(x)

        params = self.encoder(x)

        z, _ = self.reparametrize(*self.split_params(params))
        return z

    def decode(self, z):
        z = self.pad_latent(z)

        y = self.decoder(z, add_noise=True)
        if self.pqmf is not None:
            y = self.pqmf.inverse(y)
        return y

    def validation_step(self, batch, batch_idx, loader_idx):
            
        x = batch['source'].unsqueeze(1)
        target = batch['target'].unsqueeze(1)

        if self.pqmf is not None:
            x = self.pqmf(x)
            target = self.pqmf(target)

        p = self.encoder(x)
        mean, scale = self.split_params(p)

        if loader_idx > 0:
            # z = mean
            z = torch.cat((
                mean, 
                mean + torch.randn((*mean.shape[:2], 1), device=mean.device)/2),
                0)
            _, kl = self.reparametrize(mean, scale)
        else:
            z, kl = self.reparametrize(mean, scale)

        z = self.pad_latent(z)

        y = self.decoder(z, add_noise=self.hparams['use_noise'])
        # print(x.shape, z.shape, y.shape)


        if self.pqmf is not None:
            x = self.pqmf.inverse(x)
            target = self.pqmf.inverse(target)
            y = self.pqmf.inverse(y)

        if loader_idx > 0:
            y, y2 = y.chunk(2, 0)

        distance = self.distance(target, y)
        baseline_distance = self.distance(target, x)

        if self.trainer is not None:
            # if loader_idx==0:
            # full-band distance only,
            # in contrast to training distance
            # KLD in bits per second
            self.log("valid_distance", distance)
            self.log("valid_distance/baseline", baseline_distance)
            if kl is not None:
                self.log("valid_kld_bps", self.npz_to_bps(kl.sum()))
        

        if loader_idx==0:
            return torch.cat([target, y], -1), mean, kl
        if loader_idx>0:
            return torch.cat([target, y, y2], -1), mean, None

    def block_size(self):
        return np.prod(self.hparams['ratios']) * self.hparams['data_size'] 

    def npz_to_bps(self, npz):
        """convert nats per z frame to bits per second"""
        return (npz 
            * self.hparams['sr'] / self.block_size() 
            * np.log2(np.e))      

    def crop_latent_space(self, n, decoder_latent_size=0):

        # if latent_mean is used in the PCA transform,
        # there will be some error due to zero padding,
        # (when the first decoder layer has k>1 anyway)
        use_mean = False

        # # get test value
        # x = torch.randn(1,self.hparams['data_size'],self.block_size())
        # z, _ = self.split_params(self.encoder(x))
        # # y = self.decoder(z, add_noise=self.hparams['use_noise'])
        # y = self.decoder.net[0](z)
        # # y_perturb = self.decoder(z+torch.randn_like(z)/3, add_noise=self.hparams['use_noise'])
        # y_perturb = self.decoder.net[0](z+torch.randn_like(z)/3)

        # with PCA:
        pca = self.latent_pca[:n]
        # w: (out, in, kernel)
        # b: (out)
        layer_in = self.encoder.net[-1]
        layer_prev = self.encoder.net[-4]
        if hasattr(layer_in, "weight_g"):
            remove_weight_norm(layer_in)
        if hasattr(layer_prev, "weight_g"):
            remove_weight_norm(layer_prev)
        layer_out = self.decoder.net[0]
        if hasattr(layer_out, "weight_g"):
            remove_weight_norm(layer_out)
        # project and prune the final encoder layers
        W, b = layer_in.weight, layer_in.bias
        Wp, bp = layer_prev.weight, layer_prev.bias

        # remove the scale parameters
        W, _ = W.chunk(2, 0)
        b, _ = b.chunk(2, 0)
        Wp, _ = Wp.chunk(2, 0)
        bp, _ = bp.chunk(2, 0)
        # U(WX + b - c) = (UW)X + U(b - c)
        # (out',out) @ (k,out,in) -> (k, out', in)
        W = (pca @ W.permute(2,0,1)).permute(1,2,0)
        b = pca @ ((b - self.latent_mean) if use_mean else b)
        # assign back
        layer_in.weight, layer_in.bias = nn.Parameter(W), nn.Parameter(b)
        layer_in.in_channels = layer_in.in_channels//2
        layer_in.out_channels = n
        layer_in.groups = 1
        layer_prev.weight, layer_prev.bias = nn.Parameter(Wp), nn.Parameter(bp)
        layer_prev.out_channels = layer_prev.out_channels//2

        # project the first decoder layer
        inv_pca = self.latent_pca.T
        # inv_pca = torch.linalg.inv(self.latent_pca)

        # W(UX + c) + b = (WU)X + (Wc + b)
        # (k, out, in) @ (in,in') -> (k, out, in')
        W, b = layer_out.weight, layer_out.bias
        if use_mean:
            b = W.sum(-1) @ self.latent_mean + b # sum over kernel dimension
        W = (W.permute(2,0,1) @ inv_pca).permute(1,2,0) #* 0.1

        # better initialization for noise weights 
        W = torch.cat((W[:,:n], W[:,n:]*0.01), 1)

        # finally, set the number of noise dimensions
        if decoder_latent_size:
            if decoder_latent_size < n:
                raise ValueError("""
                decoder_latent_size should not be less than cropped size
                """)
            if decoder_latent_size > self.latent_size:
                # expand
                new_dims = decoder_latent_size-self.latent_size
                W2 = torch.randn(W.shape[0], new_dims, W.shape[2], device=W.device, dtype=W.dtype)
                W2 = 0.01 * W2 / W2.norm(dim=(0,2), keepdim=True) * W.norm(dim=(0,2)).min()
                W = torch.cat((W, W2), 1)
            else:
                # crop
                W = W[:,:decoder_latent_size]

            self.latent_size = self.hparams['latent_size'] = decoder_latent_size

        # assign back
        layer_out.weight, layer_out.bias = nn.Parameter(W), nn.Parameter(b)
        layer_out.in_channels = self.latent_size

        self.cropped_latent_size = self.hparams['cropped_latent_size'] = n

        # CachedConv stuff
        for layer in (layer_in, layer_prev, layer_out):
            if hasattr(layer, 'cache'):
                layer.cache.initialized = False

        # the old PCA weights etc aren't needed anymore,
        # it would be nice to keep them around for reference but
        # the size change is causing model loading headaches
        self.init_buffers()

        # test
        # z2, _ = self.split_params(self.encoder(x))
        # # print('z (should be different)', (z-z2).norm(), z.norm(), z2.norm())
        # # y2 = self.decoder(self.pad_latent(z2), add_noise=self.hparams['use_noise'])
        # y2 = self.decoder.net[0](self.pad_latent(z2))
        # print(f'{(y-y2).norm()=}, {y.norm()=}, {y2.norm()=}, {(y-y_perturb).norm()=}')

        # # without PCA:
        # # find the n most important latent dimensions
        # keep_idxs = self.kld_idxs[:n]
        # # prune the final encoder layer
        # # w: (out, in, kernel)
        # # b: (out)
        # layer_in = self.encoder.net[-1]
        # layer_in.weight = layer_in.weight[keep_idxs]
        # layer_in.bias = layer_in.bias[keep_idxs]
        # # reorder the first decoder layer
        # # w: (out, in, kernel)
        # layer_out = self.decoder.net[0]
        # layer_out.weight = layer_out.weight[:, self.kld_idxs]

        # # now sorted
        # self.kld_idxs[:] = range(len(self.kld_idxs))
        # self.cropped_latent_size = n
        

    def validation_epoch_end(self, outs):
        for (out, tag) in zip(outs, ('valid', 'test')):

            audio, z, klds = list(zip(*out))

            # LATENT SPACE ANALYSIS
            if tag=='valid' and not self.hparams['freeze_encoder']:
                z = torch.cat(z, 0)
                z = rearrange(z, "b c t -> (b t) c")

                self.latent_mean.copy_(z.mean(0))
                z = z - self.latent_mean

                pca = PCA(z.shape[-1]).fit(z.cpu().numpy())

                components = pca.components_
                components = torch.from_numpy(components).to(z)
                self.latent_pca.copy_(components)

                var = pca.explained_variance_ / np.sum(pca.explained_variance_)
                var = np.cumsum(var)

                self.fidelity.copy_(torch.from_numpy(var).to(self.fidelity))

                var_p = [.8, .9, .95, .99]
                for p in var_p:
                    self.log(f"{p}_manifold/pca",
                            np.argmax(var > p).astype(np.float32))

                klds = sum(klds) / len(klds)
                klds, kld_idxs = klds.cpu().sort(descending=True)
                self.kld_idxs[:] = kld_idxs
                kld_p = (klds / klds.sum()).cumsum(0)
                # print(kld_p)
                for p in var_p:
                    self.log(f"{p}_manifold/kld",
                            torch.argmax((kld_p > p).long()).item())

            n = 16 if tag=='valid' else 8
            y = torch.cat(audio[:1+n//audio[0].shape[0]], 0)[:n].reshape(-1)
            self.logger.experiment.add_audio(
                f"audio_{tag}", y, self.saved_step.item(), self.sr)

        self.idx += 1
