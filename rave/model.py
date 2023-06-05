from typing import List
from time import time

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch import Tensor

import numpy as np
import pytorch_lightning as pl
from sklearn.decomposition import PCA
from einops import rearrange

import cached_conv as cc

# import torch.nn.utils.weight_norm as wn
# from torch.nn.utils import remove_weight_norm
from .weight_norm import weight_norm as wn
from .weight_norm import remove_weight_norm
from .core import multiscale_stft, Loudness, mod_sigmoid
from .core import amp_to_impulse_response, fft_convolve, get_beta_kl_cyclic_annealed, get_beta_kl, get_inst_freq_patches
from .pqmf import CachedPQMF as PQMF

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
                 group_size=128,
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
                        groups=max(1, dim//group_size)
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
        ## stride
        if ratio > 1:
            # net.append(wn(cc.Conv1d(
                # in_dim, in_dim, 1, padding=(1,0), groups=max(1, in_dim//256))))
            net.append(
                cc.CachedPadding1d(1) if cc.USE_BUFFER_CONV
                else nn.ConstantPad1d((1,0), 0.))
            net.append(
                wn(
                    nn.ConvTranspose1d(
                        in_dim,
                        out_dim,
                        2 * ratio,
                        stride=ratio,
                        padding=ratio,
                        bias=bias,
                        # groups=max(1, out_dim//256)#out_dim//8
                    )))
            net[-1].cumulative_delay = 0
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
        ## Upsample
        # if ratio > 1:
        #     net.append(nn.Upsample(scale_factor=ratio))
        # net.append(
        #     wn(
        #         cc.Conv1d(
        #             in_dim,
        #             out_dim,
        #             2*ratio+1,
        #             padding=cc.get_padding(2*ratio+1, mode=padding_mode),
        #             bias=bias,
        #         )))
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
                wn(
                cc.Conv1d(
                    channels[i],
                    channels[i + 1],
                    # 3,
                    # padding=cc.get_padding(3, r, mode=padding_mode),
                    # stride=r,
                    # 2 * r + 1,
                    # padding = (r + 1, 0),
                    # stride=r,
                    r, stride=r,
                    cumulative_delay=cum_delay,
                )
                )
            )
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
        amp = amp.permute(0, 2, 1) # batch, time, channel
        amp = amp.reshape(amp.shape[0], amp.shape[1], self.data_size, -1)
        # batch, time/prod(ratios), pqmf_band, noise_band

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1
        # batch, time/prod(ratios), pqmf_band, prod(ratios)

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        # batch, pqmf_band, time/prod(ratios), prod(ratios)

        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        # batch, pqmf_band, time
        return noise


class Generator(nn.Module):
    def __init__(self,
                 latent_size,
                 capacity,
                 boom,
                 group_size,
                 data_size,
                 ratios,
                 narrow,
                 loud_stride,
                 noise_ratios,
                 noise_bands,
                 padding_mode,
                 bias=False,
                 script=True,
                 linear_path=True,
                 block_depth=2
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

            prev_layer_idx = -1
            
            if i==0:
                prev_layer_idx -= 1
            elif not linear_path:
                net.append(nn.LeakyReLU(.2))
                prev_layer_idx -= 1

            net.append(
                UpsampleLayer(
                    in_dim,
                    out_dim,
                    r,
                    padding_mode,
                    cumulative_delay=net[prev_layer_idx].cumulative_delay,
                ))
            net.append(
                ResidualStack(
                    out_dim,
                    3,
                    padding_mode,
                    depth=block_depth,
                    cumulative_delay=net[-1].cumulative_delay,
                    boom=boom,
                    group_size=group_size,
                    script=script
                ))

        # self.net = maybe_script(cc.CachedSequential(*net))
        self.net = cc.CachedSequential(*net)

        wave_gen = wn(
            cc.Conv1d(
                out_dim,
                data_size,
                7,
                padding=cc.get_padding(7, mode=padding_mode),
                bias=bias,
            ))

        r = loud_stride
        loud_gen = wn(
            cc.Conv1d(
                out_dim,
                1,
                2 * r + 1,
                padding = (r + 1, 0),
                stride=r,
                # 2 * loud_stride + 1,
                # stride=loud_stride,
                # padding=cc.get_padding(2 * loud_stride + 1,
                #                        loud_stride,
                #                        mode=padding_mode),
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
        

    def parts(self, x):
        x = self.net(x)

        if self.has_noise_branch:
            waveform, loudness, noise = self.synth(x)
        else:
            waveform, loudness = self.synth(x)
            noise = torch.zeros_like(waveform)

        loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform) 
        loudness = mod_sigmoid(loudness)

        return waveform, loudness, noise

    def forward(self, x, add_noise: bool = True):
        waveform, loudness, noise = self.parts(x)

        waveform = waveform * loudness

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
                 group_size,
                 latent_size,
                 ratios,
                 narrow,
                 padding_mode,
                 latent_params=2, # mean, scale
                 norm=None,
                 bias=False,
                 script=True,
                 linear_path=True
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

        for i,(r, n) in enumerate(zip(ratios, narrow)):
            in_dim = out_dim
            out_dim = out_dim * r // n

            prev_layer_idx = -1
            if norm is not None:
                net.append(norm(in_dim))
                prev_layer_idx -= 1

            if not linear_path:
                net.append(nn.LeakyReLU(.2))
                prev_layer_idx -= 1

            net.append(
                ResidualStack(
                    in_dim,
                    3,
                    padding_mode,
                    cumulative_delay=net[prev_layer_idx].cumulative_delay,
                    depth=1,
                    boom=boom,
                    group_size=group_size,
                    script=script
                ))
            net.append(maybe_wn(
                cc.Conv1d(
                    in_dim,
                    out_dim, 
                    2 * r + 1,
                    # padding=cc.get_padding(2 * r + 1, r, mode=padding_mode),
                    padding = (r + 1, 0),
                    stride=r,
                    bias=bias,
                    cumulative_delay=net[-1].cumulative_delay,
                )))
            # net.append(nn.AvgPool1d(r,r))
            # net.append(nn.MaxPool1d(r,r))

        net.append(nn.LeakyReLU(0.2)) 

        final_layer = cc.Conv1d(
                out_dim,
                latent_size * latent_params,
                3,
                padding=(2,0),
                groups=latent_params,
                bias=bias,
                cumulative_delay=net[-2].cumulative_delay,
                # cumulative_delay=net[-3].cumulative_delay,
            )
        with torch.no_grad():
            final_layer.weight.mul_(1e-2)

        net.append(maybe_wn(final_layer))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay
        
    def forward(self, x, double:bool=False):
        z = self.net(x)
        # duplicate along batch dimension
        if double:
            z = z.repeat(2, 1, 1)
        return z

class Prior(nn.Module):
    def __init__(self, latent_size, latent_params, n_layers=2, h=512, k_in=5, k=3):
        super().__init__()

        net = [wn(cc.Conv1d(latent_size, h, k_in, padding=(k_in-1,0)))]

        for _ in range(n_layers):
            net.append(Residual(cc.CachedSequential(
                nn.LeakyReLU(0.2),
                wn(cc.Conv1d(h, h, k, padding=(k-1,0)))
                )))

        net.append(nn.LeakyReLU(0.2))

        final_layer = cc.Conv1d(h, latent_size*latent_params, k, padding=(k-1,0))
        with torch.no_grad():
            final_layer.weight.mul_(1e-2)
        net.append(wn(final_layer))

        self.net = cc.CachedSequential(*net)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, in_size, capacity, multiplier, n_layers, norm=None):
        super().__init__()

        out_size = capacity

        def get_norm(s):
            if norm=='layer':
                return (LayerNorm1d(),)
            elif norm=='batch':
                return (nn.BatchNorm1d(s),)
            elif not norm:
                return tuple()
            raise ValueError(f'unknown discriminator norm: "{norm}"')

        net = [nn.Sequential(
            wn(cc.Conv1d(in_size, out_size, 15, padding=cc.get_padding(15))),
            *get_norm(out_size)
        )]

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
                    )),
                *get_norm(out_size)))


        net.append(nn.Sequential(
            nn.LeakyReLU(.2),
            wn(
                cc.Conv1d(
                    out_size,
                    out_size,
                    5,
                    padding=cc.get_padding(5),
                )),
            *get_norm(out_size)))

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

class Gimbal(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.log_a = nn.Parameter(torch.zeros(size, 1))
        self.b = nn.Parameter(torch.zeros(size, 1))

    def forward(self, mean, log_scale):
        return mean * self.log_a.float().exp() + self.b, log_scale + self.log_a

    def inv(self, z):
        return (z-self.b) / self.log_a.exp()

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
                 use_norm_dist,
                 warmup,
                #  kl_cycle,
                 mode,
                 adversarial_weight=1.0,
                 feature_match_weight=10.0,
                 d_norm=None,
                 gimbal=False,
                 group_size=64,
                 linear_path=True,
                 encoder_norm=None,
                 no_latency=False,
                 min_beta=1e-6,
                 max_beta=1e-1,
                 min_beta_prior=1e-3,
                 max_beta_prior=1e-1,
                 cropped_latent_size=0,
                 feature_match=True,
                 sr=48000,
                 gen_lr=1e-4,
                 dis_lr=1e-4,
                 gen_adam_betas=(0.5,0.9),
                 dis_adam_betas=(0.5,0.9),
                 grad_clip=None,
                 script=True,
                 amp=False
                ):
        super().__init__()
        self.save_hyperparameters()

        maybe_script = torch.jit.script if script else lambda _:_

        if data_size == 1:
            self.pqmf = None
        else:
            self.pqmf = PQMF(70 if no_latency else 100, data_size)

        self.loudness = Loudness(sr, 512)

        latent_params = 2

        self.encoder = Encoder(
            data_size,
            capacity,
            boom,
            group_size,
            latent_size,
            ratios,
            narrow,
            "causal" if no_latency else "centered",
            latent_params,
            encoder_norm,
            bias,
            script,
            linear_path=linear_path
        )
        self.decoder = Generator(
            latent_size,
            capacity,
            boom,
            group_size,
            data_size,
            list(reversed(ratios)),
            list(reversed(narrow)),
            loud_stride,
            noise_ratios,
            noise_bands,
            "causal" if no_latency else "centered",
            bias,
            script,
            linear_path=linear_path
        )

        if gimbal:
            self.gimbal = Gimbal(latent_size)
        else:
            self.gimbal = None

        self.prior = Prior(latent_size, latent_params)

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
                n_layers=d_n_layers,
                norm=d_norm
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

        self.cropped_latent_size = cropped_latent_size

        self.feature_match = feature_match

        self.register_buffer("saved_step", torch.tensor(0))

        self.init_buffers()

        if cropped_latent_size:
            self.crop_latent_space(cropped_latent_size)

        self.scaler = GradScaler(enabled=amp)

    def init_buffers(self):
        self.register_buffer("latent_pca", torch.eye(self.latent_size))
        self.register_buffer("latent_mean", torch.zeros(self.latent_size))
        self.register_buffer("fidelity", torch.zeros(self.latent_size))
        # this will track the most informative dimensions of latent space
        # by KLD, in descending order, computed at each validation step
        self.register_buffer("kld_idxs", 
            torch.zeros(self.latent_size, dtype=torch.long))
        self.register_buffer("kld_values", torch.zeros(self.latent_size))

    def configure_optimizers(self):
        gen_p = list(self.encoder.parameters())
        gen_p += list(self.decoder.parameters())
        gen_p += list(self.prior.parameters())
        if self.gimbal is not None:
            gen_p += list(self.gimbal.parameters())

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
        # return (norm(x - y) / norm(x)).mean()
        return (norm(x - y) / (1e-3+norm(x))).mean()

    def log_distance(self, x, y):
        return abs(torch.log(x + 1e-7) - torch.log(y + 1e-7)).mean()

    def phase_distance(self, x, y, y2=None,  weight=True, norm_ord=1):
        if y2 is not None:
            raise NotImplementedError("implement phase_distance+GED")
        assert x.shape[1]==1
        assert y.shape[1]==1
        phase, mag = get_inst_freq_patches(torch.cat((x[:,0], y[:,0])))
        x, y = phase.chunk(2)
        m, _ = mag.chunk(2)
        d = (x - y).abs().mean(-1) # mean over patch
        scale = 1e3

        norm = lambda s: torch.linalg.vector_norm(s, 
            ord=norm_ord, dim=tuple(range(1, s.ndim)))
        
        if weight:
            m = m.sqrt() # patch weight
            scale = scale / (1e-3+norm(m)) # norm by total spectrogram weight
            d = d * m

        return (norm(d) * scale).mean()

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
            dist = (
                self.norm_lin_distance if self.hparams['use_norm_dist'] 
                else self.lin_distance)
            x, y = zip(*(s.chunk(2) for s in stfts))
            lin = sum(map(dist, x, y))
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

    def clip_log_scale(self, log_scale):
        return log_scale.clamp(-14, 14)

    def reparametrize(self, mean, scale, temp=1):
        if self.cropped_latent_size > 0:
            return mean
        else:
            if scale is None:
                raise ValueError("""
                    in `reparametrize`:
                    `scale` should not be None while `self.cropped_latent_size` is 0
                """)
            log_std = self.clip_log_scale(scale)
            u = torch.randn_like(mean)
            return u * log_std.exp() * temp + mean

    def log_dens(self, z, mean, scale):
        log_std = self.clip_log_scale(scale)
        return -0.5 * (z-mean)**2 * (-2*log_std).exp() - log_std

    def kld(self, z, q_mean, q_scale, p_mean=None, p_scale=None):
        if p_mean is None:
            p_mean = torch.zeros_like(q_mean)
        if p_scale is None:
            p_scale = torch.zeros_like(q_scale)

        dens_p = self.log_dens(z, p_mean, p_scale)
        dens_q = self.log_dens(z, q_mean, q_scale)
        kl = dens_q - dens_p
        return kl.mean((0,2)) # mean over batch and time

    def moment_loss(self, z):
        return z.mean((0,2)).abs() + (z.std((0,2)) - 1).abs()

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

    def shift_latent(self, z):
        # left pad and slice off end before sending to prior
        # to align prior with posterior
        return torch.cat((z.new_zeros((*z.shape[:2], 1)), z[...,:-1]), -1)

    def training_step(self, batch, batch_idx, opt=True):
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

        def encode():
            q_params = self.split_params(self.encoder(x, double=double_z))
            if self.gimbal is not None:
                q_params = self.gimbal(*q_params)
            z = self.reparametrize(*q_params)
            shift_z = self.shift_latent(z.detach())
            p_params = self.split_params(self.prior(shift_z))

            kl = self.kld(z, *q_params, *p_params)
            # TODO: try not differentiating through sampling here?
            kl_prior = self.kld(self.reparametrize(*p_params), *p_params)

            kl = kl.sum() if kl is not None else 0
            kl_prior = kl_prior.sum() if kl_prior is not None else 0

            if self.gimbal is not None:
                z = self.gimbal.inv(z)
            z = self.pad_latent(z)

            return z, kl, kl_prior

        # ENCODE INPUT
        with autocast(enabled=self.hparams['amp']):
            if freeze_encoder:
                self.encoder.eval()
                with torch.no_grad():
                    z, kl, kl_prior = encode()
            else:
                z, kl, kl_prior = encode()

        # DECODE LATENT
        with autocast(enabled=self.hparams['amp']):
            y = self.decoder(z, add_noise=self.hparams['use_noise'])

        if double_z:
            y, y2 = y.chunk(2)
        else:
            y2 = None
        p.tick("decode")

        # DISTANCE BETWEEN INPUT AND OUTPUT

        if self.pqmf is not None:  # FULL BAND RECOMPOSITION
            distance = self.distance(target, y, y2 if use_ged else None)
            # do inverse pqmf on target to cancel any delay it introduces
            target = self.pqmf.inverse(target)
            y = self.pqmf.inverse(y)
            if y2 is not None:
                y2 = self.pqmf.inverse(y2)
            p.tick("mb distance")

        distance = distance + self.distance(target, y, y2 if use_ged else None)
        phase_distance = self.phase_distance(target, y, y2 if use_ged else None)
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

            with autocast(enabled=self.hparams['amp']):
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
                feature_matching_distance = (
                    self.hparams['feature_match_weight']
                    * sum(map(dist, feature_maps)) / len(feature_maps))

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
        #     min_beta=self.hparams['min_beta'],
        #     max_beta=self.hparams['max_beta'],
        # )
        beta = get_beta_kl(
            self.global_step, self.hparams['warmup'],
            self.hparams['min_beta'], self.hparams['max_beta'])
        loss_kld = beta * kl

        beta_prior = get_beta_kl(
            self.global_step, self.hparams['warmup'], 
            self.hparams['min_beta_prior'], self.hparams['max_beta_prior'])
        # beta_prior = self.hparams['beta_prior']
        loss_kld_prior = beta_prior * kl_prior

        # loss_moment = self.moment_loss(z).mean() * self.hparams['beta_moment']

        loss_gen = distance + loss_kld + loss_kld_prior + phase_distance#+ loss_moment
        if self.hparams['adversarial_loss']:
            loss_gen = loss_gen + loss_adv*self.hparams['adversarial_weight']
        if self.feature_match:
            loss_gen = loss_gen + feature_matching_distance
        p.tick("gen loss compose")

        if not opt: 
            return y, loss_kld, loss_kld_prior, phase_distance, distance

        # OPTIMIZATION
        is_disc_step = self.global_step % 2 and use_discriminator
        grad_clip = self.hparams['grad_clip']

        if use_discriminator:
            gen_opt, dis_opt = self.optimizers()
        else:
            gen_opt = self.optimizers()

        if is_disc_step:
            dis_opt.zero_grad()
            self.scaler.scale(loss_dis).backward()
            # loss_dis.backward()

            if grad_clip is not None:
                dis_grad = nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), grad_clip)
                self.log('grad_norm_discriminator', dis_grad)

            self.scaler.step(dis_opt)
            # dis_opt.step()
        else:
            gen_opt.zero_grad()
            self.scaler.scale(loss_gen).backward()
            # loss_gen.backward()

            if grad_clip is not None:
                if not freeze_encoder:
                    enc_grad = nn.utils.clip_grad_norm_(
                        self.encoder.parameters(), grad_clip)
                    self.log('grad_norm_encoder', enc_grad)
                dec_grad = nn.utils.clip_grad_norm_(
                    self.decoder.parameters(), grad_clip)
                self.log('grad_norm_generator', dec_grad)

            self.scaler.step(gen_opt)
            # gen_opt.step()

        self.scaler.update()
               
        p.tick("optimization")

        # LOGGING
        # total generator loss
        self.log("loss_generator", loss_gen)

        # KLD loss (KLD in nats per z * beta)
        self.log("loss_kld", loss_kld)
        self.log("loss_kld_prior", loss_kld_prior)
        # moment matching loss
        # self.log("loss_moment", loss_moment)
        # spectral + loudness distance loss
        self.log("loss_distance", distance)
        self.log("loss_phase_distance", phase_distance)
        # loudness distance loss
        self.log("loss_loudness", loud_dist)

        # KLD in bits per second
        self.log("kld_bps", self.npz_to_bps(kl))
        self.log("kld_bps_prior", self.npz_to_bps(kl_prior))
        # beta-VAE parameter
        self.log("beta", beta)
        self.log("beta_prior", beta_prior)

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
        params = p.chunk(2,1)
        return params

    def encode(self, x, temp=1):
        if self.pqmf is not None:
            x = self.pqmf(x)

        params = self.encoder(x)
        params = self.split_params(params)

        if self.gimbal is not None:
            params = self.gimbal(*params)

        z = self.reparametrize(*params, temp=temp)
        return z

    def decode(self, z):
        z = self.pad_latent(z)
        if self.gimbal is not None:
            z = self.gimbal.inv(z)

        y = self.decoder(z, add_noise=self.hparams['use_noise'])
        if self.pqmf is not None:
            y = self.pqmf.inverse(y)
        return y

    def decode_parts(self, z, pqmf=True):
        z = self.pad_latent(z)
        if self.gimbal is not None:
            z = self.gimbal.inv(z)

        wav, loud, noise = self.decoder.parts(z)
        if pqmf and self.pqmf is not None:
            wav = self.pqmf.inverse(wav)
            loud = self.pqmf.inverse(loud)
            noise = self.pqmf.inverse(noise)
        return wav, loud, noise

    def validation_step(self, batch, batch_idx, loader_idx):
            
        x = batch['source'].unsqueeze(1)
        target = batch['target'].unsqueeze(1)

        if self.pqmf is not None:
            x = self.pqmf(x)
            target = self.pqmf(target)

        q_mean, q_scale = self.split_params(self.encoder(x))
        if self.gimbal is not None:
            q_mean, q_scale = self.gimbal(q_mean, q_scale)

        if loader_idx > 0:
            z = self.reparametrize(q_mean, q_scale)
            # z = q_mean
            to_decode = torch.cat((
                z, 
                z + torch.randn((*q_mean.shape[:2], 1), device=q_mean.device)/2),
                0)
        else:
            z = to_decode = self.reparametrize(q_mean, q_scale)

        p_mean, p_scale = self.split_params(self.prior(self.shift_latent(z)))

        kl = self.kld(z, q_mean, q_scale, p_mean, p_scale)

        if self.gimbal is not None:
            to_decode = self.gimbal.inv(to_decode)
        to_decode = self.pad_latent(to_decode)

        y = self.decoder(to_decode, add_noise=self.hparams['use_noise'])
        # print(x.shape, z.shape, y.shape)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x)
            target = self.pqmf.inverse(target)
            y = self.pqmf.inverse(y)

        if loader_idx > 0:
            y, y2 = y.chunk(2, 0)

        distance = self.distance(target, y)
        baseline_distance = self.distance(target, x)

        phase_distance = self.phase_distance(target, y)

        if self.trainer is not None:
            # if loader_idx==0:
            # full-band distance only,
            # in contrast to training distance
            # KLD in bits per second
            self.log("valid_distance", distance)
            self.log("valid_distance/baseline", baseline_distance)
            self.log("valid_phase_distance", phase_distance)
            if kl is not None:
                self.log("valid_kld_bps", self.npz_to_bps(kl.sum()))

        if loader_idx==0:
            return torch.cat([y, target], -1), q_mean, kl
        if loader_idx>0:
            return torch.cat([y, target, y2], -1), q_mean, None

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
        layer_prev = self.encoder.net[-3]
        # layer_prev = self.encoder.net[-4]
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
                self.kld_values[:] = klds
                kld_p = (klds / klds.sum()).cumsum(0)
                # print(kld_p)
                for p in var_p:
                    self.log(f"{p}_manifold/kld",
                            torch.argmax((kld_p > p).long()).float().item())

            n = 16 if tag=='valid' else 8
            y = torch.cat(audio[:1+n//audio[0].shape[0]], 0)[:n].reshape(-1)
            self.logger.experiment.add_audio(
                f"audio_{tag}", y, self.saved_step.item(), self.sr)

        self.idx += 1



