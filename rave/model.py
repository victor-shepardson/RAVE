import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as wn
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
                 boom=2):
        super().__init__()
        net = []

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

        self.net = cc.CachedSequential(*net)
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
                 bias=False):
        super().__init__()

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
                    boom=boom
                ))

        self.net = cc.CachedSequential(*net)

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
                 bias=False):
        super().__init__()
        maybe_wn = (lambda x:x) if norm=='batch' else wn

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
            norm = lambda d: nn.InstanceNorm1d(d)


        for r, n in zip(ratios, narrow):
            in_dim = out_dim
            out_dim = out_dim * r // n

            if norm is not None:
                net.append(norm(in_dim))
            net.append(
                ResidualStack(
                    in_dim,
                    3,
                    padding_mode,
                    cumulative_delay=net[-2 if norm is not None else -1].cumulative_delay,
                    depth=1,
                    boom=boom
                ))
            net.append(maybe_wn(
                cc.Conv1d(
                    in_dim,
                    out_dim,
                    2 * r + 1,
                    padding=cc.get_padding(2 * r + 1, r, mode=padding_mode),
                    stride=r,
                    bias=bias,
                    cumulative_delay=net[-1].cumulative_delay,
                )))
            
        net.append(nn.LeakyReLU(0.2))
        net.append(maybe_wn(
            cc.Conv1d(
                out_dim,
                2 * latent_size,
                3,
                padding=cc.get_padding(3, mode=padding_mode),
                groups=2,
                bias=bias,
                cumulative_delay=net[-2].cumulative_delay,
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
        return torch.split(z, z.shape[1] // 2, 1)

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

class GLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, 1)
        return a.sigmoid() * b

class Discriminator(nn.Module):
    def __init__(self,
                data_size,
                capacity,
                boom,
                ratios,
                narrow,
                padding_mode="centered"):
        super().__init__()
        out_dim = capacity

        self.root = nn.Sequential(
            wn(nn.Conv1d(data_size, out_dim*2, 7)),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm1d(out_dim*2),
            wn(nn.Conv1d(out_dim*2, out_dim, 1)),
        )

        self.stem = nn.ModuleList()
        self.heads = nn.ModuleList()

        for r, n in zip(ratios, narrow):
            in_dim = out_dim
            out_dim = out_dim * r // n

            self.stem.append(nn.Sequential(
                ResidualStack(
                    in_dim,
                    3,
                    padding_mode=padding_mode,
                    depth=3,
                    boom=boom
                ),
                nn.InstanceNorm1d(in_dim),
                wn(nn.Conv1d(
                    in_dim,
                    out_dim,
                    2 * r + 1,
                    stride=r,
                )) # no padding needed here for discriminator
            ))

            # self.heads.append(nn.Conv1d(out_dim, 1, 3))
            self.heads.append(nn.Sequential(
                nn.Conv1d(out_dim, out_dim*2, 1, groups=2),
                GLU(),
                nn.Conv1d(out_dim, 1, 3)
            ))

    def forward(self, x):
        scores = []
        x = self.root(x)
        for layer, head in zip(self.stem, self.heads):
            x = layer(x)
            scores.append(head(x))
        return scores


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


# class StackDiscriminators(nn.Module):
#     def __init__(self, n_dis, *args, factor=2, **kwargs):
#         super().__init__()
#         self.factor = factor
#         self.discriminators = nn.ModuleList(
#             [Discriminator(*args, **kwargs) for i in range(n_dis)], )

#     def forward(self, x):
#         features = []
#         for layer in self.discriminators:
#             features.append(layer(x))
#             x = nn.functional.avg_pool1d(x, self.factor)
#         return features


class RAVE(pl.LightningModule):
    def __init__(self,
                 data_size,
                 capacity,
                 boom,
                 latent_size,
                 ratios,
                 narrow,
                 bias,
                 encoder_norm,
                 loud_stride,
                 use_noise,
                 noise_ratios,
                 noise_bands,
                 d_capacity,
                 d_warmup,
                 max_gamma,
                 pair_discriminator,
                 ged,
                 adversarial_loss,
                 freeze_encoder,
                 warmup,
                #  kl_cycle,
                 mode,
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
                 grad_clip=None
                ):
        super().__init__()
        self.save_hyperparameters()

        if data_size == 1:
            self.pqmf = None
        else:
            self.pqmf = PQMF(70 if no_latency else 100, data_size)

        self.loudness = Loudness(sr, 512)

        encoder_out_size = cropped_latent_size if cropped_latent_size else latent_size

        self.encoder = Encoder(
            data_size,
            capacity,
            boom,
            encoder_out_size,
            ratios,
            narrow,
            "causal" if no_latency else "centered",
            encoder_norm,
            bias,
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
        )

        print('encoder')
        for n,p in self.encoder.named_parameters():
            print(f'{n}: {p.numel()}')
        print('generator')
        for n,p in self.decoder.named_parameters():
            print(f'{n}: {p.numel()}')

        if adversarial_loss or feature_match:
            # self.discriminator = StackDiscriminators(
            #     3, factor=d_stack_factor,
            #     in_size=2 if pair_discriminator else 1,
            #     capacity=d_capacity,
            #     multiplier=d_multiplier,
            #     n_layers=d_n_layers
            #     )
            self.discriminator = Discriminator(
                data_size,
                d_capacity,
                boom,
                ratios,
                narrow,
            )
        else:
            self.discriminator = None

        self.idx = 0

        self.register_buffer("latent_pca", torch.eye(encoder_out_size))
        self.register_buffer("latent_mean", torch.zeros(encoder_out_size))
        self.register_buffer("fidelity", torch.zeros(encoder_out_size))

        self.register_buffer("discriminator_step", torch.tensor(0.))

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

        if self.hparams['sample_kl']:
            log_std = scale.clamp(-7, 7)
            u = torch.randn_like(mean)
            z = u * log_std.exp() + mean
            if self.hparams['path_derivative']:
                log_std = log_std.detach()
                mean = mean.detach()
                _u = (z - mean) / log_std.exp()
                kl = (0.5*(z*z - _u*_u) - log_std).sum(1).mean()
            else:
                kl = (0.5*(z*z - u*u) - log_std).sum(1).mean()
        else:
            std = nn.functional.softplus(scale) + 1e-4
            var = std * std
            logvar = torch.log(var)
            z = torch.randn_like(mean) * std + mean
            kl = 0.5 * (mean * mean + var - logvar - 1).sum(1).mean()

        if self.cropped_latent_size:
            noise = torch.randn(
                z.shape[0],
                self.latent_size - self.cropped_latent_size,
                z.shape[-1],
                device = z.device
            )#.to(z.device)
            z = torch.cat([z, noise], 1)
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
        is_disc_step = self.global_step % 2 and use_discriminator
        freeze_encoder = self.hparams['freeze_encoder']
        double_z = use_ged or (use_pairs and use_discriminator)

        # ENCODE INPUT
        if freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                z, kl = self.reparametrize(*self.encoder(x, double=double_z))
        else:
            z, kl = self.reparametrize(*self.encoder(x, double=double_z))

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

        feature_matching_distance = 0.
        if use_discriminator:  # DISCRIMINATION
            # note -- could run x and target both through discriminator here
            # shouldn't matter which one is used (?)
            # also -- only y is needed for generator, when not using feature_match
            # -- probably not worth the ~1/8 time saving right now though
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
            # discs_features = self.discriminator(to_disc)
            scores = self.discriminator(to_disc)

            p.tick("discriminator")

            loss_dis = 0
            loss_adv = 0
            pred_true = 0
            pred_fake = 0

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

            p.tick("adversarial loss")

            if self.feature_match:
                raise NotImplementedError("""
                    this branch uses multi-head discriminator instead of feature matching
                    """)
                # if use_ged:
                #     def dist(fm):
                #         real, fake, fake2 = fm.chunk(3)
                #         return (
                #             (real-fake).abs().mean()
                #             + (real-fake2).abs().mean()
                #             - (fake-fake2).abs().mean()
                #         )
                # else:
                #     def dist(fm):
                #         real, fake = fm.chunk(2)
                #         return (real-fake).abs().mean()
                # feature_matching_distance = 10*sum(
                #     map(dist, feature_maps)) / len(feature_maps)
                # p.tick("feature matching distance")

        else:
            pred_true = x.new_zeros(1)
            pred_fake = x.new_zeros(1)
            loss_dis = x.new_zeros(1)
            loss_adv = x.new_zeros(1)



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

        # linear warmup adversarial loss
        max_gamma = self.hparams['max_gamma']
        gamma = (
            self.discriminator_step / self.hparams['d_warmup'] 
            * max_gamma).clamp(0,max_gamma)
        loss_adv = gamma * loss_adv

        loss_gen = distance + loss_kld
        if self.hparams['adversarial_loss']:
            loss_gen = loss_gen + loss_adv
        if self.feature_match:
            loss_gen = loss_gen + feature_matching_distance
        p.tick("gen loss compose")

        # OPTIMIZATION
        grad_clip = self.hparams['grad_clip']

        if self.trainer is None: return

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

            self.discriminator_step += 1
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
        # discriminator warmup parameter
        self.log("gamma", gamma)

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

    def encode(self, x):
        if self.pqmf is not None:
            x = self.pqmf(x)

        mean, scale = self.encoder(x)
        z, _ = self.reparametrize(mean, scale)
        return z

    def decode(self, z):
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

        mean, scale = self.encoder(x)

        if loader_idx>0:
            # z = mean
            z = torch.cat((
                mean, 
                mean + torch.randn((*mean.shape[:2], 1), device=mean.device)/2),
                0)
        else:
            z, kl = self.reparametrize(mean, scale)

        y = self.decoder(z, add_noise=self.hparams['use_noise'])
        # print(x.shape, z.shape, y.shape)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x)
            target = self.pqmf.inverse(target)
            y = self.pqmf.inverse(y)

        distance = self.distance(target, y)
        baseline_distance = self.distance(target, x)

        if loader_idx==0 and self.trainer is not None:
            # full-band distance only,
            # in contrast to training distance
            # KLD in bits per second
            self.log("valid_distance", distance)
            self.log("valid_distance/baseline", baseline_distance)
            self.log("valid_kld_bps", self.npz_to_bps(kl))

        if loader_idx==0:
            return torch.cat([target, y], -1), mean
        if loader_idx>0:
            return torch.cat([target, *y.chunk(2, 0)], -1), mean

    def npz_to_bps(self, npz):
        """convert nats per z frame to bits per second"""
        return (npz * self.hparams['sr'] 
            / np.prod(self.hparams['ratios']) 
            / self.hparams['data_size'] 
            * np.log2(np.e))

    def validation_epoch_end(self, outs):
        for (out, tag) in zip(outs, ('valid', 'test')):

            audio, z = list(zip(*out))

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

                var_percent = [.8, .9, .95, .99]
                for p in var_percent:
                    self.log(f"{p}%_manifold",
                            np.argmax(var > p).astype(np.float32))

            n = 32 if tag=='valid' else 8
            y = torch.cat(audio, 0)[:n].reshape(-1)
            self.logger.experiment.add_audio(
                f"audio_{tag}", y, self.saved_step.item(), self.sr)

        self.idx += 1
