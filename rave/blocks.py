from functools import partial
import math
from typing import Callable, Optional, Sequence, Union

import cached_conv as cc
import gin
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torchaudio.transforms import Spectrogram

from .core import amp_to_impulse_response, fft_convolve, mod_sigmoid


@gin.configurable
def normalization(module: nn.Module, mode: str = 'identity'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        return weight_norm(module)
    else:
        raise Exception(f'Normalization mode {mode} not supported')


class SampleNorm(nn.Module):

    def forward(self, x):
        return x / torch.norm(x, 2, 1, keepdim=True)


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


class ResidualLayer(nn.Module):

    def __init__(
        self,
        dim,
        kernel_size,
        dilations,
        cumulative_delay=0,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()
        net = []
        cd = 0
        for d in dilations:
            net.append(activation(dim))
            net.append(
                normalization(
                    cc.Conv1d(
                        dim,
                        dim,
                        kernel_size,
                        dilation=d,
                        padding=cc.get_padding(kernel_size, dilation=d),
                        cumulative_delay=cd,
                    )))
            cd = net[-1].cumulative_delay
        self.net = Residual(
            cc.CachedSequential(*net),
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)

@gin.configurable
class DilatedUnit(nn.Module):

    def __init__(
        self,
        dim: int,
        kernel_size: int,
        dilation: int,
        group_size: int = 2**16,
        boom: int = 1,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)
    ) -> None:
        super().__init__()
        groups = max(1, dim//group_size)
        net = [
            activation(dim),
            normalization(
                cc.Conv1d(dim,
                          dim*boom,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          groups=groups,
                          padding=cc.get_padding(
                              kernel_size,
                              dilation=dilation,
                          ))),
            activation(dim*boom),
            normalization(cc.Conv1d(dim*boom, dim, kernel_size=1)),
        ]

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = net[1].cumulative_delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 dilations_list,
                 cumulative_delay=0) -> None:
        super().__init__()
        layers = []
        cd = 0

        for dilations in dilations_list:
            layers.append(
                ResidualLayer(
                    dim,
                    kernel_size,
                    dilations,
                    cumulative_delay=cd,
                ))
            cd = layers[-1].cumulative_delay

        self.net = cc.CachedSequential(
            *layers,
            cumulative_delay=cumulative_delay,
        )
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        return self.net(x)


@gin.configurable
class ResidualStack(nn.Module):

    def __init__(self,
                 dim,
                 kernel_sizes,
                 dilations_list,
                 cumulative_delay=0) -> None:
        super().__init__()
        blocks = []
        for k in kernel_sizes:
            blocks.append(ResidualBlock(dim, k, dilations_list))
        self.net = cc.AlignBranches(*blocks, cumulative_delay=cumulative_delay)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        x = self.net(x)
        x = torch.stack(x, 0).sum(0)
        return x


class UpsampleLayer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        ratio,
        cumulative_delay=0,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2)):
        super().__init__()
        net = [activation(in_dim)]
        if ratio > 1:
            net.append(
                normalization(
                    cc.ConvTranspose1d(in_dim,
                                       out_dim,
                                       2 * ratio,
                                       stride=ratio,
                                       padding=ratio // 2)))
        else:
            net.append(
                normalization(
                    cc.Conv1d(in_dim, out_dim, 3, padding=cc.get_padding(3))))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay + cumulative_delay * ratio

    def forward(self, x):
        return self.net(x)


@gin.configurable
class NoiseGenerator(nn.Module):

    def __init__(self, in_size, data_size, ratios, noise_bands):
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
                    padding=cc.get_padding(3, r),
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


class NoiseGeneratorV2(nn.Module):

    def __init__(
        self,
        in_size: int,
        hidden_size: int,
        data_size: int,
        ratios: int,
        noise_bands: int,
        n_channels: int = 1,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
    ):
        super().__init__()
        net = []
        self.n_channels = n_channels
        channels = [in_size]
        channels.extend((len(ratios) - 1) * [hidden_size])
        channels.append(data_size * noise_bands * n_channels)

        for i, r in enumerate(ratios):
            net.append(
                cc.Conv1d(
                    channels[i],
                    channels[i + 1],
                    2 * r,
                    padding=(r, 0),
                    stride=r,
                ))
            if i != len(ratios) - 1:
                net.append(activation(channels[i + 1]))

        self.net = nn.Sequential(*net)
        self.data_size = data_size

        self.register_buffer(
            "target_size",
            torch.tensor(np.prod(ratios)).long(),
        )

    def forward(self, x):
        amp = mod_sigmoid(self.net(x) - 5)
        amp = amp.permute(0, 2, 1)
        amp = amp.reshape(amp.shape[0], amp.shape[1], self.n_channels * self.data_size, -1)

        ir = amp_to_impulse_response(amp, self.target_size)
        noise = torch.rand_like(ir) * 2 - 1

        noise = fft_convolve(noise, ir).permute(0, 2, 1, 3)
        noise = noise.reshape(noise.shape[0], noise.shape[1], -1)
        return noise


class GRU(nn.Module):

    def __init__(self, latent_size: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=latent_size,
            hidden_size=latent_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.register_buffer("gru_state", torch.tensor(0))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled: return x
        x = x.permute(0, 2, 1)
        x = self.gru(x)[0]
        x = x.permute(0, 2, 1)
        return x

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True


class Generator(nn.Module):

    def __init__(
        self,
        latent_size,
        capacity,
        data_size,
        ratios,
        loud_stride,
        use_noise,
        n_channels: int = 1,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        keep_compute: bool = False,
    ):
        super().__init__()

        def get_size(i):
            if keep_compute:
                s = math.prod(ratios[:i])**0.5
                # capacity is set so a BRAVE model with RATIOS=[2,2,2,1] 
                # will be similar in size
                return int(round(s*2.4*capacity/8)*8)
            else:
                return 2**i * capacity

        net = [
            normalization(
                cc.Conv1d(
                    latent_size,
                    get_size(len(ratios)),
                    # 2**len(ratios) * capacity,
                    7,
                    padding=cc.get_padding(7),
                ))
        ]

        if recurrent_layer is not None:
            net.append(
                recurrent_layer(
                    dim=get_size(len(ratios)),
                    # dim=2**len(ratios) * capacity,
                    cumulative_delay=net[0].cumulative_delay,
                ))

        for i, r in enumerate(ratios):
            in_dim = get_size(len(ratios) - i)
            out_dim = get_size(len(ratios) - i - 1)
            # in_dim = 2**(len(ratios) - i) * capacity
            # out_dim = 2**(len(ratios) - i - 1) * capacity

            # NOTE: this layer is redundant when keep_compute and r=1?
            # (nearly -- it does increase receptive field slightly)
            net.append(
                UpsampleLayer(
                    in_dim,
                    out_dim,
                    r,
                    cumulative_delay=net[-1].cumulative_delay,
                ))
            net.append(
                ResidualStack(out_dim,
                              cumulative_delay=net[-1].cumulative_delay))

        self.net = cc.CachedSequential(*net)

        wave_gen = normalization(
            cc.Conv1d(out_dim, data_size * n_channels, 7, padding=cc.get_padding(7)))

        loud_gen = normalization(
            cc.Conv1d(
                out_dim,
                1,
                2 * loud_stride + 1,
                stride=loud_stride,
                padding=cc.get_padding(2 * loud_stride + 1, loud_stride),
            ))

        branches = [wave_gen, loud_gen]

        if use_noise:
            noise_gen = NoiseGenerator(out_dim, data_size * n_channels)
            branches.append(noise_gen)

        self.synth = cc.AlignBranches(
            *branches,
            cumulative_delay=self.net.cumulative_delay,
        )

        self.use_noise = use_noise
        self.loud_stride = loud_stride
        self.cumulative_delay = self.synth.cumulative_delay

        self.register_buffer("warmed_up", torch.tensor(0))

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x):
        x = self.net(x)

        if self.use_noise:
            waveform, loudness, noise = self.synth(x)
        else:
            waveform, loudness = self.synth(x)
            noise = torch.zeros_like(waveform)

        if self.loud_stride != 1:
            loudness = loudness.repeat_interleave(self.loud_stride)
        loudness = loudness.reshape(x.shape[0], 1, -1)

        waveform = torch.tanh(waveform) * mod_sigmoid(loudness)

        if self.warmed_up and self.use_noise:
            waveform = waveform + noise

        return waveform


class Encoder(nn.Module):

    def __init__(
        self,
        data_size,
        capacity,
        latent_size,
        ratios,
        n_out,
        sample_norm,
        repeat_layers,
        n_channels: int = 1,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        keep_compute:bool = False,
        # retro-compatiblity
        spectrogram = None
    ):
        super().__init__()
        data_size = data_size or n_channels

        def get_size(i):
            if keep_compute:
                s = math.prod(ratios[:i])**0.5
                # capacity is set so a BRAVE model with RATIOS=[2,2,2,1] 
                # will be similar in size
                return int(round(s*2.4*capacity/8)*8)
            else:
                return 2**i * capacity

        net = [cc.Conv1d(data_size * n_channels, get_size(0), 7, padding=cc.get_padding(7))]

        # store this for computing block_size
        self.downsample_factor = math.prod(ratios)

        for i, r in enumerate(ratios):
            in_dim = get_size(i)
            out_dim = get_size(i+1)

            if sample_norm:
                net.append(SampleNorm())
            else:
                net.append(nn.BatchNorm1d(in_dim))
            net.append(nn.LeakyReLU(.2))
            net.append(
                cc.Conv1d(
                    in_dim,
                    out_dim,
                    2 * r + 1,
                    padding=cc.get_padding(2 * r + 1, r),
                    stride=r,
                    cumulative_delay=net[-3].cumulative_delay,
                ))

            for i in range(repeat_layers - 1):
                if sample_norm:
                    net.append(SampleNorm())
                else:
                    net.append(nn.BatchNorm1d(out_dim))
                net.append(nn.LeakyReLU(.2))
                net.append(
                    cc.Conv1d(
                        out_dim,
                        out_dim,
                        3,
                        padding=cc.get_padding(3),
                        cumulative_delay=net[-3].cumulative_delay,
                    ))

        net.append(nn.LeakyReLU(.2))

        if recurrent_layer is not None:
            net.append(
                recurrent_layer(
                    dim=out_dim,
                    cumulative_delay=net[-2].cumulative_delay,
                ))
            net.append(nn.LeakyReLU(.2))

        net.append(
            cc.Conv1d(
                out_dim,
                latent_size * n_out,
                5,
                padding=cc.get_padding(5),
                groups=n_out,
                cumulative_delay=net[-2].cumulative_delay,
            ))

        self.net = cc.CachedSequential(*net)
        self.cumulative_delay = self.net.cumulative_delay

    def forward(self, x):
        z = self.net(x)
        return z


def normalize_dilations(dilations: Union[Sequence[int],
                                         Sequence[Sequence[int]]],
                        ratios: Sequence[int]):
    if isinstance(dilations[0], int):
        dilations = [dilations for _ in ratios]
    return dilations


class EncoderV2(nn.Module):

    def __init__(
        self,
        data_size: Union[int, None],
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        n_out: int,
        kernel_size: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        keep_compute: bool = False,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        n_channels: int = 1,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
        adain: Optional[Callable[[int], nn.Module]] = None,
        spectrogram = None,
        group_size: int = 2**16,
        group_resample: bool = False,
        boom:int = 1,
    ) -> None:
        super().__init__()
        dilations_list = normalize_dilations(dilations, ratios)
        data_size = data_size or n_channels

        # store this for computing block_size
        self.downsample_factor = math.prod(ratios)

        def get_size(i):
            if keep_compute:
                s = math.prod(ratios[:i])**0.5
                # capacity is set so a RAVE model with RATIOS=[4,4,4,2] 
                # will be slightly smaller
                return int(round(s*capacity/8)*8)
            elif keep_dim:
                return math.prod(ratios[:i]) * capacity
            else:
                return 2**i * capacity

        net = [
            normalization(
                cc.Conv1d(
                    data_size * n_channels,
                    get_size(0),
                    kernel_size=kernel_size * 2 + 1,
                    padding=cc.get_padding(kernel_size * 2 + 1),
                )),
        ]

        # num_channels = capacity
        for i,(r, dilations) in enumerate(zip(ratios, dilations_list)):
            num_channels = get_size(i)
            out_channels = get_size(i+1)
            # ADD RESIDUAL DILATED UNITS
            for d in dilations:
                if adain is not None:
                    net.append(adain(dim=num_channels))
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=num_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                            group_size=group_size,
                            boom=boom
                        )))

            # ADD DOWNSAMPLING UNIT
            net.append(activation(num_channels))

            # if keep_dim:
            #     out_channels = num_channels * r
            # else:
            #     out_channels = num_channels * 2

            if group_resample:
                if keep_compute:
                    groups = max(1,out_channels//8) 
                else:
                    groups = max(1,min(num_channels, out_channels)//8)
            else:
                groups = 1 

            net.append(
                normalization(
                    cc.Conv1d(
                        num_channels,
                        out_channels,
                        kernel_size=2 * r,
                        stride=r,
                        groups=groups,
                        padding=cc.get_padding(2 * r, r),
                    )))

            # num_channels = out_channels

        net.append(activation(out_channels))
        net.append(
            normalization(
                cc.Conv1d(
                    out_channels,
                    latent_size * n_out,
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size),
                )))
        # # init close to prior for VAE
        # if prior_init:
        #     with torch.no_grad():
        #         try:
        #             net[-1].weight_g.mul_(3e0)
        #         except AttributeError:
        #             net[-1].weight.mul_(3e0)

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size * n_out))

        self.net = cc.CachedSequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x


class GeneratorV2(nn.Module):

    def __init__(
        self,
        capacity: int,
        ratios: Sequence[int],
        latent_size: int,
        kernel_size: int,
        dilations: Sequence[int],
        keep_dim: bool = False,
        keep_compute: bool = False,
        data_size: Union[int, None] = None,
        recurrent_layer: Optional[Callable[[], nn.Module]] = None,
        n_channels: int = 1,
        amplitude_modulation: bool = False,
        noise_module: Optional[NoiseGeneratorV2] = None,
        activation: Callable[[int], nn.Module] = lambda dim: nn.LeakyReLU(.2),
        adain: Optional[Callable[[int], nn.Module]] = None,
        causal_convtranspose: bool = False,
        group_size: int = 2**16,
        group_resample: bool = False,
        boom:int = 1,
        loudness_modulation: bool = False,
        clip: Optional[str] = 'tanh'
    ) -> None:
        super().__init__()
        self.clip = clip
        if data_size is None:
            data_size = n_channels
        else:
            data_size = data_size * n_channels 
        dilations_list = normalize_dilations(dilations, ratios)[::-1]
        # ratios = ratios[::-1]

        def get_size(i):
            if keep_compute:
                s = math.prod(ratios[:i])**0.5
                # capacity is set so a RAVE model with RATIOS=[4,4,4,2] 
                # will be slightly smaller
                return int(round(s*capacity/8)*8)
            elif keep_dim:
                return math.prod(ratios[:i]) * capacity
            else:
                return 2**i * capacity

        # if keep_dim:
        #     num_channels = np.prod(ratios) * capacity
        # else:
        #     num_channels = 2**len(ratios) * capacity

        net = []

        if recurrent_layer is not None:
            net.append(recurrent_layer(latent_size))

        net.append(
            normalization(
                cc.Conv1d(
                    latent_size,
                    get_size(len(ratios)),
                    kernel_size=kernel_size,
                    padding=cc.get_padding(kernel_size),
                )), )

        for i,(r, dilations) in enumerate(zip(ratios[::-1], dilations_list)):
            num_channels = get_size(len(ratios)-i)
            out_channels = get_size(len(ratios)-i-1)
            # ADD UPSAMPLING UNIT
            # if keep_dim:
            #     out_channels = num_channels // r
            # else:
            #     out_channels = num_channels // 2
            net.append(activation(num_channels))
            
            groups = max(1,out_channels//8) if group_resample else 1

            if r > 1:
                net.append(
                    normalization(cc.ConvTranspose1d(
                        num_channels, out_channels,
                        2*r, stride=r, padding=r//2,
                        groups=groups,
                        causal=causal_convtranspose)))
            else:
                net.append(
                    normalization(cc.Conv1d(
                        num_channels, out_channels, 3,
                        groups=groups,
                        padding=cc.get_padding(3))))

            # num_channels = out_channels

            # ADD RESIDUAL DILATED UNITS
            for d in dilations:
                if adain is not None:
                    net.append(adain(out_channels))
                net.append(
                    Residual(
                        DilatedUnit(
                            dim=out_channels,
                            kernel_size=kernel_size,
                            dilation=d,
                            group_size=group_size,
                            boom=boom
                        )))

        net.append(activation(out_channels))

        waveform_module = normalization(
            cc.Conv1d(
                out_channels,
                # data_size * 2 if amplitude_modulation else data_size,
                data_size 
                    + (data_size if amplitude_modulation else 0) 
                    + (1 if loudness_modulation else 0),
                kernel_size=kernel_size * 2 + 1,
                padding=cc.get_padding(kernel_size * 2 + 1),
            ))

        self.noise_module = None
        self.waveform_module = None

        if noise_module is not None:
            self.waveform_module = waveform_module
            self.noise_module = noise_module(out_channels, n_channels = n_channels)
        else:
            net.append(waveform_module)

        self.net = cc.CachedSequential(*net)

        self.amplitude_modulation = amplitude_modulation
        self.loudness_modulation = loudness_modulation

        if self.loudness_modulation:
            k = 127
            self.loudness_filter = cc.Conv1d(
                1, 1, k, bias=False, positive=True, padding=cc.get_padding(k))
        else:
            self.loudness_filter = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)

        noise = 0.

        if self.noise_module is not None:
            noise = self.noise_module(x)
            x = self.waveform_module(x)

        if self.loudness_modulation:
            x, loudness = x.split([x.shape[1]-1,1],1)
        else:
            loudness = None

        if self.amplitude_modulation:
            x, amplitude = x.split(x.shape[1] // 2, 1)
            x = x * torch.sigmoid(amplitude)

        if loudness is not None and self.loudness_filter is not None:
            loudness = loudness.sigmoid()
            # learnable lowpass filter
            loudness = self.loudness_filter(loudness)
            x = x * loudness

        x = x + noise

        if self.clip is not None: # this guard needed for torchscript
            if self.clip=='tanh':
                x = x.tanh()

        return x

    def set_warmed_up(self, state: bool):
        pass



def normal_log_dens(x, mu, sigma):
    # - log(2pi)/2 - log(sigma) - (x-mu)**2 / 2sigma**2
    return - math.log(2*math.pi)/2 - sigma.log() - ((x-mu)/sigma)**2/2

def normal_log_dens2(x, mu, logsigma):
    # - log(2pi)/2 - log(sigma) - (x-mu)**2 / 2sigma**2
    return - math.log(2*math.pi)/2 - logsigma - ((x-mu)/logsigma.exp())**2/2

def log_sigmoid(x):
    return -(-x).exp().log1p()

class PriorV1(nn.Module):
    """mixture of brownian motion with fixed normal,
    learnable params
    """
    def __init__(self, latent_size):
        super().__init__()
        # unnormalized weight of brownian component
        self.pi = nn.Parameter(torch.tensor(0.0)) 
        # stddev of brownian component per latent
        self.log_std = nn.Parameter(torch.zeros(latent_size)) 

    def params(self, z):
        """causal map from latents z to next prior params"""
        # previous value
        return torch.cat((torch.zeros_like(z[...,:1]), z[...,:-1]),-1)

    def log_density(self, params, z):
        # log ( self.pi * N(0,1) + (1-self.pi) * N(z, std))
        # logsumexp ( logpi + logN(0,1), log(1-pi) + logN(z, std))

        mu = torch.stack((
            torch.zeros_like(z), 
            params
            ))
        logsigma = torch.stack((
            torch.zeros_like(self.log_std), 
            self.log_std
            ))[:,None,:,None]
        logpi = log_sigmoid(torch.stack((
            -self.pi, 
            self.pi
            )))[...,None,None,None]
        
        return (logpi + normal_log_dens2(z, mu, logsigma)).logsumexp(0)

    def sample(self, params):
        raise NotImplementedError
    
class PriorV0(nn.Module):
    """mixture of brownian motion with fixed normal"""
    def __init__(self):
        super().__init__()
        self.pi = 0.5 # weight of brownian component
        self.std = 0.5 # stddev of brownian component

    def params(self, z):
        """causal map from latents z to next prior params"""
        # previous value
        return torch.cat((torch.zeros_like(z[...,:1]), z[...,:-1]),-1)

    def log_density(self, params, z):
        # log ( self.pi * N(0,1) + (1-self.pi) * N(z, std))
        # logsumexp ( logpi + logN(0,1), log(1-pi) + logN(z, std))

        mu = torch.stack((
            torch.zeros_like(z), 
            params))
        sigma = torch.stack((
            torch.ones_like(z),
            torch.full_like(z, self.std)))
        logpi = torch.stack((
            torch.full_like(z, math.log(1-self.pi)),
            torch.full_like(z, math.log(self.pi))))

        return (logpi + normal_log_dens(z, mu, sigma)).logsumexp(0)

    def sample(self, params):
        raise NotImplementedError

class VariationalEncoder(nn.Module):
    def __init__(self, encoder, beta:float=1.0, n_channels=1, prior:nn.Module|None=None):
        super().__init__()
        self.encoder = encoder(n_channels=n_channels)
        self.beta = beta
        self.register_buffer("warmed_up", torch.tensor(0))
        self.prior = None if prior is None else prior()

    def params(self, z):
        mean, scale = z.chunk(2, 1)
        std = nn.functional.softplus(scale) + 1e-4
        return mean, std
    
    def rsample(self, mean, std):
        return torch.randn_like(mean) * std + mean
    
    def reparametrize(self, h):
        """
        Args:
            h: encoder output hidden state [batch, 2*latent, time]
        Returns:
            z: latent samples [batch, latent, time]
            kl: kl-divergence [batch, latent, time]
        """
        if self.prior is None:
            return self.normal_reparametrize(h)
        else:
            return self.sampled_reparametrize(h)
        
    def sampled_reparametrize(self, h):
        """
        sampled KLD: expectation under Q of logQ - logP
        """
        mean, std = self.params(h)
        z = self.rsample(mean, std)

        prior_params = self.prior.params(z)

        # log Q - log P
        kl = (
            normal_log_dens(z, mean, std)
            - self.prior.log_density(prior_params, z))

        return z, self.beta * kl

    def normal_reparametrize(self, h):
        """
        original analytic KLD for simple prior case 
        """
        mean, std = self.params(h)
        z = self.rsample(mean, std)

        var = std * std
        logvar = torch.log(var)
        kl = (mean * mean + var - logvar - 1)

        return z, self.beta * kl

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)

        if self.warmed_up:
            z = z.detach()
        return z


class WasserteinEncoder(nn.Module):

    def __init__(
        self,
        encoder_cls,
        noise_augmentation: int = 0,
        n_channels: int = 1
    ):
        super().__init__()
        self.encoder = encoder_cls(n_channels=n_channels)
        self.register_buffer("warmed_up", torch.tensor(0))
        self.noise_augmentation = noise_augmentation

    def compute_mean_kernel(self, x, y):
        kernel_input = (x[:, None] - y[None]).pow(2).mean(2) / x.shape[-1]
        return torch.exp(-kernel_input).mean()

    def compute_mmd(self, x, y):
        x_kernel = self.compute_mean_kernel(x, x)
        y_kernel = self.compute_mean_kernel(y, y)
        xy_kernel = self.compute_mean_kernel(x, y)
        mmd = x_kernel + y_kernel - 2 * xy_kernel
        return mmd

    def reparametrize(self, z):
        z_reshaped = z.permute(0, 2, 1).reshape(-1, z.shape[1])
        reg = self.compute_mmd(z_reshaped, torch.randn_like(z_reshaped))

        if self.noise_augmentation:
            noise = torch.randn(z.shape[0], self.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)

        return z, reg.mean()

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        if self.warmed_up:
            z = z.detach()
        return z


class DiscreteEncoder(nn.Module):

    def __init__(self,
                 encoder_cls,
                 vq_cls,
                 num_quantizers,
                 noise_augmentation: int = 0,
                 n_channels: int = 1):
        super().__init__()
        self.encoder = encoder_cls(n_channels=n_channels)
        self.rvq = vq_cls()
        self.num_quantizers = num_quantizers
        self.register_buffer("warmed_up", torch.tensor(0))
        self.register_buffer("enabled", torch.tensor(0))
        self.noise_augmentation = noise_augmentation

    @torch.jit.ignore
    def reparametrize(self, z):
        if self.enabled:
            z, diff, _ = self.rvq(z)
        else:
            diff = torch.zeros_like(z).mean()

        if self.noise_augmentation:
            noise = torch.randn(z.shape[0], self.noise_augmentation,
                                z.shape[-1]).type_as(z)
            z = torch.cat([z, noise], 1)

        return z, diff

    def set_warmed_up(self, state: bool):
        state = torch.tensor(int(state), device=self.warmed_up.device)
        self.warmed_up = state

    def forward(self, x):
        z = self.encoder(x)
        return z


class SphericalEncoder(nn.Module):

    def __init__(self, encoder_cls: Callable[[], nn.Module], n_channels: int = 1) -> None:
        super().__init__()
        self.encoder = encoder_cls(n_channels=n_channels)

    def reparametrize(self, z):
        norm_z = z / torch.norm(z, p=2, dim=1, keepdim=True)
        reg = torch.zeros_like(z).mean()
        return norm_z, reg

    def set_warmed_up(self, state: bool):
        pass

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return z

@gin.configurable
class Snake(nn.Module):

    def __init__(self, dim: int, init:float=1) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim, 1).mul_(init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * (self.alpha *
                                                       x).sin().pow(2)


class AdaptiveInstanceNormalization(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("mean_x", torch.zeros(cc.MAX_BATCH_SIZE, dim, 1))
        self.register_buffer("std_x", torch.ones(cc.MAX_BATCH_SIZE, dim, 1))
        self.register_buffer("learn_x", torch.zeros(1))
        self.register_buffer("num_update_x", torch.zeros(1))

        self.register_buffer("mean_y", torch.zeros(cc.MAX_BATCH_SIZE, dim, 1))
        self.register_buffer("std_y", torch.ones(cc.MAX_BATCH_SIZE, dim, 1))
        self.register_buffer("learn_y", torch.zeros(1))
        self.register_buffer("num_update_y", torch.zeros(1))

    def update(self, target: torch.Tensor, source: torch.Tensor,
               num_updates: torch.Tensor) -> None:
        bs = source.shape[0]
        target[:bs] += (source - target[:bs]) / (num_updates + 1)

    def reset_x(self):
        self.mean_x.zero_()
        self.std_x.zero_().add_(1)
        self.num_update_x.zero_()

    def reset_y(self):
        self.mean_y.zero_()
        self.std_y.zero_().add_(1)
        self.num_update_y.zero_()

    def transfer(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]

        x = (x - self.mean_x[:bs]) / (self.std_x[:bs] + 1e-5)
        x = x * self.std_y[:bs] + self.mean_y[:bs]

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return x

        if self.learn_y:
            mean = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True)

            self.update(self.mean_y, mean, self.num_update_y)
            self.update(self.std_y, std, self.num_update_y)
            self.num_update_y += 1

            return x

        else:
            if self.learn_x:
                mean = x.mean(-1, keepdim=True)
                std = x.std(-1, keepdim=True)

                self.update(self.mean_x, mean, self.num_update_x)
                self.update(self.std_x, std, self.num_update_x)
                self.num_update_x += 1

            if self.num_update_x and self.num_update_y:
                x = self.transfer(x)

            return x


def leaky_relu(dim: int, alpha: float):
    return nn.LeakyReLU(alpha)


def unit_norm_vector_to_angles(x: torch.Tensor) -> torch.Tensor:
    norms = x.flip(1).pow(2)
    norms[:, 1] += norms[:, 0]
    norms = norms[:, 1:]
    norms = norms.cumsum(1).flip(1).sqrt()
    angles = torch.arccos(x[:, :-1] / norms)
    angles[:, -1] = torch.where(
        x[:, -1] >= 0,
        angles[:, -1],
        2 * np.pi - angles[:, -1],
    )
    angles[:, :-1] = angles[:, :-1] / np.pi
    angles[:, -1] = angles[:, -1] / (2 * np.pi)
    return 2 * (angles - .5)


def angles_to_unit_norm_vector(angles: torch.Tensor) -> torch.Tensor:
    angles = (angles / 2 + .5) % 1
    angles[:, :-1] = angles[:, :-1] * np.pi
    angles[:, -1] = angles[:, -1] * (2 * np.pi)
    cos = angles.cos()
    sin = angles.sin().cumprod(dim=1)
    cos = torch.cat([
        cos,
        torch.ones(cos.shape[0], 1, cos.shape[-1]).type_as(cos),
    ], 1)
    sin = torch.cat([
        torch.ones(sin.shape[0], 1, sin.shape[-1]).type_as(sin),
        sin,
    ], 1)
    return cos * sin


def wrap_around_value(x: torch.Tensor, value: float = 1) -> torch.Tensor:
    return (x + value) % (2 * value) - value