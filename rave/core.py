import torch
import torch.nn as nn
import torch.fft as fft
from einops import rearrange
import numpy as np
from random import random
from scipy.signal import lfilter
from pytorch_lightning.callbacks import ModelCheckpoint
import librosa as li
from pathlib import Path

def gauss_window(pts, scale, device='cpu'):
    x = torch.linspace(-scale, scale, pts, device=device)
    return (-x*x).exp()

def angle_wrap(x):
    pi = np.pi
    return (x + pi)%(2*pi) - pi

# def get_inst_freq(audio, nfft=4096, overlap=31, win_param=6, k=3, s=2):
#     hop = nfft//overlap
#     win_size = nfft
#     h = gauss_window(win_size, win_param, device=audio.device)
#     Dh = gauss_window(win_size+1, win_param, device=audio.device).diff()
#     S = torch.stft(
#         audio, nfft, hop_length=hop, win_length=win_size, window=h, return_complex=True, center=False
#         )
#     dS = torch.stft(
#         audio, nfft, hop_length=hop, win_length=win_size, window=Dh, return_complex=True, center=False
#         )

#     # avoid nans
#     cond = S.abs()>1e-7
#     S_masked = S.where(cond, S.new_ones(1))

#     bin_centers = torch.linspace(0, np.pi, nfft//2+1, device=audio.device)
#     phase_dev = (dS/S_masked).imag 
#     freq = phase_dev + bin_centers[:,None]

#     # reduce
#     freq = freq.unfold(1, k, s).unfold(2, k, s)
#     med = freq.reshape(*freq.shape[:-2], -1).median(-1)
#     freq = med.values
    
#     # could do math on the indices instead of unfolding here
#     mag = S.unfold(1, k, s).unfold(2, k, s)
#     mag = mag.reshape(*mag.shape[:-2], -1)
#     mag = mag.gather(-1, med.indices[...,None])[...,0].abs()
    
#     # return angle_wrap(torch.nan_to_num(phase_dev, posinf=0, neginf=0)), S.abs()
#     # return torch.nan_to_num(phase_dev, posinf=0, neginf=0), S.abs()
#     return torch.nan_to_num(freq, posinf=0, neginf=0), mag

def get_inst_freq_patches(audio, 
        nfft=2048, overlap=17, win_param=5, k=5, s=4, drop=2):
    hop = nfft//overlap
    win_size = nfft
    h = gauss_window(win_size, win_param, device=audio.device)
    Dh = gauss_window(win_size+1, win_param, device=audio.device).diff()
    S = torch.stft(
        audio, nfft, hop_length=hop, win_length=win_size, window=h, return_complex=True, center=False)
    dS = torch.stft(
        audio, nfft, hop_length=hop, win_length=win_size, window=Dh, return_complex=True, center=False)

    # avoid nans
    cond = S.abs()>1e-7
    S_masked = S.where(cond, S.new_ones(1))

    bin_centers = torch.linspace(0, np.pi, nfft//2+1, device=audio.device)
    freq_dev = (dS/S_masked).imag 
    freq = freq_dev + bin_centers[:,None]

    # get sorted patches
    patches = freq.unfold(1, k, s).unfold(2, k, s)
    patches = patches.reshape(*patches.shape[:-2], -1)
    patches = patches.sort(-1).values[...,drop:-drop] # drop extrema
    
    mag = nn.functional.avg_pool2d(S.abs(), k, s)
    # mag = S.unfold(1, k, s).unfold(2, k, s)
    # mag = mag.reshape(*mag.shape[:-2], -1).mean(-1)
    
    return torch.nan_to_num(patches, posinf=0, neginf=0), mag



def mod_sigmoid(x):
    return 2 * torch.sigmoid(x)**2.3 + 1e-7


def multiscale_stft(signal, scales, overlap):
    """
    Compute a stft on several scales, with a constant overlap value.
    Parameters
    ----------
    signal: torch.Tensor
        input signal to process ( B X C X T )
    
    scales: list
        scales to use
    overlap: float
        overlap between windows ( 0 - 1 )
    """
    bc = signal.shape[:2]
    signal = rearrange(signal, "b c t -> (b c) t")
    stfts = []
    for s in scales:
        S = torch.stft(
            signal,
            s,
            int(s * (1 - overlap)),
            s,
            torch.hann_window(s, device=signal.device, dtype=signal.dtype),
            center=True,
            normalized=True,
            return_complex=True,
        ).abs()
        S = S.reshape(*bc, *S.shape[-2:]) # B x C x F x T
        stfts.append(S)
    return stfts


def random_angle(min_f=20, max_f=8000, sr=24000):
    min_f = np.log(min_f)
    max_f = np.log(max_f)
    rand = np.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * np.pi * rand / sr
    return rand


def pole_to_z_filter(omega, amplitude=.9):
    z0 = amplitude * np.exp(1j * omega)
    a = [1, -2 * np.real(z0), abs(z0)**2]
    b = [abs(z0)**2, -2 * np.real(z0), 1]
    return b, a


def random_phase_mangle(x, min_f, max_f, amp, sr):
    angle = random_angle(min_f, max_f, sr)
    b, a = pole_to_z_filter(angle, amp)
    return lfilter(b, a, x)


class EMAModelCheckPoint(ModelCheckpoint):

    def __init__(self, model: torch.nn.Module, alpha=.999, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()
        self.model = model
        self.alpha = alpha

    def on_train_batch_end(self, *args, **kwargs):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if n in self.shadow:
                    self.shadow[n] *= self.alpha
                    self.shadow[n] += (1 - self.alpha) * p.data

    def on_validation_epoch_start(self, *args, **kwargs):
        self.swap()

    def on_validation_epoch_end(self, *args, **kwargs):
        self.swap()

    def swap(self):
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                tmp = p.data.clone()
                p.data.copy_(self.shadow[n])
                self.shadow[n] = tmp

    def save_checkpoint(self, *args, **kwargs):
        self.swap()
        super().save_checkpoint(*args, **kwargs)
        self.swap()


class Loudness(nn.Module):

    def __init__(self, sr, block_size, n_fft=2048):
        super().__init__()
        self.sr = sr
        self.block_size = block_size
        self.n_fft = n_fft

        f = np.linspace(0, sr / 2, n_fft // 2 + 1) + 1e-7
        a_weight = li.A_weighting(f).reshape(-1, 1)

        self.register_buffer("a_weight", torch.from_numpy(a_weight).float())
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, x):
        x = torch.stft(
            x.squeeze(1),
            self.n_fft,
            self.block_size,
            self.n_fft,
            center=True,
            window=self.window,
            return_complex=True,
        ).abs()
        x = torch.log(x + 1e-7) + self.a_weight
        return torch.mean(x, 1, keepdim=True)


def amp_to_impulse_response(amp, target_size):
    """
    transforms frequecny amps to ir on the last dimension
    """
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(
        amp,
        (0, int(target_size) - int(filter_size)),
    )
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    """
    convolves signal by kernel on the last dimension
    """
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = fft.irfft(fft.rfft(signal) * fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


def search_for_run(run_path, mode="last"):
    if run_path is None: return None
    if ".ckpt" in run_path: return run_path
    ckpts = map(str, Path(run_path).rglob("*.ckpt"))
    ckpts = filter(lambda e: mode in e, ckpts)
    ckpts = sorted(ckpts)
    if len(ckpts): return ckpts[-1]
    else: return None


def get_beta_kl(step, warmup, min_beta, max_beta):
    if step > warmup: return max_beta
    t = step / warmup
    min_beta_log = np.log(min_beta)
    max_beta_log = np.log(max_beta)
    beta_log = t * (max_beta_log - min_beta_log) + min_beta_log
    return np.exp(beta_log)


def get_beta_kl_cyclic(step, cycle_size, min_beta, max_beta):
    return get_beta_kl(step % cycle_size, cycle_size // 2, min_beta, max_beta)


def get_beta_kl_cyclic_annealed(step, cycle_size, warmup, min_beta, max_beta):
    min_beta = get_beta_kl(step, warmup, min_beta, max_beta)
    return get_beta_kl_cyclic(step, cycle_size, min_beta, max_beta)
