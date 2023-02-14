import torch
import torch.nn as nn
from effortless_config import Config
import logging
from termcolor import colored
import cached_conv as cc

logging.basicConfig(level=logging.INFO,
                    format=colored("[%(relativeCreated).2f] ", "green") +
                    "%(message)s")

logging.info("exporting model")


class args(Config):
    RUN = None
    # audio sample rate -- the exported model will convert to/from
    # the RAVE model sample rate
    SR = None
    # should be true for realtime?
    CACHED = False
    # this is the proportion of variance to preserve when choosing num. latents
    # appears the number of dimensions is rounded up to the nearest power of 2. 
    # ignored if the checkpoint loaded has already had the latent space cropped
    FIDELITY = .95
    # included in output filename
    NAME = "vae"
    # if True, the pseudo-stereo effect is baked into the decode method
    STEREO = False
    # if True, the latent is sampled at zero temperature
    # and the noise branch of the decoder is turned off (!)
    DETERMINISTIC = False
    # 
    USE_PCA = True


args.parse_args()
cc.use_cached_conv(args.CACHED)

from rave.model import RAVE, remove_weight_norm
from rave.resample import Resampling
from rave.core import search_for_run

import numpy as np
import math


class TraceModel(nn.Module):
    def __init__(self, pretrained: RAVE, resample: Resampling,
                 fidelity: float, use_pca: bool):
        super().__init__()

        latent_size = pretrained.latent_size
        self.resample = resample

        self.pqmf = pretrained.pqmf
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        self.gimbal = pretrained.gimbal
        self.prior_net = pretrained.prior

        self.register_buffer("kld_idxs", pretrained.kld_idxs)

        self.register_buffer("latent_pca", pretrained.latent_pca)
        self.register_buffer("latent_mean", pretrained.latent_mean)
        self.register_buffer("latent_size", torch.tensor(latent_size))
        self.register_buffer(
            "sampling_rate",
            torch.tensor(self.resample.taget_sr),
        )
        try:
            self.register_buffer("max_batch_size",
                                 torch.tensor(cc.MAX_BATCH_SIZE))
        except:
            print(
                "You should upgrade cached_conv if you want to use RAVE in batch mode !"
            )
            self.register_buffer("max_batch_size", torch.tensor(1))
        self.trained_cropped = bool(pretrained.cropped_latent_size)
        self.deterministic = args.DETERMINISTIC

        self.use_pca = use_pca
        if self.trained_cropped:
            self.cropped_latent_size = pretrained.cropped_latent_size
        else:
            if int(fidelity)==fidelity and fidelity > 1:
                self.cropped_latent_size = int(fidelity)
            else:
                if use_pca:
                    latent_size = np.argmax(pretrained.fidelity.numpy() > fidelity)
                    latent_size = 2**math.ceil(math.log2(latent_size))
                    self.cropped_latent_size = latent_size
                else:
                    kld_fid = (pretrained.kld_values / pretrained.kld_values.sum()).cumsum(0)
                    latent_size = np.argmax(kld_fid.numpy() > fidelity)
                    print(pretrained.kld_values, latent_size)
                    latent_size = 2**math.ceil(math.log2(latent_size))
                    self.cropped_latent_size = latent_size

        x = torch.zeros(1, 1, 2**14)
        z = self.encode(x)
        ratio = x.shape[-1] // z.shape[-1]

        self.register_buffer("last_z",
            torch.zeros(cc.MAX_BATCH_SIZE, self.latent_size, 1))

        self.register_buffer(
            "encode_params",
            torch.tensor([
                1,
                1,
                self.cropped_latent_size,
                ratio,
            ]))

        self.register_buffer(
            "decode_params",
            torch.tensor([
                self.cropped_latent_size,
                ratio,
                2 if args.STEREO else 1,
                1,
            ]))


        self.register_buffer("forward_params",
                             torch.tensor([1, 1, 2 if args.STEREO else 1, 1]))

        self.stereo = args.STEREO

    def post_process_distribution(self, mean, scale):
        # std = nn.functional.softplus(scale) + 1e-4
        std = scale.exp()
        return mean, std

    def reparametrize(self, mean, std):
        # var = std * std
        # logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        # kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return z#, kl

    def log_dens(self, z, mean, scale):
        log_std = scale.clamp(-14, 3)
        return -0.5 * (z-mean)**2 * (-2*log_std).exp() - log_std        

    def _encode(self, x):
        x = self.resample.from_target_sampling_rate(x)

        if self.pqmf is not None:
            x = self.pqmf(x)

        mean, scale = self.encoder(x).chunk(2,1)
        if self.gimbal is not None:
            mean, scale = self.gimbal(mean, scale)        
        mean, std = self.post_process_distribution(mean, scale)

        if self.deterministic:
            z = mean
        else:
            z = self.reparametrize(mean, std)

        return z

    @torch.jit.export
    def encode(self, x):
        z = self._encode(x)

        if self.use_pca:
            z = z - self.latent_mean.unsqueeze(-1)
            z = nn.functional.conv1d(z, self.latent_pca.unsqueeze(-1))
            z = z[:, :self.cropped_latent_size]
        else:
            z = z[:, self.kld_idxs[:self.cropped_latent_size]]

        return z

    @torch.jit.export
    def perplexity(self, x):
        z = self._encode(x)

        last_z = self.last_z[:x.shape[0]]
        prior_mean, prior_scale = self.prior_net(last_z).chunk(2,1)
        prior_mean, prior_scale  = self.post_process_distribution(
            prior_mean, prior_scale)

        self.last_z[:z.shape[0]] = z

        return -self.log_dens(z, prior_mean, prior_scale).sum()

    @torch.jit.export
    def prior(self, temp: torch.Tensor, n:int=1):
        # TODO: cache prior when also using decode
        # related: prior won't work without decoder as it doesn't set last_z
        # (or won't work with it if it does set last_z)
        # need this to behave when prior+decoder are used together/separately
        last_z = self.last_z[:n]
        prior_mean, prior_scale = self.prior_net(last_z).chunk(2,1)
        prior_mean, prior_scale  = self.post_process_distribution(
            prior_mean, prior_scale)
        z = self.reparametrize(prior_mean, prior_scale*temp)

        self.last_z[:z.shape[0]] = z

        return z[:, self.kld_idxs[:self.cropped_latent_size]]

    @torch.jit.export
    def decode(self, z):
        if self.trained_cropped:  # PERFORM PCA BEFORE PADDING
            z = nn.functional.conv1d(z, self.latent_pca.T.unsqueeze(-1))
            z = z + self.latent_mean.unsqueeze(-1)

        if self.stereo and z.shape[0] == 1:  # DUPLICATE LATENT PATH
            z = z.expand(2, z.shape[1], z.shape[2])

        # CAT WITH SAMPLES FROM PRIOR DISTRIBUTION
        # pad_size = self.latent_size.item() - z.shape[1]

        # run prior once for each block
        zs = []
        for z_block in z.unbind(-1):
            z_block = z_block[...,None]

            last_z = self.last_z[:z.shape[0]]
            prior_mean, prior_scale = self.prior_net(last_z).chunk(2,1)
            prior_mean, prior_scale = self.post_process_distribution(
                prior_mean, prior_scale)

            if self.deterministic:
                pad_latent = prior_mean
                # pad_latent = torch.zeros(
                #     z.shape[0],
                #     pad_size,
                #     z.shape[-1],
                #     device=z.device,
                # )
            else:
                pad_latent = self.reparametrize(prior_mean, prior_scale)
                # pad_latent = torch.randn(
                #     z.shape[0],
                #     pad_size,
                #     z.shape[-1],
                #     device=z.device,
                # )

            pad_latent = pad_latent[:, self.kld_idxs[self.cropped_latent_size:]]

            # print(last_z.shape, z.shape, pad_latent.shape)

            z = torch.cat([z_block, pad_latent], 1)
            zs.append(z)

            self.last_z[:z.shape[0]] = z

        z = torch.cat(zs, -1)

        if self.use_pca:
            if not self.trained_cropped:  # PERFORM PCA AFTER PADDING
                z = nn.functional.conv1d(z, self.latent_pca.T.unsqueeze(-1))
                z = z + self.latent_mean.unsqueeze(-1)
        else:
            z = z[:, self.kld_idxs.argsort()]

        if self.gimbal is not None:
            z = self.gimbal.inv(z)  

        x = self.decoder(z, add_noise=not self.deterministic)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x)

        x = self.resample.to_target_sampling_rate(x)

        if self.stereo:
            x = x.permute(1, 0, 2)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))


logging.info("loading model from checkpoint")

RUN = search_for_run(args.RUN)
logging.info(f"using {RUN}")
model = RAVE.load_from_checkpoint(RUN, script=False, strict=False).eval()

logging.info("flattening weights")
for m in model.modules():
    if hasattr(m, "weight_g"):
        remove_weight_norm(m)

logging.info("warmup forward pass")
x = torch.zeros(1, 1, model.block_size())
if model.pqmf is not None:
    x = model.pqmf(x)

z = model.reparametrize(*model.split_params(model.encoder(x)))

if args.STEREO:
    z = z.expand(2, *z.shape[1:])

y = model.decoder(z)

if model.pqmf is not None:
    y = model.pqmf.inverse(y)

model.discriminator = None

sr = model.sr

if args.SR is not None:
    target_sr = int(args.SR)
else:
    target_sr = sr

logging.info("build resampling model")
resample = Resampling(target_sr, sr)
x = torch.zeros(1, 1, model.block_size())
resample.to_target_sampling_rate(resample.from_target_sampling_rate(x))

logging.info("script model")
model = TraceModel(model, resample, args.FIDELITY, args.USE_PCA)
model(x)

model = torch.jit.script(model)
logging.info(f"save rave_{args.NAME}.ts")
model.save(f"rave_{args.NAME}.ts")
