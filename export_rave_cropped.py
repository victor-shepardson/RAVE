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
    # should be true for realtime?
    CACHED = True
    # included in output filename
    NAME = "cropped"

args.parse_args()
cc.use_cached_conv(args.CACHED)

from rave.model import RAVE
from rave.core import search_for_run
from rave.weight_norm import remove_weight_norm


class TraceModel(nn.Module):
    def __init__(self, pretrained: RAVE):
        super().__init__()

        self.latent_size = pretrained.latent_size
        self.cropped_latent_size = pretrained.cropped_latent_size


        self.pqmf = pretrained.pqmf
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder

        # self.sample_rate = pretrained.sr
        # self.max_batch_size = cc.MAX_BATCH_SIZE
        self.block_size = pretrained.block_size

        self.register_buffer(
            "sampling_rate", torch.tensor(pretrained.sr))
        try:
            self.register_buffer(
                "max_batch_size", torch.tensor(cc.MAX_BATCH_SIZE))
        except:
            print(
                "You should upgrade cached_conv if you want to use RAVE in batch mode !"
            )
            self.register_buffer("max_batch_size", torch.tensor(1))

        x = torch.zeros(1, 1, 2**14)
        z = self.encode(x)
        ratio = x.shape[-1] // z.shape[-1]

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
                1,
                1,
            ]))

        self.register_buffer("forward_params",
            torch.tensor([1, 1, 1, 1]))

    @torch.jit.export
    def encode(self, x):
        if self.pqmf is not None:
            x = self.pqmf(x)

        z = self.encoder(x)

        return z

    @torch.jit.export
    def decode(self, z):
        pad_size = self.latent_size - z.shape[1]
        pad_latent = torch.randn(
            z.shape[0],
            pad_size,
            z.shape[-1],
            device=z.device,
        )
        z = torch.cat([z, pad_latent], 1)

        y = self.decoder(z, add_noise=True)
        if self.pqmf is not None:
            y = self.pqmf.inverse(y)
        return y

    def forward(self, x):
        return self.decode(self.encode(x))


logging.info("loading model from checkpoint")

RUN = search_for_run(args.RUN)
logging.info(f"using {RUN}")
###
compat_kw = dict(script=False, use_norm_dist=False)
###
model = RAVE.load_from_checkpoint(RUN, **compat_kw, strict=False).eval()

x = torch.zeros(1, 1, 2**14)
model.decode(model.encode(x))

logging.info("flattening weights")
for m in model.modules():
    if hasattr(m, "weight_g"):
        remove_weight_norm(m)

model.decode(model.encode(x))


# logging.info("warmup forward pass")
# x = torch.zeros(1, 1, 2**14)
# if model.pqmf is not None:
#     x = model.pqmf(x)

# # z, _ = model.reparametrize(*model.encoder(x))
# z = model.reparametrize(*model.encode(x))

# y = model.decoder(z)

# if model.pqmf is not None:
#     y = model.pqmf.inverse(y)

model.discriminator = None

sr = model.sr



logging.info("script model")
model = TraceModel(model)
model(x)

model = torch.jit.script(model)
logging.info(f"save rave_{args.NAME}.ts")
model.save(f"rave_{args.NAME}.ts")
