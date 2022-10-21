import torch
from torch.utils.data import DataLoader, random_split

from rave.model import RAVE
from rave.core import random_phase_mangle, EMAModelCheckPoint
from rave.core import search_for_run

from udls import SimpleDataset, simple_audio_preprocess
from effortless_config import Config, setting
import pytorch_lightning as pl
from os import environ, path
import os
import numpy as np

import GPUtil as gpu

from udls.transforms import Compose, RandomApply, Dequantize, RandomCrop

if __name__ == "__main__":

    class args(Config):
        groups = ["small", "large"]

        # number of channels in PQMF filter
        DATA_SIZE = 16
        # hidden layer width for the encoder and generator
        # if CAPACITY == DATA_SIZE, data dimension is preserved throughout,
        # modulo effect of NARROW and LATENT_SIZE
        CAPACITY = setting(default=32, small=16, large=32)
        # extra inner width in resblocks
        BOOM = 2
        # number of latent dimensions before pruning
        LATENT_SIZE = 128
        # passed directly to conv layers apparently
        # guessing this might be set to false if there are normalization layers
        # that make the bias redundant,
        # not sure why you would set this though since there is no option to
        # change the norm layers (except in the encoder, below)
        BIAS = True
        # None for weight norm only,
        # 'batch' for batch norm only (original version),
        # 'instance' for weight+instance norm
        ENCODER_NORM = None
        # enables causal convolutions, also lowers quality of PQMF, which reduces latency of the inverse filter (?)
        NO_LATENCY = False
        # stride/upsample factor between blocks in the encoder and generator. also determines depth of encoder/generator
        RATIOS = setting(
            default=[4, 4, 4, 2],
            small=[4, 4, 4, 2],
            large=[4, 4, 2, 2, 2],
        )
        # reduction in data dim in encoder / increase (reversed) in generator
        NARROW = setting(
            default=[2, 2, 2, 1],
            small=[2, 2, 2, 1],
            large=[2, 2, 2, 1, 1],
        )
        #low and high values for the cyclic beta-VAE objective
        MIN_KL = 1e-1
        MAX_KL = 1e-1
        # use a different parameterization and compute the sample KLD instead of analytic
        SAMPLE_KL = False
        # use the kld term from http://arxiv.org/abs/1703.09194
        # (SAMPLE_KL must be true)
        PATH_DERIVATIVE = False
        # this is here for inference I guess? set to 0 for training?
        CROPPED_LATENT_SIZE = 0
        # whether to include the discriminator feature-matching loss as part of loss_gen
        FEATURE_MATCH = True
        # architectural parameter for the generator (specifically, the ‘loudness’ branch)
        LOUD_STRIDE = 1
        # enables the noise branch of the generator during training
        USE_NOISE = True
        # downsampling ratios / network depth for the noise branch of the generator
        NOISE_RATIOS = [4, 4, 4]
        # number of noise bands *per* PQMF band in the generator (?)
        # 0 for no noise generator at all
        NOISE_BANDS = 5

        # CAPACITY but for the discriminator
        D_CAPACITY = 16
        # interacts with D_CAPACITY and D_N_LAYERS to set the layer widths, conv groups, and strides in discriminator
        # D_MULTIPLIER = 4
        # discriminator depth
        # D_N_LAYERS = 4
        # stacked discriminator pooling factor
        # D_STACK_FACTOR = 2

        # changes the discriminator to operate on (real, fake) vs (fake, fake) 
        # pairs, which has the effect of making it a conditional GAN:
        # it learns whether y is a realistic reconstruction from z,
        # not just whether it is realistic audio.
        # using pairs in the audio domain is convenient since it requires almost
        # no change to the architecture.
        # (not sure how this interacts with FEATURE_MATCH)
        PAIR_DISCRIMINATOR = False
        # changes the distance losses to use the generalized energy distance
        # from http://arxiv.org/abs/2008.01160
        # this should correspond to a more expressive likelihood
        # (and possibly be more compatible with the adversarial loss)
        GED = False
        # use GAN loss for generator training
        # if this is False but FEATURE_MATCH is True,
        # there will still be a discriminator
        ADVERSARIAL_LOSS = False
        # type of GAN loss
        MODE = "hinge"
        # stop encoder training
        FREEZE_ENCODER = False

        # this only affects KL annealing schedule now
        WARMUP = setting(default=500000, small=500000, large=1500000)
        # steps to cycle the KLD if MIN_KL and MAX_KL differ
        # KL_CYCLE = 50000
        
        # checkpoint to resume training from
        CKPT = None
        # path to store preprocessed dataset, or to already preprocessed data
        PREPROCESSED = None
        TEST_PREPROCESSED = None
        # path to raw dataset
        WAV = None
        TEST_WAV = None
        # number of samples in test set
        N_TEST = -1
        # audio sample rate
        SR = 48000
        # end training after this many iterations
        MAX_STEPS = setting(default=3000000, small=3000000, large=6000000)
        # run validation every so many iterations
        VAL_EVERY = 10000
        
        # batch length in audio samples
        N_SIGNAL = 65536
        # batch size
        BATCH = 8
        # generator+encoder learning rate
        GEN_LR = 1e-4
        # discriminator learning rate
        DIS_LR = 1e-4
        # generator+encoder beta parameters for Adam optimizer
        GEN_ADAM_BETAS = [0.5, 0.9]
        #  discriminator beta parameters for Adam optimizer
        DIS_ADAM_BETAS = [0.5, 0.9]
        # L2 norm to clip gradient
        # (separately for encoder, generator, discriminator)
        GRAD_CLIP = None

        # descriptive name for run
        NAME = None

        LOGDIR = "runs"

    args.parse_args()

    assert args.NAME is not None
    model = RAVE(
        data_size=args.DATA_SIZE,
        capacity=args.CAPACITY,
        boom=args.BOOM,
        latent_size=args.LATENT_SIZE,
        ratios=args.RATIOS,
        narrow=args.NARROW,
        bias=args.BIAS,
        encoder_norm=args.ENCODER_NORM,
        loud_stride=args.LOUD_STRIDE,
        use_noise=args.USE_NOISE,
        noise_ratios=args.NOISE_RATIOS,
        noise_bands=args.NOISE_BANDS,
        d_capacity=args.D_CAPACITY,
        # d_multiplier=args.D_MULTIPLIER,
        # d_n_layers=args.D_N_LAYERS,
        # d_stack_factor=args.D_STACK_FACTOR,
        pair_discriminator=args.PAIR_DISCRIMINATOR,
        ged=args.GED,
        adversarial_loss=args.ADVERSARIAL_LOSS,
        freeze_encoder=args.FREEZE_ENCODER,
        warmup=args.WARMUP,
        # kl_cycle=args.KL_CYCLE,
        mode=args.MODE,
        no_latency=args.NO_LATENCY,
        sr=args.SR,
        min_kl=args.MIN_KL,
        max_kl=args.MAX_KL,
        sample_kl=args.SAMPLE_KL,
        path_derivative=args.PATH_DERIVATIVE,
        cropped_latent_size=args.CROPPED_LATENT_SIZE,
        feature_match=args.FEATURE_MATCH,
        gen_lr=args.GEN_LR,
        dis_lr=args.DIS_LR,
        gen_adam_betas=args.GEN_ADAM_BETAS,
        dis_adam_betas=args.DIS_ADAM_BETAS,
        grad_clip=args.GRAD_CLIP,
    )

    x = {
        'source':torch.zeros(args.BATCH, 2**14),
        'target':torch.zeros(args.BATCH, 2**14)
    }

    model.validation_step(x, 0, 0)

    preprocess = lambda name: simple_audio_preprocess(
        args.SR,
        2 * args.N_SIGNAL,
    )(name).astype(np.float16)

    dataset = SimpleDataset(
        args.PREPROCESSED,
        args.WAV,
        preprocess_function=preprocess,
        split_set="full",
        transforms=Compose([
            lambda x: x.astype(np.float32),
            RandomCrop(args.N_SIGNAL),
            # RandomApply(
            lambda x: {
                tag: Dequantize(16)(
                    random_phase_mangle(x, 20, 2000, .99, args.SR)
                    ).astype(np.float32)
                for tag in ('source', 'target')}
            ])
        )

    def test_preprocess(name):
        s = simple_audio_preprocess(
            args.SR,
            4 * args.N_SIGNAL,
        )(name)
        return None if s is None else s.astype(np.float16)
    # test_preprocess = lambda name: simple_audio_preprocess(
    #     args.SR,
    #     8 * args.N_SIGNAL,
    # )(name).astype(np.float16)

    test = SimpleDataset(
        args.TEST_PREPROCESSED,
        args.TEST_WAV,
        preprocess_function=test_preprocess,
        split_set="full",
        transforms=Compose([
            lambda x: x.astype(np.float32),
            # RandomCrop(args.N_SIGNAL),
            # RandomApply(
            #     lambda x: random_phase_mangle(x, 20, 2000, .99, args.SR),
            #     p=.8,
            # ),
            Dequantize(16),
            lambda x: {
                'source': x.astype(np.float32),
                'target': x.astype(np.float32),
                },
        ]),
    )

    val = max((2 * len(dataset)) // 100, 1)
    train = len(dataset) - val
    train, val = random_split(
        dataset,
        [train, val],
        generator=torch.Generator().manual_seed(42),
    )
    if args.N_TEST > 0:
        test = torch.utils.data.Subset(test, torch.randperm(
            len(test), generator=torch.Generator().manual_seed(42))[:args.N_TEST])

    # train = torch.utils.data.Subset(train, range(1600)) ###DEBUG

    num_workers = 0 if os.name == "nt" else 8
    train = DataLoader(train, args.BATCH, True, drop_last=True, num_workers=num_workers)
    val = DataLoader(val, args.BATCH, False, num_workers=num_workers)
    test = DataLoader(test, args.BATCH//2, False, num_workers=num_workers)

    # CHECKPOINT CALLBACKS
    # validation_checkpoint = pl.callbacks.ModelCheckpoint(
    #     monitor="valid_distance",
    #     filename="best",
    # )
    regular_checkpoint = pl.callbacks.ModelCheckpoint(
        filename="{epoch}", save_top_k=-1, every_n_epochs=30
        )
    last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")

    # fix torch device order to be same as nvidia-smi order
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    CUDA = gpu.getAvailable(maxMemory=.05)
    VISIBLE_DEVICES = environ.get("CUDA_VISIBLE_DEVICES", "")

    if VISIBLE_DEVICES:
        use_gpu = int(int(VISIBLE_DEVICES) >= 0)
    elif len(CUDA):
        environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])
        use_gpu = 1
    elif torch.cuda.is_available():
        print("Cuda is available but no fully free GPU found.")
        print("Training may be slower due to concurrent processes.")
        use_gpu = 1
    else:
        print("No GPU found.")
        use_gpu = 0

    val_check = {}
    if len(train) > args.VAL_EVERY:
        val_check["val_check_interval"] = args.VAL_EVERY
    else:
        nepoch = args.VAL_EVERY // len(train)
        val_check["check_val_every_n_epoch"] = nepoch
    print(val_check)

    trainer = pl.Trainer(
        logger=pl.loggers.TensorBoardLogger(
            path.join(args.LOGDIR, args.NAME), name="rave"),
        gpus=use_gpu,
        callbacks=[regular_checkpoint, last_checkpoint],
        # callbacks=[validation_checkpoint, last_checkpoint],
        max_epochs=100000,
        max_steps=args.MAX_STEPS,
        num_sanity_val_steps=4,
        log_every_n_steps=10,
        **val_check,
    )

    # run = search_for_run(args.CKPT, mode="epoch")
    run = search_for_run(args.CKPT, mode="last")
    # if run is None: run = search_for_run(args.CKPT, mode="best")
    if run is not None:
        step = torch.load(run, map_location='cpu')["global_step"]
        trainer.fit_loop.epoch_loop._batches_that_stepped = step #???

    trainer.fit(model, train, [val, test], ckpt_path=run)
