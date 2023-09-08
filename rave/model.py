import math
from time import time
from typing import Callable, Dict, Optional

import gin
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from sklearn.decomposition import PCA

import rave.core

from . import blocks

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


class WarmupCallback(pl.Callback):

    def __init__(self) -> None:
        super().__init__()
        self.state = {'training_steps': 0}

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:
        if self.state['training_steps'] >= pl_module.warmup:
            pl_module.warmed_up = True
        self.state['training_steps'] += 1

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)


class QuantizeCallback(WarmupCallback):

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:

        if pl_module.warmup_quantize is None: return

        if self.state['training_steps'] >= pl_module.warmup_quantize:
            if isinstance(pl_module.encoder, blocks.DiscreteEncoder):
                pl_module.encoder.enabled = torch.tensor(1).type_as(
                    pl_module.encoder.enabled)
        self.state['training_steps'] += 1


@gin.configurable
class BetaWarmupCallback(pl.Callback):

    def __init__(self, initial_value: float, target_value: float,
                 warmup_len: int) -> None:
        super().__init__()
        self.state = {'training_steps': 0}
        self.warmup_len = warmup_len
        self.initial_value = initial_value
        self.target_value = target_value

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:
        self.state['training_steps'] += 1
        if self.state["training_steps"] >= self.warmup_len:
            pl_module.beta_factor = self.target_value
            return

        warmup_ratio = self.state["training_steps"] / self.warmup_len

        beta = math.log(self.initial_value) * (1 - warmup_ratio) + math.log(
            self.target_value) * warmup_ratio
        pl_module.beta_factor = math.exp(beta)

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

# TODO: add to gin
class PitchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            'subharmonics', torch.arange(2,8).log2())
        self.register_buffer(
            'intervals', torch.tensor([1/2, 1, 6/7, 6/5, 3, 6, 12]))

    def forward(self, pitch):
        """transformation from pitch in hz to latent space for decoder"""
        # return pitch.log() - np.log(50.)
        octave = pitch.log2()
        z = torch.cat(( # subharmonics
            octave,
            octave - self.subharmonics[0],
            octave - self.subharmonics[1],
            octave - self.subharmonics[2],
            octave - self.subharmonics[3],
            octave - self.subharmonics[4],
            octave - self.subharmonics[5],
        ), 1) 
        z = torch.cat(( # musical intervals
            z * self.intervals[0], # 1 octave difference
            z * self.intervals[1], # chroma
            z * self.intervals[2], # fifths
            z * self.intervals[3], # fourths
            z * self.intervals[4], # whole tones
            z * self.intervals[5], # semitones
            z * self.intervals[6], # microtones
        ), 1)
        z = (torch.cat((z, z+0.25), 1) * 2*np.pi).cos() # quadrature
        z = torch.cat((
            z,
            octave - 7.3, # log
            (pitch - 50)/225 - 1, # linear
        ), 1)
        return z

@gin.configurable
class RAVE(pl.LightningModule):

    def __init__(
        self,
        latent_size,
        sampling_rate,
        encoder,
        decoder,
        discriminator,
        phase_1_duration,
        gan_loss,
        valid_signal_crop,
        feature_matching_fun,
        num_skipped_features,
        audio_distance: Callable[[], nn.Module],
        multiband_audio_distance: Callable[[], nn.Module],
        weights: Dict[str, float],
        warmup_quantize: Optional[int] = None,
        pqmf: Optional[Callable[[], nn.Module]] = None,
        update_discriminator_every: int = 2,
        enable_pqmf_encode: bool = True,
        enable_pqmf_decode: bool = True,
        use_crepe:bool = False,
        latent_aug:bool = False,
    ):
        super().__init__()

        self.pqmf = None
        if pqmf is not None:
            self.pqmf = pqmf()

        self.encoder = encoder()
        self.decoder = decoder()
        self.discriminator = discriminator()

        if use_crepe:
            import torchcrepe
            self.pitch = torchcrepe.CrepePredict(
                sample_rate=sampling_rate, 
                hop_length=self.block_size,
                fmin=50, fmax=550, 
                pad=True,
                model='full')
            self.hz_to_z = PitchEmbed()
        else:
            self.pitch = None

        self.audio_distance = audio_distance()
        self.multiband_audio_distance = multiband_audio_distance()

        self.gan_loss = gan_loss

        self.register_buffer("latent_pca", torch.eye(latent_size))
        self.register_buffer("latent_mean", torch.zeros(latent_size))
        self.register_buffer("fidelity", torch.zeros(latent_size))

        self.latent_size = latent_size

        self.automatic_optimization = False

        # SCHEDULE
        self.warmup = phase_1_duration
        self.warmup_quantize = warmup_quantize
        self.weights = weights

        self.warmed_up = False

        # CONSTANTS
        self.sr = sampling_rate
        self.valid_signal_crop = valid_signal_crop
        self.feature_matching_fun = feature_matching_fun
        self.num_skipped_features = num_skipped_features
        self.update_discriminator_every = update_discriminator_every
        self.latent_aug = latent_aug

        self.eval_number = 0
        self.beta_factor = 1.
        self.integrator = None

        self.enable_pqmf_encode = enable_pqmf_encode
        self.enable_pqmf_decode = enable_pqmf_decode

        self.register_buffer("receptive_field", torch.tensor([0, 0]).long())

    def configure_optimizers(self):
        gen_p = list(self.encoder.parameters())
        gen_p += list(self.decoder.parameters())
        dis_p = list(self.discriminator.parameters())

        gen_opt = torch.optim.Adam(gen_p, 1e-4, (.5, .9))
        dis_opt = torch.optim.Adam(dis_p, 1e-4, (.5, .9))

        return gen_opt, dis_opt

    def split_features(self, features):
        feature_real = []
        feature_fake = []
        for scale in features:
            true, fake = zip(*map(
                lambda x: torch.split(x, x.shape[0] // 2, 0),
                scale,
            ))
            feature_real.append(true)
            feature_fake.append(fake)
        return feature_real, feature_fake
    
    def split_features_aug(self, features):
        feature_real = []
        feature_fake = []
        feature_fake_aug = []
        for scale in features:
            true, fake, fake_aug = zip(*map(
                lambda x: torch.split(x, x.shape[0] // 3, 0),
                scale,
            ))
            feature_real.append(true)
            feature_fake.append(fake)
            feature_fake_aug.append(fake_aug)
        return feature_real, feature_fake, feature_fake_aug
    
    @property
    def block_size(self):
        bs = self.encoder.encoder.downsample_factor
        if self.pqmf is not None: 
            bs = bs * self.pqmf.n_band
        return bs
    
    def npz_to_bps(self, npz):
        """convert (double) nats per z frame to bits per second
        """
        # VariationalEncoder computes 2 * KLD as the reg term, 
        # leaving that alone for compatibility but compensating here
        npz = npz / 2
        return (npz 
            * self.sr / self.block_size
            * np.log2(np.e))

    def training_step(self, batch, batch_idx):
        p = Profiler()
        gen_opt, dis_opt = self.optimizers()

        if self.pitch is not None:
            pitch = self.pitch(batch)['sample'][:,None,:]
        else:
            pitch = None

        x = batch.unsqueeze(1)

        if self.pqmf is not None:
            x_multiband = self.pqmf(x)
        else:
            x_multiband = x
        p.tick('decompose')

        self.encoder.set_warmed_up(self.warmed_up)
        self.decoder.set_warmed_up(self.warmed_up)

        # ENCODE INPUT
        if self.enable_pqmf_encode:
            z_pre_reg = self.encoder(x_multiband)
        else:
            z_pre_reg = self.encoder(x)

        z, reg = self.encoder.reparametrize(z_pre_reg)[:2]
        p.tick('encode')

        if self.latent_aug:
            scale = torch.rand(z.shape[0], 1, 1, device=z.device)*2
            bias = torch.randn(z.shape[0], z.shape[1], 1, device=z.device)/2
            z_aug = z * scale + bias
            z = torch.cat((z,z_aug))
            if pitch is not None:
                pitch_scale = 2**(torch.randn(pitch.shape[0], 1, 1, device=z.device)/3)
                pitch_aug = pitch*pitch_scale
                pitch = torch.cat((pitch, pitch_aug))

        # DECODE LATENT
        if pitch is not None:
            pitch_z = self.hz_to_z(pitch)
            z = torch.cat((z, pitch_z), -2)

        y_multiband = self.decoder(z)

        p.tick('decode')

        if self.valid_signal_crop and self.receptive_field.sum():
            x_multiband = rave.core.valid_signal_crop(
                x_multiband,
                *self.receptive_field,
            )
            y_multiband = rave.core.valid_signal_crop(
                y_multiband,
                *self.receptive_field,
            )
            if reg.ndim>2:
                # TODO: this possibly crops too much?
                # some of these latents are within receptive field of
                # the uncropped reconstruction,
                # meaning the model should learn to make them arbitrarily
                # precise,
                # since they are useful for reconstruction but unregularized.
                # i think this can only happen if the encoder receptive field is 
                # longer than the generator though?
                # otherwise the model can't distingish such parts of the posterior, as they would be out of r.f. of the zero padding.
                # maybe it just distorts the mean KLD?
                reg = rave.core.valid_signal_crop(
                    reg,
                    *self.receptive_field,
                    dim=self.block_size
                )

        if self.latent_aug:
            y_multiband, y_multiband_aug = torch.chunk(y_multiband,2)

        if reg.ndim>2:
            # sum over latent
            reg = reg.sum(1)

        # mean over batch, time
        reg = reg.mean()

        p.tick('crop')

        # DISTANCE BETWEEN INPUT AND OUTPUT
        distances = {}

        if self.pqmf is not None:
            multiband_distance = self.multiband_audio_distance(
                x_multiband, y_multiband)
            p.tick('mb distance')

            x = self.pqmf.inverse(x_multiband)
            y = self.pqmf.inverse(y_multiband)
            if self.latent_aug:
                y_aug = self.pqmf.inverse(y_multiband_aug)
            p.tick('recompose')

            for k, v in multiband_distance.items():
                distances[f'multiband_{k}'] = v
        else:
            x = x_multiband
            y = y_multiband
            if self.latent_aug:
                y_aug = y_multiband_aug

        fullband_distance = self.audio_distance(x, y)
        p.tick('fb distance')

        for k, v in fullband_distance.items():
            distances[f'fullband_{k}'] = v

        feature_matching_distance = 0.

        if self.warmed_up:  # DISCRIMINATION
            if self.latent_aug:
                xy = torch.cat([x, y, y_aug], 0)
                features = self.discriminator(xy)
                feature_real, feature_fake = self.split_features(features)
                feature_aug = feature_fake
            else:
                xy = torch.cat([x, y], 0)
                features = self.discriminator(xy)
                feature_real, feature_fake, feature_aug = self.split_features_aug(features)


            loss_dis = 0
            loss_adv = 0

            pred_real = 0
            pred_fake = 0
            pred_aug = 0

            for scale_real, scale_fake, scale_aug in zip(feature_real, feature_fake, feature_aug):
                current_feature_distance = sum(
                    map(
                        self.feature_matching_fun,
                        scale_real[self.num_skipped_features:],
                        scale_fake[self.num_skipped_features:],
                    )) / len(scale_real[self.num_skipped_features:])

                feature_matching_distance = feature_matching_distance + current_feature_distance

                _dis, _adv = self.gan_loss(scale_real[-1], scale_aug[-1])

                pred_real = pred_real + scale_real[-1].mean()
                pred_fake = pred_fake + scale_fake[-1].mean()
                pred_aug = pred_aug + scale_aug[-1].mean()

                loss_dis = loss_dis + _dis
                loss_adv = loss_adv + _adv

            feature_matching_distance = feature_matching_distance / len(
                feature_real)

        else:
            pred_real = torch.tensor(0.).to(x)
            pred_fake = torch.tensor(0.).to(x)
            pred_aug = torch.tensor(0.).to(x)
            loss_dis = torch.tensor(0.).to(x)
            loss_adv = torch.tensor(0.).to(x)
        p.tick('discrimination')

        # COMPOSE GEN LOSS
        loss_gen = {}
        loss_gen.update(distances)
        p.tick('update loss gen dict')

        if reg.item():
            loss_gen['regularization'] = reg * self.beta_factor

        if isinstance(self.encoder, blocks.VariationalEncoder):
            # log the KLD in bits/second
            self.log("kld_bps", self.npz_to_bps(reg.item()))

        if self.warmed_up:
            loss_gen['feature_matching'] = feature_matching_distance
            loss_gen['adversarial'] = loss_adv

        # OPTIMIZATION
        if not (batch_idx %
                self.update_discriminator_every) and self.warmed_up:
            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()
            p.tick('dis opt')
        else:
            gen_opt.zero_grad()
            loss_gen_value = 0.
            for k, v in loss_gen.items():
                loss_gen_value += v * self.weights.get(k, 1.)
            loss_gen_value.backward()
            gen_opt.step()

        # LOGGING
        self.log("beta_factor", self.beta_factor)

        if self.warmed_up:
            self.log("loss_dis", loss_dis)
            self.log("pred_real", pred_real.mean())
            self.log("pred_fake", pred_fake.mean())
            self.log("pred_aug", pred_aug.mean())

        self.log_dict(loss_gen)
        p.tick('logging')

    #### TODO: handle pitch in encode_dist?
    ### these are not used by either training or export?
    ### they are only for testing?

    def encode(self, x, **pitch_kw):
        """returns: latent tensor or tuple of latent,pitch"""
        if self.pqmf is not None and self.enable_pqmf_encode:
            x_bands = self.pqmf(x)
        z, = self.encoder.reparametrize(self.encoder(x_bands))[:1]

        if self.pitch is not None:
            pitch = self.pitch(x.squeeze(1), **pitch_kw)['sample'][:,None,:]
            return z, pitch

        return z
    
    def encode_dist(self, x):
        if self.pqmf is not None and self.enable_pqmf_encode:
            x = self.pqmf(x)
        return self.encoder.params(self.encoder(x))

    def decode(self, z):
        """
        z: latent tensor or tuple of latent,pitch
        """
        if self.pitch is not None:
            z, pitch = z
            pitch_z = self.hz_to_z(pitch)
            z = torch.cat((z, pitch_z), -2)

        y = self.decoder(z)
        if self.pqmf is not None and self.enable_pqmf_decode:
            y = self.pqmf.inverse(y)
        return y

    def forward(self, x):
        return self.decode(self.encode(x))

    def validation_step(self, batch, batch_idx):
        if self.pitch is not None:
            pitch = self.pitch(batch)['sample'][:,None,:]
        else:
            pitch = None

        x = batch.unsqueeze(1)

        if self.pqmf is not None:
            x_multiband = self.pqmf(x)

        if self.enable_pqmf_encode:
            z = self.encoder(x_multiband)

        else:
            z = self.encoder(x)

        if isinstance(self.encoder, blocks.VariationalEncoder):
            mean = torch.split(z, z.shape[1] // 2, 1)[0]
        else:
            mean = None

        z = self.encoder.reparametrize(z)[0]

        if pitch is not None:
            pitch_z = self.hz_to_z(pitch)
            z = torch.cat((z, pitch_z), -2)

        y = self.decoder(z)

        if self.valid_signal_crop and self.receptive_field.sum():
            x_multiband = rave.core.valid_signal_crop(
                x_multiband, *self.receptive_field)
            y = rave.core.valid_signal_crop(y, *self.receptive_field)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x_multiband)
            y = self.pqmf.inverse(y)

        distance = self.audio_distance(x, y)

        full_distance = sum(distance.values())

        if self.trainer is not None:
            self.log('validation', full_distance)

        # put reconstruction first for better logs
        return torch.cat([y, x], -1), mean

    def validation_epoch_end(self, out):
        if not self.receptive_field.sum():
            print("Computing receptive field for this configuration...")
            lrf, rrf = rave.core.get_rave_receptive_field(self)
            self.receptive_field[0] = lrf
            self.receptive_field[1] = rrf
            print(
                f"Receptive field: {1000*lrf/self.sr:.2f}ms <-- x --> {1000*rrf/self.sr:.2f}ms"
            )

        if not len(out): return

        audio, z = list(zip(*out))
        audio = list(map(lambda x: x.cpu(), audio))

        # LATENT SPACE ANALYSIS
        if not self.warmed_up and isinstance(self.encoder,
                                             blocks.VariationalEncoder):
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
                self.log(
                    f"fidelity_{p}",
                    np.argmax(var > p).astype(np.float32),
                )

        y = torch.cat(audio, 0)[:8].reshape(-1).numpy()

        if self.integrator is not None:
            y = self.integrator(y)

        self.logger.experiment.add_audio("audio_val", y, self.eval_number,
                                         self.sr)
        self.eval_number += 1

    def on_fit_start(self):
        tb = self.logger.experiment

        config = gin.operative_config_str()
        config = config.split('\n')
        config = ['```'] + config + ['```']
        config = '\n'.join(config)
        tb.add_text("config", config)

        model = str(self)
        model = model.split('\n')
        model = ['```'] + model + ['```']
        model = '\n'.join(model)
        tb.add_text("model", model)
