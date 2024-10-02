last merged upstream at v2.3.1 (multichannel)

# Changes in this fork:
* `--val_db_path` option in `rave train` to use a separate preprocessed dataset instead of the 2% training split
* refactor and fix cropping to valid portion of reconstruction and regularization losses
* `--join_short_files` option in `rave preprocess` to use shorter training files by concatenating them before preprocessing
* also log mono mix of audio to tensorboard for multichannel models
* new `rave adapt` script for fitting linear adapters between different models
* sign normalization at export (flip polarity of latents to generally correlate with loudness/brightness)
* changes to cached_conv to reduce latency of `--causal` models by one block
* log KLD measured in bits/second in tensorboard
* in audio logs, place the reconstruction before the original for less biased listening
* trim audio logs to the valid portion
* trim latent space to the valid portion when computing regularization loss
* scale beta (regularization parameter) appropriately with block size
* add several data augmentation options
    * RandomEQ (randomized parametric EQ)
    * RandomDelay (randomized comb delay)
    * RandomGain (randomize gain without peaking over 1)
    * RandomSpeed (random resampling to different speeds)
    * RandomDistort (random drive + dry mix tanh waveshaping)
* add random cropping option to the spectral loss
* reduce default training window slightly (turns random cropping on by default)
* option not to freeze encoder once warmed up
* transfer learning: option to initialize just weights from another checkpoint
* export: option to specify exact number of latent dimensions (instead of fidelity)
* export: add an optional temperature parameter to `encode` (only works for VAE models)
* export: also export an `encode_dist` function which returns both a sample from and parameters of the posterior (only works for VAE models)
* add random seed option to train.py

## Installation

clone the git repo and run `RAVE_VERSION=2.4.0b CACHED_CONV_VERSION=2.6.0b pip install -e RAVE`

## Transfer Learning

See https://huggingface.co/Intelligent-Instruments-Lab/rave-models for pretrained checkpoints.

To use transfer learning, you add 3 flags: `--transfer_ckpt /path/to/checkpoint/ --config /path/to/checkpoint/config.gin --config transfer` . make sure to use all 3. `transfer_ckpt` and the first config will generally be the same path (less the `config.gin` part).

for example:

```python
FROM=XXX/rave-models/checkpoints/organ_archive_b512_r48000 rave train --config $FROM/config.gin --config transfer --config mid_beta  --transfer_ckpt $FROM --db_path XXX --name XXX --gpu XXX
```

this would do transfer learning from the low latency (512 sample block) organ model. You can also add more configs; in the above example `--config mid_beta` is resetting the regularization strength (the pretrained model used a low beta value). You could also adjust the sample rate or do other non-architectural changes. make sure to add these after the first `--config` with the checkpoint path.

## Adapters

`rave adapt` takes two exported RAVE models and produces a linear adapter which converts the output of one encoder to approximate the output of the other. For example,

```rave adapt --from_model /path/to/model1.ts --to_model /path/to/model2.ts --db_path /path/to/train/data --name adapt_1to2```

Will make it so that `model2.decode(adapt_1to2(model1.encode(x)))` approximately reproduces x. The exported adapter is a `nn~` model with a `forward` method which operates in the latent space of the models you pass in.

## Sign Normalization

If you add the `--normalize_signs` flag when using `rave export`, it will flip the sign of each latent so that it correlates with a measure of loudness and brighteness of the audio. This should make the first latent variable always louder in the positive direction, for example, and the others somewhat more predictable in behavior. 

# --- original README follows below ---
![rave_logo](docs/rave.png)

# RAVE: Realtime Audio Variational autoEncoder

Official implementation of _RAVE: A variational autoencoder for fast and high-quality neural audio synthesis_ ([article link](https://arxiv.org/abs/2111.05011)) by Antoine Caillon and Philippe Esling.

If you use RAVE as a part of a music performance or installation, be sure to cite either this repository or the article !

If you want to share / discuss / ask things about RAVE you can do so in our [discord server](https://discord.gg/dhX73sPTBb) !

Please check the FAQ before posting an issue!

## Previous versions

The original implementation of the RAVE model can be restored using

```bash
git checkout v1
```

## Installation

Install RAVE using

```bash
pip install acids-rave
```

**Warning** It is strongly advised to install `torch` and `torchaudio` before `acids-rave`, so you can choose the appropriate version of torch on the [library website](http://www.pytorch.org). For future compatibility with new devices (and modern Python environments), `rave-acids` does not enforce torch==1.13 anymore.

You will need **ffmpeg** on your computer. You can install it locally inside your virtual environment using

```bash
conda install ffmpeg
```

<!-- Detailed instructions to setup a training station for this project are available [here](docs/training_setup.md). -->

## Colab

A colab to train RAVEv2 is now available thanks to [hexorcismos](https://github.com/moiseshorta) !
[![colab_badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ih-gv1iHEZNuGhHPvCHrleLNXvooQMvI?usp=sharing)

## Usage

Training a RAVE model usually involves 3 separate steps, namely _dataset preparation_, _training_ and _export_.

### Dataset preparation

You can know prepare a dataset using two methods: regular and lazy. Lazy preprocessing allows RAVE to be trained directly on the raw files (i.e. mp3, ogg), without converting them first. **Warning**: lazy dataset loading will increase your CPU load by a large margin during training, especially on Windows. This can however be useful when training on large audio corpus which would not fit on a hard drive when uncompressed. In any case, prepare your dataset using

```bash
rave preprocess --input_path /audio/folder --output_path /dataset/path --channels X (--lazy)
```

### Training

RAVEv2 has many different configurations. The improved version of the v1 is called `v2`, and can therefore be trained with

```bash
rave train --config v2 --db_path /dataset/path --out_path /model/out --name give_a_name --channels X
```

We also provide a discrete configuration, similar to SoundStream or EnCodec

```bash
rave train --config discrete ...
```

By default, RAVE is built with non-causal convolutions. If you want to make the model causal (hence lowering the overall latency of the model), you can use the causal mode

```bash
rave train --config discrete --config causal ...
```

New in 2.3, data augmentations are also available to improve the model's generalization in low data regimes. You can add data augmentation by adding augmentation configuration files with the `--augment` keyword

```bash
rave train --config v2 --augment mute --augment compress
```

Many other configuration files are available in `rave/configs` and can be combined. Here is a list of all the available configurations & augmentations :

<table>
<thead>
<tr>
<th>Type</th>
<th>Name</th>
<th>Description</th>
</tr>
</thead>
<tbody>

<tr>
<td rowspan=8>Architecture</td>
<td>v1</td>
<td>Original continuous model</td>
</tr>

<tr>
<td>v2</td>
<td>Improved continuous model (faster, higher quality)</td>
</tr>

<tr>
<td>v2_small</td>
<td>v2 with a smaller receptive field, adpated adversarial training, and noise generator, adapted for timbre transfer for stationary signals</td>
</tr>

<tr>
<td>v2_nopqmf</td>
<td>(experimental) v2 without pqmf in generator (more efficient for bending purposes)</td>
</tr>

<tr>
<td>v3</td>
<td>v2 with Snake activation, descript discriminator and Adaptive Instance Normalization for real style transfer</td>
</tr>

<tr>
<td>discrete</td>
<td>Discrete model (similar to SoundStream or EnCodec)</td>
</tr>

<tr>
<td>onnx</td>
<td>Noiseless v1 configuration for onnx usage</td>
</tr>

<tr>
<td>raspberry</td>
<td>Lightweight configuration compatible with realtime RaspberryPi 4 inference</td>
</tr>

<tr>
<td rowspan=3>Regularization (v2 only)</td>
<td>default</td>
<td>Variational Auto Encoder objective (ELBO)</td>
</tr>

<tr>
<td>wasserstein</td>
<td>Wasserstein Auto Encoder objective (MMD)</td>
</tr>

<tr>
<td>spherical</td>
<td>Spherical Auto Encoder objective</td>
</tr>

<tr>
<td rowspan=1>Discriminator</td>
<td>spectral_discriminator</td>
<td>Use the MultiScale discriminator from EnCodec.</td>
</tr>

<tr>
<td rowspan=3>Others</td>
<td>causal</td>
<td>Use causal convolutions</td>
</tr>

<tr>
<td>noise</td>
<td>Enables noise synthesizer V2</td>
</tr>

<tr>
<td>hybrid</td>
<td>Enable mel-spectrogram input</td>
</tr>

<tr>
<td rowspan=3>Augmentations</td>
<td>mute</td>
<td>Randomly mutes data batches (default prob : 0.1). Enforces the model to learn silence</td>
</tr>

<tr>
<td>compress</td>
<td>Randomly compresses the waveform (equivalent to light non-linear amplification of batches)</td>
</tr>

<tr>
<td>gain</td>
<td>Applies a random gain to waveform (default range : [-6, 3]) </td>
</tr>

</tbody>
</table>

### Export

Once trained, export your model to a torchscript file using

```bash
rave export --run /path/to/your/run (--streaming)
```

Setting the `--streaming` flag will enable cached convolutions, making the model compatible with realtime processing. **If you forget to use the streaming mode and try to load the model in Max, you will hear clicking artifacts.**

## Prior

For discrete models, we redirect the user to the `msprior` library [here](https://github.com/caillonantoine/msprior). However, as this library is still experimental, the prior from version 1.x has been re-integrated in v2.3.

### Training

To train a prior for a pretrained RAVE model :

```bash
rave train_prior --model /path/to/your/run --db_path /path/to/your_preprocessed_data --out_path /path/to/output
```

this will train a prior over the latent of the pretrained model `path/to/your/run`, and save the model and tensorboard logs to folder `/path/to/output`.

### Scripting

To script a prior along with a RAVE model, export your model by providing the `--prior` keyword to your pretrained prior :

```bash
rave export --run /path/to/your/run --prior /path/to/your/prior (--streaming)
```

## Pretrained models

Several pretrained streaming models [are available here](https://acids-ircam.github.io/rave_models_download). We'll keep the list updated with new models.

## Realtime usage

This section presents how RAVE can be loaded inside [`nn~`](https://acids-ircam.github.io/nn_tilde/) in order to be used live with Max/MSP or PureData.

### Reconstruction

A pretrained RAVE model named `darbouka.gin` available on your computer can be loaded inside `nn~` using the following syntax, where the default method is set to forward (i.e. encode then decode)

<img src="docs/rave_method_forward.png" width=400px/>

This does the same thing as the following patch, but slightly faster.

<img src="docs/rave_encode_decode.png" width=210px />

### High-level manipulation

Having an explicit access to the latent representation yielded by RAVE allows us to interact with the representation using Max/MSP or PureData signal processing tools:

<img src="docs/rave_high_level.png" width=310px />

### Style transfer

By default, RAVE can be used as a style transfer tool, based on the large compression ratio of the model. We recently added a technique inspired from StyleGAN to include Adaptive Instance Normalization to the reconstruction process, effectively allowing to define _source_ and _target_ styles directly inside Max/MSP or PureData, using the attribute system of `nn~`.

<img src="docs/rave_attribute.png" width=550px>

Other attributes, such as `enable` or `gpu` can enable/disable computation, or use the gpu to speed up things (still experimental).

## Offline usage

A batch generation script has been released in v2.3 to allow transformation of large amount of files

```bash
rave generate model_path path_1 path_2 --out out_path
```

where `model_path` is the path to your trained model (original or scripted), `path_X` a list of audio files or directories, and `out_path` the out directory of the generations.

## Discussion

If you have questions, want to share your experience with RAVE or share musical pieces done with the model, you can use the [Discussion tab](https://github.com/acids-ircam/RAVE/discussions) !

## Demonstration

### RAVE x nn~

Demonstration of what you can do with RAVE and the nn~ external for maxmsp !

[![RAVE x nn~](http://img.youtube.com/vi/dMZs04TzxUI/mqdefault.jpg)](https://www.youtube.com/watch?v=dMZs04TzxUI)

### embedded RAVE

Using nn~ for puredata, RAVE can be used in realtime on embedded platforms !

[![RAVE x nn~](http://img.youtube.com/vi/jAIRf4nGgYI/mqdefault.jpg)](https://www.youtube.com/watch?v=jAIRf4nGgYI)

# Frequently Asked Question (FAQ)

**Question** : my preprocessing is stuck, showing `0it[00:00, ?it/s]`<br/>
**Answer** : This means that the audio files in your dataset are too short to provide a sufficient temporal scope to RAVE. Try decreasing the signal window with the `--num_signal XXX(samples)` with `preprocess`, without forgetting afterwards to add the `--n_signal XXX(samples)` with `train`

**Question** : During training I got an exception resembling `ValueError: n_components=128 must be between 0 and min(n_samples, n_features)=64 with svd_solver='full'`<br/>
**Answer** : This means that your dataset does not have enough data batches to compute the intern latent PCA, that requires at least 128 examples (then batches). 


# Funding

This work is led at IRCAM, and has been funded by the following projects

- [ANR MakiMono](https://acids.ircam.fr/course/makimono/)
- [ACTOR](https://www.actorproject.org/)
- [DAFNE+](https://dafneplus.eu/) N° 101061548

<img src="https://ec.europa.eu/regional_policy/images/information-sources/logo-download-center/eu_co_funded_en.jpg" width=200px/>
