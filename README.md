![rave_logo](docs/rave.png)

# RAVE: Realtime Audio Variational autoEncoder

Official implementation of _RAVE: A variational autoencoder for fast and high-quality neural audio synthesis_ ([article link](https://arxiv.org/abs/2111.05011)) by Antoine Caillon and Philippe Esling.

If you use RAVE as a part of a music performance or installation, be sure to cite either this repository or the article !

## Colab

We propose a Google Colab handling the training of a RAVE model on a custom dataset !

[![colab_badge](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aK8K186QegnWVMAhfnFRofk_Jf7BBUxl?usp=sharing)

## Installation

RAVE needs `python 3.9`. Install the dependencies using

```bash
pip install -r requirements.txt
```

Detailed instructions to setup a training station for this project are available [here](docs/training_setup.md).

## Preprocessing

RAVE comes with two command line utilities, `resample` and `duration`. `resample` allows to pre-process (silence removal, loudness normalization) and augment (compression) an entire directory of audio files (.mp3, .aiff, .opus, .wav, .aac). `duration` prints out the total duration of a .wav folder.

## Training

Both RAVE and the prior model are available in this repo. For most users we recommand to use the `cli_helper.py` script, since it will generate a set of instructions allowing the training and export of both RAVE and the prior model on a specific dataset.

```bash
python cli_helper.py
```

However, if you want to customize even more your training, you can use the provided `train_{rave, prior}.py` and `export_{rave, prior}.py` scripts manually.

### train_rave.py

settings: small = shorter warmup/max_steps, smaller capacity. but notice the default capacity is large while the default steps is small…?

- DATA_SIZE: number of channels in PQMF filter
- CAPACITY: *not* a latent variable capacity in bits, but a rough ‘model capacity’ which just scales hidden layer sizes
- LATENT_SIZE: number of latent dimensions before pruning
- BIAS: passed directly to conv layers apparently, no idea why you would need this or ever set it to False
- NO_LATENCY: enables causal convolutions, also a lower quality PQMF (?)
- RATIOS: stride/upsample factor between blocks in the encoder and generator. also determines depth of encoder/generator
- MIN_KL, MAX_KL: low and high values for the cyclic beta-VAE objective
- CROPPED_LATENT_SIZE — this is here for inference I guess? appears set to 0 for training
- FEATURE_MATCH: whether to include the discriminator feature-matching loss as part of loss_gen
- LOUD_STRIDE: does something to the architecture of the generator (specifically, the ‘loudness’ branch)
    - the loudness branch is much less of a ‘branch’ than the noise branch, more of an elaborate activation function
- USE_NOISE: enables the noise branch of the generator
- NOISE_RATIOS: downsampling ratios / network depth for the noise branch of the generator
    - the noise branch again downsamples the already-upsampled latents to get ‘control-rate’ noise coefficients
- NOISE_BANDS: number of noise bands *per* PQMF band in the generator ?
- D_CAPACITY: like CAPACITY but for the discriminator
- D_MULTIPLIER: interacts with D_CAPACITY and D_N_LAYERS to set the layer widths, conv groups, and strides in discriminator
- D_N_LAYERS: discriminator depth
- WARMUP: number of VAE-only iterations
- MODE: type of GAN loss
- CKPT: checkpoint to resume training from
- PREPROCESSED: path to store preprocessed dataset, or to already preprocessed data
- WAV: path to raw dataset
- SR: audio sample rate
- N_SIGNAL: batch length in audio samples
- MAX_STEPS: end training after this many iterations
- VAL_EVERY: run validation every so many iterations
- BATCH: batch size
- NAME: descriptive name for run

## Reconstructing audio

Once trained, you can reconstruct an entire folder containing wav files using

```bash
python reconstruct.py --ckpt /path/to/checkpoint --wav-folder /path/to/wav/folder
```

You can also export RAVE to a `torchscript` file using `export_rave.py` and use the `encode` and `decode` methods on tensors.

## Realtime usage

**UPDATE**

If you want to use the realtime mode, you should update your dependencies !

```bash
pip install -r requirements.txt
```

RAVE and the prior model can be used in realtime on live audio streams, allowing creative interactions with both models.

### [nn~](https://github.com/acids-ircam/nn_tilde)

RAVE is compatible with the **nn~** max/msp and PureData external.

![max_msp_screenshot](docs/maxmsp_screenshot.png)

An audio example of the prior sampling patch is available in the `docs/` folder.

### [RAVE vst](https://github.com/acids-ircam/rave_vst)

You can also use RAVE as a VST audio plugin using the RAVE vst !

![plugin_screenshot](https://github.com/acids-ircam/rave_vst/blob/main/assets/rave_screenshot_audio_panel.png?raw=true)

## Discussion

If you have questions, want to share your experience with RAVE or share musical pieces done with the model, you can use the [Discussion tab](https://github.com/acids-ircam/RAVE/discussions) !

## Demonstation

### RAVE x nn~

Demonstration of what you can do with RAVE and the nn~ external for maxmsp !

[![RAVE x nn~](http://img.youtube.com/vi/dMZs04TzxUI/mqdefault.jpg)](https://www.youtube.com/watch?v=dMZs04TzxUI)

### embedded RAVE

Using nn~ for puredata, RAVE can be used in realtime on embedded platforms !

[![RAVE x nn~](http://img.youtube.com/vi/jAIRf4nGgYI/mqdefault.jpg)](https://www.youtube.com/watch?v=jAIRf4nGgYI)
