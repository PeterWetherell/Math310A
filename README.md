# Audio Denoising with CNNs

This repository contains code and data for a project focused on denoising audio using Convolutional Neural Networks (CNNs). The goal is to explore the effects of custom noise profiles on speech intelligibility and to evaluate how well different denoising strategies,  particularly CNN-based models, can recover intelligible audio from noisy inputs.

This was developed for Math 310A as part of the "Fourier Folks" group, this project integrates signal processing, deep learning, and automated speech recognition.

The project generates .wav files with noise using custom profiles, it then checks to see how distorted the audio file is using whisper (an AI text to speach program) to give a baseline for how it affected its discernability. We can then apply temporal and frequency based filters to do our best to remove the noise and run it back through Wisper to see how effective the denoising is at recovering the origional sound file.

## Project Overview

1. **Noise Generation**: 
   - Custom noise profiles are generated and added to `.wav` audio files.
   - The `AddNoise.py` script handles the injection of noise into clean audio samples.

2. **Speech Intelligibility Evaluation**:
   - [Whisper](https://github.com/openai/whisper), OpenAI’s speech recognition model, is used to transcribe both noisy and denoised audio.
   - This provides a quantifiable measure of how noise and subsequent denoising affect intelligibility.

3. **Denoising with CNNs**:
   - A CNN is trained or applied to spectrogram representations of noisy audio.
   - The model outputs a cleaned version of the spectrogram, which is converted back into audio.
   - Scripts like `CleanAudioV1.py`, `CleanAudioV2.py`, etc., implement different iterations of the denoising approach.

4. **Evaluation**:
   - `compare.py` automates comparison between original, noisy, and denoised transcriptions to evaluate denoising performance.

## Setup

Due to the large size of audio files, these could not be stored in this repository.

Instead you will need to download the [**ESC-50 audio database**](https://github.com/karolpiczak/ESC-50) for adding specific noise to the background of the audio files.

Additionally, we recomend using audio books from [**LibriVox**](https://librivox.org/) in order to get large amounts of relatively clean audio. For this project we used Pride and Prejudice and The Yellow Wallpaper as training and validation data but you could choose to use other audio books or add more to increase the effectiveness of the training.

## Running the Project

1. **Generate Noisy Audio**:

```bash
python AddNoise.py
```

3. **Denoise the Audio**:
Run one of the available versions:
```bash
python CleanAudioVX.py
```

5. **Compare Transcripts**:

```bash
python compare.py
```

This will output Whisper transcriptions for each stage: original, noisy, and cleaned. It will then run a cosine similarity between all of them and give you a comparison to assess intelligibility.

## Presentation
For an in-depth explanation of the techniques and results, check out our presentation:
[**Project Slides**](https://docs.google.com/presentation/d/1Q_uZKm6z2Zj_uadfP3kVS3UTvn-Lq4S_WaxEmmott2k/edit?slide=id.g35f391192_00#slide=id.g35f391192_00)
