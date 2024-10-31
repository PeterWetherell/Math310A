# Compress the files by reducing their complexity
# For this project we will use mono, 16 bit depth, 22050 Hz

targetSampleRate = 22050

import wave
import numpy as np
import librosa
import soundfile as sf

filePath = input("Enter path to the wav file: ")

with wave.open(filePath, "rb") as wav_file:
    # Get parameters
    n_channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()
    n_frames = wav_file.getnframes()
    
    # Read audio data
    audio_data = wav_file.readframes(n_frames)

    # Reformat into the shape of the sample width
    audio_char_array = np.frombuffer(audio_data, dtype=np.uint8).reshape(-1, sample_width) # This converts it into a byte (unsigned char) array
    audio_array = np.zeros(audio_char_array.shape[0], np.int32)
    for i in range(sample_width):
        audio_array += (audio_char_array[:, i].astype(np.int32) << (8*i))
    # Adjust for signed audio
    audio_array[audio_array >= (1 << (sample_width * 8 - 1))] -= (1 << (8 * sample_width))

if n_channels > 1: # Reshape if stereo
    audio_array = np.reshape(audio_array, (-1, n_channels))
    # Convert it to mono
    audio_array = audio_array.mean(axis=1)

audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=frame_rate, target_sr=targetSampleRate)

if (sample_width > 2): #convert to 16 bit audio
    audio_array = ((audio_array[:].astype(np.int32) >> ((sample_width - 2)*8)) & 0xFFFF ).astype(np.int16)

sf.write("output_downscaled.wav", audio_array, targetSampleRate)