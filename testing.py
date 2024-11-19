import numpy as np
from scipy.io.wavfile import read
from scipy.signal import stft

import ProjectUtils

# Example signal: 2-second sine wave at 440 Hz
fs = 44100  # Sampling frequency
t = np.linspace(0, 2, 2 * fs, endpoint=False)  # Time vector

# Load the .wav file
#sampling_rate, audio_array = read("./NormalizedSoundData/Clean/YW.wav")
audio_array, sampling_rate  = ProjectUtils.load_wav("./NormalizedSoundData/Clean/YW.wav")

audio_array = audio_array / 32768.0  # Convert to range [-1, 1]

print(audio_array.shape)

# Perform STFT
nperseg = 128  # Fourier window size
noverlap = nperseg // 2  # 50% overlap
f, t, Zxx = stft(audio_array, sampling_rate, nperseg=nperseg, noverlap=noverlap, window='hamming')

print(f"STFT output shape (frequency_bins, time_segments): {Zxx.shape}")

# Transpose to desired shape (time_segments, frequency_bins)
Zxx = Zxx.T
print(f"Transposed STFT output shape (time_segments, frequency_bins): {Zxx.shape}")

from scipy.signal import istft

# Perform the inverse STFT
_, reconstructed_audio = istft(Zxx, fs, nperseg=nperseg, noverlap=noverlap, window='hamming')

from scipy.io.wavfile import write

# Scale reconstructed signal to 16-bit PCM
reconstructed_audio_int16 = (reconstructed_audio * 32767).astype(np.int16)

# Save as WAV
write("reconstructed_audio.wav", fs, reconstructed_audio_int16)
print("Reconstructed audio saved to reconstructed_audio.wav")


