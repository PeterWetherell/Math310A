import numpy as np
from scipy.signal import stft

# Example signal: 2-second sine wave at 440 Hz
fs = 44100  # Sampling frequency
t = np.linspace(0, 2, 2 * fs, endpoint=False)  # Time vector
from scipy.io.wavfile import read

# Load the .wav file
sampling_rate, audio_array = read("./NormalizedSoundData/Clean/PAP1.wav")

# Normalize the audio if it's in integer format
if audio_array.dtype == np.int16:  # For 16-bit PCM
    audio_array = audio_array / 32768.0  # Convert to range [-1, 1]
elif audio_array.dtype == np.int32:  # For 32-bit PCM
    audio_array = audio_array / 2147483648.0
# Perform STFT
nperseg = 128  # Fourier window size
noverlap = nperseg // 2  # 50% overlap
f, t, Zxx = stft(audio_array, fs, nperseg=nperseg, noverlap=noverlap, window='hamming')

print(f"STFT computed. Shape of STFT matrix: {Zxx.shape}")

from scipy.signal import istft

# Perform the inverse STFT
_, reconstructed_audio = istft(Zxx, fs, nperseg=nperseg, noverlap=noverlap, window='hamming')

# Compare original and reconstructed signals
import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))
# plt.plot(audio_array, label='Original Signal')
# plt.plot(reconstructed_audio, linestyle='dashed', label='Reconstructed Signal')
# plt.legend()
# plt.title('Comparison of Original and Reconstructed Signal')
# plt.show()

from scipy.io.wavfile import write

# Scale reconstructed signal to 16-bit PCM
reconstructed_audio_int16 = (reconstructed_audio * 32767).astype(np.int16)

# Save as WAV
write("reconstructed_audio.wav", fs, reconstructed_audio_int16)
print("Reconstructed audio saved to reconstructed_audio.wav")


