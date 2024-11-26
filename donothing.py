import ProjectUtils
from scipy.io.wavfile import write
import numpy as np

wav_x_data,fr1 = ProjectUtils.load_wav("./NormalizedSoundData/Noisy/YWP.wav")
x_data_complex = ProjectUtils.scipy_STFT(wav_x_data, fr1, 245)
x_data = ProjectUtils.scipy_iSTFT(x_data_complex, fr1, 245)
reconstructed_audio_int16 = (x_data * 32768).astype(np.int16)
write("./Outputs/donothing.wav", 44100, reconstructed_audio_int16)

wav_x_data_padded = np.pad(wav_x_data, (0, len(reconstructed_audio_int16) - len(wav_x_data)), mode='constant', constant_values=0)

print(np.array_equal(wav_x_data_padded, reconstructed_audio_int16))
diff = wav_x_data_padded - reconstructed_audio_int16
print(f"Max difference: {np.max(np.abs(diff))}")
