import ProjectUtils
from scipy.io.wavfile import write
import numpy as np

wav_x_data,fr1 = ProjectUtils.load_wav("./NormalizedSoundData/Noisy/YWP.wav")
x_data_complex = ProjectUtils.scipy_STFT(wav_x_data, fr1, 245)
x_data = ProjectUtils.scipy_iSTFT(x_data_complex, fr1, 245)
reconstructed_audio_int16 = (x_data * 32767).astype(np.int16)
write("./Outputs/donothing.wav", 44100, reconstructed_audio_int16)