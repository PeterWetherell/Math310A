import ProjectUtils
from scipy.io.wavfile import write
import numpy as np

data,fr1 = ProjectUtils.load_wav("./NormalizedSoundData/Noisy/YWP.wav")
from scipy.io import wavfile

# Read a WAV file
# fr1, data = wavfile.read("./NormalizedSoundData/Noisy/YWP.wav")
# x_data_complex = ProjectUtils.scipy_STFT(data, fr1, 245)
# x_data = ProjectUtils.scipy_iSTFT(x_data_complex, fr1, 245)
reconstructed_audio_int16 = (data * 32767).astype(np.int16)
write("./Outputs/donothing.wav", 44100, reconstructed_audio_int16)