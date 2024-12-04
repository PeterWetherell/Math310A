import numpy as np

import ProjectUtils

width = 1000

# Load the .wav file
#sampling_rate, audio_array = read("./NormalizedSoundData/Clean/YW.wav")
audio_array, sampling_rate  = ProjectUtils.load_wav("./NormalizedSoundData/Clean/YWP.wav")

stft = ProjectUtils.scipy_STFT(audio_array, sampling_rate, width)

untransformed_audio = ProjectUtils.scipy_iSTFT(stft, sampling_rate, width)

print(np.max(np.abs(np.pad(audio_array,(0,len(untransformed_audio)-len(audio_array)),mode='constant',constant_values=0)-untransformed_audio)))