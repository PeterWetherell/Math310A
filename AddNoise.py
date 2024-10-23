import wave
import numpy as np
import scipy.signal as signal
from scipy.io.wavfile import write

# For debugging we get more items when printing
np.set_printoptions(edgeitems=30)
debug = False

filePath = input("Enter path to the wav file: ")

# Open the WAV file
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
    if (debug):
        print(audio_char_array)
    audio_array = np.zeros(audio_char_array.shape[0], np.int32)
    for i in range(sample_width):
        audio_array += (audio_char_array[:, i].astype(np.int32) << (8*i))
    # Adjust for signed audio
    audio_array[audio_array >= (1 << (sample_width * 8 - 1))] -= (1 << (8 * sample_width))
    if (debug):
        print(audio_array)

# Reshape if stereo
if n_channels > 1:
    audio_array = np.reshape(audio_array, (-1, n_channels)) # need to redo this -> needs to interleve instead of just splitting first and second half


""" # trying to just get the audio file to come out the same way it came in
"""

# Define the profile for noise -> it is just normally distributed above this cuttoff
cutoff_freq = input("Enter the lower bound cuttoff for the noise: (Generally 300 to 2000) ") # Hz
cutoff_freq = float(cutoff_freq)

# Design a Butterworth high-pass filter
nyquist_rate = frame_rate / 2.0
normal_cutoff = cutoff_freq / nyquist_rate
b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)


# Generate random noise
noise = np.random.normal(loc=0.0, scale=1.0, size=audio_array.shape)
filtered_signal = np.zeros(audio_array.shape)

for i in range(n_channels):
    # Apply the filter using filtfilt (zero-phase filtering)
    filtered_signal[:,i] = signal.filtfilt(b, a, noise[:,i])

# Get the maximum amplitude we want for the noise
percentNoise = input("Percent noise (Ex: 0.01 is 1%): ")
percentNoise = float(percentNoise)
maxNoiseAmplitude = (1 << (sample_width*8 -1)) * percentNoise

#Scale the noise and add it to the wav file data
noisy_audio_data = (filtered_signal*maxNoiseAmplitude/filtered_signal.max()) + (audio_array*(1.0-percentNoise)) #make sure we scale down the audio array to avoid issues with overflow

output_data = noisy_audio_data
# Collapse the audio again
output_data = output_data.reshape(-1)

output_file = "output.wav"

output_data = np.clip(output_data, -1 << (sample_width * 8 - 1), (1 << (sample_width * 8 - 1)) - 1).astype(np.int32)  # Clip the size based on the byte depth

with wave.open(output_file, "wb") as wav_out:
    wav_out.setnchannels(n_channels)
    wav_out.setsampwidth(sample_width)
    wav_out.setframerate(frame_rate)

    # Create a byte array from the int32 audio_array
    audio_bytes = np.zeros((output_data.size * sample_width,), dtype=np.uint8)
    for i in range(sample_width):
        audio_bytes[i::sample_width] = (output_data >> (i * 8)) & 0xFF  # Convert from the output data by grabbing each byte at a time

    if (debug):
        print(audio_bytes)

    wav_out.writeframes(audio_bytes.tobytes())