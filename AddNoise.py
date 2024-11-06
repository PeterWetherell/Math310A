import wave
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.io import wavfile

def load_wav(filePath):
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
        audio_array = np.zeros(audio_char_array.shape[0], np.int32)
        for i in range(sample_width):
            audio_array += (audio_char_array[:, i].astype(np.int32) << (8*i))
        # Adjust for signed audio
        audio_array[audio_array >= (1 << (sample_width * 8 - 1))] -= (1 << (8 * sample_width))
    
    # Reshape if stereo
    if n_channels > 1:
        audio_array = np.reshape(audio_array, (-1, n_channels)) # need to redo this -> needs to interleve instead of just splitting first and second half

    return (audio_array, frame_rate)

# Takes in an audio array and a sample rate and adds noise
# WARNING: modifies the audio_array
def add_noise(audio_array, frame_rate):
    index = 0
    n_samples = audio_array.shape[0]
    n_channels = 1
    if (len(audio_array.shape) != 1):
        n_channels = audio_array.shape[1]

    cutoff_freq = np.random.normal(loc=400, scale=50) #generally noise is between 300 to 2000 hz so we will put the cutoff at generally around 400
    percentNoise = np.random.normal(loc=0.15, scale=0.5)

    while (index < n_samples):
        time = np.random.normal(loc=0.2,scale=0.05)
        length = frame_rate * time
        length = int(length)
        length = min(length,n_samples-index)
        if (length < 1200): # we don't really have enough data to be able to add noise to this
            return audio_array

        # Define the profile for noise -> it is just flatly distributed above this cuttoff
        cutoff_freq = np.random.normal(loc=cutoff_freq, scale=25) # This will give us a more gradual transition
        cutoff_freq = max(cutoff_freq,40)
        
        # Design a Butterworth high-pass filter
        nyquist_rate = frame_rate / 2.0
        normal_cutoff = cutoff_freq / nyquist_rate
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)

        # Generate random noise
        audioDim = (length)
        if (n_channels != 1):
            audioDim = (length, n_channels)
        
        noise = np.random.normal(loc=0.0, scale=1.0, size=audioDim)
        high_pass_noise = np.zeros(shape=audioDim)


        if n_channels != 1:
            for i in range(n_channels):
                # Apply the filter using filtfilt (zero-phase filtering)
                high_pass_noise[:,i] = signal.filtfilt(b, a, noise[:,i])
        else:
            high_pass_noise = signal.filtfilt(b, a, noise)
            
        # Get the percentage of the origional audio we want to convert to noise
        percentNoise = np.random.normal(loc=percentNoise, scale=0.02)
        percentNoise = min(max(percentNoise,0.05),0.3)

        # Scale the noise and add it to the wav file data
        static_noise_scalar = audio_array.max()/high_pass_noise.max() * percentNoise
        audio_array[index:index+length] = (high_pass_noise*static_noise_scalar) + (audio_array[index:index+length]*(1.0-percentNoise))
        index += length
    return audio_array

# Assumes that the data coming in is already in standard form (fr 44100, 16  bit, mono)
def add_sound_effects(audio_array, frame_rate):
    #Adding known noise samples over top
    #Grab load in all of the data from the csv
    n_samples = audio_array.shape[0]
    data = pd.read_csv("esc50.csv")
    percent_seound_effect = 80 # % likelihood to use a sound effect
    index = 0
    while (index + 6 * frame_rate < n_samples): # Each clip is 5 sec so we must have space for it (thats why we use 6)
        if np.random.uniform(0,100) < percent_seound_effect:
            audio_percent_seound_effect = np.random.normal(loc=0.3, scale=0.05)
            audio_percent_seound_effect = max(audio_percent_seound_effect,0.1)
            random_element =  data.iloc[np.random.choice(data.index)]
            soundFileName = "./ESC-50-audio/" + random_element["filename"]
            print(soundFileName)
            sound_effect_sample_rate, sound_effect_data = wavfile.read(soundFileName)
            length = int(sound_effect_data.shape[0])
            audio_array[index:index+length] = sound_effect_data*audio_percent_seound_effect + audio_array[index:index+length]*(1.0-audio_percent_seound_effect)
            index += length
        else: 
            time = np.random.normal(loc=7, scale=1)
            time = max(time,0)
            index += int(time * frame_rate)
    return audio_array

def write_wav(output_data, frame_rate, sample_width, filePath):
    n_channels = 1
    if (len(output_data.shape) != 1):
        n_channels = output_data.shape[1]
        # Collapse the audio again
        output_data = output_data.reshape(-1)

    output_data = np.clip(output_data, -1 << (sample_width * 8 - 1), (1 << (sample_width * 8 - 1)) - 1).astype(np.int32)  # Clip the size based on the byte depth

    with wave.open(filePath, "wb") as wav_out:
        wav_out.setnchannels(n_channels)
        wav_out.setsampwidth(sample_width)
        wav_out.setframerate(frame_rate)

        # Create a byte array from the int32 audio_array
        audio_bytes = np.zeros((output_data.size * sample_width,), dtype=np.uint8)
        for i in range(sample_width):
            audio_bytes[i::sample_width] = (output_data >> (i * 8)) & 0xFF  # Convert from the output data by grabbing each byte at a time
        wav_out.writeframes(audio_bytes.tobytes())


y, fr = load_wav("./NormalizedSoundData/Clean/NeonDrive.wav")
y = add_sound_effects(y, fr)
y = add_noise(y, fr)
write_wav(y,fr,2,"./NormalizedSoundData/Noisy/NeonDrive.wav")