import wave
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.io import wavfile

import ProjectUtils

# Takes in an audio array and a sample rate and adds noise
# This gives a varying background noise that veries with the audio of who is talking
# This mimics a microphone that is not very good and therefore collects a lot of noise when someone talks
# WARNING: modifies the audio_array
def add_noise(audio_array, frame_rate):
    index = 0
    n_samples = audio_array.shape[0]
    n_channels = 1
    if (len(audio_array.shape) != 1):
        n_channels = audio_array.shape[1]

    cutoff_freq = max(np.random.normal(loc=400, scale=50),100) #generally noise is between 300 to 2000 hz so we will put the cutoff at generally around 400
    maxPercentNoise = 0.5
    percentNoise = max(min(np.random.normal(loc=maxPercentNoise/2, scale=maxPercentNoise/8),maxPercentNoise),0)

    while (index < n_samples):
        time = max(np.random.normal(loc=0.2,scale=0.05),0.1)
        length = frame_rate * time
        length = int(max(length,1000))
        if (index + length > n_samples): # we don't really have enough data to be able to add noise to this
            return audio_array

        # Define the profile for noise -> it is just flatly distributed above this cuttoff
        cutoff_freq = np.random.normal(loc=cutoff_freq, scale=25) # This will give us a more gradual transition
        cutoff_freq = min(max(cutoff_freq,40),4000)
        
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
        percentNoise = min(max(np.random.normal(loc=percentNoise, scale=maxPercentNoise/16),0),maxPercentNoise)

        # Scale the noise and add it to the wav file data
        static_noise_scalar = audio_array[index:index+length].max()/high_pass_noise.max() * percentNoise
        audio_array[index:index+length] = (high_pass_noise*static_noise_scalar) + (audio_array[index:index+length]*(1.0-percentNoise))
        index += length
    return audio_array

def getMult():
    mult = np.random.normal(loc=1,scale=0.05)
    if (mult < 1): # Make the mult work so that it produces the correct inverses so the average mult value is 1
        mult = 1/(2-mult) 
    return mult

# This function models a slowly changing background noise
# This is generally a lower pitched noise and has a more continuous variance of volume and frequency cutoff with time
def add_background_noise(audio_array, frame_rate):
    n_samples = audio_array.shape[0]
    time_index = [0]
    frequency_min = [30]
    frequency_max = [3000]
    maxPercentNoise = 0.125
    percentNoise = [min(max(np.random.normal(loc=maxPercentNoise/2,scale=maxPercentNoise/8),0),maxPercentNoise)]
    
    while time_index[-1] < n_samples:
        time = max(np.random.normal(loc=4,scale=1),0.5)
        time = time*frame_rate
        time = int(time)

        time_index.append(time_index[-1] + time)
        frequency_min.append(frequency_min[-1]*getMult())
        frequency_max.append(frequency_max[-1]*getMult())

        frequency_min[-1] = min(max(frequency_min[-1],20),200)
        if frequency_max[-1] < frequency_min[-1] + 100: # whenever we get too close together we need to seperate again
            frequency_max[-1] += max(np.random.normal(loc=1000,scale=100),500)
        frequency_max[-1] = min(frequency_max[-1],6000)

        percentNoise.append(min(max(percentNoise[-1] + np.random.normal(loc=0.0,scale=maxPercentNoise/16),0),maxPercentNoise))
    
    maxAudio = audio_array.max()

    index = 0
    linearInterpIndex = 0
    while (index < n_samples):
        time = max(np.random.normal(loc=0.2,scale=0.05),0.1)
        time = frame_rate * time
        time = max(int(time),1000)
        if (index + time > n_samples):
            return audio_array
        
        noise = np.random.normal(loc=0.0, scale=1.0, size=(time))

        # Deal with linear interpolation
        while (time_index[linearInterpIndex+1] <= index): # move the window forward when we pass the window 
            linearInterpIndex += 1
        k = (time_index[linearInterpIndex+1]-index)/(time_index[linearInterpIndex+1]-time_index[linearInterpIndex])
        if (k < 0 or k > 1):
            print(k)

        # Band-pass filter parameters
        low_cutoff = frequency_min[linearInterpIndex]*k + frequency_min[linearInterpIndex+1]*(1-k)   # Low cutoff frequency in Hz
        high_cutoff = frequency_max[linearInterpIndex]*k + frequency_max[linearInterpIndex+1]*(1-k)  # High cutoff frequency in Hz
        
        #print(low_cutoff, " ", high_cutoff)

        r = percentNoise[linearInterpIndex]*k + percentNoise[linearInterpIndex+1]*(1-k)
        order = 4  # Filter order

        # Design Butterworth band-pass filter
        nyquist = 0.5 * frame_rate
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist
        b, a = signal.butter(order, [low, high], btype='band')

        # Apply the filter
        filtered_signal = signal.lfilter(b, a, noise)

        # Scale the noise and add it to the wav file data
        static_noise_scalar = maxAudio/filtered_signal.max() * r
        audio_array[index:index+time] = (filtered_signal*static_noise_scalar) + (audio_array[index:index+time]*(1.0-r))
        index += time
    
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
            audio_percent_seound_effect = min(max(np.random.normal(loc=0.15, scale=0.05),0.05),0.2)
            random_element =  data.iloc[np.random.choice(data.index)]
            soundFileName = "./ESC-50-audio/" + random_element["filename"]
            #print(soundFileName)
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

fileArray = ["YW","PAP1","PAP2","SAS1","SAS2"]

for file in fileArray:
    print("Loading in the file: ", file)
    y, fr = ProjectUtils.load_wav("./NormalizedSoundData/Clean/" + file + ".wav")
    print("Adding sound effects") 
    y = add_sound_effects(y, fr)
    print("Adding noise")
    y = add_noise(y, fr)
    print("Adding background noise")
    y = add_background_noise(y,fr)
    print("Writing to the file")
    write_wav(y,fr,2,"./NormalizedSoundData/Noisy/" + file + ".wav")