import wave
import numpy as np
import scipy.signal as signal

def add_noise_from_filepath(filePath):
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

    return (add_noise(audio_array, frame_rate), frame_rate)

# Takes in an audio array and a sample rate and adds noise
# WARNING: modifies the audio_array
def add_noise(audio_array, frame_rate):
    n_samples = audio_array.shape[0]
    index = 0
    n_channels = audio_array.shape[1]

    while (index < n_samples):
        length = np.random.normal(loc=10000, scale=1000)
        length = int(length)
        length = min(length,n_samples-index)
        if (length < 1200): # we don't really have enough data to be able to add noise to this
            return audio_array

        # Define the profile for noise -> it is just flatly distributed above this cuttoff
        cutoff_freq = np.random.normal(loc=400, scale=50) #generally noise is between 300 to 2000 hz so we will put the cutoff at generally around 400

        # Design a Butterworth high-pass filter
        nyquist_rate = frame_rate / 2.0
        normal_cutoff = cutoff_freq / nyquist_rate
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)

        # Generate random noise
        audioDim = (length, n_channels)
        noise = np.random.normal(loc=0.0, scale=1.0, size=audioDim)
        high_pass_noise = np.zeros(audioDim)

        if n_channels != 1:
            for i in range(n_channels):
                # Apply the filter using filtfilt (zero-phase filtering)
                high_pass_noise[:,i] = signal.filtfilt(b, a, noise[:,i])
        else:
            high_pass_noise = signal.filtfilt(b, a, noise)
            
        # Get the percentage of the origional audio we want to convert to noise
        percentNoise = np.random.normal(loc=0.35, scale=0.085)
        percentNoise = min(max(percentNoise,0.2),0.5)

        # Get the percentage of the noise that we want to be the static
        percentStatic = np.random.normal(loc=0.35, scale=0.1)
        percentStatic = min(max(percentStatic,0.1),0.75)

        """
        profiled_noise = np.zeros(size)
        time = np.linspace(start=0, stop=length/frame_rate, num=length) #from 0 to numFrames/(frames/sec) with length # of frames
        for i in range(n_channels):
            #TODO: Replace these frequence and amplitudes with a profile based on different noises
            freq = 10
            amp = 10
            phase_shift = np.random.uniform(0,2*np.pi)
            profiled_noise[:,i] += amp*np.cos(freq*time + phase_shift)
        """
        # Scale the noise and add it to the wav file data
        static_noise_scalar = audio_array.max()/high_pass_noise.max() * percentNoise * percentStatic
        #profiled_noise_scalar = audio_array.max()/profiled_noise.max() * percentNoise * (1.0-percentStatic)
        audio_array[index:index+length,:] = (high_pass_noise*static_noise_scalar) + (audio_array[index:index+length,:]*(1.0-percentNoise)) # + (profiled_noise*profiled_noise_scalar)
        index += length

    return audio_array

def write_audio(output_data, frame_rate, sample_width, filePath):
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


y, fr = add_noise_from_filepath("./SoundData/NeonDrive.wav")
write_audio(y,fr,3,"complexNoise.wav")