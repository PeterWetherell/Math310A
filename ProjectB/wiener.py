import numpy as np
from scipy.signal import wiener
import soundfile as sf

def apply_wiener_filter(audio, mysize=15, noise=0.5):
    """
    Apply Wiener filter to an audio signal.
    
    Parameters:
    - audio (np.array): Input audio signal as a numpy array.
    - mysize (int): Size of the Wiener filter window.
    - noise (float): Estimate of the noise power.
    
    Returns:
    - filtered_audio (np.array): Wiener filtered audio signal.
    """
    # Apply Wiener filter
    filtered_audio = wiener(audio, mysize=mysize, noise=noise)
    return filtered_audio

# Example of loading and applying Wiener filter to an audio file
audio_path = "/Users/vivim./Desktop/Fall2024/Mathematical Modeling/Project/output.wav"
audio, sample_rate = sf.read(audio_path)  # Load audio file

filtered_audio = apply_wiener_filter(audio)  # Apply Wiener filter

# Save the filtered audio to a new file
output_path = "/Users/vivim./Desktop/Fall2024/Mathematical Modeling/Project/ProjectB/wiener.wav"
sf.write(output_path, filtered_audio, sample_rate)
print("Filtered audio saved at:", output_path)
