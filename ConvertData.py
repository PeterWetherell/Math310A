import numpy as np
import math

def compute_STFT(audio_array, fourierSize, overlap=0.5, window_func=np.hamming):
    length = len(audio_array)
    hop_size = int(fourierSize * (1 - overlap))  # Step size based on overlap
    # (Lenth - fourierSize) / hop_size --> this gets us the number of hops. We then add 1 for the final step
    outputLength = math.ceil((length - fourierSize) / hop_size) + 1

    output_array = np.zeros((outputLength, fourierSize), dtype=complex)
    window = window_func(fourierSize)
    
    for i in range(outputLength): # Perform the STFT
        start_index = i * hop_size
        end_index = start_index + fourierSize
        # Zero the segment
        segment = np.zeros(fourierSize)
        # Grab the last data into the segment (avoiding the last case when the length - start index is less than the fourier size)
        segment[:min(fourierSize, length - start_index)] = audio_array[start_index:end_index]
        
        # Apply window and compute FFT
        windowed_segment = segment * window # Window function is needed to reduce spectral leakage
        output_array[i] = np.fft.fft(windowed_segment)
    
    return output_array