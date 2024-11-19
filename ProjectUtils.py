import numpy as np
import math
import wave
from scipy.signal import stft
from scipy.signal import istft

from scipy.io.wavfile import read

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


def scipy_STFT(audio_array, sampling_rate, window_size, overlap=0.5, window_func='hamming'):
    # Perform STFT
    noverlap = int(window_size * overlap)  # 50% overlap
    f, t, Zxx = stft(audio_array/32768, sampling_rate, nperseg=window_size, noverlap=noverlap, window=window_func)
    return Zxx.T

def scipy_iSTFT(stft_matrix, sampling_rate, window_size, overlap=0.5, window_func=np.hamming):
    noverlap = int(window_size * overlap)  # 50% overlap
    _, reconstructed_audio = istft(stft_matrix.T, sampling_rate, nperseg=window_size, noverlap=noverlap, window='hamming')
    return reconstructed_audio

