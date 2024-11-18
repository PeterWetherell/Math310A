import numpy as np
import math
import wave

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


from scipy.signal import stft
from scipy.io.wavfile import read
def stft(data_path ,window_size)
    # Example signal: 2-second sine wave at 440 Hz
    fs = 44100  # Sampling frequency
    t = np.linspace(0, 2, 2 * fs, endpoint=False)  # Time vector


    # Load the .wav file
    sampling_rate, audio_array = read(data_path)

    # Normalize the audio if it's in integer format
    if audio_array.dtype == np.int16:  # For 16-bit PCM
        audio_array = audio_array / 32768.0  # Convert to range [-1, 1]
    elif audio_array.dtype == np.int32:  # For 32-bit PCM
        audio_array = audio_array / 2147483648.0
    # Perform STFT
    nperseg = window_size  # Fourier window size
    noverlap = nperseg // 2  # 50% overlap
    f, t, Zxx = stft(audio_array, fs, nperseg=nperseg, noverlap=noverlap, window='hamming')
    return Zxx
import numpy as np

def reverse_STFT(stft_matrix, fourierSize, overlap=0.5, window_func=np.hamming):
    """
    Reverse STFT to reconstruct the time-domain signal.
    
    Parameters:
        stft_matrix (ndarray): The STFT matrix (time_frames x fourierSize).
        fourierSize (int): The size of the FFT window used in the STFT.
        overlap (float): The overlap ratio used in the STFT.
        window_func (callable): The window function used in the STFT.

    Returns:
        ndarray: Reconstructed time-domain signal.
    """
    # Derive hop_size from overlap
    hop_size = int(fourierSize * (1 - overlap))
    n_frames = stft_matrix.shape[0]
    signal_length = hop_size * (n_frames - 1) + fourierSize

    # Initialize output signal and synthesis window
    signal = np.zeros(signal_length)
    window = window_func(fourierSize)
    window_norm = np.zeros(signal_length)

    # Overlap-add synthesis
    for i in range(n_frames):
        start_index = i * hop_size
        end_index = start_index + fourierSize

        # Perform the inverse FFT
        time_segment = np.fft.ifft(stft_matrix[i]).real
        signal[start_index:end_index] += time_segment * window
        window_norm[start_index:end_index] += window ** 2

    # Normalize by the window norm
    window_norm = np.where(window_norm > 1e-10, window_norm, 1.0)  # Avoid division by zero
    return signal / window_norm

