import numpy as np
import math
import wave
from scipy.signal import stft
from scipy.signal import istft

from scipy.io.wavfile import read

import tensorflow as tf

epsilon = 1e-12 # Chosen so that with this amplitude it produces no audio

 # 137.16 = 0.54 * 254 (max value for amplitude) -> 274.32 (2x safety factor) (old max clip comment if we ever need to go back to it)

def log_spectral_distance(clean_magnitude, denoised_magnitude):
     # We treat the denoised_magnitude as exponents and therefore there is no need to mess with it
    clean_magnitude = tf.clip_by_value(clean_magnitude, epsilon, math.inf)
    log_clean = tf.math.log(clean_magnitude)
    log_dist = tf.sqrt(tf.reduce_mean(tf.square(log_clean - denoised_magnitude))) # multiply by 20/ln(10) to get it into decible
    return log_dist

def log_spectral_distancell(clean_magnitude, denoised_magnitude):
    clean_magnitude = tf.clip_by_value(clean_magnitude, epsilon, math.inf)
    log_clean = tf.math.log(clean_magnitude)
    denoised_magnitude = tf.clip_by_value(denoised_magnitude, epsilon, math.inf)
    log_denoised = tf.math.log(denoised_magnitude)
    log_dist = tf.sqrt(tf.reduce_mean(tf.square(log_clean - log_denoised))) # multiply by 20/ln(10) to get it into decible
    return log_dist


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

def convert_To_Log(magnitude):
    magnitude = tf.clip_by_value(magnitude, epsilon, math.inf) # Clip so we don't get NaN logarithms
    return np.log(magnitude) # take the natural log of the mangitude

def convert_To_Amplitude(log_magnitude):
    magnitude = np.exp(log_magnitude) # Exponentiate the logarithm
    # magnitude[magnitude <= 5*epsilon] = 0 # If its close to 0 -> just make it 0
    return magnitude

def scipy_STFT(audio_array, sampling_rate, window_size, overlap=0.5, window_func='hamming'):
    noverlap = int(window_size * overlap)  # 50% overlap
    # / 32768
    f, t, Zxx = stft(audio_array, sampling_rate, nperseg=window_size, noverlap=noverlap, window=window_func) # , boundary=None
    return Zxx.T

def scipy_iSTFT(stft_matrix, sampling_rate, window_size, overlap=0.5, window_func='hamming'):
    noverlap = int(window_size * overlap)  # 50% overlap
    _, reconstructed_audio = istft(stft_matrix.T, sampling_rate, nperseg=window_size, noverlap=noverlap, window=window_func) # , boundary=None
    return reconstructed_audio

