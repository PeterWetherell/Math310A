from tensorflow.keras.models import load_model
from scipy.io.wavfile import write
import numpy as np
import ProjectUtils
import time

model = load_model('./Models/deonoiserV7-18-7.keras')

audio_array, sampling_rate  = ProjectUtils.load_wav("./NormalizedSoundData/Noisy/YWP.wav")

stft_sample_width = 510

width = (stft_sample_width // 2) + 1
height = 137 # Was chosen in order to have 0.4 sec -> (0.4 * 44100)/(254 * 0.5) [bottom is times 0.5 due to the 0.5 overlap]
channels = 1

x_data_complex = ProjectUtils.scipy_STFT(audio_array, sampling_rate, stft_sample_width)

# Get each individual window -> run it throught the model -> create the wav file
data_length = x_data_complex.shape[0]
total_segments = data_length - height + 1

complex_result = np.zeros((total_segments, width), dtype=np.complex64)

start_time = time.time()
last_time = time.time()

batch_size = 256
batch_segments = []

for i in range(total_segments):
    curr_time = time.time()
    if curr_time - last_time > 15:
        last_time = curr_time
        time_remaining = (curr_time - start_time) * ((total_segments/max(i,1)) - 1)
        
        progress_percentage = (i / total_segments) * 100
        minutes_remaining = int(time_remaining // 60)
        seconds_remaining = time_remaining % 60

        print(f"{progress_percentage:>5.2f}% finished.\t{minutes_remaining}:{seconds_remaining:04.1f} minutes remain")

    # Get the log of the amplitude of the window
    input_window = np.abs(x_data_complex[i:i+height, :]) # Should be(137, 128)
    input_window = ProjectUtils.convert_To_Log(input_window)

    # Reshape to match model input
    input_window = np.expand_dims(np.expand_dims(input_window, axis=0), axis=-1)  # Add channel dim: (1, 137, 128, 1)

    # Load into the batch
    batch_segments.append(input_window) # This adds the batch dimension

    # Check if its time to load out of the batch
    if len(batch_segments) == batch_size or i == total_segments -1:
        batch_array = np.concatenate(batch_segments, axis=0)  # Create numpy batch array
        predicted_batch = model.predict(batch_array, verbose=0)  # Predict the batch -> clip to be within parameters of exponential values

        for j, predicted_amplitude in enumerate(predicted_batch):
            index = i - len(batch_segments) + 1 + j
            # Get the angle of the window
            input_angle = np.angle(x_data_complex[index + height // 2, :])
            amplitude = ProjectUtils.convert_To_Amplitude(predicted_amplitude)
            amplitude[amplitude <= 1e-4] = 0
            complex_result[index, :] = amplitude * np.exp(1j * input_angle)

        batch_segments = []  # Clear the batch)

print("100% finished")
print("Started inverse Transform")
reconstructed_audio = ProjectUtils.scipy_iSTFT(complex_result, sampling_rate, stft_sample_width) # Undo the STFT to get data
reconstructed_audio = reconstructed_audio - np.mean(reconstructed_audio)
reconstructed_audio = np.int16(reconstructed_audio / np.max(np.abs(reconstructed_audio)) * 32767)
write('output.wav', sampling_rate, reconstructed_audio)