import numpy as np
from tensorflow.keras.models import load_model
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import ProjectUtils
import time

width = 128
height = 137
channels = 1

num_segments_per_save = 10

stft_sample_width = (width - 1) * 2
batch_size = 128
epochs = 3

# Grab all of the raw data
print("Loading raw data")
wav_x_data,fr1 = ProjectUtils.load_wav("./NormalizedSoundData/Noisy/SAS2.wav")
wav_y_data,fr2 = ProjectUtils.load_wav("./NormalizedSoundData/Clean/SAS2.wav")

if (fr1 != fr2):
    print("Error with frame rate of both files: BIG ISSUE")
    exit()

version = 4
currSeg = 590
model = load_model(f'./Models2/deonoiserV{version}-{currSeg // num_segments_per_save}-{currSeg % num_segments_per_save}.keras')

# Compile the model
initial_learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss="mean_squared_error")

segment_length = 120 # Number of seconds in the segment length
sub_segment_length = 3 # Number of seconds in the subsegment
num_sub_segments = segment_length // sub_segment_length
segment_frames = segment_length * fr1
num_segments = len(wav_x_data) // segment_frames
sub_segment_order = np.linspace(0,num_segments * num_sub_segments - 1, num_segments * num_sub_segments) # Get a list of each sub segment we can choose from
np.random.shuffle(sub_segment_order) # Shuffle it arround -> gets a random order for each second
sub_segment_order = sub_segment_order * sub_segment_length # convert subsegments back into seconds

start_time = time.time()

speedup = 3

sub_seg_training_size = (((fr1*sub_segment_length)//stft_sample_width + 1) * 2 - height) // speedup
training_size = num_sub_segments * sub_seg_training_size
# Preallocate space for the input windows and target -> stops memory fragmentation & improves efficiency
input_windows = np.zeros(shape=(training_size,height,width))
targets = np.zeros(shape=(training_size,width))

minVal = 1
maxVal = 0

for seg_num in range(num_segments):
    currSeg += 1
    print(f'Converting segment {seg_num + 1} out of {num_segments} from raw data into STFT')
    curr_time = time.time()
    if (curr_time - start_time > 30):
        print(f'Estimated Time Remaining {((curr_time - start_time) * ((num_segments/max(seg_num,1)) - 1)/60):.2f} minutes')
    
    for sub_seg_num in range(num_sub_segments):
        start = int(sub_segment_order[seg_num*num_sub_segments + sub_seg_num])
        start_index = start*fr1
        end_index = start_index + ((fr1*sub_segment_length)//stft_sample_width + 1) * stft_sample_width # make it so that we don't need any padding
        # Convert a specific segment into spectrograms (we don't have the memory to convert all 6 hours)
        x_data_complex = ProjectUtils.scipy_STFT(wav_x_data[start_index:end_index], fr1, stft_sample_width)
        y_data_complex = ProjectUtils.scipy_STFT(wav_y_data[start_index:end_index], fr1, stft_sample_width)
        # Convert the spectrograms into amplitude only (abs does magnitude for some reason)
        x_data = np.abs(x_data_complex)
        y_data = np.abs(y_data_complex)
        minVal = min(np.min(y_data), minVal)
        maxVal = max(np.max(y_data), maxVal)
        # Convert the amplitude into log amplitude
        x_data = ProjectUtils.convert_To_Log(x_data)
        y_data = ProjectUtils.convert_To_Log(y_data)
        # Convert into windows with corresponding targets
        data_length = x_data.shape[0]
        if ((data_length - height + 1 - 2) // speedup != sub_seg_training_size):
            print(f"Issue with calculating the sub segment training size. It is currently {sub_seg_training_size} and it should be {data_length - height + 1 - 2}")
            exit()
        for i in range(1, data_length - height, speedup): # Remove the two padding frames
            input_index = i // speedup
            if (sub_seg_training_size <= input_index):
                continue
            input_windows[sub_seg_training_size*sub_seg_num + input_index] = x_data[i:i + height, :]
            targets[sub_seg_training_size*sub_seg_num + input_index] = y_data[i + height // 2, :]
    # X_train, y_train, X_val, y_val are your training and validation datasets
    print(minVal, maxVal)
    X_train, X_val, y_train, y_val = train_test_split(input_windows, targets, test_size=0.2, random_state=None)
    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
    if (currSeg % num_segments_per_save == 0 and currSeg != 0):
        model.save(f'./Models2/deonoiserV4-{currSeg // num_segments_per_save}-0.keras')
model.save(f'./Models2/deonoiserV4-{currSeg // num_segments_per_save}-{currSeg % num_segments_per_save}.keras')