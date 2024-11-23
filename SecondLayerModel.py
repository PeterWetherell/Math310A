import numpy as np
import math
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import ProjectUtils
import time

stft_sample_width = 254 # 128

width = (stft_sample_width // 2) + 1
height = 137 # Was chosen in order to have 0.4 sec -> (0.4 * 44100)/(254 * 0.5) [bottom is times 0.5 due to the 0.5 overlap]
channels = 1

# Grab all of the raw data
print("Loading raw data")
wav_x_data,fr1 = ProjectUtils.load_wav("./NormalizedSoundData/Noisy/PAP1.wav")
wav_y_data,fr2 = ProjectUtils.load_wav("./NormalizedSoundData/Clean/PAP1.wav")
layer_one_model = load_model('./Models/deonoiserV4-18-7.h5')

if (fr1 != fr2):
    print("Error with frame rate of both files: BIG ISSUE")
    exit()

# Initialize the model
model = Sequential()
# Add convolutional layers
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten and add dense layers
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=width, activation='linear'))
# Set to width -> we want to produce that many distinct outputs to correspond with our STFT

# Compile the model
initial_learning_rate = 0.001
model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss="mean_squared_error")
batch_size = 256
epochs = 3

segment_length = 120 # Number of seconds in the segment length
sub_segment_length = 4 # Number of seconds in the subsegment
num_sub_segments = segment_length // sub_segment_length
segment_frames = segment_length * fr1
num_segments = len(wav_x_data) // segment_frames
sub_segment_order = np.linspace(0,num_segments * num_sub_segments - 1, num_segments * num_sub_segments) # Get a list of each sub segment we can choose from
np.random.shuffle(sub_segment_order) # Shuffle it arround -> gets a random order for each second
sub_segment_order = sub_segment_order * sub_segment_length # convert subsegments back into seconds

num_segments_per_save = 10
start_time = time.time()

for seg_num in range(num_segments):
    print(f'Converting segment {seg_num} out of {num_segments} from raw data into STFT')
    curr_time = time.time()
    if (curr_time - start_time > 30):
        print(f'Estimated Time Remaining {((curr_time - start_time) * ((num_segments/max(seg_num,1)) - 1)/60):.2f} minutes')
    
    sub_seg_training_size = int(math.ceil((fr1*sub_segment_length)/(0.5*stft_sample_width)) + 1 - 2*(height-1)) # should already be an int but python is goofy
    training_size = num_sub_segments * sub_seg_training_size
    # Preallocate space for the input windows and target -> stops memory fragmentation & improves efficiency
    input_windows = np.zeros(shape=(training_size,height,width))
    targets = np.zeros(shape=(training_size,width))

    for sub_seg_num in range(num_sub_segments):
        start = int(sub_segment_order[seg_num*num_sub_segments + sub_seg_num])
        # Convert a specific segment into spectrograms (we don't have the memory to convert all 6 hours)
        x_data_complex = ProjectUtils.scipy_STFT(wav_x_data[start*fr1:(start + sub_segment_length)*fr1], fr1, stft_sample_width)
        y_data_complex = ProjectUtils.scipy_STFT(wav_y_data[start*fr1:(start + sub_segment_length)*fr1], fr1, stft_sample_width)
        # Convert the spectrograms into amplitude only (abs does magnitude for some reason)
        x_data = np.abs(x_data_complex)
        y_data = np.abs(y_data_complex)
        # Convert the amplitude into log amplitude
        x_data = ProjectUtils.convert_To_Log(x_data)
        y_data = ProjectUtils.convert_To_Log(y_data)
        # Convert batch into windows
        data_length = x_data.shape[0]
        batch_array = np.zeros(shape=(data_length - height + 1, height, width, 1))
        for i in range(data_length - height + 1):
            #batch_array[i] = np.expand_dims(x_data[i:i+height, :], axis=-1)
            batch_array[i] = x_data[i:i+height, :, np.newaxis]
        # Run it throught the layer 1 model -> initial denoise
        x_data = layer_one_model.predict(batch_array, verbose=0)
        # Recreate the corresponding window and target
        data_length = x_data.shape[0]
        if (data_length - height + 1 != sub_seg_training_size):
            print("Messed up on the sub_seg_training_size calcs")
            exit()
        for i in range(data_length - height + 1): # load each of the input windows and corresponding targets into final np array
            input_windows[sub_seg_training_size*sub_seg_num + i] = (x_data[i:i+height, :])
            targets[sub_seg_training_size*sub_seg_num + i] = (y_data[i + 2*(height // 2), :]) # we take off two of these heights for each time -> therefore 2 * height // 2

    # X_train, y_train, X_val, y_val are your training and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(input_windows, targets, test_size=0.2, random_state=None)

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
    if (seg_num % num_segments_per_save == 0 and seg_num != 0):
        model.save(f'./Models/Layer2/denoiserV1-{seg_num // num_segments_per_save}-0.keras')
model.save(f'./Models/Layer2/donoiserV1-{seg_num // num_segments_per_save}-{seg_num % num_segments_per_save}.keras')