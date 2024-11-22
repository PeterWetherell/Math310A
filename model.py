import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

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

if (fr1 != fr2):
    print("Error with frame rate of both files: BIG ISSUE")
    exit()

# Initialize the model
model = Sequential()
# Add convolutional layers
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# Flatten and add dense layers
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=width, activation='linear')) # tanh
# Set to width -> we want to produce that many distinct outputs to correspond with our STFT

# Compile the model
initial_learning_rate = 0.001
model.compile(optimizer=Adam(learning_rate=initial_learning_rate), loss="mean_squared_error") # ProjectUtils.log_spectral_distance

segment_length = 120 # Number of seconds in the segment length
segment_frames = segment_length * fr1
num_segments = len(wav_x_data) // segment_frames

sub_segment_order = np.linspace(0,num_segments * segment_length - 1, num_segments * segment_length) # Get a list of each second we can choose from
np.random.shuffle(sub_segment_order) # Shuffle it arround

num_segments_per_save = 10
start_time = time.time()

batch_size = 256 # 512
epochs = 3 # 6

for seg_num in range(num_segments):
    input_windows = []
    targets = []
    
    print(f'Converting segment {seg_num} out of {num_segments} from raw data into STFT')
    curr_time = time.time()
    if (curr_time - start_time > 30):
        print(f'Estimated Time Remaining {((curr_time - start_time) * ((num_segments/max(seg_num,1)) - 1)/60):.2f} minutes')

    for i in range(segment_length):
        start = int(sub_segment_order[seg_num*segment_length+i])
        # Convert a specific segment into spectrograms (we don't have the memory to convert all 6 hours)
        x_data_complex = ProjectUtils.scipy_STFT(wav_x_data[start*fr1:(start + 1)*fr1], fr1, stft_sample_width)
        y_data_complex = ProjectUtils.scipy_STFT(wav_y_data[start*fr1:(start + 1)*fr1], fr1, stft_sample_width)

        """
        # Convert the spectrograms into amplitude only (abs does magnitude for some reason)
        x_data = np.abs(x_data_complex)
        y_data = np.abs(y_data_complex)
        """
        # Convert the spectrograms into log amplitude
        x_data = ProjectUtils.convert_To_Log(np.abs(x_data_complex))
        y_data = ProjectUtils.convert_To_Log(np.abs(y_data_complex))

        # Convert into windows with corresponding targets
        data_length = x_data.shape[0]
        for i in range(data_length - height + 1):
            input_windows.append(x_data[i:i+height, :])
            targets.append(y_data[i + height // 2, :])

    # Convert from python list to numpy list
    input_windows = np.array(input_windows)
    targets = np.array(targets)

    # X_train, y_train, X_val, y_val are your training and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(input_windows, targets, test_size=0.2, random_state=None)

    # Train the model
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
    if (seg_num % num_segments_per_save == 0 and seg_num != 0):
        model.save(f'./Models/deonoiserV5-{seg_num // num_segments_per_save}-0.keras')
model.save(f'./Models/deonoiserV5-{seg_num // num_segments_per_save}-{seg_num % num_segments_per_save}.keras')