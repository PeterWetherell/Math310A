import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import ProjectUtils


width = 128
height = 137 # Was chosen in order to have 0.2 sec -> (0.2 * 44100)/(128 * 0.5) [bottom is times 0.5 due to the 0.5 overlap]
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
model.add(Dense(units=width, activation='linear')) # Set to width -> we want to produce that many distinct outputs to correspond with our STFT

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

segment_length = 10 # Number of seconds in the segment length
segment_frames = segment_length * fr1
num_segments = len(wav_x_data) // segment_frames

sub_segment_order = np.linspace(0,num_segments * 10 - 1,num_segments * 10) # Get a list of each second we can choose from
np.random.shuffle(sub_segment_order) # Shuffle it arround

#print(sub_segment_order)

for seg_num in range(num_segments):
    input_windows = []
    targets = []
    
    print(f'Converting segment {seg_num} from raw data into STFT')

    for i in range(segment_length):
        start = int(sub_segment_order[seg_num*10+i])
        #print(start)
        # Convert a specific segment into spectrograms (we don't have the memory to convert all 6 hours)
        x_data_complex = ProjectUtils.compute_STFT(wav_x_data[start*fr1:(start + 1)*fr1], width)
        y_data_complex = ProjectUtils.compute_STFT(wav_y_data[start*fr1:(start + 1)*fr1], width)

        if (x_data_complex.shape != y_data_complex.shape):
            print("Error with conversion into STFT. Data must have the same shape")
            exit()

        # Convert the spectrograms into amplitude only (abs does magnitude for some reason)
        #print("Converting STFT to amplitude only")
        x_data = np.abs(x_data_complex)
        y_data = np.abs(y_data_complex)

        # Convert into windows with corresponding targets
        #print("Switching into windowed data for training")
        data_length = x_data.shape[0]
        for i in range(data_length - height + 1):
            input_windows.append(x_data[i:i+height, :])
            targets.append(y_data[int(i+height/2), :])

    # Convert from python list to numpy list
    input_windows = np.array(input_windows)
    targets = np.array(targets)


    print("window shape ", input_windows.shape)
    print("targets shape ", targets.shape)

    # X_train, y_train, X_val, y_val are your training and validation datasets
    X_train, X_val, y_train, y_val = train_test_split(input_windows, targets, test_size=0.2, random_state=42)

    # Train the model
    batch_size = 128
    epochs = 3
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))
    # TODO: SAVE THE MODEL!!! We need to save after each subset of the data is put through it

    # Evaluate the model
    #loss = model.evaluate(X_val, y_val)
    #print(f'Validation Loss: {loss}')