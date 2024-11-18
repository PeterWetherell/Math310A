import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import AddNoise
import ConvertData

# Grab all of the raw data
wav_x_data,fr1 = AddNoise.load_wav("./NormalizedSoundData/Noisy/PAP1.wav")
wav_y_data,fr2 = AddNoise.load_wav("./NormalizedSoundData/Clean/PAP1.wav")

if (fr1 != fr2):
    print("Error with frame rate of both files: BIG ISSUE")
    exit()

width = 256
height = 137 # Was chosen in order to have 0.4 sec -> (0.4 * 44100)/(256 * 0.5) [bottom is times 0.5 due to the 0.5 overlap]
channels = 1

# Convert into spectrograms
x_data_complex = ConvertData.compute_STFT(wav_x_data, width)
y_data_complex = ConvertData.compute_STFT(wav_y_data, width)

if (x_data_complex.shape != y_data_complex.shape):
    print("Error with conversion into STFT. Data must have the same shape")
    exit()

# Convert the spectrograms into amplitude only (abs does magnitude for some reason)
x_data = np.abs(x_data_complex)
y_data = np.abs(y_data_complex)

# Convert into windows with corresponding targets
data_length = x_data.shape[0]
input_windows = []
targets = []
for i in range(data_length - height + 1):
    input_windows.append(x_data[i:i+height, :])
    targets.append(y_data[i+height/2, :])

# X_train, y_train, X_val, y_val are your training and validation datasets
X_train, X_val, y_train, y_val = train_test_split(input_windows, targets, test_size=0.2, random_state=42)

# Initialize the model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten and add dense layers
model.add(Flatten())
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=width, activation='linear')) # Set to width -> we want to produce that many distinct outputs to correspond with our STFT

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Data augmentation (optional, if needed)
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

batch_size = 10
epochs = 3

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), epochs=epochs, validation_data=(X_val, y_val))

# Evaluate the model
loss = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}')