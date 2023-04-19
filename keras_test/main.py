import librosa
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

mp3_files = []
labels = []
mp3_ext = '.mp3'
num_mfcc = 13
num_frames = 50

df = pd.read_excel('sheets_with_data.xlsx',engine='openpyxl',dtype=object,header=0)
excelList:list = df.values.tolist()

for i in range(74):
    oneRow=excelList[i]
    mp3_files.append(oneRow[8] + mp3_ext)
    labels.append(oneRow[9])

print(mp3_files)
print(len(mp3_files))
print(labels)
print(len(labels))

# Load and preprocess data
X = []
y = []

for mp3_file, label in zip(mp3_files, labels):
    audio, sr = librosa.load(mp3_file)
    audio = librosa.util.fix_length(data=audio, size=num_frames * sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=2048, hop_length=512)
    X.append(mfcc)
    y.append(label)

# Stack the MFCC coefficients for each audio file into a 3D array
X = np.stack(X)

# X = np.array(X)
# y = np.array(y)
# Create label mapping
label_to_int = {label: i for i, label in enumerate(np.unique(y))}
int_to_label = {i: label for label, i in label_to_int.items()}
y = np.array([label_to_int[label] for label in y])

# Split data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Reshape the input data
X_train = X_train.reshape(X_train.shape[0], num_mfcc, num_frames, 1)
X_test = X_test.reshape(X_test.shape[0], num_mfcc, num_frames, 1)
print(X_train.shape)

# Build model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(num_mfcc, num_frames, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(y), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)