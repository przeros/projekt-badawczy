import tensorflow.compat.v2 as tf
import numpy as np
import librosa
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# TRAINING
train_data_amount = 100
train_dir = 'train/'
test_dir = 'test/'
mp3_files = []
labels = []
mp3_ext = '.mp3'
num_mfcc = 13
num_frames = 50 #50

df = pd.read_excel('sheets_with_data.xlsx',engine='openpyxl',dtype=object,header=0)
excelList:list = df.values.tolist()

for i in range(train_data_amount):
    oneRow=excelList[i]
    mp3_files.append(train_dir + oneRow[8] + mp3_ext)
    labels.append(oneRow[9])

print(mp3_files)
print(len(mp3_files))
print(labels)
print(len(labels))

# Load and preprocess data
X = []
y = []

i = 0
for mp3_file, label in zip(mp3_files, labels):
    audio, sr = librosa.load(mp3_file)
    audio = librosa.util.fix_length(data=audio, size=num_frames * sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=2048, hop_length=512)
    X.append(mfcc)
    y.append(label)
    i += 1

# Convert data to numpy arrays
X = np.array(X)
y = np.array(y)

# Stack the MFCC coefficients for each audio file into a 3D array
X = np.stack(X)

# Add extra dimension to represent number of channels
X = np.expand_dims(X, axis=-1)

# Split the data into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding all labels
label_map = {}
num_labels = 0
for label in y:
    if label not in label_map:
        label_map[label] = num_labels
        num_labels += 1
print(y)
y_encoded = [label_map[label] for label in y]
y = to_categorical(y_encoded, num_classes=len(y))
print(y_encoded)


print(X.shape)
print(len(y))
# Build model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(len(y), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Train model
model.fit(X, y, batch_size=32, epochs=25)

# TEST

test_data_amount = 30
mp3_files_test = []
labels_test = []

for i in range(test_data_amount):
    oneRow=excelList[i]
    mp3_files_test.append(test_dir + str(i) + mp3_ext)
    labels_test.append(oneRow[9])

# Load test data
X_test = []

i = 0
for mp3_file, label in zip(mp3_files_test, labels_test):
    audio, sr = librosa.load(mp3_file)
    audio = librosa.util.fix_length(data=audio, size=num_frames * sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=2048, hop_length=512)
    X_test.append(mfcc)
    i += 1

print(mp3_files_test)
print(labels_test)
# Convert data to numpy arrays
X_test = np.array(X_test)

# Stack the MFCC coefficients for each audio file into a 3D array
X_test = np.stack(X_test)

# Add extra dimension to represent number of channels
X_test = np.expand_dims(X_test, axis=-1)

# Evaluate model on test data
predictions = np.argmax(model.predict(X_test), axis=1)
real_vals = labels_test
correct_classifications = 0
all_classifications = len(predictions)
for prediction, real in zip(predictions, real_vals):
    print(f"predicted: {prediction}    real: {label_map.get(real)}")
    correct_classifications += (prediction == label_map.get(real))

accuracy = round(correct_classifications / all_classifications * 100, 2)
print(f'Accuracy = {accuracy}%')