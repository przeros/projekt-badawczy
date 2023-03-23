import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense
import glob

mp3_files = []

# find all mp3 files with names 'C01', 'C02', and 'C03'
for filename in glob.glob('C*.mp3'):
    mp3_files.append(filename)

# Load and preprocess data
X = []
y = []

print(mp3_files)
for mp3_file, label in zip(mp3_files, labels):
    audio, sr = librosa.load(mp3_file)
    mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=20)
    X.append(mfccs.T)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Create label mapping
label_to_int = {label: i for i, label in enumerate(np.unique(y))}
int_to_label = {i: label for label, i in label_to_int.items()}
y = np.array([label_to_int[label] for label in y])

# Split data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(len(label_to_int), activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)