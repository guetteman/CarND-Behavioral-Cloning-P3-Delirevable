import csv
import cv2
import numpy as np

# Load images paths from csv file
lines = [];

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        lines.append(line)
    
# Fix images path to use in another machine like AWS
images = []
measurements = []

for line in lines:
    for i in range(3):

        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename

        image = cv2.imread(current_path)
        images.append(image)

        measurement = float(line[3])
        measurements.append(measurement)

# Data augmentation
augmented_images, augmented_measurements = [], []

for image, measurement in zip(images, measurements):
    
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement * -1.0)

# Convert inputs and labels as np array
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# Build network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

#Train model
model.save('model.h5')
