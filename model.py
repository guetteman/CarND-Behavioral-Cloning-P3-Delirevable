import csv
import cv2
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

# Definitions

def load_data_from_csv(csv_file):
    data = []    

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            data.append(line)
        
        return data

def get_images(lines, base_path):

    images = []
    measurements = []

    for line in lines:
        for i in range(3):

            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = base_path + filename

            image = cv2.imread(current_path)
            images.append(image)

            # angle corrections
            if i == 0:
                measurement = float(line[3])
            elif i == 1:
                measurement = float(line[3]) + 0.2
            else:
                measurement = float(line[3]) - 0.2

            measurements.append(measurement)
    
    return images,measurements

def augment_data(images, measurements):

    augmented_images, augmented_measurements = [], []

    for image, measurement in zip(images, measurements):
    
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement * -1.0)

    return augmented_images,augmented_measurements

def generator(images, measurements, batch_size=32):
    
    num_samples = len(images)
    
    while 1:
        for offset in range(0, num_samples, batch_size):
            batch_images = images[offset:offset+batch_size]
            batch_measurements = measurements[offset:offset+batch_size]

            # trim image to only see section with road
            X_train = np.array(batch_images)
            y_train = np.array(batch_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# Load images paths from csv file
track1_lines = load_data_from_csv('data/driving_log.csv')
track2_lines = load_data_from_csv('data1/driving_log.csv')
   
# Fix images path to use in another machine like AWS
track1_images, track1_measurements = get_images(track1_lines, 'data/IMG/')
track2_images, track2_measurements = get_images(track2_lines, 'data1/IMG/')

images = track1_images + track2_images
measurements = track1_measurements + track2_measurements

# Data augmentation
augmented_images, augmented_measurements = augment_data(images, measurements)

train_images, validation_images, train_measurements, validation_measurements = train_test_split(augmented_images, augmented_measurements, test_size=0.20)

# compile and train the model using the generator function
train_generator = generator(train_images, train_measurements, batch_size=32)
validation_generator = generator(validation_images, validation_measurements, batch_size=32)

# Build network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20),(0,0))))
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
# model.fit_generator(train_generator, samples_per_epoch= len(train_images), validation_data=validation_generator, validation_steps=len(validation_images), nb_epoch=5, verbose = 1)

model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_images), validation_data=validation_generator, \
                 nb_val_samples=len(validation_images), nb_epoch=5, verbose=1)

#Train model
model.save('model.h5')
