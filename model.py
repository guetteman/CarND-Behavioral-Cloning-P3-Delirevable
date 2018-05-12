import csv
import cv2
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random

# Definitions

def load_data_from_csv(csv_file):
    data = []    

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:
            data.append(line)
        
        return data

def angle_correction(measurement, index):
    if index == 1:
        measurement = measurement + 0.2
    elif index == 2:
        measurement = measurement - 0.2
    
    return measurement

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
            measurement = angle_correction(float(line[3]), i)
            measurements.append(round(measurement, 2))
    
    images_measurements = list(zip(images, measurements))
    random.shuffle(images_measurements)

    return zip(*images_measurements)

def plot_distribution_chart(x, y, xlabel, ylabel, width, color, location):
    plt.figure(figsize=(15,7))
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.bar(x, y, width, color=color)
    plt.savefig(location)

def random_flip(image, measurement):
    if np.random.rand() > 0.5:
        image = cv2.flip(image,1)
        measurement = measurement * -1.0

    return image, measurement 

def random_translation(image, measurement, trans_range):
    rows,cols,ch = image.shape
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    image = cv2.warpAffine(image,Trans_M,(cols,rows))
    measurement += round(tr_x * 0.002, 2)

    return image, measurement

def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
    random_value = np.random.rand()

    if random_value > 0.5:
        ratio = 1 + random_value - 0.5
    else:
        ratio = random_value
    
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return image

def add_to_augmented_data(augmented_image, augmented_measurement, augmented_images, augmented_measurements):
    if augmented_image is not None and augmented_measurement is not None:
            augmented_images.append(augmented_image)
            augmented_measurements.append(augmented_measurement)
    
    return augmented_images, augmented_measurements

def augment_data(images, measurements, _classes, counts):

    augmented_images, augmented_measurements = [], []

    for image, measurement in zip(images, measurements):
    
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        
        max_counts = np.amax(counts)
        i, = np.where(_classes == measurement)

        if counts[i]/max_counts < 0.1:
            for i in range(5):
                augmented_image, augmented_measurement = random_flip(image, measurement)
                augmented_image, augmented_measurement = random_translation(augmented_image, augmented_measurement, 5)
                augmented_image = random_brightness(augmented_image)
                
                add_to_augmented_data(augmented_image, augmented_measurement, augmented_images, augmented_measurements)

        if counts[i] < 0.05:
            for i in range(10):
                augmented_image, augmented_measurement = random_flip(image, measurement)
                augmented_image, augmented_measurement = random_translation(augmented_image, augmented_measurement, 5)
                augmented_image = random_brightness(augmented_image)
                
                add_to_augmented_data(augmented_image, augmented_measurement, augmented_images, augmented_measurements)

    return augmented_images,augmented_measurements

def generator(lines, batch_size=32):
    
    images, measurements = get_images(lines, 'data/IMG/')

    while 1:

        for offset in range(0, len(lines), batch_size):
            batch_images = images[offset:offset+batch_size]
            batch_measurements = measurements[offset:offset+batch_size]

            _classes, counts = np.unique(np.array(measurements), return_counts=True)
            batch_images, batch_measurements = augment_data(batch_images, batch_measurements, _classes, counts)

            X_train = np.array(batch_images)
            y_train = np.array(batch_measurements)

            yield sklearn.utils.shuffle(X_train, y_train)



#Main code

# Load images paths from csv file
lines = load_data_from_csv('data/driving_log.csv')
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Build network
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(160, 320, 3)))
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
model.summary()

model.compile(optimizer=Adam(), loss='mse')

checkpoint = ModelCheckpoint(
    'model-{epoch:03d}.h5', 
    monitor='val_loss', 
    verbose=0, 
    save_best_only=False, 
    save_weights_only=False, 
    mode='auto', 
    period=1)

model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_samples)*10, validation_data=validation_generator, \
                 nb_val_samples=len(validation_samples)*10, nb_epoch=10, callbacks=[checkpoint], verbose=1)
