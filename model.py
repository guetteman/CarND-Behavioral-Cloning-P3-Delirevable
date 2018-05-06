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
            measurements.append(measurement)
    
    return images,measurements

def random_flip(image, measurement):
    if np.random.rand() > 0.5:
        image = cv2.flip(image,1)
        measurement = measurement * -1.0
    
        return image, measurement
    else:
        return None, None

def random_translation(image, measurement, trans_range):
    if np.random.rand() > 0.5:
        rows,cols,ch = image.shape
        tr_x = trans_range*np.random.uniform()-trans_range/2
        tr_y = trans_range*np.random.uniform()-trans_range/2
        Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

        image = cv2.warpAffine(image,Trans_M,(cols,rows))
        measurement += tr_x * 0.002
    
        return image, measurement

    else: 
        return None, None

def random_brightness(image):
    if np.random.rand() > 0.5 :

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
    if augmented_image is not None:
            augmented_images.append(augmented_image)
            augmented_measurements.append(augmented_measurement)
    
    return augmented_images, augmented_measurements

def augment_data(images, measurements):

    augmented_images, augmented_measurements = [], []

    for image, measurement in zip(images, measurements):
    
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        
        augmented_image, augmented_measurement = random_flip(image, measurement)
        augmented_images, augmented_measurement = add_to_augmented_data(augmented_image, augmented_measurement, augmented_images, augmented_measurements)        

        augmented_image, augmented_measurement = random_translation(image, measurement, 5)
        augmented_images, augmented_measurement = add_to_augmented_data(augmented_image, augmented_measurement, augmented_images, augmented_measurements)
        
        #augmented_image = random_brightness(augmented_image)

        #augmented_images.append(augmented_image)
        #augmented_measurements.append(augmented_measurement)

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
lines = load_data_from_csv('data/driving_log.csv')

# Fix images path to use in another machine like AWS
images, measurements = get_images(lines, 'data/IMG/')

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
from keras.optimizers import Adam

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

model.compile(loss='mean_squared_error', optimizer=Adam(lr=1.0e-3))
# model.fit_generator(train_generator, samples_per_epoch= len(train_images), validation_data=validation_generator, validation_steps=len(validation_images), nb_epoch=5, verbose = 1)

model.fit_generator(train_generator, samples_per_epoch= \
                 len(train_images), validation_data=validation_generator, \
                 nb_val_samples=len(validation_images), nb_epoch=4, verbose=1)

#Train model
model.save('model.h5')
