import csv
import csv2
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
    
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    
    image = cv2.imread(current_path)
    images.append(image)

    measurement = float(line[3])
    measurements.append(measurement)

# Convert inputs and labels as np array
X_train = np.array(images)
y_train = np.array(measurements)

# Build network
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

#Train model
model.save('model.h5')