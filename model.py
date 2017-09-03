import csv
import cv2
import numpy as np

lines = []
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter = ' ')
    for line in reader:
        lines.append(line)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, MaxPooling2D

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(lines, test_size = 0.3)
batch_size = 32
correction = 0.08

def generator(samples, batch_size = batch_size):
    '''
    generator to generate input for each batch
    '''
    n_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset : offset + batch_size]
        
            images = []
            measurements = []

            for sample in batch_samples:
                # center camera image
                image = cv2.imread("./data/IMG/" + sample[0].split('\\')[-1])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # left camera image
                image_l = cv2.imread("./data/IMG/" + sample[1].split('\\')[-1])
                image = cv2.cvtColor(image_l, cv2.COLOR_BGR2RGB)
                # right camera image
                image_r = cv2.imread("./data/IMG/" + sample[2].split('\\')[-1])
                image = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
                # steering angle
                steering = float(sample[3])

                images.append(image)
                measurements.append(steering)
                # flipping image
                images.append(np.fliplr(image))
                measurements.append(-steering)
                # left camera
                images.append(image_l)
                measurements.append(steering + correction)
                # right camera
                images.append(image_r)
                measurements.append(steering - correction)

            X = np.array(images)
            y = np.array(measurements)

            yield (X, y)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model = Sequential()
# cropping image before any processing
model.add(Cropping2D(cropping = ((50, 20), (0, 0)), input_shape = (160, 320, 3)))
# normalization
model.add(Lambda(lambda x: (x / 255.0 - 0.5)))

# LeNet model
#model.add(Convolution2D(6, 5, 5, activation = 'relu'))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6, 5, 5, activation = 'relu'))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(120, activation = 'relu'))
#model.add(Dense(84, activation = 'relu'))
#model.add(Dense(1))

# Nvidia model
model.add(Convolution2D(24, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample = (2, 2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit_generator(train_generator, samples_per_epoch = 4 * len(train_samples), \
    validation_data = validation_generator, nb_val_samples = 4 * len(validation_samples), \
    nb_epoch = 5)

model.save('model.h5')

