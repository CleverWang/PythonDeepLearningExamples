#! /usr/bin/python3

from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical

# prepare train set and test set
print('load data...')
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('data sets normalize...')
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255
print('data labels categorize...')
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# define model
print('define model...')
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))

# compile model
print('compile model...')
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# train model
print('train model...')
network.fit(train_images, train_labels, epochs = 5, batch_size = 128)

#evaluate
print('evaluate model...')
result = network.evaluate(test_images, test_labels)
print('test loss: %f\ntest accuracy: %f' % tuple(result))
