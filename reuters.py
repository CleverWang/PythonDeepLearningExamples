#! /usr/bin/python3

from keras.datasets import reuters
from keras import models, layers
from keras import optimizers, losses, metrics
from keras.utils import to_categorical

from utils import vectorize_seqs, to_one_hot
from validation_plot import validation_plot

num_words = 10000
num_categories = 46


# load data
print('load data...')
(train_data, train_labels), (test_data,
                             test_labels) = reuters.load_data(num_words=num_words)

# restore to original data
""" word_idx=reuters.get_word_index()
reverse_word_idx=dict([(v,k) for (k,v) in word_idx.items()])
decode_newswire=' '.join([reverse_word_idx.get(i-3,'?') for i in train_data[0]])
print(decode_newswire) """

# vectorize seqs
print('vetorize...')
x_train = vectorize_seqs(train_data, num_words)
x_test = vectorize_seqs(test_data, num_words)

# vectorize labels
one_hot_train_labels = to_one_hot(train_labels, num_categories)
one_hot_test_labels = to_one_hot(test_labels, num_categories)
""" one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels) """
""" import numpy as np
y_train = np.array(train_labels)
y_test = np.array(test_labels) """

# define model
print('define model...')
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(num_words,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# compile model
print('compile model...')
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
""" model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']) """

""" # reserve validation datasets
print('reserve validation datasets...')
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# train the model with validation
print('train model...')
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

# plot
validation_plot(history) """

# train model
print('train model...')
history = model.fit(x_train,
                    one_hot_train_labels,
                    epochs=9,
                    batch_size=512)
""" history = model.fit(x_train,
                    y_train,
                    epochs=9,
                    batch_size=512) """

# evaluate test dataset
print('evaluate...')
results = model.evaluate(x_test, one_hot_test_labels)
print('result: ', end='')
print(results)
""" print('evaluate...')
results = model.evaluate(x_test, y_test)
print('result: ', end='')
print(results) """

""" # predict
print('predict...')
predicts = model.predict(x_test)
print('predicts: ', end='')
print(predicts)
print('reality: ', end='')
print(one_hot_test_labels) """
