#! /usr/bin/python3
# print("hello world!")

from keras.datasets import imdb
from keras import models, layers
from keras import optimizers, losses, metrics
import numpy as np
import matplotlib.pyplot as plt

num_words = 10000


def vectorize_seqs(seqs, dim=num_words):
    """
    vectorize seqs to the seqs of shape (len(seqs), dim).
    """
    res = np.zeros((len(seqs), dim))
    for i, seq in enumerate(seqs):
        res[i, seq] = 1
    return res


# load data
print('load data...')
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=num_words)

# restore to original data
""" word_idx=imdb.get_word_index()
reverse_word_idx=dict([(v,k) for (k,v) in word_idx.items()])
decode_review=' '.join([reverse_word_idx.get(i-3,'?') for i in train_data[0]])
print(decode_review) """

# vectorize the seqs
print('vetorize...')
x_train = vectorize_seqs(train_data)
x_test = vectorize_seqs(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# define the model
print('define model...')
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(num_words,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# compile the model
print('compile model...')
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
""" model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metric=[metrics.binary_accuracy]) """

# reserve the validation dataset
""" print('reserve validation dataset...')
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:] """

# train the model with validation
""" print('train model...')
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val)) """

""" history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.subplot(1, 2, 2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show() """

# train the model
print('train model...')
history = model.fit(x_train,
                    y_train,
                    epochs=4,
                    batch_size=512)

# evaluate the test dataset
print('evaluate...')
results = model.evaluate(x_test, y_test)
print('result: ', end='')
print(results)

#predict
print('predict...')
predicts = model.predict(x_test)
print('predicts: ', end='')
print(predicts)
print('reality: ', end='')
print(y_test)