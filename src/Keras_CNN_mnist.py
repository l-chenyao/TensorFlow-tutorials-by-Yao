import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adam

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

# Another way to build your CNN
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
# model.add(Conv2D(
#     batch_input_shape=(64, 1, 28, 28),
#     filters=32,
#     kernel_size=5,
#     strides=1,
#     padding='same',     # Padding method
#     data_format='channels_first',
# ))
# model.add(Activation('relu'))

model.add(Conv2D(32, 5, 5, activation='relu', input_shape=(1, 28, 28)))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
# model.add(MaxPooling2D(
#     pool_size=2,
#     strides=2,
#     padding='same',    # Padding method
#     data_format='channels_first',
# ))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

# Conv layer 2 output shape (64, 14, 14)
# model.add(Conv2D(64, 5, strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
model.add(Conv2D(64, 5, 5, activation='relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
# model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, batch_size=64, nb_epoch=10 )

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)