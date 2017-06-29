#Readme:  this is the code come from    http://gonzalopla.com/deep-learning-nonlinear-regression/

# Numeric Python Library.
import numpy as np
# Python Data Analysis Library.
import pandas
# Scikit-learn Machine Learning Python Library modules.
#   Preprocessing utilities.
from sklearn import preprocessing
#   Cross-validation utilities.
from sklearn import cross_validation
# Python graphical library
from matplotlib import pyplot
 
# Keras perceptron neuron layer implementation.
from keras.layers import Dense
# Keras Dropout layer implementation.
from keras.layers import Dropout
# Keras Activation Function layer implementation.
from keras.layers import Activation
# Keras Model object.
from keras.models import Sequential


# Training model with train data. Fixed random seed:
np.random.seed(3)

# Slicing all rows, second column...
X = np.linspace(-100, 100, 100000)[:, np.newaxis]

noise = np.random.normal(0, 0.02, size = X.shape)
# Slicing all rows, first column...
y = np.power(X ,1)+ 2*np.power(X ,7)+ 2*np.power(X ,5)+ 2*np.power(X ,2)+ noise
 
# Data Scaling from 0 to 1, X and y originally have very different scales.
X_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
y_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_scaled = ( X_scaler.fit_transform(X.reshape(-1,1)))
y_scaled = (y_scaler.fit_transform(y.reshape(-1,1)))
 
# Preparing test and train data: 60% training, 40% testing.
X_train, X_test, y_train, y_test = cross_validation.train_test_split( \
    X_scaled, y_scaled, test_size=0.40, random_state=3)

# build a neural network from the 1st layer to the last layer
# New sequential network structure.
model = Sequential()
 
# Input layer with dimension 1 and hidden layer i with 128 neurons. 
model.add(Dense(128, input_dim=1, activation='relu'))
# Dropout of 20% of the neurons and activation layer.
#model.add(Dropout(.2))
model.add(Activation("linear"))

# Hidden layer j with 64 neurons plus activation layer.
model.add(Dense(64, activation='relu'))
model.add(Activation("linear"))
# Hidden layer k with 64 neurons.
model.add(Dense(64, activation='relu'))
# Output Layer.
model.add(Dense(1))
 
# Model is derived and compiled using mean square error as loss
# function, accuracy as metric and gradient descent optimizer.
model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])
 

model.fit(X_train, y_train, nb_epoch=10, batch_size=5, verbose=2)

# Predict the response variable with new data
predicted = model.predict(X_scaled)
 
# Plot in blue color the predicted adata and in green color the
# actual data to verify visually the accuracy of the model.
#pyplot.plot(y_scaler.inverse_transform(predicted), color="blue")
#pyplot.plot(y_scaler.inverse_transform(y_test), color="yellow")
pyplot.plot(predicted, color="blue")
pyplot.plot(y_scaled, color="yellow")

# pyplot.plot(predicted, color="blue")
# pyplot.plot(y, color="yellow")
pyplot.show()