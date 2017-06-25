'''
Dependencies:
tensorflow: 1.2.0
matplotlib

Introduction about the activation function for NN 

'''

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt
#from IPython.core.pylabtools import figsize

#fake data
x = np.linspace(-5, 5, 200) #x data, shape = (200, 1))

#following are popular Activation functions
y_relu = tf.nn.relu(x)
y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)
y_softmax = tf.nn.softmax(x)

sess = tf.Session()
y_relu, y_sigmoid, y_tanh, y_softplus, y_softmax= sess.run([y_relu, y_sigmoid, y_tanh, y_softplus, y_softmax]) # use [] as one input parameter

# plt to visualize these Activation
plt.figure(1, figsize = (15, 15))
plt.subplot(321)
plt.plot(x, y_relu, c='red', label = 'relu')
plt.ylim((-1, 5))
plt.legend(loc = 'best')

plt.subplot(322)
plt.plot(x, y_sigmoid, c='red', label = 'sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc = 'best')

plt.subplot(323)
plt.plot(x, y_tanh, c='red', label = 'tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc = 'best')

plt.subplot(324)
plt.plot(x, y_softplus, c='red', label = 'softplus')
plt.ylim((-0.2, 6))
plt.legend(loc = 'best')

plt.subplot(325)
plt.plot(x, y_softmax, c='red', label = 'softmax')
plt.ylim((-0,0.1))
plt.legend(loc = 'best')


plt.show()