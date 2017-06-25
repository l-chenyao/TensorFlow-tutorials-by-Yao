"""
The first regression method use TensorFlow 
Dependencies:
tensorflow: 1.2.0
matplotlib
numpy

"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 

tf.set_random_seed(1)                           #To control the random distribution every time use the same number 
np.random.seed(1)                               #https://www.tensorflow.org/api_docs/python/tf/set_random_seed, TensorFlow API have the clear means 

#fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]      #shape(100, 1))
noise = np.random.normal(0, 0.1, size = x.shape)
y = np.power(x , 2) + noise                      #shape (100, 1) + some nosie

#plot data
plt.scatter(x, y)
plt.show()

tf_x = tf.placeholder(tf.float32, x.shape)      #input x
tf_y = tf.placeholder(tf.float32, y.shape)      #input Y

#neural network layers
l1 = tf.layers.dense(tf_x, 10, tf.nn.relu )       #hidden layer
output = tf.layers.dense(l1, 1)                   #ouput layer     
 
loss = tf.losses.mean_squared_error(tf_y, output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train_op = optimizer.minimize(loss)

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph


plt.ion()                                       #open interactive 
 
for step in range(200):
    #train and net output
    _, l, pred = sess.run([train_op, loss , output], {tf_x: x, tf_y:y})             #each var get one value from the function
    if step % 5 == 0:
        #plot and show learning process
        plt.cla()
        plt.scatter(x,y)
        plt.plot(x, pred, 'r-', lw = 5)
        plt.text(0.5, 0, 'Loss = %.4f' % l, fontdict= {'size':20, 'color':'red'})
        plt.pause(0.1)
         
plt.ioff()    
plt.show()