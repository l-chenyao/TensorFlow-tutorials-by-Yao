'''
Two points 
    - placeholder need feed_dict to putin value
    - 
'''
import tensorflow as tf

x1 = tf.placeholder(dtype = tf.float32, shape = None)
y1 = tf.placeholder(dtype = tf.float32, shape = None)
z1 = x1+y1

x2 = tf.placeholder(dtype =tf.float32, shape = None)
y2 = tf.placeholder(dtype = tf.float32, shape = None)
z2 = tf.matmul(x2, y2)

with tf.Session() as sess:
    #when only one operation to run
    z1_value = sess.run(z1,feed_dict = {x1:1, y1:2})
    print("z1 value = ", z1_value)
    #when run multiple operatons
    z1_value, z2_value = sess.run(
        [z1, z2],   #run them together
        feed_dict = {
            x1:2, y1:5,
            x2:[[2], [2]], y2:[[3,3]]  #x2 is a matrix 2*1,,  y2 is a vector 1*2
        })
    print("z1 value = ", z1_value)
    print("z2 value = ", z2_value)