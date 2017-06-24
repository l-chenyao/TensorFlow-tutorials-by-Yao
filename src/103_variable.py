#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import tensorflow as tf

State = tf.Variable(0, name = 'counter')
#print(State.name)
one = tf.constant(1)

new_value = tf.add(State, one, name = 'function') #Blank is not accept
update = tf.assign(State, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(State))