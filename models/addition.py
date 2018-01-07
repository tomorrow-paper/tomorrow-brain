#!/usr/bin/python3

import tensorflow as tf

x = tf.placeholder(tf.float32, name = 'x')
y = tf.placeholder(tf.float32, name = 'y')
z = tf.add(x, y, name = 'z')

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

tf.train.write_graph(session.graph_def, './models', 'addition.pb', as_text = False)