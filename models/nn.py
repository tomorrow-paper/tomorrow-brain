#!/usr/bin/python3

from __future__ import print_function

import csv
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("datasets/", one_hot = True)

# Training Parameters
learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256    # 1st layer number of neurons
n_hidden_2 = 256    # 2nd layer number of neurons
num_input = 784     # MNIST data input (img shape: 28*28)
num_classes = 10    # MNIST total classes (0-9 digits)

# Inputs
X = tf.placeholder("float", [None, num_input], name = 'x')
Y = tf.placeholder("float", [None, num_classes], name = 'y')

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    
    return tf.identity(out_layer, name = 'output')

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train = optimizer.minimize(loss)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as session:
    # Run the initializer
    session.run(init)

    for step in range(1, num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop)
        session.run(train, feed_dict={X: batch_x, Y: batch_y})

        # Calculate batch loss and accuracy
        if step % display_step == 0 or step == 1:
            l, acc = session.run([loss, accuracy], feed_dict = {
                X: batch_x,
                Y: batch_y
            })
            print(
                "Step " + str(step) + ", " +
                "Minibatch Loss = " + "{:.4f}".format(l) + ", " + 
                "Training Accuracy = " + "{:.3f}".format(acc)
            )

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        session.run(accuracy, feed_dict={X: mnist.test.images,
        Y: mnist.test.labels}))
    
    tf.train.write_graph(session.graph_def, './models', 'nn.pb', as_text = False)