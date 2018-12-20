# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:49:14 2018

@author: yaua
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import *

import math
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error

dataframe = pd.read_csv('AlexData1.csv')
#all data:
#x = dataframe.iloc[:,1:7]
#12/5/1988 onward (exclude all dates with NaN)
x = dataframe.iloc[2329:10091,1:7]

x_diff = x.pct_change()
x_diff = x_diff.iloc[1:,]
##x_diff[isnan(x_diff)] = 0
x['Index'] = range(1, len(x) + 1)
y = x.iloc[:,6]

#x_diff = x_diff + 0.5

x_dataset = x_diff.values
x_dataset = x_dataset.astype('float32')
y_dataset = y.values
y_dataset = y_dataset.astype('float32')

x_train_size = int(len(x_dataset) * 0.7)
x_test_size = len(x_dataset) - x_train_size
y_train_size = int(len(y_dataset) * 0.7)
y_test_size = len(y_dataset) - y_train_size
x_train, x_test = x_dataset[0:x_train_size,:], x_dataset[x_train_size:len(x_dataset),:]
y_train, y_test = y_dataset[0:y_train_size,], y_dataset[y_train_size:len(y_dataset),]


####################  NN starts here

def create_train_model(hidden_nodes, epochs):
    
    tf.reset_default_graph()
    
    # fed through the NN
    inputs = tf.placeholder(shape = [None, x_train.shape[1]], dtype = tf.float32)
    #inputs that are not fed through, used to calculate loss
    labels = tf.placeholder(shape = [None, x_train.shape[1]], dtype = tf.float32)
    
    # weights of each layer
    weights_encoder = tf.Variable(tf.random_normal([x_train.shape[1], hidden_nodes]))
    weights_decoder = tf.Variable(tf.random_normal([hidden_nodes, x_train.shape[1]]))
    
    # neural net graph of each layer
    encoder = tf.matmul(inputs, weights_encoder)
    ## Variational component
    z_mean, z_var = tf.nn.moments(encoder, axes=[1])
    z_stddev = tf.sqrt(z_var)
    
    samples = tf.random_normal([100, hidden_nodes],0,1,dtype=tf.float32)
    sampled_z = z_mean + (z_stddev * samples)
    
    decoder = tf.matmul(sampled_z, weights_decoder)
    
    # network loss, KL divergence and total loss
    # loss = tf.reduce_mean(tf.square(decoder - labels))
    loss = -tf.reduce_sum(labels * tf.log(1e-8 + decoder) + (1-labels) * tf.log(1e-8 + 1 - decoder),1)
    latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - [1,1])
    cost = tf.reduce_mean(loss + latent_loss)
    
    lr = 0.0005
    
    # optimizer function
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    
    # initialize & run session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        sess.run(train_op, feed_dict = {inputs: x_test, labels: x_test})
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={inputs: x_test, labels: x_test}))
        weights1 = sess.run(weights_encoder)
        weights2 = sess.run(weights_decoder)

    print("loss (hidden nodes: %d, epochs: %d): %.9f" % (hidden_nodes, epochs, loss_plot[hidden_nodes][-1]))
    sess.close()

    return weights1, weights2

num_hidden_nodes = [1, 2]
loss_plot = {1: [], 2: []}
weights1 = {1: None, 2: None}
weights2 = {1: None, 2: None}
epochs = 1000

plt.figure(figsize=(12,8)) 
for hidden_nodes in num_hidden_nodes:  
    weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model(hidden_nodes, epochs)
    plt.plot(range(epochs), loss_plot[hidden_nodes], label="nn: 4-%d-3" % hidden_nodes)

plt.xlabel('Iteration', fontsize=12)  
plt.ylabel('Loss', fontsize=12)  
plt.legend(fontsize=12) 







# Evaluate models on the test set
X = tf.placeholder(shape=[None, x_test.shape[1]], dtype=tf.float32)  
y = tf.placeholder(shape=[None, x_test.shape[1]], dtype=tf.float32)

for hidden_nodes in num_hidden_nodes:

    # Forward propagation
    W1 = tf.Variable(weights1[hidden_nodes])
    W2 = tf.Variable(weights2[hidden_nodes])
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Calculate the predicted outputs
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        x_est_np = sess.run(y_est, feed_dict={X: Xtest, y: ytest})

    # Calculate the prediction accuracy
    correct = [estimate.argmax(axis=0) == target.argmax(axis=0) 
               for estimate, target in zip(y_est_np, ytest.as_matrix())]
    accuracy = 100 * sum(correct) / len(correct)
    print('Network architecture 4-%d-3, accuracy: %.2f%%' % (hidden_nodes, accuracy))

'''


batch_size = 100

for epoch in range(epochs):
    avg_cost = 0
    total_batch = int(x_train.shape[0] / batch_size)
    x_batches = np.array_split(x_train, total_batch)
    for i in range(total_batch):
        batch_x = x_batches[i]
        _, c = sess.run([train_op, loss], 
                        feed_dict={inputs: batch_x, labels: batch_x})
        avg_cost += c / total_batch
    print("Epoch:", (epoch + 1), "cost =", "{:.9f}".format(avg_cost))
print(sess.run(train_op, feed_dict = {inputs: x_test[:100,], labels: x_test[:100]}))

'''


print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
sess.close()



if sess.run(weights_decoder) != sess.run(weights_decoder):
    raise ValueError("Check Weights")

