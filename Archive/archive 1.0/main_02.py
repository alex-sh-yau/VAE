# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 11:49:14 2018

@author: yaua
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.contrib.slim import fully_connected as fc

#import seaborn as sns
#from math import floor, ceil
#from pylab import rcParams
##fix nan:
##from numpy import *
#import math
#from sklearn.model_selection import train_test_split
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
# 
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


### Building NN:

tf.reset_default_graph()

# inputs fed through the NN
inputs = tf.placeholder(shape = [None, x_train.shape[1]], dtype = tf.float32)
# inputs that are not fed through, used to calculate loss
labels = tf.placeholder(shape = [None, x_train.shape[1]], dtype = tf.float32)

hidden_neurons = 1
lr = 0.0005
epochs = 100
batch_size = 100


#######################################################################

# weights of each layer
weights_encoder = tf.Variable(tf.random_normal([x_train.shape[1], hidden_neurons]))
weights_decoder = tf.Variable(tf.random_normal([hidden_neurons, x_train.shape[1]]))

# neural net graph of encoder
encoder = tf.matmul(inputs, weights_encoder)
# add bias?

## Variational component

'''

tfd = tf.contrib.distributions

hidden = tf.layers.dense(inputs, 3, tf.nn.relu)
mean = tf.layers.dense(hidden, hidden_neurons, None)
dist = tfd.MultivariateNormalDiag(mean, tf.ones_like(mean))

'''
###
'''

f1 = fc(inputs, 3, activation_fn=tf.nn.elu)
z_mean = fc(f1, hidden_neurons)
z_log_stddev = fc(f1, hidden_neurons)

'''

z_mean, z_var = tf.nn.moments(encoder, axes=[1])
z_mean = tf.reshape(z_mean, shape=(-1,1))
# z_var = tf.exp(z_log_stddev)
z_var = tf.reshape(z_var, shape=(-1,1))
z_stddev = tf.sqrt(z_var)

#samples = tf.random_normal([batch_size, hidden_neurons],0,1,dtype=tf.float32)
samples = tf.random_normal(shape=tf.shape(z_stddev),
                               mean=0, stddev=1, dtype=tf.float32)
sampled_z = z_mean + (z_stddev * samples)

decoder = tf.matmul(sampled_z, weights_decoder)

# network loss, KL divergence and total loss
# loss = tf.reduce_mean(tf.square(decoder - labels))
loss = -tf.reduce_sum(labels * tf.log(1e-8 + decoder) + (1-labels) * tf.log(1e-8 + 1 - decoder),1)
latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - [1,1])
cost = tf.reduce_mean(loss + latent_loss)

# optimizer function
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

#######################################################################

'''

batch_x = batch_x[:-1,]

sess.run(z_mean, feed_dict={inputs: batch_x}).shape
sess.run(z_stddev, feed_dict={inputs: batch_x}).shape
sess.run(samples, feed_dict={inputs: batch_x}).shape

(sess.run(z_stddev, feed_dict={inputs: batch_x}).reshape(-1,1) * sess.run(samples, feed_dict={inputs: batch_x})).shape

sess.run(sampled_z, feed_dict={inputs: batch_x}).shape
sess.run(weights_decoder).shape

print(sess.run(z_mean, feed_dict={inputs: batch_x}))

'''

# initialize & run session
sess = tf.Session()
sess.run(tf.global_variables_initializer())



# training + errors
for epoch in range(epochs):
    avg_cost = 0
    total_batch = int(x_train.shape[0] / batch_size)
    x_batches = np.array_split(x_train, total_batch)
    for i in range(total_batch):
        batch_x = x_batches[i]
        _, c = sess.run([train_op, cost], 
                        feed_dict={inputs: batch_x, labels: batch_x})
        avg_cost += c / total_batch
#   if sess.run(weights_decoder) != sess.run(weights_decoder):
#       raise ValueError("Check Weights")
    print("Epoch:", (epoch + 1), "cost =", "{:.9f}".format(avg_cost))
print(sess.run(train_op, feed_dict = {inputs: x_test[:100,], labels: x_test[:100]}))

# accuracy + predictions
correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(decoder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
x_out = tf.argmax(decoder,1)
with sess.as_default():
    print("Accuracy:", accuracy.eval({inputs: x_test, labels: x_test}))
    print("predictions", x_out.eval(feed_dict={inputs: x_test}))
    plt.plot(x_out.eval(feed_dict={inputs: x_test[:100, ]}))
    
# visualizing output
decoded_series = sess.run(decoder, {inputs: x_test[:100, ]})

plt.plot(x_test[:100, 0], label = "Actual")
plt.plot(decoded_series[:,0], label = "Decoder")
plt.legend(loc = 1)
plt.show()


merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./graphs', sess.graph)
print(sess.run(decoder, feed_dict = {inputs: x_test}))
writer.close()


# Terminal:
# cd OneDrive - OPTrust/Variational Autoencoder Project
# tensorboard --logdir="./graphs" --port 6006 
