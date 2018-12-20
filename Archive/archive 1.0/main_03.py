# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 08:41:44 2018

@author: yaua
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

dataframe = pd.read_csv('AlexData1.csv')

#12/5/1988 onward (exclude all dates with NaN)
x_dataset = dataframe.iloc[2329:10091,1:7]

x_diff = x_dataset.pct_change()
x_diff = x_diff.iloc[1:,]

x = x_diff.values
x = x.astype('float32')

x_train_size = int(len(x) * 0.7)
x_test_size = len(x) - x_train_size
x_train, x_test = x[0:x_train_size,:], x[x_train_size:len(x),:]


### Building NN:

hidden_neurons = 1
lr = 0.005
epochs = 1000
batch_size = 100


tf.reset_default_graph()

inputs = tf.placeholder(shape = [None, x_train.shape[1]], dtype = tf.float32)
labels = tf.placeholder(shape = [None, x_train.shape[1]], dtype = tf.float32)

w_encoder = tf.Variable(tf.random_normal([x_train.shape[1], hidden_neurons]))
w_decoder = tf.Variable(tf.random_normal([hidden_neurons, x_train.shape[1]]))

#encoder = tf.matmul(inputs, w_encoder)

z_hidden = tf.layers.dense(inputs, 3, tf.nn.relu)
encoder = tf.layers.dense(z_hidden, 1, tf.nn.relu)
z_mean = tf.layers.dense(encoder, hidden_neurons)
z_stddev = tf.layers.dense(encoder, hidden_neurons, tf.nn.sigmoid) #sigmoid relu or softplus??

samples = tf.random_normal(shape=tf.shape(z_stddev),
                               mean=0, stddev=1, dtype=tf.float32)
sampled_z = z_mean + (z_stddev * samples)


'''
x_hidden = tf.layers.dense(sampled_z, 3)
x_mean = tf.layers.dense(x_hidden, 6)
x_stddev = tf.layers.dense(x_hidden, 6, tf.nn.softplus)

x_samples = tf.random_normal(shape=tf.shape(x_stddev),
                               mean=0, stddev=1, dtype=tf.float32)
sampled_x = x_mean + (x_stddev * x_samples)
'''


decoder = tf.matmul(sampled_z, w_decoder)

loss = tf.reduce_mean(tf.square(decoder - labels))
#loss = -tf.reduce_sum(labels * tf.log(1e-8 + decoder) + (1-labels) * tf.log(1e-8 + 1 - decoder),1)
latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - [1,1])
cost = tf.reduce_mean(loss + tf.reshape(latent_loss, shape=(-1,)))

train_op = tf.train.AdamOptimizer(lr).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training + Errors
for epoch in range(epochs):
    avg_cost = 0
    total_batch = int(x_train.shape[0] / batch_size)
    x_batches = np.array_split(x_train, total_batch)
    for i in range(total_batch):
        batch_x = x_batches[i]
        _, c = sess.run([train_op, cost], 
                        feed_dict={inputs: batch_x, labels: batch_x})
        avg_cost += c / total_batch
    print("Epoch:", (epoch + 1), "cost =", "{:.9f}".format(avg_cost))

# Accuracy + Predictions
correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(decoder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
x_out = tf.argmax(decoder,1)
with sess.as_default():
    print("Accuracy:", accuracy.eval({inputs: x_test, labels: x_test}))
    print("predictions", x_out.eval(feed_dict={inputs: x_test}))
#    plt.plot(x_out.eval(feed_dict={inputs: x_test[:100, ]}))
    
# Visualizing output
decoded_series = sess.run(decoder, {inputs: x_test[:100, ]})

plt.plot(x_test[:100, 0], label = "Actual")
plt.plot(decoded_series[:,0], label = "Decoder")
plt.legend(loc = 1)
plt.show()



'''

print(sess.run(z_mean, feed_dict={inputs: batch_x}))


merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./graphs', sess.graph)
print(sess.run(decoder, feed_dict = {inputs: x_test}))
writer.close()


# Terminal:
# cd OneDrive - OPTrust/Variational Autoencoder Project
# tensorboard --logdir="./graphs" --port 6006 

'''