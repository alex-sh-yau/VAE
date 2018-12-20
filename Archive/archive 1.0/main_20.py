# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 11:39:23 2018

@author: yaua
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

class VariationalAutoencoder():
    
    def __init__(self, train_inputs):
        self.lr = 0.005
        self.n_hidden = 8
        self.n_z = 1
        self.epochs = 80
        self.batchsize = 300

        self.inputs = tf.placeholder(shape = [None, train_inputs.shape[1]], dtype = tf.float32)
        self.labels = tf.placeholder(shape = [None, train_inputs.shape[1]], dtype = tf.float32)
        z_mean, z_stddev = self.encoder(self.inputs)
        samples = tf.random_normal(shape=tf.shape(z_stddev),mean=0,stddev=1,dtype=tf.float32)
        sampled_z = z_mean + (z_stddev * samples)
        
#        self.decoded_data = tf.get_variable("output", initializer=tf.zeros(shape=tf.shape(sampled_z.shape)))
        self.decoded_data = self.decoder(sampled_z)

        self.decoded_loss = tf.reduce_mean(tf.square(self.decoded_data - self.labels))
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - 
                                               tf.log(tf.square(z_stddev)) - [1,1])
        self.cost = tf.reduce_mean(self.decoded_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    # encoder
    def encoder(self, inputs):
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            h1 = tf.layers.dense(inputs, self.n_hidden, tf.nn.relu)
            w_mean = tf.layers.dense(h1, self.n_z, tf.nn.relu)
            w_stddev = tf.layers.dense(h1, self.n_z, tf.nn.softplus)
        return w_mean, w_stddev

    # decoder
    def decoder(self, z):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):         
            h1 = tf.layers.dense(z, self.n_hidden, tf.nn.relu)
            w_mean = tf.layers.dense(h1, self.inputs.shape[1], tf.nn.relu)
            w_stddev = tf.layers.dense(h1, self.inputs.shape[1], tf.nn.softplus)
            w_samples = tf.random_normal(shape=tf.shape(w_stddev), mean=0, stddev=1, dtype=tf.float32)
            h2 = w_mean + (w_stddev * w_samples)
        return h2

    # train model
    def train(self, train_inputs):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                avg_cost = 0
                np.random.shuffle(train_inputs)
                total_batch = int(train_inputs.shape[0] / self.batchsize)
                x_batches = np.array_split(train_inputs, total_batch)
                for i in range(total_batch):
                    batch_x = x_batches[i]
                    np.random.shuffle(batch_x)
                    _, c, dec_loss, lat_loss = sess.run([self.optimizer, self.cost, self.decoded_loss, self.latent_loss], 
                                                        feed_dict={self.inputs: batch_x, self.labels: batch_x})
                    avg_cost += c / total_batch
                print("Epoch:", (epoch + 1), "| Cost =", "{:.9f}".format(avg_cost), 
                      "| Generative loss =", "{:.9f}".format(dec_loss), 
                      "| Latent loss =", "{:.9f}".format(lat_loss))

    # %returns to prices
    def chain_returns(self, n_out, x_test, x_test_price):
        returns = np.zeros((n_out, x_test.shape[0], x_test.shape[1]))
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for k in range(n_out):
                returns[k,:,:] = sess.run(self.decoded_data, feed_dict={self.inputs: x_test})
            
            chained_returns = np.zeros((n_out, x_test.shape[0], x_test.shape[1]))
            for k in range(n_out):
                for j in range(chained_returns.shape[2]):
                    for i in range(chained_returns.shape[1]):
                        if i == 0:
                            chained_returns[k,i,j] = x_test_price[0,j]
                        else:
                            chained_returns[k,i,j] = returns[k,i-1,j]
#                            chained_returns[k,i,j] = chained_returns[k,i-1,j] * (1 + returns[k,i-1,j])

            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('./graphs', sess.graph)
            print(sess.run(self.decoded_data, feed_dict = {self.inputs: x_test}))
            writer.close()
        return chained_returns




#def chain_returns(n_out, x_test, x_test_price):
#    returns = np.zeros((n_out, x_test.shape[0], x_test.shape[1]))
#
#    with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        for k in range(n_out):
#            returns[k,:,:] = sess.run(model.decoded_data, feed_dict={x_test})
#        
#    chained_returns = np.zeros((n_out, x_test.shape[0], x_test.shape[1]))
#    for k in range(n_out):
#        for j in range(chained_returns.shape[2]):
#            for i in range(chained_returns.shape[1]):
#                if i == 0:
#                    chained_returns[k,i,j] = x_test_price[0,j]
#                else:
#                    chained_returns[k,i,j] = chained_returns[k,i-1,j] * (1 + returns[k,i-1,j])
#    return chained_returns
    
# inputs
def data(df):
    # Find all index rows that contain NaN 
    null_rows = pd.isnull(df).any(axis=1).nonzero()[0]
    x_dataset = dataframe.iloc[(null_rows[-1]+1):dataframe.shape[0],1:dataframe.shape[1]]
    
    x_price = x_dataset.values
    x_price = x_price.astype('float32')
    x_diff = x_dataset.pct_change()
    x_diff = x_diff.iloc[1:,]
    x_r = x_diff.values
    x_r = x_r.astype('float32')

    x_train_size = int(len(x_r) * 0.7)
    x_train, x_test = x_r[0:x_train_size,:], x_r[x_train_size:len(x_r),:]
    x_train_price, x_test_price = x_price[0:x_train_size,:], x_price[x_train_size:len(x_r),:]

    return x_train, x_test, x_train_price, x_test_price    

dataframe = pd.read_csv('AlexData.csv')  # bigger data set (23 inputs)
x_train, x_test, x_train_price, x_test_price = data(dataframe)

model = VariationalAutoencoder(x_train)
model.train(x_train)

n_out = 9
y = np.zeros((n_out, x_test.shape[0], x_test.shape[1]))
y = model.chain_returns(n_out, x_test, x_test_price)



asset_index = 0
plt.plot(y[:,:,asset_index].T, color = 'lightgrey')
plt.plot(x_test_price[:,asset_index])
#
#plt.plot(y[:,:,7].T, color = 'lightgrey')
#plt.plot(y[:,:,9].T, color = 'orange')