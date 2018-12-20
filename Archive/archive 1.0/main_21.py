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
    
    def __init__(self, sess, n_features, lr, n_hidden):
        self.sess = sess
        self._n_features = n_features
        self._lr = lr
        self._n_hidden = n_hidden
        self._n_z = 1
                
        self.inputs = tf.placeholder(shape = [None, self._n_features], dtype = tf.float32)
        self.labels = tf.placeholder(shape = [None, self._n_features], dtype = tf.float32)
        
        # encoder
        self.z_hidden_weights = tf.Variable(tf.random_normal([self._n_features, self._n_hidden], 
                                                        stddev = tf.sqrt(2/(self._n_features + self._n_hidden))))
        self.z_hidden = tf.nn.relu(tf.matmul(self.inputs, self.z_hidden_weights))
        
        self.z_hidden_mean = tf.Variable(tf.random_normal([self._n_hidden, self._n_z], 
                                                        stddev = tf.sqrt(2/(self._n_hidden + self._n_z))))        
        self.z_hidden_stddev = tf.Variable(tf.random_normal([self._n_hidden, self._n_z], 
                                                        stddev = tf.sqrt(2/(self._n_hidden + self._n_z))))
        self.z_mean = tf.nn.relu(tf.matmul(self.z_hidden, self.z_hidden_mean))
        self.z_stddev = tf.nn.softplus(tf.matmul(self.z_hidden, self.z_hidden_stddev))
        
        self.z_samples = tf.random_normal(shape = tf.shape(self.z_stddev), 
                                                    stddev = tf.sqrt(2/(self._n_hidden + self._n_z)),
                                                    dtype = tf.float32)
        self.sampled_z = self.z_mean + (self.z_stddev * self.z_samples)
        
        # decoder
        self.x_hidden_weights = tf.Variable(tf.random_normal([self._n_z, self._n_hidden], 
                                                        stddev = tf.sqrt(2/(self._n_z + self._n_hidden))))
        self.x_hidden = tf.nn.relu(tf.matmul(self.sampled_z, self.x_hidden_weights))
        
        self.decoder_layer = tf.Variable(tf.random_normal([self._n_hidden, self._n_features], 
                                                        stddev = tf.sqrt(2/(self._n_hidden + self._n_features))))        
#        self.x_hidden_stddev = tf.Variable(tf.random_normal([self._n_hidden, self._n_features], 
#                                                        stddev = tf.sqrt(2/(self._n_hidden + self._n_features))))
        self.decoded_data = tf.matmul(self.x_hidden, self.decoder_layer)
#        self.x_stddev = tf.nn.softplus(tf.matmul(self.x_hidden, self.x_hidden_stddev))
#        
#        self.x_samples = tf.random_normal(shape = tf.shape(self.x_stddev), mean = 0,
#                                                    stddev = tf.sqrt(2/(self._n_hidden + self._n_features)),
#                                                    dtype = tf.float32)
#        
#        # output, parameters
#        self.decoded_data = self.x_mean + (self.x_stddev * self.x_samples)

        self.decoded_loss = tf.reduce_mean(tf.square(self.decoded_data - self.labels))
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) + tf.square(self.z_stddev) - 
                                               tf.log(tf.square(self.z_stddev)) - 1)
        self.cost = tf.reduce_mean(self.decoded_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(self._lr).minimize(self.cost)

    # train model
    def train(self, train_inputs, epochs, batchsize):
        for epoch in range(epochs):
            avg_cost = 0
            np.random.shuffle(train_inputs)
            total_batch = int(train_inputs.shape[0] / batchsize)
            x_batches = np.array_split(train_inputs, total_batch)
            for i in range(total_batch):
                batch_x = x_batches[i]
#                np.random.shuffle(batch_x)
                _, c, dec_loss, lat_loss = self.sess.run([self.optimizer, self.cost, self.decoded_loss, self.latent_loss], 
                                                    feed_dict={self.inputs: batch_x, self.labels: batch_x})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "| Cost =", "{:.9f}".format(avg_cost), 
                  "| Generative loss =", "{:.9f}".format(dec_loss), 
                  "| Latent loss =", "{:.9f}".format(lat_loss))
            
    def returns_test(self, i):
        d = np.zeros((i.shape[0], i.shape[1]))
        d = self.sess.run(self.decoded_data, {self.inputs: i})
        return d

    # %returns to prices
    def chain_returns(self, n_out, x_test, x_test_price):
        returns = np.zeros((n_out, x_test.shape[0], x_test.shape[1]))
        
        for k in range(n_out):
            returns[k,:,:] = self.sess.run(self.decoded_data, feed_dict={self.inputs: x_test})
        
        chained_returns = np.zeros((n_out, x_test.shape[0], x_test.shape[1]))
        for k in range(n_out):
            for j in range(chained_returns.shape[2]):
                for i in range(chained_returns.shape[1]):
                    if i == 0:
                        chained_returns[k,i,j] = x_test_price[0,j]
                    else:
#                        chained_returns[k,i,j] = returns[k,i-1,j]
                        chained_returns[k,i,j] = chained_returns[k,i-1,j] * (1 + returns[k,i-1,j])

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./graphs', self.sess.graph)
        print(self.sess.run(self.decoded_data, feed_dict = {self.inputs: x_test}))
        writer.close()
        return chained_returns

# inputs
def data(df):
    # Find all index rows that contain NaN 
    null_rows = pd.isnull(df).any(axis=1).nonzero()[0]
    X_dataset = dataframe.iloc[(null_rows[-1]+1):dataframe.shape[0],1:dataframe.shape[1]]
    
    X_price = X_dataset.values
    X_price = X_price.astype('float32')
    X_diff = X_dataset.pct_change()
    X_diff = X_diff.iloc[1:,]
    X_r = X_diff.values
    X_r = X_r.astype('float32')

    X_train_size = int(len(X_r) * 0.7)
    X_train, X_test = X_r[0:X_train_size,:], X_r[X_train_size:len(X_r),:]
    X_train_price, X_test_price = X_price[0:X_train_size,:], X_price[X_train_size:len(X_r),:]

    return X_train, X_test, X_train_price, X_test_price    

dataframe = pd.read_csv('AlexData.csv')  # bigger data set (23 inputs)
X_train, X_test, X_train_price, X_test_price = data(dataframe)

FEATURE_SIZE = X_train.shape[1]
LEARNING_RATE = 0.005
NEURONS = 8
EPOCHS = 100
BATCH_SIZE = 250
N_OUTPUTS = 9

tf.reset_default_graph()
sess = tf.Session()


model = VariationalAutoencoder(sess, FEATURE_SIZE, LEARNING_RATE, NEURONS)

sess.run(tf.global_variables_initializer())

model.train(X_train, EPOCHS, BATCH_SIZE)

y = np.zeros((N_OUTPUTS, X_test.shape[0], X_test.shape[1]))
y = model.chain_returns(N_OUTPUTS, X_test, X_test_price)

decoded_series = np.zeros((100, X_test.shape[1]))
decoded_series = model.returns_test(X_test[:100,:])


plt.figure(0)
plt.plot(X_test[:100, 0], label = "Actual")
plt.plot(decoded_series[:,0], label = "Decoder")
plt.legend(loc = 1)
plt.show()

plt.figure(1)
asset_index = 0
plt.plot(y[:,:,asset_index].T, color = 'lightgrey')
plt.plot(X_test_price[:,asset_index])
plt.show()
#
#plt.plot(y[:,:,7].T, color = 'lightgrey')
#plt.plot(y[:,:,9].T, color = 'orange')