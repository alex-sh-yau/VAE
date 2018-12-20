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
        
        self.z_hidden_mean = tf.Variable(tf.zeros([self._n_hidden, self._n_z]))
        self.z_hidden_stddev = tf.Variable(tf.random_normal([self._n_hidden, self._n_z], 
                                                            stddev = tf.sqrt(2/(self._n_hidden + self._n_z))))
        self.z_mean = tf.nn.relu(tf.matmul(self.z_hidden, self.z_hidden_mean))
        self.z_stddev = tf.nn.softplus(tf.matmul(self.z_hidden, self.z_hidden_stddev))
        
        self.z_samples = tf.random_normal(shape = tf.shape(self.z_stddev), 
                                          stddev = tf.sqrt(2/(self._n_hidden + self._n_z)),
                                          dtype = tf.float32)
        self.sampled_z = self.z_mean + (self.z_stddev * self.z_samples)

        # decoder       
        self.decoder_layer = tf.Variable(tf.random_normal([self._n_z, self._n_features], 
                                                        stddev = tf.sqrt(2/(self._n_z + self._n_features))))
        self.decoded_data = tf.matmul(self.sampled_z, self.decoder_layer)

        self.decoded_loss = tf.reduce_mean(tf.square(self.decoded_data - self.labels))
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) + tf.square(self.z_stddev) - 
                                               tf.log(tf.square(self.z_stddev)) - 1)
        self.cost = tf.reduce_mean(self.decoded_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(self._lr).minimize(self.cost)

# train model
def train(model, sess, train_inputs, epochs, batchsize):
    for epoch in range(epochs):
        avg_cost = 0
        np.random.shuffle(train_inputs)
        total_batch = int(train_inputs.shape[0] / batchsize)
        x_batches = np.array_split(train_inputs, total_batch)
        for i in range(total_batch):
            batch_x = x_batches[i]
#                np.random.shuffle(batch_x)
            _, c, dec_loss, lat_loss = sess.run([model.optimizer, model.cost, model.decoded_loss, model.latent_loss], 
                                                feed_dict={model.inputs: batch_x, model.labels: batch_x})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "| Cost =", "{:.9f}".format(avg_cost), 
              "| Generative loss =", "{:.9f}".format(dec_loss), 
              "| Latent loss =", "{:.9f}".format(lat_loss))
            
def returns_test(model, sess, i):
    d = np.zeros((i.shape[0], i.shape[1]))
    d = sess.run(model.decoded_data, {model.inputs: i})
    return d

    # %returns to prices
def chain_returns(model, sess, n_out, x_test, x_test_price):
    returns = np.zeros((n_out, x_test.shape[0], x_test.shape[1]))
    
    for k in range(n_out):
        returns[k,:,:] = sess.run(model.decoded_data, feed_dict={model.inputs: x_test})
    
    chained_returns = np.zeros((n_out, x_test.shape[0], x_test.shape[1]))
    for k in range(n_out):
        for j in range(chained_returns.shape[2]):
            for i in range(chained_returns.shape[1]):
                if i == 0:
                    chained_returns[k,i,j] = x_test_price[0,j]
                else:
                    chained_returns[k,i,j] = chained_returns[k,i-1,j] * (1 + returns[k,i-1,j])

#        merged = tf.summary.merge_all()
#        writer = tf.summary.FileWriter('./graphs', self.sess.graph)
#        print(self.sess.run(self.decoded_data, feed_dict = {self.inputs: x_test}))
#        writer.close()
    return chained_returns, returns

# inputs
def data(df):
    # Find all index rows that contain NaN 
    null_rows = pd.isnull(df).any(axis=1).nonzero()[0]
    X_dataset = df.iloc[(null_rows[-1]+1):df.shape[0],1:df.shape[1]]
    
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

dataframe = pd.read_csv('AlexStock.csv')  # 9 inputs
X_train, X_test, X_train_price, X_test_price = data(dataframe)

FEATURE_SIZE = X_train.shape[1]
LEARNING_RATE = 0.4
NEURONS = 8
EPOCHS = 80
BATCH_SIZE = 200
N_OUTPUTS = FEATURE_SIZE

#tf.reset_default_graph()

g_1 = tf.Graph()
with g_1.as_default():
    
#    tf.set_random_seed(1)
    
    sess_1 = tf.Session()
    model = VariationalAutoencoder(sess_1, FEATURE_SIZE, LEARNING_RATE, NEURONS)
    sess_1.run(tf.global_variables_initializer())
    train(model, sess_1, X_train, EPOCHS, BATCH_SIZE)
    
    y = np.zeros((N_OUTPUTS, X_test.shape[0], X_test.shape[1]))
    y_r = np.zeros((N_OUTPUTS, X_test.shape[0], X_test.shape[1]))
    y, y_r = chain_returns(model, sess_1, N_OUTPUTS, X_test, X_test_price)
    
    decoded_series = np.zeros((100, X_test.shape[1]))
    decoded_series = returns_test(model, sess_1, X_test[:100,:])
    
    plt.figure(0)
    plt.plot(X_test[:100, 0], label = "Actual")
    plt.plot(decoded_series[:,0], label = "Decoder")
    plt.legend(loc = 1)
    plt.show()
    

    f, axes = plt.subplots(2,2, figsize=(15,9))
    axes[0,0].plot(y[:,:,0].T, color = 'lightgrey')
    axes[0,0].plot(X_test_price[:,0])
    axes[0,0].set_title("{}".format(dataframe.columns[1]))
    axes[0,1].plot(y[:,:,1].T, color = 'lightgrey')
    axes[0,1].plot(X_test_price[:,1])
    axes[0,1].set_title("{}".format(dataframe.columns[2]))
    axes[1,0].plot(y[:,:,4].T, color = 'lightgrey')
    axes[1,0].plot(X_test_price[:,4])
    axes[1,0].set_title("{}".format(dataframe.columns[5]))
    axes[1,1].plot(y[:,:,8].T, color = 'lightgrey')
    axes[1,1].plot(X_test_price[:,8])
    axes[1,1].set_title("{}".format(dataframe.columns[9]))
    plt.show()
    
    
    chart = 1
    plt.figure(2)
    for n in range (N_OUTPUTS):
        sns.kdeplot(y_r[n,:,chart], shade=True, color = 'lightgrey')
    sns.kdeplot(X_test[:,chart], shade=True, label = "Input")
    plt.legend(loc = 1)
    plt.title("Distributions of {}".format(dataframe.columns[chart+1]))
    plt.show()


#
#plt.plot(y[:,:,7].T, color = 'lightgrey')
#plt.plot(y[:,:,9].T, color = 'orange')