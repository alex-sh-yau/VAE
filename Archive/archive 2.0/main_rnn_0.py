# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:37:59 2018

@author: yaua
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import charts_1_0 as charts
import tensorflow.contrib.rnn as rnn

import seaborn as sns

num_periods = 20

def data_preprocess(df):
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
    
    #Reshape for RNN 
    X_train = np.reshape(X_train[:(len(X_train)-(len(X_train)%num_periods))], (-1,num_periods,9))
    X_test = np.reshape(X_test[:(len(X_test)-(len(X_test)%num_periods))], (-1,num_periods,9))
    X_train_price = np.reshape(X_train_price[:(len(X_train_price)-(len(X_train_price)%num_periods))], (-1,9))
    X_test_price = np.reshape(X_test_price[:(len(X_test_price)-(len(X_test_price)%num_periods))], (-1,9))
    return X_train, X_test, X_train_price, X_test_price

dataframe = pd.read_csv('AlexCurr.csv')
X_tr, X_te, X_tr_p, X_te_p = data_preprocess(dataframe)

tf.reset_default_graph()

num_periods = 20 # batch splitting (size per batch)
n_features = 9 ### X_tr.shape[1]
learning_rate = 0.0015
hidden_neurons = 2

lstm_hidden = 50

inputs = tf.placeholder(tf.float32, [None, num_periods, n_features])
labels = tf.placeholder(tf.float32, [None, num_periods, n_features])

# Encoder LSTM
enc_lstm_cell = rnn.BasicLSTMCell(lstm_hidden, forget_bias=1.0, activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
enc_lstm_output, states = tf.nn.dynamic_rnn(enc_lstm_cell, inputs, dtype=tf.float32)
enc_stacked_lstm_output = tf.reshape(enc_lstm_output, [-1,lstm_hidden])
enc_stacked_outputs = tf.layers.dense(enc_stacked_lstm_output, n_features)

# Encoder latent distribution
z_mean = tf.layers.dense(enc_stacked_outputs, hidden_neurons)
z_stddev = tf.layers.dense(enc_stacked_outputs, hidden_neurons, tf.nn.softplus)
samples = tf.random_normal(shape=tf.shape(z_stddev), mean=0, stddev=1, dtype=tf.float32)
sampled_z = z_mean + (z_stddev * samples)
z_output = tf.layers.dense(sampled_z, n_features)
enc_outputs = tf.reshape(z_output,[-1,num_periods,n_features])

# Decoder LSTM
dec_lstm_cell = rnn.BasicLSTMCell(lstm_hidden, forget_bias=1.0, activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
dec_rnn_output, states = tf.nn.dynamic_rnn(dec_lstm_cell, enc_outputs, dtype=tf.float32)
dec_stacked_rnn_output = tf.reshape(dec_rnn_output, [-1,lstm_hidden])
dec_stacked_outputs = tf.layers.dense(dec_stacked_rnn_output, n_features)

# Decoded outputs
outputs = tf.reshape(dec_stacked_outputs,[-1,num_periods,n_features])

loss = tf.reduce_sum(tf.square(outputs-labels))
latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - [1,1])
total_loss = tf.reduce_mean(loss + tf.reshape(latent_loss, shape=(-1,)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(total_loss)

init = tf.global_variables_initializer()

epochs = 250

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={inputs: X_tr, labels: X_tr})
        if ep % 10 == 0:
            c, dec_loss, lat_loss = sess.run([total_loss, loss, latent_loss], feed_dict={inputs: X_tr, labels: X_tr})
            print("Epoch:", (ep + 1), "| Total loss =", "{:.9f}".format(c), 
                  "| Generative loss =", "{:.9f}".format(dec_loss), 
                  "| Latent loss =", "{:.9f}".format(lat_loss))    
    y_pred = np.reshape(sess.run(outputs, feed_dict={inputs: X_te}), (-1,9))
    print(y_pred)
    
        
    output_var = 9  # Number of decoded outputs to generate
    output_returns = np.zeros((output_var, X_te.shape[0] * X_te.shape[1], X_te.shape[2]))
    
    for k in range(output_var):
        output_returns[k,:,:] = np.reshape(sess.run(outputs, feed_dict={inputs: X_te}), (-1,9))
    
    def chain_returns(returns):
        chained_returns = np.zeros((output_var, returns.shape[1], returns.shape[2]))
        for k in range(output_var):
            for j in range(chained_returns.shape[2]):
                for i in range(chained_returns.shape[1]):
                    if i == 0:
                        chained_returns[k,i,j] = X_te_p[0,j]
                    else:
                        chained_returns[k,i,j] = chained_returns[k,i-1,j] * (1 + returns[k,i-1,j])
        return chained_returns
    
    y = np.zeros((output_var, X_te.shape[0], X_te.shape[1]))
    y = chain_returns(output_returns)

def price_charts(df):
    f, axes = plt.subplots(3,3, figsize=(15,9))
    axes = axes.ravel()
    for k in range (df.shape[1]-1):
        axes[k].plot(y[:,:,k].T, color = 'lightgrey')
        axes[k].plot(X_te_p[:,k])
        axes[k].set_title("{}".format(df.columns[k+1]))
    f.subplots_adjust(hspace=0.5)
    plt.legend(loc = 1)
    f.suptitle("Price charts of input vs all outputs", fontsize=14)
    return f

price_charts(dataframe).show()
