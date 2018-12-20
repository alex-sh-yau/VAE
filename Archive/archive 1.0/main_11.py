# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:37:59 2018

@author: yaua
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''
dataframe1 = pd.read_csv('AlexData1.csv') #smaller data set (6 inputs)

#12/6/1988 onward (exclude all dates with NaN)
x_dataset = dataframe.iloc[2329:10091,1:7]
'''

dataframe = pd.read_csv('AlexData.csv') #bigger data set (23 inputs)

#12/31/1992 onward (exclude all dates with NaN)
x_dataset = dataframe.iloc[3391:dataframe.shape[0],1:dataframe.shape[1]]

x_diff = x_dataset.pct_change()
x_diff = x_diff.iloc[1:,]
x = x_diff.values
x = x.astype('float32')

x_train_size = int(len(x) * 0.7)
x_test_size = len(x) - x_train_size
x_train, x_test = x[0:x_train_size,:], x[x_train_size:len(x),:]

# Actual price train/test set for conversions and checking
x_price_set = x_dataset.values
x_price_set = x_price_set.astype('float32')
x_train_price, x_test_price = x_price_set[0:x_train_size,:], x_price_set[x_train_size:len(x),:]

### Building NN:
hidden_neurons = 1
lr = 0.005
epochs = 100
batch_size = 300

tf.reset_default_graph()

inputs = tf.placeholder(shape = [None, x_train.shape[1]], dtype = tf.float32)
labels = tf.placeholder(shape = [None, x_train.shape[1]], dtype = tf.float32)

# Encoder layer
z_hidden = tf.layers.dense(inputs, 8, tf.nn.relu)
z_mean = tf.layers.dense(z_hidden, hidden_neurons, tf.nn.relu)
z_stddev = tf.layers.dense(z_hidden, hidden_neurons, tf.nn.softplus)

samples = tf.random_normal(shape=tf.shape(z_stddev),
                               mean=0, stddev=1, dtype=tf.float32)
sampled_z = z_mean + (z_stddev * samples)

# Decoder layer
x_hidden = tf.layers.dense(sampled_z, 8, tf.nn.relu)
x_mean = tf.layers.dense(x_hidden, x_train.shape[1], tf.nn.relu)
x_stddev = tf.layers.dense(x_hidden, x_train.shape[1], tf.nn.softplus)

x_samples = tf.random_normal(shape=tf.shape(x_stddev),
                               mean=0, stddev=1, dtype=tf.float32)
decoder = x_mean + (x_stddev * x_samples)

# Cost, Optimizer
loss = tf.reduce_mean(tf.square(decoder - labels))
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

'''
# Accuracy + Predictions
correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(decoder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
x_out = tf.argmax(decoder,1)
with sess.as_default():
    print("Accuracy:", accuracy.eval({inputs: x_test, labels: x_test}))
    print("predictions", x_out.eval(feed_dict={inputs: x_test}))
#    plt.plot(x_out.eval(feed_dict={inputs: x_test[:100, ]}))
'''

#### Visualized returns input vs decoded test-check
decoded_series = sess.run(decoder, {inputs: x_test[:100, ]})

plt.plot(x_test[:100, 0], label = "Actual")
plt.plot(decoded_series[:,0], label = "Decoded")
plt.legend(loc = 1)
plt.show()
##################################################

# Run this line to generate new set of decoded outputs
output_returns = sess.run(decoder, {inputs: x_test})

def chain_returns(returns):
    chained_returns = np.zeros([returns.shape[0], returns.shape[1]])
    for j in range(chained_returns.shape[1]):
        for i in range(chained_returns.shape[0]):
            if i == 0:
                chained_returns[i,j] = x_test_price[0,j]
            else:
                chained_returns[i,j] = chained_returns[i-1,j] * (1 + returns[i-1,j])
    return chained_returns

y = chain_returns(output_returns)



#### Visualized price test-check
datarow = 9
plt.plot(x_test_price[:,datarow], label = "Actual")
plt.plot(y[:,datarow], label = "Decoded")
plt.legend(loc = 1)
plt.title(dataframe.columns[datarow+1])
plt.show()
##########################

names = list(dataframe.columns.values)
names = pd.DataFrame(names)
dates = dataframe.iloc[8082:dataframe.shape[0],0] #Truncated dates of test set
dates = pd.DataFrame(dates)
dates = dates.reset_index(drop=True)

df = pd.DataFrame(y)
df = dates.join(df)
df.columns = names
df.to_csv("decoded_output.csv")

'''
def chain_returns(returns):
    chained_returns = np.zeros(returns.shape[0]+1)
    for i in range(chained_returns.shape[0]):
        if i == 0:
            chained_returns[i] = 1
        else:
            chained_returns[i] = chained_returns[i-1] * (1+returns[i-1])
    return chained_returns-1
'''

#output = x_dataset.iloc[0:1,:] * (1 + tf.transpose(decoded_series).eval())

#print(sess.run(z_mean, feed_dict={inputs: batch_x}))