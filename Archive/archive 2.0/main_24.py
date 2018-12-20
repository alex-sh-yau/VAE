# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 09:26:37 2018

@author: dasilvb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import seaborn as sns
'''
dataframe1 = pd.read_csv('AlexData1.csv') #smaller data set (6 inputs)

#12/6/1988 onward (exclude all dates with NaN)
x_dataset = dataframe.iloc[2329:10091,1:7]
'''

dataframe = pd.read_csv('AlexData.csv')  # bigger data set (23 inputs)

# 12/31/1992 onward (exclude all dates with NaN)
x_dataset = dataframe.iloc[3391:dataframe.shape[0],1:dataframe.shape[1]]

# % Returns data - normalized and easier to work with
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

class VariationalAutoencoder():
    def __init__(self, n_features, hidden_neurons, lr):
        self.n_features = n_features
        self.hidden_neurons = hidden_neurons
        self.lr = lr
        
        self.inputs = tf.placeholder(shape = [None,self.n_features], dtype = tf.float32)
        self.labels = tf.placeholder(shape = [None, self.n_features], dtype = tf.float32)
        
        # Encoder layer
        self.z_hidden = tf.layers.dense(self.inputs, 8, tf.nn.relu)
        self.z_mean = tf.layers.dense(self.z_hidden, self.hidden_neurons, tf.nn.relu)
        self.z_stddev = tf.layers.dense(self.z_hidden, self.hidden_neurons, tf.nn.softplus)

        samples = tf.random_normal(shape=tf.shape(self.z_stddev), mean=0, stddev=1, dtype=tf.float32)
        self.sampled_z = self.z_mean + (self.z_stddev * samples)
        
        # Decoder layer
        self.x_hidden = tf.layers.dense(self.sampled_z, 8, tf.nn.relu)
        self.x_mean = tf.layers.dense(self.x_hidden, self.n_features, tf.nn.relu)
        self.x_stddev = tf.layers.dense(self.x_hidden, self.n_features, tf.nn.softplus)
        
        x_samples = tf.random_normal(shape=tf.shape(self.x_stddev), mean=0, stddev=1, dtype=tf.float32)
        self.decoder = self.x_mean + (self.x_stddev * x_samples)
        
        # Cost, Optimizer
        self.loss = tf.reduce_mean(tf.square(self.decoder - self.labels))
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) + tf.square(self.z_stddev) - tf.log(tf.square(self.z_stddev)) - [1,1])
        self.cost = tf.reduce_mean(self.loss + tf.reshape(self.latent_loss, shape=(-1,)))
        
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

# Building NN:
hidden_neurons = 1
lr = 0.005
epochs = 80
batch_size = 300

tf.reset_default_graph()
sess = tf.Session()

generative_model = VariationalAutoencoder(x_train.shape[1], hidden_neurons, lr)

sess.run(tf.global_variables_initializer())


# Training + Errors
for epoch in range(epochs):
    avg_cost = 0
    np.random.shuffle(x_train)
    total_batch = int(x_train.shape[0] / batch_size)
    x_batches = np.array_split(x_train, total_batch)
    for i in range(total_batch):
        batch_x = x_batches[i]
        np.random.shuffle(batch_x)
        _, c, dec_loss, lat_loss = sess.run([generative_model.train_op, generative_model.cost, generative_model.loss, generative_model.latent_loss], 
                                            feed_dict={generative_model.inputs: batch_x, generative_model.labels: batch_x})
        avg_cost += c / total_batch
    print("Epoch:", (epoch + 1), "| Cost =", "{:.9f}".format(avg_cost), 
                      "| Generative loss =", "{:.9f}".format(dec_loss), 
                      "| Latent loss =", "{:.9f}".format(lat_loss))


################ Output generation from Decoder tensor #######################
    
output_var = 9  # Number of decoded outputs to generate
output_returns = np.zeros((output_var, x_test.shape[0], x_test.shape[1]))

for k in range(output_var):
    output_returns[k,:,:] = sess.run(generative_model.decoder, {generative_model.inputs: x_test})

#### Input (returns) is the Decoder tensor processed into output_returns
def chain_returns(returns):
    chained_returns = np.zeros((output_var, returns.shape[1], returns.shape[2]))
    for k in range(output_var):
        for j in range(chained_returns.shape[2]):
            for i in range(chained_returns.shape[1]):
                if i == 0:
                    chained_returns[k,i,j] = x_test_price[0,j]
                else:
                    chained_returns[k,i,j] = chained_returns[k,i-1,j] * (1 + returns[k,i-1,j])
    return chained_returns

y = np.zeros((output_var, x_test.shape[0], x_test.shape[1]))
y = chain_returns(output_returns)

#asset_index = 0
#plt.plot(y[:,:,asset_index].T, color = 'lightgrey')
#plt.plot(x_test_price[:,asset_index])
#
#plt.plot(y[:,:,7].T, color = 'lightgrey')
#plt.plot(y[:,:,9].T, color = 'orange')


#### Visualized price test-check
datarow = 0  # Decoded output number (3rd dimension)
for n in range(output_var):
    plt.plot(y[n,:,datarow], color = 'lightgrey')
plt.plot(x_test_price[:,datarow])
plt.legend(loc = 1)
plt.title(dataframe.columns[datarow+1])
plt.show()
##########################


##### Dumping to csv file
names = list(dataframe.columns.values)
names = pd.DataFrame(names)
dates = dataframe.iloc[8082:dataframe.shape[0],0] #Truncated dates within test set
dates = pd.DataFrame(dates)
dates = dates.reset_index(drop=True)

df = pd.DataFrame(y[0,:,:]) 
df = dates.join(df)
df.columns = names
df.to_csv("decoded_output.csv")
################################
## Next step: get 3D dataframe to store all decoded output


####################### Correlation of outputs #############################

#datarow1 = 0  # Decoded output number (3rd dimension, output_var in total)
chart = 0  # Chart number (2nd dimension, input amount (23) in total)
df2 = pd.DataFrame(x_test_price)


###### SMA crossover 3m vs 1y of input
plt.figure(0)

SMA90_input = df2.rolling(90).mean()
SMA90_input = SMA90_input.values
SMA90_input = SMA90_input.astype('float32')
SMA360_input = df2.rolling(360).mean()
SMA360_input = SMA360_input.values
SMA360_input = SMA360_input.astype('float32')
plt.plot(SMA360_input[:,chart] - SMA90_input[:,chart], label = "Input")

###### SMA crossover 3m vs 1y of all outputs
for n in range(output_var):
    dfn = pd.DataFrame(y[n,:,:])
    SMA90_output = dfn.rolling(90).mean()
    SMA90_output = SMA90_output.values
    SMA90_output = SMA90_output.astype('float32')
    SMA360_output = dfn.rolling(360).mean()
    SMA360_output = SMA360_output.values
    SMA360_output = SMA360_output.astype('float32')
    plt.plot(SMA360_output[:,chart] - SMA90_output[:,chart], label = "Output {}".format(n+1))
plt.legend(loc = 1)
plt.title("SMA crossover 3m vs 1y of {}".format(dataframe.columns[chart+1]))
plt.show()


###### Std dev difference between outputs and input, averaged
plt.figure(1)

chart_var = 6
chart = 0

df2r = pd.DataFrame(x_test)
stddev_x = df2r.rolling(180).std()
stddev_x = stddev_x.values
stddev_x = stddev_x.astype('float32')
sigma_store = np.zeros((x_test.shape[0], x_test.shape[1]))
for n in range (output_var):
    dfn = pd.DataFrame(output_returns[n,:,:])
    stddev_out = dfn.rolling(180).std()
    stddev_out = stddev_out.values
    stddev_out = stddev_out.astype('float32')
    sigma_store = sigma_store + (stddev_out - stddev_x)
sigma_store = sigma_store / output_var
for m in range (chart_var):
    plt.plot(np.abs(sigma_store[:,m]), label = dataframe.columns[m+1])
plt.legend(loc = 1)
plt.title("Stddev difference avg of all outputs")
plt.show()
#######################


output_var = 9  # Number of decoded outputs to generate
chart = 1


######## All distributions of input and outputs in one plot
plt.figure(2)

sns.kdeplot(output_returns[0,:,chart], shade=True, label = "Output")
for n in range (output_var):
    sns.kdeplot(output_returns[n,:,chart], shade=True, label = "Output")
plt.legend(loc = 1)
plt.title("Distributions of {}".format(dataframe.columns[chart+1]))
plt.show()


####### Distributions of each input vs its output (multiplot)


sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(8, 3, figsize=(12, 9), sharex=True)
sns.despine(left=True)
for n in range (output_returns.shape[2]):
    sns.kdeplot(output_returns[1,:,n], shade=True, 
                label = "Output {}".format(dataframe.columns[n+1]), 
                ax=axes[n%8,n%3])
    sns.kdeplot(x_test[:,n], shade=True, label = "Input", ax=axes[n%8,n%3])
    ################## The order is weird but it works ###################
plt.legend(loc = 1)
#f.suptitle("Distributions of {}".format(dataframe.columns[chart+1]))
plt.show()



###### Rolling correlation of each output vs input
plt.figure(5)

df3 = pd.DataFrame(x_test_price)
for n in range (3): #output_var
    dfn = pd.DataFrame(y[n,:,:])
    corr_out = dfn.rolling(180).corr(df3)    
    corr_out = corr_out.values
    corr_out = corr_out.astype('float32')
    plt.plot(corr_out[:,chart], label = "Rolling correlation output {}".format(n+1))
plt.legend(loc = 1)
plt.title(dataframe.columns[chart+1])
plt.show()
#######################


#### Mean and std dev of input and outputs RETURNS ###########
mean_input_r = np.zeros(x_test.shape[1])
stddev_input_r = np.zeros(x_test.shape[1])

for m in range (x_test.shape[1]):
    for n in range (x_test.shape[0]):
        mean_input_r[m] = mean_input_r[m] + x_test[n,m]
mean_input_r = mean_input_r / x_test.shape[0]

for m in range (x_test.shape[1]):
    for n in range (x_test.shape[0]):
        stddev_input_r[m] = stddev_input_r[m] + np.sqrt(np.abs(mean_input_r[m] - x_test[n,m])/x_test.shape[0])
        
        
mean_output_r = np.zeros((output_var, x_test.shape[1]))
stddev_output_r = np.zeros((output_var, x_test.shape[1]))

for p in range (output_var):
    for m in range (x_test.shape[1]):
        for n in range (x_test.shape[0]):
            mean_output_r[p,m] = mean_output_r[p,m] + output_returns[p,n,m]
mean_output_r = mean_output_r / x_test.shape[0]
mean_output_avg_r = np.zeros(x_test.shape[1])
for k in range (x_test.shape[1]):
    for j in range (output_var):
        mean_output_avg_r[k] = mean_output_avg_r[k] + mean_output_r[j,k]
mean_output_avg_r = mean_output_avg_r / output_var

for p in range (output_var):
    for m in range (x_test.shape[1]):
        for n in range (x_test.shape[0]):
            stddev_output_r[p,m] = stddev_output_r[p,m] + np.sqrt(np.abs(mean_output_r[p,m] 
            - output_returns[p,n,m])/x_test.shape[0])
stddev_output_avg_r = np.zeros(x_test.shape[1])
for k in range (x_test.shape[1]):
    for j in range (output_var):
        stddev_output_avg_r[k] = stddev_output_avg_r[k] + stddev_output_r[j,k]
stddev_output_avg_r = stddev_output_avg_r / output_var
##########################################################

indexes = names.iloc[1:24]
indexes = pd.DataFrame(indexes)
comp = np.vstack((mean_output_avg_r, stddev_output_avg_r, mean_input_r, stddev_input_r))
comp = np.transpose(comp)
df_comp = pd.DataFrame(comp)

df_comp = np.column_stack((indexes, df_comp))
df_comp = pd.DataFrame(df_comp)
df_comp_columns = ['Asset', 'Output mean', 'Output stddev', 'Input mean', 'Input stddev']
df_comp.columns = df_comp_columns
