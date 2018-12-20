# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 10:20:26 2018

@author: yaua
"""

'''
###### Accuracy + Predictions
###### Goes right after NN training for n epochs as a check
correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(decoder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
x_out = tf.argmax(decoder,1)
with sess.as_default():
    print("Accuracy:", accuracy.eval({inputs: x_test, labels: x_test}))
    print("predictions", x_out.eval(feed_dict={inputs: x_test}))
#    plt.plot(x_out.eval(feed_dict={inputs: x_test[:100, ]}))


#### Visualized returns input vs decoded test-check
decoded_series = sess.run(decoder, {inputs: x_test[:100, ]})

plt.plot(x_test[:100, 0], label = "Actual")
plt.plot(decoded_series[:,0], label = "Decoded")
plt.legend(loc = 1)
plt.show()
##################################################

'''

'''

#### Output generation from Decoder tensor: multiplying with inputs
#### Input (returns) is the Decoder tensor

####### Not working #######
for k in range(output_var):
    output_returns[k,:,:] = sess.run(decoder, {inputs: x_test})

def chain_returns(returns):
    chained_returns = np.zeros(returns.shape[0]+1)
    for i in range(chained_returns.shape[0]):
        if i == 0:
            chained_returns[i] = 1
        else:
            chained_returns[i] = chained_returns[i-1] * (1+returns[i-1])
    return chained_returns-1
###########################
'''



'''
# SMA 3m vs 1y
df01 = pd.DataFrame(x_test_price)
rmean90_in = df01.rolling(90).mean()
rmean90_in = rmean90_in.values
rmean90_in = rmean90_in.astype('float32')
rmean360_in = df01.rolling(360).mean()
rmean360_in = rmean360_in.values
rmean360_in = rmean360_in.astype('float32')

df1 = pd.DataFrame(y[datarow1,:,:]) 
rmean90_out = df1.rolling(90).mean()
rmean90_out = rmean90_out.values
rmean90_out = rmean90_out.astype('float32')
rmean360_out = df1.rolling(360).mean()
rmean360_out = rmean360_out.values
rmean360_out = rmean360_out.astype('float32')

#plt.plot(y[datarow1,:,chart], label = "Output price")
plt.plot(rmean90_in[:,chart], label = "Input SMA90")
plt.plot(rmean360_in[:,chart], label = "Input SMA360")
plt.plot(rmean90_out[:,chart], label = "Output SMA90")
plt.plot(rmean360_out[:,chart], label = "Output SMA360")
plt.legend(loc = 1)
plt.title(dataframe.columns[chart+1])
plt.show()
################
'''

'''
###### SMA 3m for all outputs
plt.plot(SMA90_input[:,chart], label = "Input SMA90")
for n in range(output_var):
    dfn = pd.DataFrame(y[n,:,:])
    SMA_output = dfn.rolling(90).mean()
    SMA_output = SMA_output.values
    SMA_output = SMA_output.astype('float32')
    plt.plot(SMA_output[:,chart], label = "Decoded SMA90")
plt.legend(loc = 1)
plt.title(dataframe.columns[chart+1])
plt.show()
'''


'''
###### Visualized price std. dev. check
stddev_x = df2.rolling(180).std()
stddev_x = stddev_x.values
stddev_x = stddev_x.astype('float32')
plt.plot(stddev_x[:,chart], label = "Input stddev")
for n in range (output_var):
    dfn = pd.DataFrame(y[n,:,:])
    stddev_out = dfn.rolling(180).std()
    stddev_out = stddev_out.values
    stddev_out = stddev_out.astype('float32')
    plt.plot(stddev_out[:,chart], label = "Output stddev")
plt.legend(loc = 1)
plt.title(dataframe.columns[chart+1])
plt.show()
#######################
'''

#
#
#
#
#
#
#
#
#

'''
###### Std dev difference between outputs and input, averaged
plt.figure(1)

chart = 0

stddev_x = df2.rolling(180).std()
stddev_x = stddev_x.values
stddev_x = stddev_x.astype('float32')
sigma_store = np.zeros((x_test.shape[0], x_test.shape[1]))
for n in range (output_var):
    dfn = pd.DataFrame(y[n,:,:])
    stddev_out = dfn.rolling(180).std()
    stddev_out = stddev_out.values
    stddev_out = stddev_out.astype('float32')
    sigma_store = sigma_store + (stddev_out - stddev_x)
sigma_store = sigma_store / output_var
plt.plot(sigma_store[:,chart], label = "Stddev difference avg of all outputs")
plt.legend(loc = 1)
plt.title(dataframe.columns[chart+1])
plt.show()
#######################
'''



'''
#### Mean and std dev of input and outputs PRICES #############
mean_input = np.zeros(x_test.shape[1])
stddev_input = np.zeros(x_test.shape[1])

for m in range (x_test.shape[1]):
    for n in range (x_test.shape[0]):
        mean_input[m] = mean_input[m] + x_test_price[n,m]
mean_input = mean_input / x_test.shape[0]

for m in range (x_test.shape[1]):
    for n in range (x_test.shape[0]):
        stddev_input[m] = stddev_input[m] + np.sqrt(np.abs(mean_input[m] - x_test_price[n,m])/x_test.shape[0])
        
        
mean_output = np.zeros((output_var, x_test.shape[1]))
stddev_output = np.zeros((output_var, x_test.shape[1]))

for p in range (output_var):
    for m in range (x_test.shape[1]):
        for n in range (x_test.shape[0]):
            mean_output[p,m] = mean_output[p,m] + y[p,n,m]
mean_output = mean_output / x_test.shape[0]
mean_output_avg = np.zeros(x_test.shape[1])
for k in range (x_test.shape[1]):
    for j in range (output_var):
        mean_output_avg[k] = mean_output_avg[k] + mean_output[j,k]
mean_output_avg = mean_output_avg / output_var

for p in range (output_var):
    for m in range (x_test.shape[1]):
        for n in range (x_test.shape[0]):
            stddev_output[p,m] = stddev_output[p,m] + np.sqrt(np.abs(mean_output[p,m] - y[p,n,m])/x_test.shape[0])
stddev_output_avg = np.zeros(x_test.shape[1])
for k in range (x_test.shape[1]):
    for j in range (output_var):
        stddev_output_avg[k] = stddev_output_avg[k] + stddev_output[j,k]
stddev_output_avg = stddev_output_avg / output_var
######################################################
'''



'''
#### This boy ain't right

######### Distribution of average of outputs vs distribution of input
plt.figure(3)

sns.kdeplot(x_test[:,chart], shade=True, label = "Input")
dist_store = np.zeros((x_test.shape[0], x_test.shape[1]))
for n in range (output_var):
    dist_store = dist_store + output_returns[n,:,:]
dist_store = dist_store / output_var
sns.kdeplot(dist_store[:,chart], shade=True, label = "Avg of outputs")
plt.legend(loc = 1)
plt.title("Distributions of {}".format(dataframe.columns[chart+1]))
plt.show()
'''


#sns.set(style="white", palette="muted", color_codes=True)
#f, axes = plt.subplots(7, 3, figsize=(12, 9), sharex=True)
#sns.despine(left=True)
#for n in range (output_returns.shape[2] % 3):
#    for m in range (output_returns.shape[2] % 7):
#        sns.kdeplot(output_returns[0,:,n+m], shade=True, 
#                    label = "Output {}".format(dataframe.columns[n+m+1]), 
#                    ax=axes[n,m])
#        sns.kdeplot(x_test[:,n+m], shade=True, label = "Input", ax=axes[n,m])
#plt.legend(loc = 1)
##f.suptitle("Distributions of {}".format(dataframe.columns[chart+1]))
#plt.show()




##############@@@@@@@@@@@@@@@@@@@@@@###############
##############      main_20         ###############
##############@@@@@@@@@@@@@@@@@@@@@@###############

#    # encoder
#    def encoder(self, inputs):
#        with tf.variable_scope("encoder"):
#            h1 = dense(inputs, self.inputs.shape[1], self.n_hidden, "z_1")
#            h1 = tf.nn.relu(h1)
#            w_mean = dense(h1, self.n_hidden, self.n_z, "mean_z_1")
#            w_mean = tf.nn.relu(w_mean)
#            w_stddev = dense(h1, self.n_hidden, self.n_z, "std_z_1")
#            w_stddev = tf.nn.softplus(w_stddev)
#            w_mean = tf.nn.relu(w_mean)
#        return w_mean, w_stddev
#
#    # decoder
#    def decoder(self, z):
#        with tf.variable_scope("decoder"):         
#            h1 = dense(z, self.n_z, self.n_hidden, "x_1")
#            h1 = tf.nn.relu(h1)
#            w_mean = dense(h1, self.n_hidden, self.inputs.shape[1], "mean_x_1")
#            w_stddev = dense(h1, self.n_hidden, self.inputs.shape[1], "std_x_1")
#            w_mean = tf.nn.relu(w_mean)
#            w_stddev = tf.nn.softplus(w_stddev)
#            w_samples = tf.random_normal(shape=tf.shape(w_stddev), mean=0, stddev=1, dtype=tf.float32)
#            h2 = w_mean + (w_stddev * w_samples)
#        return h2



## fully-connected dense layer
#def dense(x, inputFeatures, outputFeatures, scope=None, with_w=False):
#    with tf.variable_scope(scope or "Linear"):
#        matrix = tf.get_variable("Matrix", [inputFeatures, outputFeatures], tf.float32, tf.random_normal_initializer(stddev=0.02))
#        bias = tf.get_variable("bias", [outputFeatures], initializer=tf.constant_initializer(0.0))
#        if with_w:
#            return tf.matmul(x, matrix) + bias, matrix, bias
#        else:
#            return tf.matmul(x, matrix) + bias






#    # %returns to prices
#    def chain_returns(self, n_out, x_test, x_test_price):
#        returns = np.zeros((n_out, x_test.shape[0], x_test.shape[1]))
#        
#        with tf.Session() as sess:
#            sess.run(tf.global_variables_initializer())
#            for k in range(n_out):
#                returns[k,:,:] = sess.run(self.decoded_data, feed_dict={self.inputs: x_test})
#            
#        chained_returns = np.zeros((n_out, x_test.shape[0], x_test.shape[1]))
#        for k in range(n_out):
#            for j in range(chained_returns.shape[2]):
#                for i in range(chained_returns.shape[1]):
#                    if i == 0:
#                        chained_returns[k,i,j] = x_test_price[0,j]
#                    else:
#                        chained_returns[k,i,j] = returns[k,i-1,j]
##                        chained_returns[k,i,j] = chained_returns[k,i-1,j] * (1 + returns[k,i-1,j])
#        return chained_returns