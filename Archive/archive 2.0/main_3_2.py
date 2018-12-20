# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:10:27 2018

@author: yaua
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import charts


class VariationalAutoencoder():

    def __init__(self, n_features, hidden_neurons, lr):
        
        self.n_features = n_features
        self.hidden_neurons = hidden_neurons
        self.lr = lr
        
        self.inputs = tf.placeholder(shape = [None, self.n_features], dtype = tf.float32)
        self.labels = tf.placeholder(shape = [None, self.n_features], dtype = tf.float32)
        
        # Encoder layer
        self.z_hidden = tf.layers.dense(self.inputs, self.hidden_neurons[0], tf.nn.relu)
        self.z_mean = tf.layers.dense(self.z_hidden, self.hidden_neurons[1], tf.nn.relu)
        self.z_stddev = tf.layers.dense(self.z_hidden, self.hidden_neurons[1], tf.nn.softplus)

        samples = tf.random_normal(shape = tf.shape(self.z_stddev), mean = 0, 
                                   stddev = tf.sqrt(2/(self.hidden_neurons[0] + self.hidden_neurons[1])),
                                   dtype = tf.float32)
        self.sampled_z = self.z_mean + (self.z_stddev * samples)
        
        # Decoder layer
        self.x_hidden = tf.layers.dense(self.sampled_z, self.hidden_neurons[0], tf.nn.relu)
        self.x_mean = tf.layers.dense(self.x_hidden, self.n_features, tf.nn.relu)
        self.x_stddev = tf.layers.dense(self.x_hidden, self.n_features, tf.nn.softplus)
        
        x_samples = tf.random_normal(shape = tf.shape(self.x_stddev), mean = 0, 
                                   stddev = tf.sqrt(2/(self.hidden_neurons[0] + self.n_features)),
                                   dtype = tf.float32)
        self.decoded_data = self.x_mean + (self.x_stddev * x_samples)
        
        # Cost, Optimizer
        self.decoded_loss = tf.reduce_mean(tf.square(self.decoded_data - self.labels))
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) + tf.square(self.z_stddev) - 
                                               tf.log(tf.square(self.z_stddev)) - 1)
        self.cost = tf.reduce_mean(self.decoded_loss + tf.reshape(self.latent_loss, shape=(-1,)))
        
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        
        
def train(model, sess, train_inputs, epochs, batchsize):
    for epoch in range(epochs):
        avg_cost = 0
        np.random.shuffle(train_inputs)
        total_batch = int(train_inputs.shape[0] / batchsize)
        x_batches = np.array_split(train_inputs, total_batch)
        for i in range(total_batch):
            batch_x = x_batches[i]
            _, c, dec_loss, lat_loss = sess.run([model.optimizer, model.cost, model.decoded_loss, model.latent_loss], 
                                                feed_dict={model.inputs: batch_x, model.labels: batch_x})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "| Cost =", "{:.9f}".format(avg_cost), 
              "| Generative loss =", "{:.9f}".format(dec_loss), 
              "| Latent loss =", "{:.9f}".format(lat_loss))


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
    return chained_returns, returns


class ModelGen():

    def __init__(self, X_train, X_test, X_train_price, X_test_price, dataframe):

        self.FEATURE_SIZE = X_train.shape[1]
        self.LEARNING_RATE = 0.005
        self.NEURONS = [int(self.FEATURE_SIZE / 2), int(self.FEATURE_SIZE / 4)]
        self.EPOCHS = 50
        self.BATCH_SIZE = 300
        self.N_OUTPUTS = self.FEATURE_SIZE
        
        #tf.reset_default_graph()

        sess_1 = tf.Session()
        model = VariationalAutoencoder(self.FEATURE_SIZE, self.NEURONS, self.LEARNING_RATE)
        sess_1.run(tf.global_variables_initializer())
        train(model, sess_1, X_train, self.EPOCHS, self.BATCH_SIZE)
        
        y = np.zeros((self.N_OUTPUTS, X_test.shape[0], X_test.shape[1]))
        y_r = np.zeros((self.N_OUTPUTS, X_test.shape[0], X_test.shape[1]))
        y, y_r = chain_returns(model, sess_1, self.N_OUTPUTS, X_test, X_test_price)
        
        # Individual output/feature number
        self.output_var = 1
        self.feature_var = 5
        
        # Output/features to loop through
        self.output_charts = 3
        self.feature_charts = 4
        
        #### Next step - code for individual selective initialization
        
        visualizer = charts.Charts(X_test_price, X_test, y, y_r, dataframe)
        
        self.model_stddev_check = visualizer.stddev_check(self.output_var)
        self.price_check = visualizer.price_charts()
        self.dist_check = visualizer.returns_dist(self.N_OUTPUTS, self.feature_var)
        self.sma_cross_check = visualizer.sma_cross(90, 360, self.feature_var, self.output_charts)
        self.avg_stddev_check = visualizer.avg_stddev(180, self.N_OUTPUTS, self.feature_charts)
        self.output_dists_check = visualizer.all_returns_dist(self.FEATURE_SIZE, self.output_var)
        self.output_corr_check = visualizer.rolling_corr(180, self.feature_var, self.output_charts)
        self.in_vs_out_corr_check = visualizer.corr_in_vs_out(180, self.output_var, 4, 5)
        self.spread_check = visualizer.spread(self.output_var, 4, 5)
        

def data(df):
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


def mean_stddev(df):
    df_v = df.iloc[:,1:].values
    mean_input = np.zeros(df_v.shape[1])
    stddev_input = np.zeros(df_v.shape[1])
    n_datapoints = np.zeros(df_v.shape[1])
    
    for m in range (df_v.shape[1]):
        for n in range (df_v.shape[0]):
            if not np.isnan(df_v[n,m]):
                mean_input[m] = mean_input[m] + df_v[n,m]
    mean_input = mean_input / df_v.shape[0]
    
    for m in range (df_v.shape[1]):
        for n in range (df_v.shape[0]):
            if not np.isnan(df_v[n,m]):
                stddev_input[m] = stddev_input[m] + np.square(np.abs(df_v[n,m] - mean_input[m]))
                n_datapoints[m] += 1
    stddev_input = np.sqrt(stddev_input / df_v.shape[0])
    
    indexes = pd.DataFrame(df.iloc[:,1:].columns.values)
    comp = np.vstack((mean_input, stddev_input, n_datapoints))
    df_comp = pd.DataFrame(comp)
    df_comp.columns = indexes
    df_comp.index = ['Mean', 'Std Dev', 'Datapoints']
    return df_comp

    
#if __name__ == '__main__':

df1 = pd.read_csv('AlexCurr.csv')  # 9 inputs
X_tr_1, X_te_1, X_tr_p_1, X_te_p_1 = data(df1)

df2 = pd.read_csv('AlexComm.csv')
X_tr_2, X_te_2, X_tr_p_2, X_te_p_2 = data(df2)

df3 = pd.read_csv('AlexIndex.csv')
X_tr_3, X_te_3, X_tr_p_3, X_te_p_3 = data(df3)

df4 = pd.read_csv('AlexStock.csv')
X_tr_4, X_te_4, X_tr_p_4, X_te_p_4 = data(df4)


mean_stddev(df1).to_csv("Currency Mean and Std_dev.csv")
mean_stddev(df2).to_csv("Commodities Mean and Std_dev.csv")
mean_stddev(df3).to_csv("Indices Mean and Std_dev.csv")
mean_stddev(df4).to_csv("S&P500 Stocks Mean and Std_dev.csv")


g_1 = tf.Graph()
with g_1.as_default():

    vae_1 = ModelGen(X_tr_1, X_te_1, X_tr_p_1, X_te_p_1, df1)
    vae_2 = ModelGen(X_tr_2, X_te_2, X_tr_p_2, X_te_p_2, df2)
    vae_3 = ModelGen(X_tr_3, X_te_3, X_tr_p_3, X_te_p_3, df3)
    vae_4 = ModelGen(X_tr_4, X_te_4, X_tr_p_4, X_te_p_4, df4)

#    vae_1.price_check.show()
#    vae_2.price_check.show()
#    vae_3.price_check.show()
#    vae_4.price_check.show()
#
#    vae_3.sma_cross_check.show()
#    vae_4.avg_stddev_check.show()
#    vae_4.output_corr_check.show()
#    vae_4.in_vs_out_corr_check.show()
#    
#    vae_1.output_dists_check.show()
#    vae_2.output_dists_check.show()
#    vae_3.output_dists_check.show()
#    vae_4.output_dists_check.show()
    
#    vae_4.dist_check.show()
#    
#    vae_1.spread_check.show()
#    vae_2.spread_check.show()
#    vae_3.in_vs_out_corr_check.show()
