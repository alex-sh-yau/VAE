# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:10:27 2018

@author: yaua
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import charts_1_0 as charts

class VariationalAutoencoder():

    def __init__(self, n_features, hidden_neurons, lr):
        
        self.n_features = n_features
        self.hidden_neurons = hidden_neurons
        self.lr = lr
        
        self.inputs = tf.placeholder(shape = [None, self.n_features], dtype = tf.float32)
        self.labels = tf.placeholder(shape = [None, self.n_features], dtype = tf.float32)
        
        # Encoder layer
        self.z_hidden_weights = tf.Variable(
                tf.random_normal(
                        [self.n_features, self.hidden_neurons[0]], 
                        stddev = tf.sqrt(2/(self.n_features + self.hidden_neurons[0]))))
        self.z_hidden = tf.nn.relu(tf.matmul(self.inputs, self.z_hidden_weights))
    
        self.z_mean_weights = tf.Variable(
                tf.random_normal(
                        [self.hidden_neurons[0], self.hidden_neurons[1]], 
                        stddev = tf.sqrt(2/(self.hidden_neurons[0] + self.hidden_neurons[1]))))
        self.z_mean = tf.nn.relu(tf.matmul(self.z_hidden, self.z_mean_weights))

        self.z_stddev_weights = tf.Variable(
                tf.random_normal(
                        [self.hidden_neurons[0], self.hidden_neurons[1]], 
                        stddev = tf.sqrt(2/(self.hidden_neurons[0] + self.hidden_neurons[1]))))
        self.z_stddev = tf.nn.softplus(tf.matmul(self.z_hidden, self.z_stddev_weights))

        samples = tf.random_normal(shape = tf.shape(self.z_stddev), 
                                   mean = 0, 
                                   stddev = 1,
                                   dtype = tf.float32)
        self.sampled_z = self.z_mean + (self.z_stddev * samples)
        
        # Decoder layer
        # Initializing the decoder weights and layers separately like in the encoder causes the generated outputs to go nuts
        self.x_hidden = tf.layers.dense(self.sampled_z, self.hidden_neurons[0], tf.nn.relu)
        self.x_mean = tf.layers.dense(self.x_hidden, self.n_features, tf.nn.relu)
        self.x_stddev = tf.layers.dense(self.x_hidden, self.n_features, tf.nn.softplus)
        
        x_samples = tf.random_normal(shape = tf.shape(self.x_stddev), 
                                     mean = 0, 
                                     stddev = 1,
                                     dtype = tf.float32)
        self.decoded_data = self.x_mean + (self.x_stddev * x_samples)
        
        # Cost, Optimizer
        self.decoded_loss = tf.reduce_mean(tf.square(self.decoded_data - self.labels))
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(self.z_mean) + 
                                               tf.square(self.z_stddev) - 
                                               tf.log(tf.square(self.z_stddev)) - 1)
        self.cost = tf.reduce_mean(self.decoded_loss + tf.reshape(self.latent_loss, shape=(-1,)))
        
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        
        
# Train the model - shuffles training data every epoch
def train(model, sess, train_inputs, epochs, batchsize):
    print("Begin model training")
    for epoch in range(epochs):
        avg_cost = 0
        np.random.shuffle(train_inputs)
        total_batch = int(train_inputs.shape[0] / batchsize)
        x_batches = np.array_split(train_inputs, total_batch)
        for i in range(total_batch):
            batch_x = x_batches[i]
            _, c, dec_loss, lat_loss = sess.run([model.optimizer, 
                                                 model.cost, 
                                                 model.decoded_loss, 
                                                 model.latent_loss], 
                                                feed_dict={model.inputs: batch_x, model.labels: batch_x})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "| Cost =", "{:.9f}".format(avg_cost), 
              "| Generative loss =", "{:.9f}".format(dec_loss), 
              "| Latent loss =", "{:.9f}".format(lat_loss))
    print("Training complete \n")

# Chain the returns calculations from the first price in the test set
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

        
# Import and preprocess data, split into train/test set
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

    return X_train, X_test, X_train_price, X_test_price   


# Calculate mean and stddev of each column in data
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


# Define parameters, generate and train the model, generate the outputs, visualize with various charts
class ModelGen():

    def __init__(self, X_train, X_test, X_train_price, X_test_price):
        
        self.FEATURE_SIZE = X_train.shape[1] # Number of input features in dataset
        self.LEARNING_RATE = 0.005
        self.NEURONS = [int(self.FEATURE_SIZE / 2), int(self.FEATURE_SIZE / 4)]
        self.EPOCHS = 50
        self.BATCH_SIZE = 300
        self.N_OUTPUTS = self.FEATURE_SIZE # Number of outputs for the model to generate
        
        #tf.reset_default_graph()

        sess_1 = tf.Session()
        model = VariationalAutoencoder(self.FEATURE_SIZE, self.NEURONS, self.LEARNING_RATE)
        sess_1.run(tf.global_variables_initializer())
        train(model, sess_1, X_train, self.EPOCHS, self.BATCH_SIZE)
        
        self.y = np.zeros((self.N_OUTPUTS, X_test.shape[0], X_test.shape[1]))
        self.y_r = np.zeros((self.N_OUTPUTS, X_test.shape[0], X_test.shape[1]))
        self.y, self.y_r = chain_returns(model, sess_1, self.N_OUTPUTS, X_test, X_test_price)


if __name__ == '__main__':
    
    df = [pd.read_csv('AlexCurr.csv'),
          pd.read_csv('AlexComm.csv'), 
          pd.read_csv('AlexIndex.csv'), 
          pd.read_csv('AlexStock.csv')]
    
    X_tr, X_te, X_tr_p, X_te_p = [0]*4, [0]*4, [0]*4, [0]*4
    for n in range (4):
        X_tr[n], X_te[n], X_tr_p[n], X_te_p[n] = data_preprocess(df[n])
    
#    mean_stddev(df[0]).to_csv("Currency Mean and Std_dev.csv")
#    mean_stddev(df[1]).to_csv("Commodities Mean and Std_dev.csv")
#    mean_stddev(df[2]).to_csv("Indices Mean and Std_dev.csv")
#    mean_stddev(df[3]).to_csv("S&P500 Stocks Mean and Std_dev.csv")
    
    g_1 = tf.Graph()
    with g_1.as_default():
    
        vae = [ModelGen(X_tr[0], X_te[0], X_tr_p[0], X_te_p[0]),
               ModelGen(X_tr[1], X_te[1], X_tr_p[1], X_te_p[1]),
               ModelGen(X_tr[2], X_te[2], X_tr_p[2], X_te_p[2]),
               ModelGen(X_tr[3], X_te[3], X_tr_p[3], X_te_p[3])]
        visualizer = [charts.Charts(X_te_p[0], X_te[0], vae[0].y, vae[0].y_r, df[0]),
                      charts.Charts(X_te_p[1], X_te[1], vae[1].y, vae[1].y_r, df[1]),
                      charts.Charts(X_te_p[2], X_te[2], vae[2].y, vae[2].y_r, df[2]),
                      charts.Charts(X_te_p[3], X_te[3], vae[3].y, vae[3].y_r, df[3])]
        
        ''' Refer to charts_1_0.py for input parameters
            Note: current charted inputs and outputs are generated using test set data '''
#        visualizer[0].stddev_check(output_v=1).show()
#        visualizer[0].sma_cross(M_window=90, Y_window=360, asset=5, out_charts=3).show()
#        visualizer[0].avg_stddev(window=180, n_out=vae[0].N_OUTPUTS, asset_charts=4).show()
#        visualizer[0].rolling_corr(window=180, asset=5, out_charts=3).show()
#        visualizer[0].corr_in_vs_out(window=180, output_v=1, asset_1=4, asset_2=5).show()
#        visualizer[0].spread(output_v=1, asset_1=0, asset_2=2).show()
        visualizer[0].price_charts().show()
        visualizer[1].price_charts().show()
        visualizer[2].price_charts().show()
        visualizer[3].price_charts().show()
        
        df_kl_r = [visualizer[0].KL_returns(n_out=vae[0].N_OUTPUTS),
                   visualizer[1].KL_returns(n_out=vae[1].N_OUTPUTS),
                   visualizer[2].KL_returns(n_out=vae[2].N_OUTPUTS),
                   visualizer[3].KL_returns(n_out=vae[3].N_OUTPUTS)]
#        visualizer[0].all_returns_all_dist(n_features=vae[0].FEATURE_SIZE, n_out=vae[0].N_OUTPUTS, df_kl=df_kl_r[0]).show()
#        visualizer[1].all_returns_all_dist(n_features=vae[1].FEATURE_SIZE, n_out=vae[1].N_OUTPUTS, df_kl=df_kl_r[1]).show()
#        visualizer[2].all_returns_all_dist(n_features=vae[2].FEATURE_SIZE, n_out=vae[2].N_OUTPUTS, df_kl=df_kl_r[2]).show()
#        visualizer[3].all_returns_all_dist(n_features=vae[3].FEATURE_SIZE, n_out=vae[3].N_OUTPUTS, df_kl=df_kl_r[3]).show()
        
        df_kl_p = [visualizer[0].KL_price(n_out=vae[0].N_OUTPUTS),
                   visualizer[1].KL_price(n_out=vae[1].N_OUTPUTS),
                   visualizer[2].KL_price(n_out=vae[2].N_OUTPUTS),
                   visualizer[3].KL_price(n_out=vae[3].N_OUTPUTS)]
#        visualizer[0].all_prices_all_dist(n_features=vae[0].FEATURE_SIZE, n_out=vae[0].N_OUTPUTS, df_kl=df_kl_p[0]).show()
#        visualizer[1].all_prices_all_dist(n_features=vae[1].FEATURE_SIZE, n_out=vae[1].N_OUTPUTS, df_kl=df_kl_p[1]).show()
#        visualizer[2].all_prices_all_dist(n_features=vae[2].FEATURE_SIZE, n_out=vae[2].N_OUTPUTS, df_kl=df_kl_p[2]).show()
#        visualizer[3].all_prices_all_dist(n_features=vae[3].FEATURE_SIZE, n_out=vae[3].N_OUTPUTS, df_kl=df_kl_p[3]).show()