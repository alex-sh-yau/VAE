# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:10:27 2018

@author: yaua
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import seaborn as sns


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



class Charts():
    
    def __init__(self, input_price, input_return, output_price, output_return, dataframe):
        
        self.X_p = input_price
        self.X_r = input_return
        self.y_p = output_price
        self.y_r = output_return
        self.df = dataframe
        
    def stddev_check(self, output_v):
        # Input vs output stddev check
        f = plt.figure()
        plt.plot(self.X_r[:100, 0], label = "Actual")
        plt.plot(self.y_r[output_v,:100,0], label = "Decoder")
        plt.legend(loc = 1)
        plt.close(f)
        return f
        
    
    ########### Modify later for reusability
    def price_charts(self):
        # Input vs outputs price charts
        f, axes = plt.subplots(2,2, figsize=(15,9))
        axes[0,0].plot(self.y_p[:,:,0].T, color = 'lightgrey')
        axes[0,0].plot(self.X_p[:,0])
        axes[0,0].set_title("{}".format(self.df.columns[1]))
        axes[0,1].plot(self.y_p[:,:,1].T, color = 'lightgrey')
        axes[0,1].plot(self.X_p[:,1])
        axes[0,1].set_title("{}".format(self.df.columns[2]))
        axes[1,0].plot(self.y_p[:,:,4].T, color = 'lightgrey')
        axes[1,0].plot(self.X_p[:,4])
        axes[1,0].set_title("{}".format(self.df.columns[5]))
        axes[1,1].plot(self.y_p[:,:,8].T, color = 'lightgrey')
        axes[1,1].plot(self.X_p[:,8])
        axes[1,1].set_title("{}".format(self.df.columns[9]))
        plt.close(f)
        return f
    
    def returns_dist(self, n_out, i_feature):
        # Input vs outputs returns distributions
        f = plt.figure()
        for n in range (n_out):
            sns.kdeplot(self.y_r[n,:,i_feature], shade=True, color = 'lightgrey')
        sns.kdeplot(self.X_r[:,i_feature], shade=True, label = "Input")
        plt.legend(loc = 1)
        plt.title("Distributions of {}".format(self.df.columns[i_feature+1]))
        plt.close(f)
        return f
    
    def sma_cross(self, M_window, Y_window, i_feature, out_chart):
        # SMA crossover 3m vs 1y of input
        f = plt.figure()
    
        df = pd.DataFrame(self.X_p)
        SMA_M_input = df.rolling(M_window).mean()
        SMA_M_input = SMA_M_input.values
        SMA_M_input = SMA_M_input.astype('float32')
        SMA_Y_input = df.rolling(Y_window).mean()
        SMA_Y_input = SMA_Y_input.values
        SMA_Y_input = SMA_Y_input.astype('float32')
        
        # SMA crossover 3m vs 1y of all outputs
        for n in range(out_chart):
            dfn = pd.DataFrame(self.y_p[n,:,:])
            SMA_M_output = dfn.rolling(M_window).mean()
            SMA_M_output = SMA_M_output.values
            SMA_M_output = SMA_M_output.astype('float32')
            SMA_Y_output = dfn.rolling(Y_window).mean()
            SMA_Y_output = SMA_Y_output.values
            SMA_Y_output = SMA_Y_output.astype('float32')
            plt.plot(SMA_Y_output[:,i_feature] - SMA_M_output[:,i_feature], 
                     label = "Output {}".format(n+1), color = 'lightgrey')
            
        plt.plot(SMA_Y_input[:,i_feature] - SMA_M_input[:,i_feature], label = "Input")
        plt.legend(loc = 1)
        plt.title("SMA crossover 3m vs 1y of {}".format(self.df.columns[i_feature+1]))
        plt.close(f)
        return f

    def avg_stddev(self, window, out_charts, feat_charts):
        # Std dev difference between outputs and input, averaged
        f = plt.figure()
        
        df = pd.DataFrame(self.X_r)
        stddev_x = df.rolling(window).std()
        stddev_x = stddev_x.values
        stddev_x = stddev_x.astype('float32')
        sigma_store = np.zeros((self.X_r.shape[0], self.X_r.shape[1]))
        for n in range (out_charts):
            dfn = pd.DataFrame(self.y_r[n,:,:])
            stddev_out = dfn.rolling(window).std()
            stddev_out = stddev_out.values
            stddev_out = stddev_out.astype('float32')
            sigma_store = sigma_store + (stddev_out - stddev_x)
        sigma_store = sigma_store / out_charts
        for m in range (feat_charts):
            plt.plot(np.abs(sigma_store[:,m]), label = self.df.columns[m+1])
        plt.legend(loc = 1)
        plt.title("Stddev difference avg of all outputs")
        plt.close(f)
        return f
        
    def all_returns_dist(self, features, output_v):
        # Distributions of each input vs its output (multiplot)
        k = math.ceil(features/3)
        sns.set(style="white", palette="muted", color_codes=True)
        f, axes = plt.subplots(k, 3, figsize=(12, 9))
        axes = axes.ravel()
        sns.despine(left=True)
        for n in range (self.y_r.shape[2]):
            sns.kdeplot(self.y_r[output_v,:,n], shade=True, 
                        label = "Output", ax=axes[n])
            sns.kdeplot(self.X_r[:,n], shade=True, label = "Input", ax=axes[n])
            axes[n].set_title("{}".format(self.df.columns[n+1]))
        f.subplots_adjust(hspace=0.5)
        plt.legend(loc = 1)
        plt.close(f)
        return f
    
    def rolling_corr(self, window, i_feature, out_chart):
        # Rolling correlation of each output vs input
        f = plt.figure()
        
        df = pd.DataFrame(self.X_p)
        for n in range (out_chart): 
            dfn = pd.DataFrame(self.y_p[n,:,:])
            corr_out = dfn.rolling(window).corr(df)    
            corr_out = corr_out.values
            corr_out = corr_out.astype('float32')
            plt.plot(corr_out[:,i_feature], label = "Output {}".format(n+1))
        plt.legend(loc = 1)
        plt.title("{}".format(window) + " Day Rolling correlation of {}".format(self.df.columns[i_feature+1]) +
                  " Input vs Output")
        plt.close(f)
        return f
    
    def corr_in_vs_out(self, window, output_v, asset_1, asset_2):
        # Rolling correlation between inputs
        f = plt.figure()
        
        df_in_1 = pd.DataFrame(self.X_p[:,asset_1])
        df_out_1 = pd.DataFrame(self.y_p[output_v,:,asset_1])

        df_in_2 = pd.DataFrame(self.X_p[:,asset_2])
        corr_in = df_in_2.rolling(window).corr(df_in_1) 
        corr_in = corr_in.values
        corr_in = corr_in.astype('float32')
        plt.plot(corr_in, label = "Input")
        df_out_2 = pd.DataFrame(self.y_p[output_v,:,asset_2])
        corr_out = df_out_2.rolling(window).corr(df_out_1)
        corr_out = corr_out.values
        corr_out = corr_out.astype('float32')
        plt.plot(corr_out, label = "Output")
            
        plt.legend(loc = 1)
        plt.title("{}".format(window) + " Day Rolling Correlation of {}".format(self.df.columns[asset_1+1]) + 
                  " vs {}".format(self.df.columns[asset_2+1]))
        plt.close(f)
        return f
    
    def spread(self, output_v, asset_1, asset_2):
        # Spread between 2 assets
        f = plt.figure()
        
        plt.plot(self.X_p[:,asset_1] - self.X_p[:,asset_2], label = "Input")
        plt.plot(self.y_p[output_v,:,asset_1] - self.y_p[output_v,:,asset_2], label = "Output")
        plt.legend(loc = 1)
        plt.title("Price spread of {}".format(self.df.columns[asset_1+1]) + 
                  " vs {}".format(self.df.columns[asset_2+1]))
        plt.close(f)
        return f


class ModelGen():

    def __init__(self, X_train, X_test, X_train_price, X_test_price, dataframe):

        self.FEATURE_SIZE = X_train.shape[1]
        self.LEARNING_RATE = 0.005
        self.NEURONS = [int(self.FEATURE_SIZE / 2), int(self.FEATURE_SIZE / 4)]
        self.EPOCHS = 50
        self.BATCH_SIZE = 300
        self.N_OUTPUTS = self.FEATURE_SIZE
        
    #    tf.reset_default_graph()
    #        
        sess_1 = tf.Session()
        model = VariationalAutoencoder(self.FEATURE_SIZE, self.NEURONS, self.LEARNING_RATE)
        sess_1.run(tf.global_variables_initializer())
        train(model, sess_1, X_train, self.EPOCHS, self.BATCH_SIZE)
        
        y = np.zeros((self.N_OUTPUTS, X_test.shape[0], X_test.shape[1]))
        y_r = np.zeros((self.N_OUTPUTS, X_test.shape[0], X_test.shape[1]))
        y, y_r = chain_returns(model, sess_1, self.N_OUTPUTS, X_test, X_test_price)
        
        
        # Individual output/feature number
        self.output_var = 1
        self.feature_var = 3
        
        # Output/features to loop through
        self.output_charts = 3
        self.feature_charts = 4
        
        #### Next step - code for reusability and individual initialization
        
        
        visualizer = Charts(X_test_price, X_test, y, y_r, dataframe)
        
        self.model_stddev_check = visualizer.stddev_check(self.output_var)
        self.price_check = visualizer.price_charts()
        self.dist_check = visualizer.returns_dist(self.N_OUTPUTS, self.feature_var)
        self.sma_cross_check = visualizer.sma_cross(90, 360, self.feature_var, self.output_charts)
        self.avg_stddev_check = visualizer.avg_stddev(180, self.N_OUTPUTS, self.feature_charts)
        self.output_dists_check = visualizer.all_returns_dist(self.FEATURE_SIZE, self.output_var)
        self.output_corr_check = visualizer.rolling_corr(180, self.feature_var, self.output_charts)
        self.in_vs_out_corr_check = visualizer.corr_in_vs_out(180, self.output_var, 0, 1)
        self.spread_check = visualizer.spread(self.output_var, 0, 1)
    
    
#if __name__ == '__main__':

df1 = pd.read_csv('AlexCurr.csv')  # 9 inputs
X_tr_1, X_te_1, X_tr_p_1, X_te_p_1 = data(df1)

df2 = pd.read_csv('AlexComm.csv')
X_tr_2, X_te_2, X_tr_p_2, X_te_p_2 = data(df2)

df3 = pd.read_csv('AlexIndex.csv')
X_tr_3, X_te_3, X_tr_p_3, X_te_p_3 = data(df3)

df4 = pd.read_csv('AlexStock.csv')
X_tr_4, X_te_4, X_tr_p_4, X_te_p_4 = data(df4)

g_1 = tf.Graph()
with g_1.as_default():

    vae_1 = ModelGen(X_tr_1, X_te_1, X_tr_p_1, X_te_p_1, df1)
    vae_2 = ModelGen(X_tr_2, X_te_2, X_tr_p_2, X_te_p_2, df2)
    vae_3 = ModelGen(X_tr_3, X_te_3, X_tr_p_3, X_te_p_3, df3)
    vae_4 = ModelGen(X_tr_4, X_te_4, X_tr_p_4, X_te_p_4, df4)
    
    vae_1.price_check.show()
    vae_2.price_check.show()
    vae_3.price_check.show()
    vae_4.price_check.show()
    
    vae_1.output_dists_check.show()
    vae_2.output_dists_check.show()
    vae_3.output_dists_check.show()
    vae_4.output_dists_check.show()

    vae_1.spread_check.show()
    vae_2.spread_check.show()
    vae_3.in_vs_out_corr_check.show()
    
