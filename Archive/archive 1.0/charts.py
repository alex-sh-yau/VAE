# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:58:33 2018

@author: yaua
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

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
        
    ### Change size depending on number of features
    def price_charts(self):
        # Input vs outputs price charts
        f, axes = plt.subplots(3,3, figsize=(15,9))
        axes = axes.ravel()
        for k in range (self.df.shape[1]-1):
            axes[k].plot(self.y_p[:,:,k].T, color = 'lightgrey')
            axes[k].plot(self.X_p[:,k])
            axes[k].set_title("{}".format(self.df.columns[k+1]))
        f.subplots_adjust(hspace=0.5)
        plt.legend(loc = 1)
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
        f, axes = plt.subplots(k, 3, figsize=(15, 9))
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