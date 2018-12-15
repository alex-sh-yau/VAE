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
        self.X_r = np.reshape(input_return, (-1,9))
        self.y_p = output_price
        self.y_r = output_return
        self.df = dataframe
    
    ''' Input vs output stddev check '''
    # output_v = single selected output out of all that were generated, from 0 to N_OUTPUTS
    def stddev_check(self, output_v):
        f = plt.figure()
        plt.plot(self.X_r[:100, 0], label = "Actual")
        plt.plot(self.y_r[output_v,:100,0], label = "Decoder")
        plt.legend(loc = 1)
        plt.title("Stddev Check: Returns of inputs vs outputs")
        return f
        
    ''' Input vs outputs price charts '''
    ### Change size depending on number of input features
    ### Current datasets have 9 assets each - Outputs charts for each asset in 3x3 plot
    def price_charts(self):
        f, axes = plt.subplots(3,3, figsize=(15,9))
        axes = axes.ravel()
        for k in range (self.df.shape[1]-1):
            axes[k].plot(self.y_p[:,:,k].T, color = 'lightgrey')
            axes[k].plot(self.X_p[:,k])
            axes[k].set_title("{}".format(self.df.columns[k+1]))
        f.subplots_adjust(hspace=0.5)
        plt.legend(loc = 1)
        f.suptitle("Price charts of input vs all outputs", fontsize=14)
        return f
    
    ''' SMA crossovers '''
    # M_window = smaller window, size in number of days, typically month-wide
    # Y_window = bigger window, size in number of days, typically year-wide
    # asset = single selected input feature in the dataset, from 0 to FEATURE_SIZE
    # out_charts = number of outputs to loop through and chart, from 0 to N_OUTPUTS
    def sma_cross(self, M_window, Y_window, asset, out_charts):
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
        for n in range(out_charts):
            dfn = pd.DataFrame(self.y_p[n,:,:])
            SMA_M_output = dfn.rolling(M_window).mean()
            SMA_M_output = SMA_M_output.values
            SMA_M_output = SMA_M_output.astype('float32')
            SMA_Y_output = dfn.rolling(Y_window).mean()
            SMA_Y_output = SMA_Y_output.values
            SMA_Y_output = SMA_Y_output.astype('float32')
            plt.plot(SMA_Y_output[:,asset] - SMA_M_output[:,asset], 
                     label = "Output {}".format(n+1), color = 'lightgrey')
        plt.plot(SMA_Y_input[:,asset] - SMA_M_input[:,asset], label = "Input")
        plt.legend(loc = 1)
        plt.title("SMA crossover 3m vs 1y of {}".format(self.df.columns[asset+1]))
        return f
    
    ''' Std dev difference between outputs and input, averaged '''
    # window = window size, size in number of days, typically 180
    # n_out = total number of generated outputs
    # asset_charts = number of input features to loop through, from 0 to FEATURE_SIZE
    def avg_stddev(self, window, n_out, asset_charts):
        f = plt.figure()
        df = pd.DataFrame(self.X_r)
        stddev_x = df.rolling(window).std()
        stddev_x = stddev_x.values
        stddev_x = stddev_x.astype('float32')
        sigma_store = np.zeros((self.X_r.shape[0], self.X_r.shape[1]))
        for n in range (n_out):
            dfn = pd.DataFrame(self.y_r[n,:,:])
            stddev_out = dfn.rolling(window).std()
            stddev_out = stddev_out.values
            stddev_out = stddev_out.astype('float32')
            sigma_store = sigma_store + (stddev_out - stddev_x)
        sigma_store = sigma_store / n_out
        for m in range (asset_charts):
            plt.plot(np.abs(sigma_store[:,m]), label = self.df.columns[m+1])
        plt.legend(loc = 1)
        plt.title("Stddev difference avg of all outputs")
        return f
        
    ''' Input vs all outputs returns distributions for each asset (multiplot)'''
    # n_features = total number of input features in the dataset
    # n_out = total number of generated outputs
    # df_kl = dataframe of displayed info (mean, std dev, KL div)
    # Output mean and std dev displayed are averages of all outputs generated
    # KL divergence is calculated between input distribution and average distribution of all outputs generated
    def all_returns_all_dist(self, n_features, n_out, df_kl):
        k = math.ceil(n_features/3)
        sns.set(style="white", palette="muted", color_codes=True)
        f, axes = plt.subplots(k, 3, figsize=(15, 9))
        axes = axes.ravel()
        sns.despine(left=True)
        text = df_kl.values
        for n in range (self.y_r.shape[2]):
            for m in range (n_out):
                sns.kdeplot(self.y_r[m,:,n], shade=True, 
                            color = 'lightgrey', ax=axes[n])
            sns.kdeplot(self.X_r[:,n], shade=True, label = "Input", ax=axes[n])
            axes[n].set_title("{}".format(self.df.columns[n+1]))
            axes[n].text(0.01, 0.99, "In \u03BC: {:.7f}\nOut \u03BC: {:.7f}\nIn \u03C3: {:.4f}\nOut \u03C3: {:.4f}\nKL Div: {:.4f}".format(text[n,2], text[n,0], text[n,3], text[n,1], text[n,4]),
                transform=axes[n].transAxes, fontsize=8, verticalalignment='top')
        f.subplots_adjust(hspace=0.5)
        plt.legend(loc = 1)
        f.suptitle("Returns distributions: Input vs outputs of all assets", fontsize=14)
        return f
    
    ''' Input vs all outputs price distributions for each asset (multiplot)'''
    # n_features = total number of input features in the dataset
    # n_out = total number of generated outputs
    # df_kl = dataframe of displayed info (mean, std dev, KL div)
    # Output mean and std dev displayed are averages of all outputs generated
    # KL divergence is calculated between input distribution and average distribution of all outputs generated
    def all_prices_all_dist(self, n_features, n_out, df_kl):
        k = math.ceil(n_features/3)
        sns.set(style="white", palette="muted", color_codes=True)
        f, axes = plt.subplots(k, 3, figsize=(15, 9))
        axes = axes.ravel()
        sns.despine(left=True)
        text = df_kl.values
        for n in range (self.y_p.shape[2]):
            for m in range (n_out):
                sns.kdeplot(self.y_p[m,:,n], shade=True, 
                            color = 'lightgrey', ax=axes[n])
            sns.kdeplot(self.X_p[:,n], shade=True, label = "Input", ax=axes[n])
            axes[n].set_title("{}".format(self.df.columns[n+1]))
            axes[n].text(0.01, 0.99, "In \u03BC: {:.7f}\nOut \u03BC: {:.7f}\nIn \u03C3: {:.4f}\nOut \u03C3: {:.4f}\nKL Div: {:.4f}".format(text[n,2], text[n,0], text[n,3], text[n,1], text[n,4]),
                transform=axes[n].transAxes, fontsize=8, verticalalignment='top')
        f.subplots_adjust(hspace=0.5)
        plt.legend(loc = 1)
        f.suptitle("Price distributions: Input vs outputs of all assets", fontsize=14)
        return f
    
    ''' Rolling correlation of each output vs input '''
    # window = window size, size in number of days, typically 180
    # asset = single selected input feature in the dataset, from 0 to FEATURE_SIZE
    # out_charts = number of outputs to loop through and chart, from 0 to N_OUTPUTS
    def rolling_corr(self, window, asset, out_charts):
        f = plt.figure()
        df = pd.DataFrame(self.X_p)
        for n in range (out_charts): 
            dfn = pd.DataFrame(self.y_p[n,:,:])
            corr_out = dfn.rolling(window).corr(df)    
            corr_out = corr_out.values
            corr_out = corr_out.astype('float32')
            plt.plot(corr_out[:,asset], label = "Output {}".format(n+1))
        plt.legend(loc = 1)
        plt.title("{}".format(window) + " Day Rolling correlation of {}".format(self.df.columns[asset+1]) +
                  " Input vs Output")
        return f
    
    ''' Rolling correlation between inputs '''
    # window = window size, size in number of days, typically 180
    # output_v = single selected output out of all that were generated, from 0 to N_OUTPUTS
    # asset_1, asset_2 = single selected input features in the dataset, from 0 to FEATURE_SIZE
    def corr_in_vs_out(self, window, output_v, asset_1, asset_2):
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
        return f
    
    ''' Spread between 2 assets '''
    # output_v = single selected output out of all that were generated, from 0 to N_OUTPUTS
    # asset_1, asset_2 = single selected input features in the dataset, from 0 to FEATURE_SIZE
    def spread(self, output_v, asset_1, asset_2):
        f = plt.figure()
        plt.plot(self.X_p[:,asset_1] - self.X_p[:,asset_2], label = "Input")
        plt.plot(self.y_p[output_v,:,asset_1] - self.y_p[output_v,:,asset_2], label = "Output")
        plt.legend(loc = 1)
        plt.title("Price spread of {}".format(self.df.columns[asset_1+1]) + 
                  " vs {}".format(self.df.columns[asset_2+1]))
        return f
    
    ''' Mean and std dev of input and output(avg) >> returns <<
        as well as KL divergence between the distributions '''
    # n_out = total number of generated outputs
    def KL_returns(self, n_out):
        mean_input = np.zeros(self.X_r.shape[1])
        stddev_input = np.zeros(self.X_r.shape[1])
        mean_output = np.zeros((n_out, self.X_r.shape[1]))
        stddev_output = np.zeros((n_out, self.X_r.shape[1]))
        
        for m in range (self.X_r.shape[1]):
            for n in range (self.X_r.shape[0]):
                mean_input[m] = mean_input[m] + self.X_r[n,m]
        mean_input = mean_input / self.X_r.shape[0]
        
        for m in range (self.X_r.shape[1]):
            for n in range (self.X_r.shape[0]):
                stddev_input[m] = stddev_input[m] + np.square(np.abs(self.X_r[n,m] - mean_input[m]))
        stddev_input = np.sqrt(stddev_input / self.X_r.shape[0])
        
        for p in range (n_out):
            for m in range (self.X_r.shape[1]):
                for n in range (self.X_r.shape[0]):
                    mean_output[p,m] = mean_output[p,m] + self.y_r[p,n,m]
        mean_output = mean_output / self.X_r.shape[0]
        mean_output_avg = np.zeros(self.X_r.shape[1])
        for k in range (self.X_r.shape[1]):
            for j in range (n_out):
                mean_output_avg[k] = mean_output_avg[k] + mean_output[j,k]
        mean_output_avg = mean_output_avg / n_out
        
        for p in range (n_out):
            for m in range (self.X_r.shape[1]):
                for n in range (self.X_r.shape[0]):
                    stddev_output[p,m] = stddev_output[p,m] + np.square(np.abs(self.y_r[p,n,m] - mean_output[p,m]))
        stddev_output = np.sqrt(stddev_output / self.X_r.shape[0])
        stddev_output_avg = np.zeros(self.X_r.shape[1])
        for k in range (self.X_r.shape[1]):
            for j in range (n_out):
                stddev_output_avg[k] = stddev_output_avg[k] + stddev_output[j,k]
        stddev_output_avg = stddev_output_avg / n_out
        
        KL_div = np.zeros(self.X_r.shape[1])
        for m in range (self.X_r.shape[1]):
            KL_div[m] = (math.log(stddev_output_avg[m]/stddev_input[m]) + 
                ((np.square(stddev_input[m]) + np.square(mean_input[m] - mean_output_avg[m])) /
                (2 * np.square(stddev_output_avg[m]))) - (1/2))
        
        indexes = pd.DataFrame(self.df.iloc[:,1:].columns.values)
        comp = np.vstack((mean_output_avg, stddev_output_avg, mean_input, stddev_input, KL_div))
        comp = np.transpose(comp)
        df_comp = pd.DataFrame(comp)
        df_comp_columns = ['Output mean - avg', 'Output stddev - avg', 
                           'Input mean', 'Input stddev', 'KL Divergence']
        df_comp.columns = df_comp_columns
        df_comp.index = indexes
        return df_comp

    ''' Mean and std dev of input and output(avg) >> prices <<
        as well as KL divergence between the distributions '''
    # n_out = total number of generated outputs
    def KL_price(self, n_out):
        mean_input = np.zeros(self.X_p.shape[1])
        stddev_input = np.zeros(self.X_p.shape[1])
        mean_output = np.zeros((n_out, self.X_p.shape[1]))
        stddev_output = np.zeros((n_out, self.X_p.shape[1]))
        
        for m in range (self.X_p.shape[1]):
            for n in range (self.X_p.shape[0]):
                mean_input[m] = mean_input[m] + self.X_p[n,m]
        mean_input = mean_input / self.X_p.shape[0]
        
        for m in range (self.X_p.shape[1]):
            for n in range (self.X_p.shape[0]):
                stddev_input[m] = stddev_input[m] + np.square(np.abs(self.X_p[n,m] - mean_input[m]))
        stddev_input = np.sqrt(stddev_input / self.X_p.shape[0])
        
        for p in range (n_out):
            for m in range (self.X_p.shape[1]):
                for n in range (self.X_p.shape[0]):
                    mean_output[p,m] = mean_output[p,m] + self.y_p[p,n,m]
        mean_output = mean_output / self.X_p.shape[0]
        mean_output_avg = np.zeros(self.X_p.shape[1])
        for k in range (self.X_p.shape[1]):
            for j in range (n_out):
                mean_output_avg[k] = mean_output_avg[k] + mean_output[j,k]
        mean_output_avg = mean_output_avg / n_out
        
        for p in range (n_out):
            for m in range (self.X_p.shape[1]):
                for n in range (self.X_p.shape[0]):
                    stddev_output[p,m] = stddev_output[p,m] + np.square(np.abs(self.y_p[p,n,m] - mean_output[p,m]))
        stddev_output = np.sqrt(stddev_output / self.X_p.shape[0])
        stddev_output_avg = np.zeros(self.X_p.shape[1])
        for k in range (self.X_p.shape[1]):
            for j in range (n_out):
                stddev_output_avg[k] = stddev_output_avg[k] + stddev_output[j,k]
        stddev_output_avg = stddev_output_avg / n_out
        
        KL_div = np.zeros(self.X_p.shape[1])
        for m in range (self.X_p.shape[1]):
            KL_div[m] = (math.log(stddev_output_avg[m]/stddev_input[m]) + 
                ((np.square(stddev_input[m]) + np.square(mean_input[m] - mean_output_avg[m])) /
                (2 * np.square(stddev_output_avg[m]))) - (1/2))
        
        indexes = pd.DataFrame(self.df.iloc[:,1:].columns.values)
        comp = np.vstack((mean_output_avg, stddev_output_avg, mean_input, stddev_input, KL_div))
        comp = np.transpose(comp)
        df_comp = pd.DataFrame(comp)
        df_comp_columns = ['Output mean - avg', 'Output stddev - avg', 
                           'Input mean', 'Input stddev', 'KL Divergence']
        df_comp.columns = df_comp_columns
        df_comp.index = indexes
        return df_comp