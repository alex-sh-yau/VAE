# Variational Autoencoder for generating financial time-series data

This repository contains the files for the latest version of the Variation Autoencoder 
project worked on at OPTrust from September-December 2018.

---------------------------------------------------------------------------------------------------------

#### Author: Alex Yau

#### Supervisor: Brandon Da Silva

---------------------------------------------------------------------------------------------------------

## Abstract

The purpose of this project is to use a deep neural network to learn historical 
time-series financial data in order to create alternate varied data for the same assets in a given asset class.

Because historical financial data is so limited, this significantly constrains ML models that rely on it.
Being able to generate varied but realistic data for something like historical price time series 
can drastically improve ML models for trading, stock picking, portfolio/risk allocation among others.

---------------------------------------------------------------------------------------------------------

## Implementation

### Model

#### Encoder
The current VAE model is made up of a neural network that takes in a certain number of assets as input nodes, X. 
The data is compressed through a hidden layer of X/2 nodes, then down to a latent space layer, z, of X/4 hidden nodes. 
The mean and standard deviation is sampled from the latent distribution of z with ReLu and SoftPlus activation functions, respectively
(z_mean and z_stddev).

Using the reparameterization trick, a randomly initialized normal distribution with a mean of 0 and standard deviation of 1 
is multiplied with z_stddev. Adding this to z_mean gives us a "newly configured" latent space z, which sufficiently allows for
backpropagation of the neural network during training. 

For a better explanation of the reparameterization trick: 
*   https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf
*   https://www.jeremyjordan.me/variational-autoencoders/]

#### Decoder
From this reparameterized z-space, data is sampled by variational inference, then decompressed back to a feature space of X nodes.
Through testing various configurations of the neural network, it was found that applying the reparameterization trick 
to this final output layer provided better results.
*   [Need to prove this]

#### The rest
The model, using the TensorFlow AdamOptimizer, optimizes over two loss functions combined:
a MSE calculated between the inputs and outputs, and a latent loss function characterized by KL-Divergence. 

Other network configurations that are currently optimized through testing for datasets of 9 features:

        self.FEATURE_SIZE = X.shape[1] 
        self.LEARNING_RATE = 0.005
        self.NEURONS = [int(self.FEATURE_SIZE / 2), int(self.FEATURE_SIZE / 4)]
        self.EPOCHS = 50
        self.BATCH_SIZE = 300


### Workflow
The input features fed through the VAE consist of individual assets within a dataset of the same asset class.
Each feature corresponds to a node in the initial encoder layer of the neural network. 
The data consists of a time series of price points for each publicly traded day for each asset.
Absolute returns are calculated between each day and fed into the network. 
The distribution of absolute returns for all of the datasets generally lie between -0.1 and 0.1, i.e. maximum of 10% per day.

After being fed through the network, a dataset of slightly varied returns for each day is produced as the output.
This dataset is converted back to price, and can then be analysed or used.


### RNN implementation



---------------------------------------------------------------------------------------------------------




## Things that were tried



---------------------------------------------------------------------------------------------------------

## Next steps

*   Sinkhorn Autoencoder (?)
*   GRU implementation
*   VAE with latent constraints: https://colab.research.google.com/notebooks/latent_constraints/latentconstraints.ipynb 

---------------------------------------------------------------------------------------------------------
