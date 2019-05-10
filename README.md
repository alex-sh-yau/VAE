# VAE for generating financial time-series data

This repository contains the files for the latest version of the Variational Autoencoder (VAE)
project used to generate synthetic time-series data in various financial markets. 
A majority of the work was done between September to December 2018.

---------------------------------------------------------------------------------------------------------

#### Author: Alex Yau

---------------------------------------------------------------------------------------------------------

## Abstract

The purpose of this project is to use a deep neural network to learn historical 
time-series financial data in order to create alternate varied data for the same assets in a given asset class.

Because historical financial data is so limited, this significantly constrains ML models that rely on it.
Being able to generate varied but realistic data for something like historical price time series 
can drastically improve ML models for trading, stock picking, portfolio/risk allocation among others.

---------------------------------------------------------------------------------------------------------

## Neural network flow

The input features fed through the VAE consist of individual assets within a dataset of the same asset class.
Each feature corresponds to a node in the initial encoder layer of the neural network. 
The data consists of a time series of price points for each publicly traded day for each asset.
Absolute returns are calculated between each day and fed into the network. 
The distribution of absolute returns for all of the datasets generally lie between -0.1 and 0.1, i.e. maximum of 10% per day.

After being fed through the network, a dataset of slightly varied returns for each day is produced as the output.
This dataset is converted back to price, and can then be analysed or used.


## MLP Implementation

### Encoder
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

### Decoder
From this reparameterized z-space, data is sampled by variational inference, then decompressed back to a feature space of X nodes.
Through testing various configurations of the neural network, it was found that applying the reparameterization trick 
to this final output layer provided better results.
*   [Need to prove this]

### The rest
The model, using the TensorFlow AdamOptimizer, optimizes over two loss functions combined:
a MSE calculated between the inputs and outputs, and a latent loss function characterized by KL-Divergence. 

Other network configurations that are currently optimized through testing for datasets of 9 features:

        self.FEATURE_SIZE = X_train.shape[1] 
        self.LEARNING_RATE = 0.005
        self.NEURONS = [int(self.FEATURE_SIZE / 2), int(self.FEATURE_SIZE / 4)]
        self.EPOCHS = 50
        self.BATCH_SIZE = 300

## RNN implementation - LSTM

### Encoder

The first part of the RNN model consists of an LSTM with 50 hidden cells. 
The input features X are fed through the LSTM in batchs of 20 timesteps, then output back to X nodes.

After running through the LSTM, the data is reshaped and fed through an MLP hidden layer into a latent space z of X/4 nodes, 
with the reparameterization trick applied the same way as in the MLP VAE encoder. 

### Decoder

Data is sampled from this z-space with variational inference and decompressed back out to X nodes, as in the MLP VAE decoder.

The outputs here are then reshaped and fed through another LSTM with the same parameters as in the encoder,
then finally output back into the feature space.

### The rest

The model, using the TensorFlow AdamOptimizer, optimizes over two loss functions combined:
a MSE calculated between the inputs and outputs, and a latent loss function characterized by KL-Divergence. 

Other network configurations that are currently optimized through testing for datasets of 9 features:

        self.FEATURE_SIZE = X_train.shape[2]
        self.LEARNING_RATE = 0.0015
        self.NEURONS = int(self.FEATURE_SIZE / 4)
        self.LSTM_HIDDEN = 50
        self.EPOCHS = 250
        self.NUM_PERIODS = 20
        self.N_OUTPUTS = self.FEATURE_SIZE

---------------------------------------------------------------------------------------------------------

## Things that were tried

Most things tried involved iterative blackbox testing of the model configurations through parameter tuning.
Current models using roughly tuned parameters that produce fairly realistic results, but there should be room
for further improvements

One notable unsolved problem involves the data structure of the code in the neural network file.
If the model training function is kept and called within the same class as the model, the results produced
are incredibly unrealistic with huge deviations in mean and std dev. 
This is found in the archived python filed main_22.py. Further analysis is needed to find out the cause of this.

Anything else may be found in the VAEupdate word docs in this repository.

---------------------------------------------------------------------------------------------------------

## Next steps & things to try

### Things to do on current code
*   Improve LSTM VAE - Results for a few assets in each dataset are skewed relative to others. See RNN VAE results figures.
*   Generate decoded outputs for entire dataset, not just the test  (Did on a previous iteration with different dataset - look within archive)
*   Output the decoded price data to .csv
*   Test generated .csv data with robust trading model for training to analyse its effects
*   Improve documentation of results

### Things to try with current code
*   Maximum Mean Discrepancy VAE: https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/ 
*   Separate Encoder and Decoder into individual classes (might solve previous problem of trying to include model training within the same class)
*   Combine VAE with GAN 
*   VAE with latent constraints (Basically VAE with CGAN): https://colab.research.google.com/notebooks/latent_constraints/latentconstraints.ipynb 
*   Sinkhorn Autoencoder (?)

### Implementations to try
*   CNN implementation: use pictures of price charts, map output price charts to real price values
*   RNN implementation: Use GRU and compare results with LSTM
*   Google Deepmind's RNN VAE: https://github.com/snowkylin/rnn-vae

---------------------------------------------------------------------------------------------------------
