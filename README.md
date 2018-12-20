# VAE
Variational Autoencoder for generating financial time-series data

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
The current VAE model is made up of a neural network that takes in a certain number of assets as input nodes, X. 
The data is compressed through a hidden layer of X/2 nodes, then down to a latent space layer, z, of X/4 hidden nodes. 
The mean and standard deviation is sampled from the latent distribution of z with ReLu and SoftPlus activation functions, respectively. 
(z_mean and z_stddev)

Using the reparameterization trick, a randomly initialized normal distribution with a mean of 0 and standard deviation of 1 
is multiplied with z_stddev. Adding this to z_mean gives us a "newly configured" latent space z, which sufficiently allows for
backpropagation of the neural network during training. 
[For a better explanation of the reparameterization trick: https://www.jeremyjordan.me/variational-autoencoders/]

the data is decompressed from the latent space through another hidden layer of X/2 nodes, 
to varied output features of X nodes with mean and standard deviations characteristic of the input features.


---------------------------------------------------------------------------------------------------------






