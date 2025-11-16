import os
import tensorflow as tf
from bicoder import BiCoder
from neural_networks import encoder_mlp, encoder_conv


class Encoder(BiCoder):
    """
    Encoder
    --------------------------
    This class is subclass of BiCoder that transforms input data x into the parametors of a probabilistic encoder q(z|x).
    As return user can get parametors mu and std. And these parametors are going to be used to sample latent vectors z
    during VAE training. Encoder class can choose different neural network architectures depending on the dataset that
    is given. For "mnist_bw", it will use "encoder_mlp" with 20 latent dimension and for "mnist_color", 
    it is going to use "encoder_conv", 50 latent dimension.

    Methods:
    
    1) __init__: Initializes the encoder and selects the latent dimension depending on the dataset that is going to be used.
    @param dset: The dataset name. It shoud be either "mnist_bw" or "mnist_color". Default value is "mnist_bw"
    
    2) call: Runs the forward pass by applying the neural network encoder with input x and do calculation to make mu and std.
            And it returns (mu, std). This method use override for "call" method in BiCoder.
    @param x: Input batch(x) of images.
    @return (mu, std): "mu" is the he mean of q(z|x) and "std" is a the standard deviation of q(z|x).
    """

    def __init__(self, dset = "mnist_bw"):
        super().__init__()
        
        if dset == "mnist_bw":
            self._encoder = encoder_mlp
            self._latent_dim = 20
        elif dset == "mnist_color":
            self._encoder = encoder_conv
            self._latent_dim = 50

    def call(self, x):        
        out = self._encoder(x)
        
        mu  = out[:,:self._latent_dim]
        log_var = out[:,self._latent_dim:]
        std = tf.math.exp(0.5*log_var)

        return mu, std
