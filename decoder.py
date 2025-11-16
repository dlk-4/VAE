import os
import tensorflow as tf
from bicoder import BiCoder
from neural_networks import decoder_mlp, decoder_conv


class Decoder(BiCoder):
    def __init__(self, dset = "mnist_bw"):
        """
        Decoder
        --------------------------
        The Decoder class takes a latent sample z and transforms it back into the distribution p(x|z). The parameters that made from
        "Decoder" can describe how to generate new images from the z latent space. Depending on the dataset, Decoder
        is going to choose different neural network architectures. For "mnist_bw", it will use "decoder_mlp" and for "mnist_color",
        it will use "decoder_conv". 

        Methods:
        
        1) __init__: Initializes the decoder depending on the dataset that is going to be used.
        @param dset: The dataset name. It shoud be either "mnist_bw" or "mnist_color". Default value is "mnist_bw"
        
        2) call: Runs the forward pass using decoder. Therefore, produces the output distribution parameters p(x|z).
                But in this project, it does not learn std. Instead, std is a constant 0.75. And it returns (mu, std). 
                This method use override for "call" method in "BiCoder".
        @param z: Latent vectors sampled from q(z|x) or from the prior p(z).
        @return (mu, std): "mu" is the reconstructed mean image and "std" is a fixed number 0.75, representing 
                            standard deviation of p(x|z).
        """


        super().__init__()

        if dset == "mnist_bw":
            self._decoder = decoder_mlp

        elif dset == "mnist_color":
            self._decoder = decoder_conv
    
    def call(self, z):
        mu = self._decoder(z)
        std = 0.75

        return mu, std

