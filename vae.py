import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from losses import kl_divergence, log_diag_mvn


class VAE(tf.keras.Model):
    """
    VAE
    --------------------------
    This class is the main Variational Autoencoder model. It inherit from tf.keras.Model. This class builds and combine 
    the Encoder and Decoder. With these, VAE uses probability distributions instead of fixed values and this allows VAE 
    to generate new images by sampling. Then calculate and return the loss function and perform training steps.

    Methods:
    1) __init__: Initializes the VAE depending on the dataset that is going to be used. Have instance variables for 
                Encoder and Decoder.
    @param dset: The dataset name. It shoud be either "mnist_bw" or "mnist_color". Default value is "mnist_bw"

    2) cal: Get mu and std of q(z|x). Calculate z where eps ~ Normal(0, I) and return z.
    @param (mu, std): "mu" is mean of q(z|x) and "std" is standard deviation of q(z|x)
    @return v: latent space z, which is sample from q(z|x) or regenrated x from latent z.

    3) call: Perform full forward pass of the VAE which is x → encoder → z → decoder → regenerated x. And computes 
            negative loss function and return it.
    @param x: Input mini batch(x) of images.
    @return loss: Negative loss function.

    4) train: Perform one training step using gradient descent.
    @param x: Input mini batch(x) of images.
    @param optimizer: Optimizer used for updating the weights.
    @return loss: The computed VAE loss for the batch.
    """
    
    def __init__(self, dset="mnist_bw", ):
        super().__init__()
        
        self.encoder = Encoder(dset)
        self.decoder = Decoder(dset)


    @tf.function
    def cal(self, mu, std):
        eps = tf.random.normal(mu.shape, dtype=mu.dtype)
        v = mu + eps*std
        return v


    @tf.function
    def call(self, x):
        e_mu, e_std = self.encoder(x)        
        z = self.cal(e_mu, e_std)

        d_mu, d_std = self.decoder(z)
        d_x = self.cal(d_mu, d_std)

        kl = kl_divergence(e_mu, tf.math.log(e_std)*2)
        ld = log_diag_mvn(x, d_x, tf.math.log(d_std))

        loss = tf.reduce_mean(ld - kl)

        return -loss


    @tf.function
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            loss = self.call(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss