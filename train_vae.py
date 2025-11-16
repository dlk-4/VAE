"""
train_vae.py
--------------------------
This is the script for training and testing the Variational Autoencoder. User have to use the command line
using argparse. Hence, it is possible to change the dataset, number of epochs, and 3 different options for visualization. 

Process:
1. Build the VAE model (Encoder + Decoder).
2. Train the VAE on the chosen dataset.
3. (Optional) Visualize the latent space with t-SNE.
4. (Optional) Generate images by sampling from the p(z).
5. (Optional) Generate images by sampling from the q(z|x).

Command line arguments:
1) --dset:
    Type: str
    Default: "mnist_bw"
    Description: Selects the dataset and architecture:
                - mnist_bw (latent_dim = 20)
                - mnist_color (latent_dim = 50)

2) --epochs:
    Type: int
    Default: 20
    Description: Number of training epochs. 

3) --num_picture:
    Type: int
    Default: 1
    Description: Number of image grids to generate when using visualizing options(generate_from_prior, generate_from_posterior). 
                For each grid, the code generates 100 images.

4) --visualize_latent:
    Type: Boolean
    Default: False
    Description: If this option is true, the script will encode the test data into latent means and use TSNE 
                to reduce the latent dimension to 2D. Then save a scatter plot of the latent space in pdf file.

5) --generate_from_prior:
    Type: Boolean
    Default: False
    Description: If this option is true, it sample z from the prior distribution p(z). Pass these latent vectors 
                through the decoder. Then Generate images and save it into pdf file.
        

6) --generate_from_posterior:
    Type: Boolean
    Default: False
    Description: If this option is true, encode batches from the test set and sample z. Then decode z back into images.
                Save grids of generated images into pdf file.
"""

from vae import VAE
import tensorflow as tf
from utils import plot_grid
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data_loader import DataLoader
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dset", type=str, default="mnist_bw")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_picture", type=int, default=1)
    parser.add_argument("--visualize_latent", action='store_true')
    parser.add_argument("--generate_from_prior", action='store_true')
    parser.add_argument("--generate_from_posterior", action='store_true')
    args = parser.parse_args()

    my_data_loder = DataLoader(dset=args.dset)
    model = VAE(dset=args.dset)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    tr_data = my_data_loder.get_training_data()
    te_data = my_data_loder.get_test_data()
        
    tr_data = tr_data.map(lambda x: tf.cast(x, tf.float32))


    try:
        for e in range(args.epochs):
            for i, tr_batch in enumerate(tr_data):
                # losses.py -> log_diag_mvn need float64
                loss = model.train(tr_batch , optimizer)
                print(loss)
        
        print("complete train")

    except Exception as e:
        print(e)
        

    try:
        if args.visualize_latent:
            for te_batch in te_data:
                e_mu, _ = model.encoder(te_batch)
                z = TSNE(n_components=2, perplexity=4).fit_transform(e_mu)
                
                plt.scatter(z[:, 0], z[:, 1], s = 1)     
            plt.savefig(f"latent_{args.dset}.pdf")
            plt.close()
            print("complete to save latent space")

    except Exception as e:
        print(e)        

            
    try:
        if args.generate_from_prior:
            num_picture = args.num_picture
            num_batch = 0

            for te_batch in te_data:
                if num_batch >= num_picture:
                    break
                
                if args.dset == "mnist_bw":
                    latent_dim = 20
                elif args.dset == "mnist_color":
                    latent_dim = 50
                
                z = tf.random.normal((100, latent_dim))
                d_mu, _ = model.decoder(z)

                if args.dset == "mnist_bw":
                    d_mu = tf.reshape(d_mu, (100, 28, 28, 1))            
                    plot_grid(d_mu, name = f"prior_bw_{num_batch}")        
                elif args.dset == "mnist_color":
                    d_mu = tf.reshape(d_mu, (100, 28, 28, 3))
                    plot_grid(d_mu, name = f"prior_color_{num_batch}")
                
                num_batch += 1
        
            print("complete to generate image from prior p(z)")

    except Exception as e:
        print(e)


    try:
        if args.generate_from_posterior:
            num_picture = args.num_picture
            num_batch = 0

            if args.dset == "mnist_bw":
                latent_dim = 20
            elif args.dset == "mnist_color":
                latent_dim = 50

            for te_batch in te_data:
                if num_batch >= num_picture:
                    break
                
                batch_size = te_batch.shape[0]
                
                e_mu, e_std = model.encoder(te_batch)
                z = model.cal(e_mu, e_std)
                d_mu, _ = model.decoder(z)

                if args.dset == "mnist_bw":
                    d_mu = tf.reshape(d_mu, (batch_size, 28, 28, 1))
                    plot_grid(d_mu, name = f"psterior_bw_{num_batch}")
                elif args.dset == "mnist_color":
                    d_mu = tf.reshape(d_mu, (batch_size, 28, 28, 3))
                    plot_grid(d_mu, name = f"psterior_color_{num_batch}")
                
                num_batch += 1
        
            print("complete to generate image from posterior p(z|x)")

    except Exception as e:
        print(e)
            
