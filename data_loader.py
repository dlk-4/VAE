import os
import wget
import pickle
import numpy as np
import tensorflow as tf


class DataLoader:
    """
    DataLoader
    --------------------------
    DataLoader class loads and preprocesses the mnist datasets(training data, test data, and label data) for the VAE process.
    This class fully automates 1) downloading the dataset if there is no data in the current directory 2) preprocessing images 
    based on dataset 3) creating batches.

    Methods:
    1) __init__: Initialize the DataLoader.
    @params dset: The dataset name. It shoud be either "mnist_bw" or "mnist_color". Default value is "mnist_bw".
    @params batch: Batch size. Default value is 128.
    @params d_key: Version key for the color MNIST dictionary(m0 ~ m4). Default value is "m1".

    2) download_data: Downloads all required files (train, test, labels) in the the current directory.

    3) get_training_data: Loads the training dataset. If the dataset is not in the the current directory, 
                        automatically downloads them.
    @return data: training dataset with batches.

    4) get_test_data(): Loads the testing dataset. If the dataset is not in the the current directory, 
                        automatically downloads them.
    @return data: testing dataset with batches.

    5) get_labels_data(): Loads the labels dataset. If the datasets is not in the the current directory, 
                        automatically downloads them.
    @return data: labels dataset with batches.
    """


    WEB = {"mnist_bw":{"train" : "https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=1",
                       "test" : "https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=1",
                       "labels" : "https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=1"},
           "mnist_color":{"train" : "https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=1",
                          "test" : "https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=1",
                          "labels": "https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=1"}}
    
    def __init__(self, dset="mnist_bw", batch = 128, d_key = "m1"):
        self._dset = dset
        self._batch = batch
        self._d_key = d_key

    def download_data(self):
        try:                
            if self._dset == "mnist_bw":
                for key, value in DataLoader.WEB[self._dset].items():
                    if key == "train":
                        wget.download(value, out="mnist_bw_train.npy")
                    elif key == "test":
                        wget.download(value, out="mnist_bw_test.npy")
                    elif key == "labels":
                        wget.download(value, out="mnist_bw_labels.npy")

            elif self._dset == "mnist_color":
                for key, value in DataLoader.WEB[self._dset].items():
                    if key == "train":
                        wget.download(value, out="mnist_color_train.pkl")
                    elif key == "test":
                        wget.download(value, out="mnist_color_test.pkl")
                    elif key == "labels":
                        wget.download(value, out="mnist_color_labels.npy")   
        except Exception as e:
            print(e)


    def get_training_data(self):
        try:
            if self._dset == "mnist_bw":
                if os.path.exists("mnist_bw_train.npy"):
                    print("Already downloaded")
                else:
                    self.download_data()
                    self.get_training_data()
            elif self._dset == "mnist_color":
                if os.path.exists("mnist_color_train.pkl"):
                    print("Already downloaded")
                else:
                    self.download_data()
                    self.get_training_data()
                
            if self._dset == "mnist_bw":
                tr =  np.load("mnist_bw_train.npy")
                tr = tr/255
                tr = np.reshape(tr, (tr.shape[0], -1))
                data = tf.data.Dataset.from_tensor_slices(tr).batch(self._batch)
                return data

            elif self._dset == "mnist_color":
                with open("mnist_color_train.pkl", "rb") as f:
                    tr = pickle.load(f)[self._d_key]
                    data = tf.data.Dataset.from_tensor_slices(tr).batch(self._batch)
                    return data
        except Exception as e:
            print(e)    


    def get_test_data(self):
        try:
            if self._dset == "mnist_bw":
                if os.path.exists("mnist_bw_test.npy"):
                    print("Already downloaded")
                else:
                    self.download_data()
                    self.get_training_data()
            elif self._dset == "mnist_color":
                if os.path.exists("mnist_color_test.pkl"):
                    print("Already downloaded")
                else:
                    self.download_data()
                    self.get_training_data()
                    
            if self._dset == "mnist_bw":
                te =  np.load("mnist_bw_test.npy")
                te = te/255
                te = np.reshape(te, (te.shape[0], -1))
                data = tf.data.Dataset.from_tensor_slices(te).batch(self._batch)
                return data

            elif self._dset == "mnist_color":
                with open("mnist_color_test.pkl", "rb") as f:
                    te = pickle.load(f)[self._d_key]
                    data = tf.data.Dataset.from_tensor_slices(te).batch(self._batch)
                    return data
        except Exception as e:
            print(e)

    def get_labels_data(self):
        try:
            if self._dset == "mnist_bw":
                if os.path.exists("mnist_bw_labels.npy"):
                    print("Already downloaded")
                else:
                    self.download_data()
                    self.get_training_data()
            elif self._dset == "mnist_color":
                if os.path.exists("mnist_color_labels.npy"):
                    print("Already downloaded")
                else:
                    self.download_data()
                    self.get_training_data()
            
            if self._dset == "mnist_bw":
                la =  np.load("mnist_bw_labels.npy")
                la = la/255 
                data = tf.data.Dataset.from_tensor_slices(la).batch(self._batch)
                return data
            
            elif self._dset == "mnist_color":
                la = np.load("mnist_color_labels.npy")
                data = tf.data.Dataset.from_tensor_slices(la).batch(self._batch)
                return data
        except Exception as e:
            print(e)
