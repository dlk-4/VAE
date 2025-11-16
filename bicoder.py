from tensorflow.keras import layers


class BiCoder(layers.Layer):
    """
    BiCoder
    --------------------------
    Base super class for both encoder and decoder in a Variational Autoencoder (VAE). 
    Therefore, here you can find the call method that encoders and decoders is going to Inherited
    This class inherit from tensorflow.keras.layers and Encoder and Decoder class is going to inherit from this 
    class so it is possible to make clean interface that works with tensorflow.keras.layers. This class 
    should not be used directly from user. Instead, you can create subclass "Encoder" and "Decoder" that extend 
    "BiCoder" and need to use override BiCoder's method.
    
    Methods:
    1) __init__: Sets up the base BiCoder layer. In this method, it simply calls the parent constructor from 
                "layers.Layer". 
    
    2) call:
    This method gives a hint for user to override this method for supclass that is going to inherit this super class.
    With this "call" method, it is possible to defines the forward pass of the layer. This method is made because of
    defining a common interface that is going to be shared across sub class.
    """
    def __init__(self):
        super().__init__()
    
    def call(self):
        print("Override this call method.")