from tensorflow.keras.datasets import mnist
from main import AutoEncoder

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 20

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalization and adding extra dimension for channel. Grey scale images lack the dimension specifying single color image
    x_train = x_train.astype("float32") / 255   # highest pixel value is 255
    x_train = x_train.reshape(x_train.shape + (1, ))     # New format :  [channels][rows][cols].

    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1, ))

    return x_train, y_train, x_test, y_test

def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = AutoEncoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder

if __name__ == "__main__":
    x_train, _, _, _ = load_mnist()     # other variables are useless for now
    autoencoder = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")       # the name of the folder is model which will be located in the current working directory
    autoencoder2 = AutoEncoder.load("model")
    autoencoder2.summary()
