import os

from numpy import ndarray

from VAutoEncoder import VAutoEncoder
import numpy as np

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150
SPECTROGRAM_DIR = "D:/Danish_Study_Material/AutoEnc/Spectrograms/"
def load_fsdd(spectrogram_path):
    # to keep things simple, we will  not split data into train and test
    x_train = []
    for root, _, filenames in os.walk(spectrogram_path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            spectrogram = np.load(file_path)       # (No_Of_FrequencyBins, No_Of_frames)
            x_train.append(spectrogram)     # now all spectrograms are loaded in x train list (not yest a np array type)
    x_train = np.array(x_train)
    # as conv layer expects arrays with 3 dims, not 2. Now we need to add 3rd dim. for MNIST, it was 1 as it was grey scale, for spectrograms, it is always 3
    x_train = x_train[..., np.newaxis]  # -> (3000, 256, 64, 1)(3000 No of samples, 256 No. of bins, 64 no. of frames, 1 newaxis). This was we fool a network used to grey scale images.
    return x_train

def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAutoEncoder(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),   # 5 layers
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder

if __name__ == "__main__":
    x_train = load_fsdd(SPECTROGRAM_DIR)     # other variables are useless for now
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("varmodel")       # the name of the folder is model which will be located in the current working directory
    autoencoder2 = VAutoEncoder.load("varmodel")
    autoencoder2.summary()
