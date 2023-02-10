from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, \
    Activation  # helps compose an input to the model which can be passed to the Model created
from tensorflow.keras import backend as k
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import os
import pickle


class AutoEncoder:
    """
    Autoencoder class responsible for creating CNN based mirrored Encoder/decoder part
    functions described accordingly:

    init: Helps pass down basic characteristics of the object created from this class anytime in future.
            It basically defines the "default" state of the object created.
    We will pass some important properties we want to use to create CNN within autoencoder such as:
    input_shape: shape of input
    Properties of each CNN layers:
    conv_filters (list or tuple): No of filters for each layer
    conv_kernels (list or tuple): No of kernels for each layer
    conv_strides (list or tuple): No of strides for each layer
    latent_space_dim : dimension of the bottleneck space
    We started with Encoder part, then Decoder, then the final Autoencoder part
    """

    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):

        # let's bind the attribute which may be passed down from object into the arguments defines by the class

        self.input_shape = input_shape  # e.g [28, 28, 1] 1 because of greyscale
        self.conv_filters = conv_filters  # e.g [2, 4, 8]   Note: Every filter has a dimension equal to the output from the previous layer
        self.conv_kernels = conv_kernels  # e.g [3, 5, 3]
        self.conv_strides = conv_strides  # e.g [1, 2, 2]
        self.latent_space_dim = latent_space_dim  # e.g 2

        self.encoder = None
        self.decoder = None
        self.model = None
        # Private attribute  accessible only with ObjName._P_num_conv_layers
        self._num_conv_layers = len(conv_filters)  # e.g. 1st layer has 2 filters, 2nd layer has 4 filters and so on..
        self._shape_before_bottleneck = None
        self._model_input = None
        self._build()

    # high level method _build and its body which is executed when called from above.
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    # Autoencoder Methods
    def _build_autoencoder(self):
        model_input = self._model_input  # It is a private variable that will hold encoder input. Go to _build_encoder() section
        model_output = self.decoder(
            self.encoder(model_input))  # input to encoder, who's output to decoder, who's output is final model output
        self.model = Model(model_input, model_output, name="autoencoder")

    # Decoder Methods

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)  # adding a dense layer after input
        reshape_layer = self._add_reshape_layer(
            dense_layer)  # Converting a dense (vector input) to a 3D array to be used by transpose-CNN later to which Dense layer is the input
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output,
                             name="decoder")  # Now decoder, just like encoder is a keras model

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)  # If shape was [1, 2, 3], we now have num_neurons = 6
        dense_layer = Dense(num_neurons, name="decoder_dense")(
            decoder_input)  # we want number of neurons same as ones we had in the Dense layer before bottleneck
        return dense_layer

    def _add_reshape_layer(self,
                           dense_layer):  # To go back to the dimension we had before bottleneck and even before Dense layer we talked about, above. The task is the opposite of what Flatten did
        return Reshape(self._shape_before_bottleneck)(dense_layer)  # reshape applied to Dense_layer

    def _add_conv_transpose_layers(self, x):
        """Here we append reshape layer and run a loop to add conv blocks,
         but in reverse order of encoder, Hence the word 'Transpose'
         Note: We stop at the first layer (at encoder side)"""
        for layer_index in reversed(range(1, self._num_conv_layers)):
            # say we had 3 conv layers in encoder indexed [0, 1, 2], we want to remove 0th layer and reverse the sequence, hence [2, 1]
            x = self._add_conv_transpose_layer(layer_index, x)

        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
            # indexing by default was Zero, we dont want confusion while desplaying
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_ReLU_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        """This is just another CNN but without ReLU or BatchNormalization. We instead use only a nonlinear sigmoid activation.
        Remember, We want the output format same as input, i.e, [28, 28, 1].
        As we know Number of filters  = output depth. we want it 1
        Also remember, this is the last layer, which is also equal to the 1st alyer of encoder, i.e at 0 index"""
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_layer_{self._num_conv_layers}"
            # indexing by default was Zero, we dont want confusion while displaying
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    # Encoder Methods
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()  # append skeleton of the input we plan to have.
        conv_layers = self._add_conv_layers(encoder_input)  # adding  skeleton for conv. layers structure after input.
        bottleneck = self._add_bottleneck(
            conv_layers)  # acts as an output of encoder, We passed graph of layers up until bottleneck
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")  # we passed input and output of the model. We will use the same keras model to predit bottleneck output later, remember

    def _add_encoder_input(self):
        return Input(shape=self.input_shape,
                     name="encoder_input")  # input is nothing but the input we passed to the object

    def _add_conv_layers(self, encoder_input):
        # creates Several connected blocks of conv layers starting with input
        x = encoder_input  # keeping track of already made skeleton in the past and appending to it the conv. structure skeleton which will have many small blocks of individual layers.
        for layer_index in range(self._num_conv_layers):
            """adds a block using the method '_insert_conv_layer' which is passed current layer index
             and current graph of layers 'x' it returns the graph of layers with a new layer attached"""

            x = self._insert_conv_layer(layer_index, x)

        return x

    def _insert_conv_layer(self, layer_index, x):
        """Here we create a convolution block containing Conv2D, Relu, Batch Normalization etc."""
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_index + 1}"
            # indexing by default was Zero, we don't want confusion while displaying
        )
        # Now after creating conv layer above, we need to apply it to the graph of layers which we return back
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_index + 1}")(x)  # instantiate ReLU and at the same time apply it to x
        x = BatchNormalization(name=f"encoder_bn_{layer_index + 1}")(x)
        return x

    def _add_bottleneck(self, x):
        """Our bottleneck is a dense layer. We need to first flatten the data we have got from previous layers before passing it in.
            Before we flatten it, we need to store the current output shape of the graph for future use in decoder for mirroring
        """
        self._shape_before_bottleneck = k.int_shape(x)[
                                        1:]  # int_shape is inbuilt method return is like a 4D array, [2, 7, 7, 32], i.e [batchSize(not important so sliced), Width, Height, NumOfChannels]

        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(
            x)  # Dense layer has neurons which is equal to the dimension of the bottleNeck, i.e latent Space
        return x

    # Summary function for any model section summary
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

# Compilation methods. Keras models need compilation before we can use them
    def compile(self, learning_rate=0.001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer,
                           loss=mse_loss)  # Remember, self.model here is Autoencoder hence we can apply method compile
        # .compile method used above is native to keras API

    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,  # .fit is native to keras API.
                       x_train,
                       # Note, input = output, as we want to train autoencoder to bring back output same as input after training
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)        # gives back latent space representation as that is the outpout of the original model when designed!
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    def save(self, save_folder="."):        # defaulted to working dir. "."
        self._create_folder_if_not_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def _create_folder_if_not_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(folder, "parameters.pkl")      # we want to add a path insider folder for a file called parameters.pkl
        with open(save_path, "wb") as f:        # "wb" stands for 'open file in write-binary' mode
            pickle.dump(parameters, f)
        """
        In the above code, parameters variable contains parameters to be saved in a file.
         We open the file in “wb” mode instead of “w” as all the operations are done using bytes in the current working directory.
          A new file who's name is stored in 'save_path' is created, which converts the data in the byte stream through variable f.
          """
    def _save_weights(self, folder):
        save_path = os.path.join(folder, "weights.h5")      # we want to add a path insider folder for a file called weights.h5. keras uses h5 format for saving weights
        self.model.save_weights(save_path)                  # model is our autoencoder model (trained) which has inbuilt function .save_weights(#path)

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = AutoEncoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)


if __name__ == "__main__":
    autoencoder = AutoEncoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
