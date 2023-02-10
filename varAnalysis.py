import numpy as np
import matplotlib.pyplot as plt
from VAutoEncoder import VAutoEncoder
from varTrain import load_mnist

def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels

def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i+1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i+num_images+1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()


def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()


if __name__ == "__main__" :
    autoencoder = VAutoEncoder.load("model")
    x_train, y_train, x_test, y_test = load_mnist()

    num_sample_images_to_show = 8
    sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    reconstructed_images, _ = autoencoder.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images)

    num_sample_images = 6000
    sample_images, sample_labels = select_images(x_test, y_test, num_sample_images)
    _, latent_representations = autoencoder.reconstruct(sample_images)
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)

    """Overall drawbacks concluded for vanilla AEs from above plot after training model on 10,000 datasets:
    1.  Plot is not symmetrical around origin. as there is no symmetry, when randomly sampling a point from graph, it doesnt lead to a concrete generation of a new image
    2.  Some labels take small areas, some take large. So when we want to generate some random numbers, there is not much diversity. the points more spread will turn out to be most common numbers.
    3.  gaps between color points. what happens if we sample a dot from region between 2 different color dots? the generated image will be poorly formed.
    
    Solution to above problems: Variational AutoEncoders!"""

