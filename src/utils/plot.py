import numpy as np
import matplotlib.pyplot as plt

def visualise_weights(weights):
    # Expecting input in (N, C, H, W)
    
    # reshape the data in (N, H, W, C)
    weights = weights.transpose(0, 2, 3, 1)

    # Normalize the weights for better visualization
    min_val = np.min(weights)
    max_val = np.max(weights)
    weights = (weights - min_val) / (max_val - min_val)

    # Number of kernels in the first layer
    num_kernels = weights.shape[0]

    # Set up the figure size depending on the number of kernels
    fig, axs = plt.subplots(nrows=int(np.sqrt(num_kernels)), ncols=int(np.sqrt(num_kernels)), figsize=(10, 10))

    # Flatten the array of axes, for easy looping
    axs = axs.flatten()

    # Plot each kernel
    for i, ax in enumerate(axs):
        # Only plot if there is a kernel to plot (important if num_kernels is not a perfect square)
        if i < num_kernels:
            # Display the kernel
            ax.imshow(weights[i])  # Indexing: [kernel_index, color_channel]
            ax.axis('off')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()

def plot_image_normalised_mm(image):
    image = image.permute(1, 2, 0)
    # Normalize the image using min max normalisation
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Plot the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def plot_image_normalised_z(image):
    image = image.permute(1, 2, 0)
    # Normalize the image using z score normalisation
    image = (image - np.mean(image)) / np.std(image)

    # Plot the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def plot_image_no_normalisation(image):
    image = image.permute(1, 2, 0)
    # Plot the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()
