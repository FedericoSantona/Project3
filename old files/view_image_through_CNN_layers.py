import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras as tfk

def view_image_through_cnn_layers(model, image):
    """
    NOTE: ABANDONED/UNFINISHED FUNCTION
    Visualizes the activations of each layer in a CNN model for a given image.

    Args:
        model (tf.keras.Model): The CNN model.
        image (numpy.ndarray): The input image.

    Returns:
        None
    """

    print(image.shape)
    # Get the layer names
    layer_names = [layer.name for layer in model.layers]

    # Create a model that outputs the activation values for the layers we want to visualize
    layer_outputs = [layer.output for layer in model.layers[1:4]]
    activation_model = tfk.models.Model(inputs=model.input, outputs=layer_outputs)

    print(layer_names[0:-1])

    # Get the activations for the image
    activations = activation_model.predict(image)

    # Visualize the activations
    images_per_row = 16
    for layer_name, layer_activation in zip(layer_names, activations):
        # Number of features in the feature map
        n_features = layer_activation.shape[-1]

        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # Tiles the activation channels in this matrix
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # Tiles each filter into a big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                # Post-process the feature to make it visually palatable
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype("uint8")
                display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect="auto", cmap="viridis")
    plt.show()
