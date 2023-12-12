import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras as tfk
from tensorflow.keras.callbacks import ModelCheckpoint

# Functions
def tensorflow_model(x_train, t_train, x_test, t_test, batch_size=64, epochs=10, eta = 0.0001, l2_lambda = 0.0001, save_results = None):
    """
    Trains a CNN model using TensorFlow.
    Builds the model with an architecture of 3 convolutional layers, 3 max pooling layers, 1 dropout layer, 1 flatten layer, and 3 dense layers.
    The model is compiled with the Adam optimizer, categorical crossentropy loss function, and accuracy metric.
    Then, the model is trained with the given training data and validated with the given testing data.
    If save_results is not None, the model is saved as a .h5 file with the given file name given by save_results as a string.
    Lastly, the trained model and training history are returned.

    Parameters:
    - x_train (numpy.ndarray): Training data input.
    - t_train (numpy.ndarray): Training data target.
    - x_test (numpy.ndarray): Testing data input.
    - t_test (numpy.ndarray): Testing data target.
    - batch_size (int): Number of samples per gradient update. Default is 64.
    - epochs (int): Number of epochs to train the model. Default is 10.
    - eta (float): Learning rate for the optimizer. Default is 0.0001.
    - l2_lambda (float): L2 regularization lambda value. Default is 0.0001.
    - save_results (str): File name to save the trained model. Default is None.

    Returns:
    - model (tensorflow.python.keras.engine.sequential.Sequential): Trained CNN model.
    - history (tensorflow.python.keras.callbacks.History): Training history containing score per epoch.
    """
    
    # Initialize CNN model
    model = tfk.Sequential([
        tfk.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1), name="conv2d_1"),
        tfk.layers.MaxPooling2D((2, 2), name="maxpooling_1"),
        tfk.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_2"),
        tfk.layers.MaxPooling2D((2, 2), name="maxpooling_2"),
        tfk.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_3"),
        tfk.layers.MaxPooling2D((2, 2), name="maxpooling_3"),
        tfk.layers.Dropout(0.3, name="dropout_1"),
        tfk.layers.Flatten(name="flatten_1"),
        tfk.layers.Dense(64, activation="relu", kernel_regularizer=tfk.regularizers.l2(l2_lambda), name="dense_1"),
        tfk.layers.Dense(7, activation="relu", kernel_regularizer=tfk.regularizers.l2(l2_lambda), name="dense_2"),
        tfk.layers.Dense(n_classes, activation="softmax", name="output")
    ])

    # Compile the model
    opt = tfk.optimizers.Adam(learning_rate=eta)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])    

    checkpoint_callback = ModelCheckpoint(
        filepath=r"best_model.pth",  # Corrected path
        monitor = "val_loss",
        verbose = 0,
        save_best_only = True,
        save_weights_only = False,
        mode = "auto",
        save_freq="epoch",
        options=None,
        initial_value_threshold=None
    )
    # Train the model
    history = model.fit(
        x_train, t_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, t_test),
        callbacks=[checkpoint_callback]  # Put the callback inside a list
    )

    if save_results is not None:
        # Save the model
        model.save(f"{save_results}.h5")

    return model, history

def initialize_data(batch_size) -> tuple:
    """
    Initialize the data by loading and preprocessing the dataset as training and test data separately in gray scale and categorical labels.
    The data is then rescaled to values between 0 and 1. Lastly, the data is split into images and labels as numpy arrays.
    The images and labels of the training and testing data are then returned as four numpy arrays in a tuple in the format (x_train, t_train, x_test, t_test).

    Args:
        batch_size (int): The batch size for training and testing.

    Returns:
        tuple: A tuple containing the training and testing data as numpy arrays.
            The tuple is in the format (x_train, t_train, x_test, t_test).
            - x_train (numpy.ndarray): The training images.
            - t_train (numpy.ndarray): The training labels.
            - x_test (numpy.ndarray): The testing images.
            - t_test (numpy.ndarray): The testing labels.
    """

    # Load the dataset from the dataset folder
    train_dataset = tfk.preprocessing.image_dataset_from_directory(
        "dataset/train",
        image_size=(48, 48),
        batch_size=batch_size,
        label_mode="categorical",
        color_mode="grayscale"
    )
    test_dataset = tfk.preprocessing.image_dataset_from_directory(
        "dataset/test",
        image_size=(48, 48),
        batch_size=batch_size,
        label_mode="categorical",
        color_mode="grayscale"
    )

    # Preprocess the dataset
    preprocessing_model = tfk.Sequential([
        tfk.layers.experimental.preprocessing.Rescaling(1./255)
    ])
    train_dataset = train_dataset.map(lambda x, y: (preprocessing_model(x), y))
    test_dataset = test_dataset.map(lambda x, y: (preprocessing_model(x), y))
    
    # Split the dataset into images and labels as numpy arrays
    x_train_list = []
    t_train_list = []
    for images, labels in train_dataset:
        x_train_list.append(images.numpy())
        t_train_list.append(labels.numpy())
    x_train = np.concatenate(x_train_list, axis=0)
    t_train = np.concatenate(t_train_list, axis=0)

    x_test_list = []
    t_test_list = []
    for images, labels in test_dataset:
        x_test_list.append(images.numpy())
        t_test_list.append(labels.numpy())
    x_test = np.concatenate(x_test_list, axis=0)
    t_test = np.concatenate(t_test_list, axis=0)

    return x_train, t_train, x_test, t_test


def grid_search_3D(axes, n_epochs):
    """
    Perform a 3D grid search for hyperparameter tuning in a convolutional neural network.
    A 3D ndarray is created from the lengths of the axes, and used to store the scores. The score of each combination of hyperparameters is then calculated. 
    The best score and their indices are updated if the next 3D grid point scored better than the last.
    Lastly, the scores and their indices are returned.

    Args:
        axes (tuple): A tuple containing the axes for the grid search. Each axis represents a hyperparameter.
                      The order of the axes should be (eta_arr, lmbda_arr, batch_arr).
        n_epochs (int): The number of epochs to train the neural network for.

    Returns:
        tuple: A tuple containing the following elements:
            - score_acc (ndarray): A 3D array containing the accuracy scores for each combination of hyperparameters.
            - score_loss (ndarray): A 3D array containing the loss scores for each combination of hyperparameters.
            - highest_acc (float): The highest accuracy score achieved during the grid search.
            - lowest_loss (float): The lowest loss score achieved during the grid search.
            - highest_acc_index (tuple): The indices of the hyperparameters that achieved the highest accuracy score.
            - lowest_loss_index (tuple): The indices of the hyperparameters that achieved the lowest loss score.
    """

    # Define dimensions
    eta_arr, lmbda_arr, batch_arr = axes
    Nx, Ny, Nz = len(eta_arr), len(lmbda_arr), len(batch_arr)
    
    # Calculate grid scores
    score_acc = np.zeros((Nx, Ny, Nz))
    highest_acc = 0
    highest_acc_index = (0, 0, 0)
    score_loss = np.zeros((Nx, Ny, Nz))
    lowest_loss = 100
    lowest_loss_index = (0, 0, 0)

    n_iter = 1
    for i, eta in enumerate(eta_arr):
        for j, lmbda in enumerate(lmbda_arr):
            for k, batch_size in enumerate(batch_arr):
                print(f"\nIteration: {n_iter}/{Nx * Ny * Nz}")
                n_iter += 1
                
                x_train, t_train, x_test, t_test = initialize_data(batch_size)
                tf_model, tf_history = tensorflow_model(x_train, t_train, x_test, t_test, eta = eta, l2_lambda=lmbda, batch_size=batch_size, epochs=n_epochs, save_results=f"CNN_eta{i}_lambda{j}_batch{k}")
                cnn_predict = tf_model.predict(x_test)
                cnn_accuracy = tf_history.history["val_accuracy"][-1]
                cnn_loss = tf_history.history["val_loss"][-1]
                score_acc[i, j, k] = cnn_accuracy
                score_loss[i, j, k] = cnn_loss
                if cnn_accuracy > highest_acc:
                    highest_acc = cnn_accuracy
                    highest_acc_index = (i, j, k)
                if cnn_loss < lowest_loss:
                    lowest_loss = cnn_loss
                    lowest_loss_index = (i, j, k)
    
    return score_acc, score_loss, highest_acc, lowest_loss, highest_acc_index, lowest_loss_index
    
def plot_3D_grid(score_acc, score_loss, axes, highest_acc, lowest_loss, highest_acc_index, lowest_loss_index, plot_name = None):
    """
    Plots a 3D grid of accuracy and loss values. 
    The accuracy and loss values are plotted as heat maps, while the best accuracy and loss values are annotated.

    Parameters:
    - score_acc (ndarray): Array of accuracy values.
    - score_loss (ndarray): Array of loss values.
    - axes (tuple): Tuple of arrays representing the axes values (eta_arr, lmbda_arr, batch_arr).
    - highest_acc (float): Highest accuracy value.
    - lowest_loss (float): Lowest loss value.
    - highest_acc_index (tuple): Indices of the highest accuracy value in the grid.
    - lowest_loss_index (tuple): Indices of the lowest loss value in the grid.
    - plot_name (str): Name of the plot (optional).

    Returns:
    - None
    """    

    # Unpack and define dimensions
    eta_arr, lmbda_arr, batch_arr = axes
    Nx, Ny, Nz = len(eta_arr), len(lmbda_arr), len(batch_arr)
    
    #X, Y, Z = np.meshgrid(eta_arr, lmbda_arr, batch_arr)
    X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), np.arange(Nz))

    # Plot accuracy heat map
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X, Y, Z, c=score_acc, s=50, cmap='plasma')
    ax.xaxis.set_ticks(np.arange(Nx))
    ax.yaxis.set_ticks(np.arange(Ny))
    ax.zaxis.set_ticks(np.arange(Nz))
    ax.xaxis.set_ticklabels(np.log10(eta_arr), fontsize=11)
    ax.yaxis.set_ticklabels(np.log10(lmbda_arr), fontsize=11)
    ax.zaxis.set_ticklabels(batch_arr, fontsize=11)
    ax.set_xlabel(r"Learning rate, $\log(\eta)$", fontsize=14)
    ax.set_ylabel(r"Regularization, $\log(\lambda)$", fontsize=14)
    ax.set_zlabel(r"Batch size", fontsize=14)
    ax.set_title("CNN accuracy", fontsize=16)
    ax.set_box_aspect(aspect=None, zoom=0.8)
    
    # Annotate
    i, j, k = highest_acc_index
    ax.plot(X[i, j, k], Y[i, j, k], Z[i, j, k], marker='x', c='k', markersize=12, zorder=100)
    ax.text(X[i, j, k]+0.05, Y[i, j, k]+0.05, Z[i, j, k], f'{highest_acc:.2F}', c='k', fontsize=10, zorder=101)
    
    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.02, pad=0.1)
    cbar.set_label(label='Accuracy', size=14)
    cbar.ax.tick_params(labelsize=14)
    
    # Save figure
    plt.savefig(plot_name+"_accuracy")
    plt.close()
   
    # Plot loss heat map
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X, Y, Z, c=score_loss, s=50, cmap='plasma')
    ax.xaxis.set_ticks(np.arange(Nx))
    ax.yaxis.set_ticks(np.arange(Ny))
    ax.zaxis.set_ticks(np.arange(Nz))
    ax.xaxis.set_ticklabels(np.log10(eta_arr), fontsize=11)
    ax.yaxis.set_ticklabels(np.log10(lmbda_arr), fontsize=11)
    ax.zaxis.set_ticklabels(batch_arr, fontsize=11)
    ax.set_xlabel(r"Learning rate, $\log(\eta)$", fontsize=14)
    ax.set_ylabel(r"Regularization, $\log(\lambda)$", fontsize=14)
    ax.set_zlabel(r"Batch size", fontsize=14)
    ax.set_title("CNN loss", fontsize=16)
    ax.set_box_aspect(aspect=None, zoom=0.8)

    # Annotate
    i, j, k = lowest_loss_index
    ax.plot(X[i, j, k], Y[i, j, k], Z[i, j, k], marker='x', c='k', markersize=12, zorder=100)
    ax.text(X[i, j, k]+0.05, Y[i, j, k]+0.05, Z[i, j, k], f'{highest_acc:.2F}', c='k', fontsize=10, zorder=101)
    
    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.02, pad=0.1)
    cbar.set_label(label='Loss', size=14)
    cbar.ax.tick_params(labelsize=14)
    
    # Save figure
    plt.savefig(plot_name+"_loss")
    plt.close()
    
def save_array(arrays: tuple, filename: str):
    """
    Save the arrays NumPy in a NumPy bytecode file (.npy).
    A meshgrid is constructed from the arrays. The arrays are then flattened, 
    concatenated and saved as a 2D array.

    Parameters:
    arrays (tuple): A tuple containing the ndarrays to be saved (x, y, z, values).
    filename (str): The name of the file to save the arrays.

    Returns:
    None
    """
    x, y, z, values = arrays
    X, Y, Z = np.meshgrid(x, y, z)
    
    print("shape before")
    print(X.shape, Y.shape, Z.shape, values.shape)
    
    values = values.ravel().reshape(-1, 1)
    X = X.ravel().reshape(-1, 1)
    Y = Y.ravel().reshape(-1, 1)
    Z = Z.ravel().reshape(-1, 1)
    
    print("shape after")
    print(X.shape, Y.shape, Z.shape, values.shape)

    XYZ = np.concatenate((X, Y, Z, values), axis=1)
    
    np.save(filename, XYZ)
    
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

eta_space = np.logspace(-1, -5, 5)
lambda_space = np.logspace(-1, -5, 5)
batch_space = np.array([8, 16, 32, 64, 128])
n_epochs = 300
target_num_images = 1000  # Set your target number of images per category
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
n_classes = len(labels)

score_acc, score_loss, highest_acc, lowest_loss, highest_acc_index, lowest_loss_index = grid_search_3D((eta_space, lambda_space, batch_space), epochs = n_epochs)
#plot_3D_grid(score_acc, score_loss, (eta_space, lambda_space, batch_space), highest_acc, lowest_loss, highest_acc_index, lowest_loss_index, plot_name = 'CNN_gridsearch_eta_lambda_batch')
save_array((eta_space, lambda_space, batch_space, score_acc), filename='CNN_score_accuracy')
save_array((eta_space, lambda_space, batch_space, score_loss), filename='CNN_score_loss')