import random
import tensorflow as tf
import zipfile
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras as tfk
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Functions
def xgboost_model(x_train, t_train, x_test, t_test, max_depth=10, eta=0.3, num_class=7, n_boosts=20):
    # Convert the data to DMatrix format
    d_train = xgb.DMatrix(x_train, label=t_train)
    d_test = xgb.DMatrix(x_test, label=t_test)

    # Set the parameters for the xgboost model
    params = {
        "seed": seed,
        "max_depth": max_depth,
        "eta": eta,
        "objective": "multi:softprob",
        "num_class": num_class,
        "eval_metric": "mlogloss",
        "lambda": l2_lambda
    }

    # Train the model
    model = xgb.train(params, d_train, n_boosts)

    return model

def tensorflow_model(x_train, t_train, x_test, t_test, batch_size=64, epochs=10, eta = 0.0001, l2_lambda = 0.0001, save_results = False):
    # Set the seed
    #tfk.utils.set_random_seed(seed)

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

    if save_results:
        # Save the model
        model.save("model.h5")

    return model, history

def augment_images(image, augmentation_datagen, aug_count) -> list:
    augmented_images = []
    img = np.expand_dims(image, 0)  # Add batch dimension
    img_gen = augmentation_datagen.flow(img, batch_size=1)
    for _ in range(aug_count):
        augmented_images.append(img_gen.next()[0])  # Get the augmented image
    return augmented_images

def balance_classes(images, labels, num_samples, augmentation_datagen) -> tuple:
    images_by_class = defaultdict(list)
    labels_by_class = defaultdict(list)

    # Separate images by class
    for i, label in enumerate(np.argmax(labels, axis=1)):
        images_by_class[label].append(images[i])
        labels_by_class[label].append(labels[i])

    balanced_images = []
    balanced_labels = []
    for label, imgs in images_by_class.items():
        current_count = len(imgs)
        if current_count < num_samples:
            # Calculate how many augmented images are needed
            aug_count = num_samples - current_count

            # Augment images
            for img in imgs:
                augmented_images = augment_images(img, augmentation_datagen, aug_count // current_count)
                balanced_images.extend(augmented_images)

            # Extend the original images and labels
            balanced_images.extend(imgs)
            balanced_labels.extend(labels_by_class[label] * (1 + aug_count // current_count))

            # Add additional augmented images if necessary
            additional_count = num_samples - len(balanced_images)
            if additional_count > 0:
                extra_augmented_images = augment_images(imgs[0], augmentation_datagen, additional_count)
                balanced_images.extend(extra_augmented_images)
                balanced_labels.extend([labels_by_class[label][0]] * additional_count)
        else:
            # Randomly select images if there are too many
            indices = np.random.choice(range(current_count), num_samples, replace=False)
            balanced_images.extend(np.array(imgs)[indices])
            balanced_labels.extend(np.array(labels_by_class[label])[indices])

    return np.array(balanced_images), np.array(balanced_labels)

def initialize_data(batch_size) -> tuple:
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

    # Augmentation setup
    augmentation_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
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

    # Balance the classes
    # x_train, t_train = balance_classes(x_train, t_train, target_num_images, augmentation_datagen)
    # x_test, t_test = balance_classes(x_test, t_test, target_num_images, augmentation_datagen)

    # Shuffle the data
    # x_train, t_train = shuffle(x_train, t_train, random_state=seed)
    # x_test, t_test = shuffle(x_test, t_test, random_state=seed)

    return x_train, t_train, x_test, t_test


def grid_search_3D(axes, epochs, plot_name = None, mode = 'loss'):
    
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

    for i, eta in enumerate(eta_arr):
        for j, lmbda in enumerate(lmbda_arr):
            for k, batch_size in enumerate(batch_arr):
                n_iter = 1
                tot_iter = Nx + Ny + Nz
                print(f"\nIteration: {n_iter}/{tot_iter}")
                n_iter += 1
                
                x_train, t_train, x_test, t_test = initialize_data(batch_size)
                tf_model, tf_history = tensorflow_model(x_train, t_train, x_test, t_test, eta = eta, l2_lambda=lmbda, batch_size=batch_size, epochs=n_epochs, save_results=False)
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
    
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

eta_space = np.logspace(-1, -5, 5)
lambda_space = np.logspace(-1, -5, 5)
batch_space = [8, 16, 32, 64, 128]
n_epochs = 300
target_num_images = 1000  # Set your target number of images per category
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
n_classes = len(labels)

grid_search_3D((eta_space, lambda_space, batch_space), epochs = n_epochs, plot_name='CNN_gridsearch_eta_lambda_batch')