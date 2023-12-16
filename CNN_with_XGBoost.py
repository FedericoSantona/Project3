import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras as tfk
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import cv2      # Only for simple rescaling of images

# Functions
def xgboost_model(x_train, t_train, x_test, t_test, max_depth=10, eta=0.3, num_class=7, n_boosts=20):
    """
    Trains an XGBoost model using the given training data and parameters.

    Parameters:
    - x_train: The training data features.
    - t_train: The training data labels.
    - x_test: The test data features.
    - t_test: The test data labels.
    - max_depth: The maximum depth of each tree in the XGBoost model. Default is 10.
    - eta: The learning rate of the XGBoost model. Default is 0.3.
    - num_class: The number of classes in the classification problem. Default is 7.
    - n_boosts: The number of boosting rounds. Default is 20.

    Returns:
    - model: The trained XGBoost model.
    """


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

def tensorflow_model(
        x_train, t_train, x_test, t_test, 
        batch_size=64, epochs=100, 
        eta = 0.0001, l2_lambda = 0.0001,
        n_filters: int = 32,
        kernel_size: tuple = (3, 3),
        pool_size: tuple = (2, 2),
        strides: tuple = (1, 1),
        dropout: float = 0.3,
        fc_neurons_1: int = 64,
        fc_neurons_2: int = 7,
        save_results = False, summary = False, seed = 42):
    """
    Trains a convolutional neural network (CNN) model using TensorFlow.

    Parameters:
    - x_train (numpy.ndarray): Training data input.
    - t_train (numpy.ndarray): Training data target.
    - x_test (numpy.ndarray): Testing data input.
    - t_test (numpy.ndarray): Testing data target.
    - batch_size (int): Number of samples per gradient update. Default is 64.
    - epochs (int): Number of epochs to train the model. Default is 10.
    - eta (float): Learning rate for the optimizer. Default is 0.0001.
    - l2_lambda (float): L2 regularization lambda value. Default is 0.0001.
    - save_results (bool): Whether to save the trained model. Default is False.
    - summary (bool): Whether to print the model summary. Default is False.

    Returns:
    - model (tensorflow.python.keras.engine.sequential.Sequential): Trained CNN model.
    - history (tensorflow.python.keras.callbacks.History): Training history.
    """

    # Set the seed
    tfk.utils.set_random_seed(seed)

    # Initialize CNN model
    model = tfk.Sequential([
        tfk.layers.Conv2D(n_filters, kernel_size, strides, activation="relu", input_shape=(48, 48, 1), name="conv2d_1"),
        tfk.layers.MaxPooling2D(pool_size, name="maxpooling_1"),
        tfk.layers.Conv2D(n_filters*2, kernel_size, strides = (1,1), activation="relu", name="conv2d_2"),
        tfk.layers.MaxPooling2D(pool_size, name="maxpooling_2"),
        tfk.layers.Conv2D(n_filters*2, kernel_size, strides = (1,1), activation="relu", name="conv2d_3"),
        tfk.layers.MaxPooling2D(pool_size, name="maxpooling_3"),
        tfk.layers.Dropout(dropout, name="dropout_1"),
        tfk.layers.Flatten(name="flatten_1"),
        tfk.layers.Dense(fc_neurons_1, activation="relu", kernel_regularizer=tfk.regularizers.l2(l2_lambda), name="dense_1"),
        tfk.layers.Dense(fc_neurons_2, activation="relu", kernel_regularizer=tfk.regularizers.l2(l2_lambda), name="dense_2"),
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

    
    if summary:
        # Print the model summary
        model.summary()
    else:
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
    """
    Augments an image using the specified augmentation data generator.

    Parameters:
    image (numpy.ndarray): The input image to be augmented.
    augmentation_datagen (ImageDataGenerator): The data generator used for augmentation.
    aug_count (int): The number of augmented images to generate.

    Returns:
    list: A list of augmented images.
    """
    augmented_images = []
    img = np.expand_dims(image, 0)  # Add batch dimension
    img_gen = augmentation_datagen.flow(img, batch_size=1)
    for _ in range(aug_count):
        augmented_images.append(img_gen.next()[0])  # Get the augmented image
    return augmented_images

def balance_classes(images, labels, num_samples, augmentation_datagen) -> tuple:
    """
    Balances the classes in a dataset by oversampling the minority classes and 
    undersampling the majority classes using a data augmentation generator.

    Args:
        images (numpy.ndarray): The input images.
        labels (numpy.ndarray): The corresponding labels for the images.
        num_samples (int): The desired number of samples per class after balancing.
        augmentation_datagen: The data augmentation generator used to augment the images.

    Returns:
        tuple: A tuple containing the balanced images and labels.

    """
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
    """
    Initialize the data by loading the dataset, preprocessing it, and splitting it into train and test sets.
    The dataset is assumed to be in a folder called "dataset" in the same directory as this script.
    Each image is turned into a 48x48 grayscale image and the pixel values are rescaled to be between 0 and 1.

    Args:
        batch_size (int): The batch size for loading the dataset.

    Returns:
        tuple: A tuple containing the train and test data arrays.
            - x_train (ndarray): The training data images.
            - t_train (ndarray): The training data labels.
            - x_test (ndarray): The test data images.
            - t_test (ndarray): The test data labels.
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

    return x_train, t_train, x_test, t_test

def downscale(scale_factor=6):
    """
    Downscale the images in the dataset using a specified scale factor.

    Parameters:
    scale_factor (int): The factor by which to downscale the images. Default is 6.

    Returns:
    x_train_downscaled (ndarray): The downscaled training images.
    x_test_downscaled (ndarray): The downscaled testing images.
    """

    # Reshape the data for XGBoost without CNN
    downscale_factor = scale_factor
    x_train_downscaled = np.array([cv2.resize(image, (48//downscale_factor, 48//downscale_factor), interpolation=cv2.INTER_AREA) for image in x_train])
    x_test_downscaled  = np.array([cv2.resize(image, (48//downscale_factor, 48//downscale_factor), interpolation=cv2.INTER_AREA) for image in x_test])
    x_train_downscaled = x_train_downscaled.reshape((x_train_downscaled.shape[0], x_train_downscaled.shape[1]**2))
    x_test_downscaled  = x_test_downscaled.reshape((x_test_downscaled.shape[0], x_test_downscaled.shape[1]**2))

    return x_train_downscaled, x_test_downscaled

def plot_dataset_balance(labels):
    """
    Plot the number of images per category after resampling.

    Parameters:
    labels (numpy.ndarray): The label array, assumed to be one-hot encoded.

    Returns:
    None
    """

    # Assuming t_train_balanced is your balanced label array and is one-hot encoded
    class_indices = np.argmax(labels, axis=1)

    # Count the occurrences of each class index
    class_counts = np.bincount(class_indices)
    categories = np.arange(len(class_counts))  # This will be your x-axis (the number of the category)

    # Plot the number of images per category
    plt.figure(figsize=(10, 6))
    plt.bar(categories, class_counts, color='skyblue')
    plt.xlabel('Category Number')
    plt.ylabel('Number of Images')
    plt.title('Number of Images per Category After Resampling')
    plt.xticks(categories)
    plt.show()

def plot_CNN_layer_params_scores(x_train, t_train, x_test, t_test, test_variable, n_epochs = 100, batch_size = 128, eta = 0.0001, l2_lambda = 0.0001):
    test_params = {
        'filters': [8, 16, 32, 64],
        'kernel_size': [(3, 3), (5, 5), (7, 7)],
        'pool_size': [(2, 2), (3, 3), (4, 4)],
        'strides': [(2, 2), (3, 3), (4, 4)],
        'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
        'fc_neurons_1': [64, 128, 256],
        'fc_neurons_2': [64, 128, 256],
        'color_cycle': ['r', 'g', 'b', 'm', 'c']
    }

    # Initialize figure
    fig_acc, ax_acc = plt.subplots(figsize=(10, 6))
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))

    # Initialize the model and train for all the values of the test variable
    for test_value in test_params[test_variable]:
        color = test_params['color_cycle'].pop(0)
        match test_variable:
            case "filters":
                model, history = tensorflow_model(x_train, t_train, x_test, t_test, n_filters=test_value, summary=False, epochs=n_epochs, batch_size=batch_size, eta=eta, l2_lambda=l2_lambda)
                ax_acc.plot(history.history['accuracy'], c=color, label=f"Acc: {test_variable} = {test_value}")
                ax_acc.plot(history.history['val_accuracy'], c=color, ls='--', label = f"Val. acc: {test_variable} = {test_value}")
                ax_loss.plot(history.history['loss'], c=color, label=f"Loss: {test_variable} = {test_value}")  
                ax_loss.plot(history.history['val_loss'], c=color, ls='--', label = f"Val. loss: {test_variable} = {test_value}")
                tfk.backend.clear_session()
            case "kernel_size":
                model, history = tensorflow_model(x_train, t_train, x_test, t_test, kernel_size=test_value, summary=False, epochs=n_epochs, batch_size=batch_size, eta=eta, l2_lambda=l2_lambda)
                ax_acc.plot(history.history['accuracy'], c=color, label=f"Acc: {test_variable} = {test_value}")
                ax_acc.plot(history.history['val_accuracy'], c=color, ls='--', label = f"Val. acc: {test_variable} = {test_value}")
                ax_loss.plot(history.history['loss'], c=color, label=f"Loss: {test_variable} = {test_value}")  
                ax_loss.plot(history.history['val_loss'], c=color, ls='--', label = f"Val. loss: {test_variable} = {test_value}")
                tfk.backend.clear_session()
            case "pool_size":
                model, history = tensorflow_model(x_train, t_train, x_test, t_test, pool_size=test_value, summary=False, epochs=n_epochs, batch_size=batch_size, eta=eta, l2_lambda=l2_lambda)
                ax_acc.plot(history.history['accuracy'], c=color, label=f"Acc: {test_variable} = {test_value}")
                ax_acc.plot(history.history['val_accuracy'], c=color, ls='--', label = f"Val. acc: {test_variable} = {test_value}")
                ax_loss.plot(history.history['loss'], c=color, label=f"Loss: {test_variable} = {test_value}")  
                ax_loss.plot(history.history['val_loss'], c=color, ls='--', label = f"Val. loss: {test_variable} = {test_value}")
                tfk.backend.clear_session()
            case "strides":
                model, history = tensorflow_model(x_train, t_train, x_test, t_test, strides=test_value, summary=False, epochs=n_epochs, batch_size=batch_size, eta=eta, l2_lambda=l2_lambda)
                ax_acc.plot(history.history['accuracy'], c=color, label=f"Acc: {test_variable} = {test_value}")
                ax_acc.plot(history.history['val_accuracy'], c=color, ls='--', label = f"Val. acc: {test_variable} = {test_value}")
                ax_loss.plot(history.history['loss'], c=color, label=f"Loss: {test_variable} = {test_value}")  
                ax_loss.plot(history.history['val_loss'], c=color, ls='--', label = f"Val. loss: {test_variable} = {test_value}")
                tfk.backend.clear_session()
            case "dropout":
                model, history = tensorflow_model(x_train, t_train, x_test, t_test, dropout=test_value, summary=False, epochs=n_epochs, batch_size=batch_size, eta=eta, l2_lambda=l2_lambda)
                ax_acc.plot(history.history['accuracy'], c=color, label=f"Acc: {test_variable} = {test_value}")
                ax_acc.plot(history.history['val_accuracy'], c=color, ls='--', label = f"Val. acc: {test_variable} = {test_value}")
                ax_loss.plot(history.history['loss'], c=color, label=f"Loss: {test_variable} = {test_value}")  
                ax_loss.plot(history.history['val_loss'], c=color, ls='--', label = f"Val. loss: {test_variable} = {test_value}")
                tfk.backend.clear_session()
            case "fc_neurons_1":
                model, history = tensorflow_model(x_train, t_train, x_test, t_test, fc_neurons_1=test_value, summary=False, epochs=n_epochs, batch_size=batch_size, eta=eta, l2_lambda=l2_lambda)
                ax_acc.plot(history.history['accuracy'], c=color, label=f"Acc: {test_variable} = {test_value}")
                ax_acc.plot(history.history['val_accuracy'], c=color, ls='--', label = f"Val. acc: {test_variable} = {test_value}")
                ax_loss.plot(history.history['loss'], c=color, label=f"Loss: {test_variable} = {test_value}")  
                ax_loss.plot(history.history['val_loss'], c=color, ls='--', label = f"Val. loss: {test_variable} = {test_value}")
                tfk.backend.clear_session()
            case "fc_neurons_2":
                model, history = tensorflow_model(x_train, t_train, x_test, t_test, fc_neurons_2=test_value, summary=False, n_epochs=n_epochs, batch_size=batch_size, eta=eta, l2_lambda=l2_lambda)
                ax_acc.plot(history.history['accuracy'], c=color, label=f"Acc: {test_variable} = {test_value}")
                ax_acc.plot(history.history['val_accuracy'], c=color, ls='--', label = f"Val. acc: {test_variable} = {test_value}")
                ax_loss.plot(history.history['loss'], c=color, label=f"Loss: {test_variable} = {test_value}")  
                ax_loss.plot(history.history['val_loss'], c=color, ls='--', label = f"Val. loss: {test_variable} = {test_value}")
                tfk.backend.clear_session()
            case _:
                print("Invalid test variable.")
                return None

    # Plot the results
    ax_acc.set_title(f"Accuracy vs. Epochs for Different {test_variable} Values", fontsize=16)
    ax_acc.set_xlabel("Epochs", fontsize=14)
    ax_acc.set_ylabel("Accuracy", fontsize=14)
    ax_acc.legend(fontsize=10, loc='best')
    fig_acc.savefig(f"acc_{test_variable}_.png")

    ax_loss.set_title(f"Loss vs. Epochs for Different {test_variable} Values", fontsize=16)
    ax_loss.set_xlabel("Epochs", fontsize=14)
    ax_loss.set_ylabel("Loss", fontsize=14)
    ax_loss.legend(fontsize=10, loc='best')
    fig_loss.savefig(f"loss_{test_variable}_.png")

def main():
    # Load the CNN model and predict
    tf_model, tf_history = tensorflow_model(x_train, t_train, x_test, t_test, eta = learning_rate, l2_lambda=l2_lambda, 
                                            batch_size=batch_size, epochs=n_epochs, save_results=False, summary=False)
    cnn_predict = tf_model.predict(x_test)
    cnn_score = accuracy_score(np.argmax(t_test, axis=1), np.argmax(cnn_predict, axis=1))
    print(f"\nTensorFlow CNN accuracy: {100*cnn_score:.2f} %")

    # Initialize the xgboost model
    xgboost_layer_model = tfk.models.Model(inputs=tf_model.input, 
                                           outputs=tf_model.get_layer("dense_1").output)
    xgboost_layer_output_train = xgboost_layer_model.predict(x_train)
    xgboost_layer_output_test = xgboost_layer_model.predict(x_test)

    # xgb_model = xgboost_model(xgboost_layer_output, np.argmax(t_train, axis=1), X_test_CNN, np.argmax(t_test, axis=1))
    xgb_model = xgboost_model(xgboost_layer_output_train, np.argmax(t_train, axis=1), xgboost_layer_output_test, np.argmax(t_test, axis=1))
    xgb_predict = xgb_model.predict(xgb.DMatrix(xgboost_layer_output_test))

    # Load XGBoost without CNN
    xgb_model_pure = xgboost_model(x_train_downscaled, 
                                   np.argmax(t_train, axis=1), 
                                   x_test_downscaled, 
                                   np.argmax(t_test, axis=1))
    xgb_predict_pure = xgb_model_pure.predict(xgb.DMatrix(x_test_downscaled))
    xgb_score_pure = accuracy_score(np.argmax(t_test, axis=1), np.argmax(xgb_predict_pure, axis=1))
    print(f"\nXGBoost without CNN accuracy: {100*xgb_score_pure:.2f} %")

    # Initialize the xgboost model
    for xgb_layer in xgboost_layer_list:
        xgboost_layer_model = tfk.models.Model(inputs=tf_model.input, 
                                            outputs=tf_model.get_layer(xgb_layer).output)
        xgboost_layer_output_train = xgboost_layer_model.predict(x_train)
        xgboost_layer_output_test = xgboost_layer_model.predict(x_test)

        # xgb_model = xgboost_model(xgboost_layer_output, np.argmax(t_train, axis=1), X_test_CNN, np.argmax(t_test, axis=1))
        xgb_model = xgboost_model(xgboost_layer_output_train, np.argmax(t_train, axis=1), xgboost_layer_output_test, np.argmax(t_test, axis=1))
        xgb_predict = xgb_model.predict(xgb.DMatrix(xgboost_layer_output_test))

        xgb_score = accuracy_score(np.argmax(t_test, axis=1), np.argmax(xgb_predict, axis=1))
        print(f"\nXGBoost\n\tLayer: {xgb_layer}\n\t Accuracy: {100*xgb_score:.2f} %")

    # Plot results
    plt.plot(tf_history.history['accuracy'], label='accuracy')
    plt.plot(tf_history.history['val_accuracy'], label = 'val_accuracy')
    plt.title('CNN accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(tf_history.history['loss'], label='loss')
    plt.plot(tf_history.history['val_loss'], label = 'val_loss')
    plt.title('CNN cross-entropy Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    # Parameters and variables
    seed = 42
    learning_rate = 0.001
    l2_lambda = 0.0001
    n_epochs = 50
    batch_size = 128
    target_num_images = 1000  # Set your target number of images per category
    xgboost_layer_list = ["dense_1", "dense_2", "output"]

    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    n_classes = len(labels)

    # Initialize the data
    x_train, t_train, x_test, t_test = initialize_data(batch_size)
    x_train_downscaled, x_test_downscaled = downscale()     # For XGBoost without CNN

    # plot_dataset_balance(t_train)
    # grid_search_eta_lambda(n_epochs)
    plot_CNN_layer_params_scores(x_train, t_train, x_test, t_test, test_variable="filters", n_epochs=n_epochs, batch_size=batch_size, eta=learning_rate, l2_lambda=l2_lambda)
    # main()