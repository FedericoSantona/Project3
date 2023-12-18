import os
import sys
import zipfile
import numpy as np
from tensorflow import keras as tfk
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import datasets, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


# print python version
print("Python VERSION:", sys.version)

# Path to the zip file and extraction directory
zip_path = r"C:\Users\sanfe\OneDrive\Desktop\Machine Learning\Project 3\archive.zip"
extraction_path = r"C:\Users\sanfe\OneDrive\Desktop\Machine Learning\Project 3\dataset"

# Unzip the dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# Set up directories
train_dir = os.path.join(extraction_path, 'train')
test_dir = os.path.join(extraction_path, 'test')

# Define image dimensions
img_width, img_height = 48, 48  # You can adjust this based on your specific dataset
#batch_size = 26 # You can adjust this based on your specific needs

# Load images using image_dataset_from_directory
train_dataset = tfk.utils.image_dataset_from_directory(
    train_dir,
    color_mode='grayscale', # or 'rgb' for colored images
    label_mode='categorical',  # or 'binary' if you have two classes
    #batch_size=batch_size,
    image_size=(img_width, img_height))

test_dataset = tfk.utils.image_dataset_from_directory(
    test_dir,
    color_mode='grayscale', # or 'rgb' for colored images
    label_mode='categorical',  # or 'binary' if you have two classes
    #batch_size=batch_size,
    image_size=(img_width, img_height))

# Data preprocessing and augmentation
# You can add your preprocessing layers here
preprocessing_model = tfk.Sequential([
    tfk.layers.Rescaling(1./255),  # Rescale pixel values
    # Add any additional preprocessing layers here
])

# Apply the preprocessing to the dataset
train_dataset = train_dataset.map(lambda x, y: (preprocessing_model(x), y))
test_dataset = test_dataset.map(lambda x, y: (preprocessing_model(x), y))
#train_dataset = train_dataset.take(200)
#test_dataset = test_dataset.take(40)

# To get the labels from the train dataset
train_labels = []
for _, labels in train_dataset:
    train_labels.extend(labels.numpy())

# Do the same for the test dataset
test_labels = []
for _, labels in test_dataset:
    test_labels.extend(labels.numpy())

# Now train_labels and test_labels contain the labels for their respective datasets

#Split the images from their target
x_train_list = []
t_train_list = []

# Iterate through the training dataset
for images, labels in train_dataset:
    # Append images and labels to the respective lists
    x_train_list.append(images.numpy())
    t_train_list.append(labels.numpy())

# Convert lists to numpy arrays
x_train = np.concatenate(x_train_list, axis=0)
t_train = np.concatenate(t_train_list, axis=0)

# Do the same for the test dataset
x_test_list = []
t_test_list = []

for images, labels in test_dataset:
    x_test_list.append(images.numpy())
    t_test_list.append(labels.numpy())

x_test = np.concatenate(x_test_list, axis=0)
t_test = np.concatenate(t_test_list, axis=0)


# Plot frequency distribution

class_indices = np.argmax(t_train, axis=1)
classes = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

# Count the occurrences of each class index
class_counts = np.bincount(class_indices)
categories = np.arange(len(class_counts))  # This will be your x-axis (the number of the category)
freq_classes = dict()
for i in categories:
    freq_classes[str(classes[i])] = class_counts[i]
print(freq_classes)
print(sum(freq_classes.values()))

# Plot the number of images per category
plt.figure(figsize=(10, 6))
plt.bar(freq_classes.keys(), freq_classes.values(), color='skyblue')
plt.xlabel('Category')
plt.ylabel('Number of Images')
plt.title('Number of Images per Category - Training Set')
#plt.xticks()
plt.show()

# CNN with different number of layers

def different_layers(num_classes, batch_size_gd, l2_lambda, n_epochs, dropout):
    """
    Creates and trains multiple models with 1, 2 and 3 number of convolution-pooling-layer pairs.
    
    Args:
        num_classes (int): Number of classes in the output layer.
        batch_size_gd (int): Batch size for gradient descent.
        l2_lambda (float): Lambda value for L2 regularization.
        n_epochs (int): Number of epochs for training.
        dropout (str): Whether to apply dropout or not. Should be either "True" or "False".
    
    Returns:
        list: List of training histories for each model.
    """

    histories = []
    for i in range(3):
        model = Sequential()
        # Convolutional Layer 1
        model.add(layers.Conv2D(32, (3, 3),
                                activation='relu',
                                input_shape=(img_width, img_width, 1)))
        # Max Pooling Layer 1
        model.add(layers.MaxPooling2D((2, 2)))
        
        if i == 1 or i == 2: 
            # Convolutional Layer 2
            model.add(layers.Conv2D(64, (3, 3)
                                    , activation='relu'))

            # Max Pooling Layer 2
            model.add(layers.MaxPooling2D((2, 2)))

            if i == 2:
                # Convolutional Layer 3
                model.add(layers.Conv2D(64, (3, 3)
                                        , activation='relu'))

                # Max Pooling Layer 3
                model.add(layers.MaxPooling2D((2, 2)))

        if dropout == "True":
            model.add(layers.Dropout(0.5))

        # Flattening Layer
        model.add(layers.Flatten())

        # Fully Connected Layer 1 with L2 Regularization
        model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))

        # Fully Connected Layer 2 with L2 Regularization
        model.add(layers.Dense(7, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)))

        # Output Layer
        model.add(layers.Dense(num_classes, activation='softmax'))

        #display model architecture
        model.summary()

        learning_rate = 0.0001

        # Compile the model
        model.compile(optimizer='adam', 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

        # Define the checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=r"best_model.pth",  # Corrected path
            monitor = "val_loss",
            verbose = 0,
            save_best_only = False,
            save_weights_only = False,
            mode = "auto",
            save_freq="epoch",
            options=None,
            initial_value_threshold=None
        )

        # Train the model with L2 regularization
        history = model.fit(
            x_train, t_train,
            batch_size=batch_size_gd,
            epochs=n_epochs,
            validation_data=(x_test, t_test),
            callbacks=[checkpoint_callback]  # Put the callback inside a list
        )

        histories.append(history) 
    return histories


n_epochs = 100
histories_without_dropout = different_layers(num_classes = 7, batch_size_gd = 50, l2_lambda = 0.0001, n_epochs = n_epochs, dropout = "False")
histories_with_dropout = different_layers(num_classes = 7, batch_size_gd = 50, l2_lambda = 0.0001, n_epochs = n_epochs, dropout = "True")


# Plotting Accuracy
epochs = np.arange(n_epochs)
colors = ["orange", "b", "g"]
plt.figure(figsize=(10, 6))
for i, history in enumerate(histories_with_dropout, 0):
    plt.plot(epochs, history.history['val_accuracy'], label = 'Val Accuracy {} Conv Layers With Dropout'.format(i+1), color = colors[i])
for i, history in enumerate(histories_without_dropout, 0):
    plt.plot(epochs, history.history['val_accuracy'], label = 'Val Accuracy {} Conv Layers Without Dropout'.format(i+1), color = colors[i], linestyle = "dashed")
    plt.title('CNN Test Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.savefig('architecture_comparison_accuracy.png')

# Plotting Loss
epochs = np.arange(n_epochs)
colors = ["orange", "b", "g"]
plt.figure(figsize=(10, 6))
for i, history in enumerate(histories_with_dropout, 0):
    plt.plot(epochs, history.history['val_loss'], label = 'Val Loss {} Conv Layers With Dropout'.format(i+1), color = colors[i])
for i, history in enumerate(histories_without_dropout, 0):
    plt.plot(epochs, history.history['val_loss'], label = 'Val Loss {} Conv Layers Without Dropout'.format(i+1), color = colors[i], linestyle = "dashed")
    plt.title('CNN Test Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.savefig('architecture_comparison_loss.png')