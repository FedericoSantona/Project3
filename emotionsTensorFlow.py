import os
import zipfile
from tensorflow import keras as tfk
from CNN import *
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from collections import Counter
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Rescaling
from tensorflow.keras.regularizers import l2




# print python version
print("Python VERSION:", sys.version)


"""
Description:
------------
    CNN application to the emotions dataset, remeber to change the path 
    to the dataset in the beginning of the code.

Parameters to tune:
------------
    I  Architecture of the CNN( Number and types of layers)
    II  Dimensions and number of the kernels (height and width) used for convolution and pooling
    III stride in the convolutional layers and pooling layers, padding or no padding
    IV  Numbers of epochs and batches ( 2 types of batches are used, one for the GD(in the fit function) and one for data preprocessing)
    V   Learning rate and regularization parameter ( lambda)
    VI  Activation functions ( sigmoid, tanh, relu, leaky relu, softmax)
    VII  Number of neurons in the fully connected layers
        """

# Path to the zip file and extraction directory
zip_path = r'C:\Users\annar\OneDrive\Desktop\FYS-STK3155\Project3\archive.zip'
extraction_path = r'C:\Users\annar\OneDrive\Desktop\FYS-STK3155\Project3\dataset'

# Unzip the dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# Set up directories
train_dir = os.path.join(extraction_path, 'train')
test_dir = os.path.join(extraction_path, 'test')

# Define image dimensions
img_width, img_height = 48, 48  # You can adjust this based on your specific dataset
batch_size = 120  # You can adjust this based on your specific needs

# Load images using image_dataset_from_directory
train_dataset = tfk.utils.image_dataset_from_directory(
    train_dir,
    color_mode='grayscale',  # or 'rgb' for colored images
    label_mode='categorical',  # or 'binary' if you have two classes
    batch_size=batch_size,
    image_size=(img_width, img_height),
    shuffle=False)  # Do not shuffle here to keep labels in order

test_dataset = tfk.utils.image_dataset_from_directory(
    test_dir,
    color_mode='grayscale', # or 'rgb' for colored images
    label_mode='categorical',  # or 'binary' if you have two classes
    batch_size=batch_size,
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

# Obtain labels for all images in the dataset
all_labels = np.concatenate([y for x, y in train_dataset], axis=0)
all_images = np.concatenate([x for x, y in train_dataset], axis=0)

# Determine the least represented class
class_counts = np.sum(all_labels, axis=0)
least_represented_class = np.argmin(class_counts)
num_samples = int(class_counts[least_represented_class])

# Function to balance classes
def balance_classes(images, labels, num_samples):
    images_by_class = defaultdict(list)
    labels_by_class = defaultdict(list)

    # Separate images by class
    for i, label in enumerate(np.argmax(labels, axis=1)):
        images_by_class[label].append(images[i])
        labels_by_class[label].append(labels[i])

    # Balance classes
    balanced_images = []
    balanced_labels = []
    for label, imgs in images_by_class.items():
        indices = np.random.choice(range(len(imgs)), num_samples, replace=False)
        balanced_images.extend(np.array(imgs)[indices])
        balanced_labels.extend(np.array(labels_by_class[label])[indices])

    return np.array(balanced_images), np.array(balanced_labels)

# Balance the training   dataset
x_train_balanced, t_train_balanced = balance_classes(all_images, all_labels, num_samples)
x_test_balanced, t_test_balanced = balance_classes(all_images, all_labels, num_samples)


# Shuffle the balanced dataset
x_train_balanced, t_train_balanced = shuffle(x_train_balanced, t_train_balanced, random_state=0)
x_test_balanced, t_test_balanced = shuffle(x_test_balanced, t_test_balanced, random_state=0)

# Print shapes of the balanced dataset
print("Balanced x_train shape:", x_train_balanced.shape)
print("Balanced t_train shape:", t_train_balanced.shape)
print("Balanced x_test shape:", x_test_balanced.shape)
print("Balanced t_test shape:", t_test_balanced.shape)


# Assuming t_train_balanced is your balanced label array and is one-hot encoded
# First, convert the one-hot encoded labels back to their class indices
class_indices = np.argmax(t_train_balanced, axis=1)

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

x_train = x_train_balanced
t_train = t_train_balanced
x_test = x_test_balanced
t_test = t_test_balanced


num_classes = 7
lambda_val = 0.00001

model = models.Sequential()

# Convolutional Layer 1
model.add(layers.Conv2D(1, (3, 4),
                         activation='relu',
                           input_shape=(img_width, img_width, 1)))

# Max Pooling Layer 1
#model.add(layers.MaxPooling2D((2, 2)))

# Convolutional Layer 2
model.add(layers.Conv2D(1, (3, 4)
                        , activation='relu'))

# Max Pooling Layer 2
#model.add(layers.MaxPooling2D((2, 2)))

# Convolutional Layer 3
#model.add(layers.Conv2D(64, (3, 3)
 #                       , activation='relu'))

# Max Pooling Layer 3
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Dropout(0.3))

# Flattening Layer
model.add(layers.Flatten())

# Fully Connected Layer 1 with L2 Regularization
model.add(layers.Dense(30, activation='relu', kernel_regularizer=regularizers.l2(lambda_val)))
#model.add(layers.Dropout(0.5))

# Fully Connected Layer 2 with L2 Regularization
model.add(layers.Dense(20, activation='relu', kernel_regularizer=regularizers.l2(lambda_val)))

# Output Layer
model.add(layers.Dense(num_classes, activation='softmax'))

#display model architecture
model.summary()


# Set your desired learning rate
learning_rate = 0.1  # Example learning rate

# Instantiate an Adam optimizer with the desired learning rate
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=adam_optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=r"C:\Users\annar\OneDrive\Desktop\FYS-STK3155\Project3\best_model.pth",
    monitor="val_accuracy",
    verbose=0,
    save_best_only=True, 
    save_weights_only=False,
    mode="auto",
    save_freq="epoch"
    # Removed the options parameter
)

n_epochs = 100
batch_size_gd = 10


# Train the model with L2 regularization
history = model.fit(
    x_train, t_train,
    batch_size=batch_size_gd,
    epochs=n_epochs,
    validation_data=(x_test, t_test),
    callbacks=[checkpoint_callback]  # Put the callback inside a list
)


epochs = np.arange(n_epochs)

plt.plot(epochs , history.history['accuracy'], label='accuracy')
plt.plot(epochs, history.history['val_accuracy'], label = 'val_accuracy')
plt.title('CNN accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Load the best saved model
best_model = load_model('best_model.pth')

def convert_to_one_hot(predictions):
    # Create a zero matrix with the same shape as the predictions
    one_hot_predictions = np.zeros_like(predictions)
    # Find the index of the max value in each row and set that to 1
    one_hot_predictions[np.arange(len(predictions)), predictions.argmax(1)] = 1
    return one_hot_predictions


# Evaluate the best model on the test data
prediction = convert_to_one_hot(best_model.predict(x_test))

print("accuracy is ", best_model.evaluate(x_test, t_test)[1])


print(f"prediction: {prediction}")
print(f"prediction shape: {prediction.shape}")
print(f"test labels: {t_test}")

plt.plot(np.argmax(prediction, axis=1), np.argmax(t_test, axis=1), 'o')
plt.xlabel('Prediction')
plt.ylabel('True label')
plt.show()