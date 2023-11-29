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
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)




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
zip_path = r"C:\Users\sanfe\OneDrive\Desktop\Machine Learning\Project 3\archive.zip"
extraction_path = r"C:\Users\sanfe\OneDrive\Desktop\Machine Learning\Project 3\dataset"

# Unzip the dataset
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# Set up directories
train_dir = os.path.join(extraction_path, 'train')
test_dir = os.path.join(extraction_path, 'test')

# Define image dimensions
img_width, img_height = 48, 48
batch_size = 120

# Load images using image_dataset_from_directory
train_dataset = tfk.utils.image_dataset_from_directory(
    train_dir,
    color_mode='grayscale',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_width, img_height),
    shuffle=False)

test_dataset = tfk.utils.image_dataset_from_directory(
    test_dir,
    color_mode='grayscale',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(img_width, img_height))

# Data preprocessing and augmentation
preprocessing_model = tfk.Sequential([
    tfk.layers.Rescaling(1./255),
    # Add any additional preprocessing layers here
])

# Apply the preprocessing to the dataset
train_dataset = train_dataset.map(lambda x, y: (preprocessing_model(x), y))
test_dataset = test_dataset.map(lambda x, y: (preprocessing_model(x), y))

# Obtain labels for all images in the dataset
train_labels = np.concatenate([y for x, y in train_dataset], axis=0)
train_images = np.concatenate([x for x, y in train_dataset], axis=0)

test_labels = np.concatenate([y for x, y in test_dataset], axis=0)
test_images = np.concatenate([x for x, y in test_dataset], axis=0)

# Set your target number of images per category
target_num_images = 4000  # Adjust this number as needed

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

# Function to augment images
def augment_images(image, augmentation_datagen, aug_count):
    augmented_images = []
    img = np.expand_dims(image, 0)  # Add batch dimension
    img_gen = augmentation_datagen.flow(img, batch_size=1)
    for _ in range(aug_count):
        augmented_images.append(img_gen.next()[0])  # Get the augmented image
    return augmented_images



# Function to balance classes
def balance_classes(images, labels, num_samples, augmentation_datagen):
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


# Balance the training and test dataset
x_train_balanced, t_train_balanced = balance_classes(train_images, train_labels, target_num_images, augmentation_datagen)
#x_test_balanced, t_test_balanced = balance_classes(all_images, all_labels, target_num_images, augmentation_datagen)

# Shuffle the balanced dataset
x_train_balanced, t_train_balanced = shuffle(x_train_balanced, t_train_balanced, random_state=0)
#x_test_balanced, t_test_balanced = shuffle(x_test_balanced, t_test_balanced, random_state=0)

# Print shapes of the balanced dataset
print("Balanced x_train shape:", x_train_balanced.shape)
print("Balanced t_train shape:", t_train_balanced.shape)

# Assuming t_train_balanced is your balanced label array and is one-hot encoded
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
x_test = test_images
t_test = test_labels


num_classes = 7
lambda_val = 0.00001

model = models.Sequential()

# Convolutional Layer 1
model.add(layers.Conv2D(32, (3, 3),
                         activation='relu',
                           input_shape=(img_width, img_width, 1)))

# Max Pooling Layer 1
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional Layer 2
model.add(layers.Conv2D(64, (3, 3)
                        , activation='relu'))

# Max Pooling Layer 2
model.add(layers.MaxPooling2D((2, 2)))

# Convolutional Layer 3
model.add(layers.Conv2D(64, (3, 3)
                        , activation='relu'))

# Max Pooling Layer 3
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))

# Flattening Layer
model.add(layers.Flatten())

# Fully Connected Layer 1 with L2 Regularization
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(lambda_val)))
#model.add(layers.Dropout(0.5))

# Fully Connected Layer 2 with L2 Regularization
model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(lambda_val)))

# Output Layer
model.add(layers.Dense(num_classes, activation='softmax'))

#display model architecture
model.summary()


# Set your desired learning rate
learning_rate = 0.001  # Example learning rate

# Instantiate an Adam optimizer with the desired learning rate
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=adam_optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=r"best_model.pth",
    monitor="val_accuracy",
    verbose=0,
    save_best_only=True, 
    save_weights_only=False,
    mode="auto",
    save_freq="epoch"
    # Removed the options parameter
)

n_epochs = 30
batch_size_gd = 50


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

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.title('CNN cross-entropy Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
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
y_pred_classes = np.argmax(prediction, axis=1)

print("accuracy is ", best_model.evaluate(x_test, t_test)[1])


print(f"prediction: {prediction}")
print(f"prediction shape: {prediction.shape}")
print(f"test labels: {t_test}")

plt.plot(np.argmax(prediction, axis=1), np.argmax(t_test, axis=1), 'o')
plt.xlabel('Prediction')
plt.ylabel('True label')
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Calculate the confusion matrix
y_true_classes = [np.argmax(i) for i in t_test]
conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(conf_mat, display_labels=np.arange(num_classes))
disp.plot(cmap='viridis', values_format='d')
plt.title('Confusion Matrix')
plt.show()

class_accuracies = np.diag(conf_mat) / np.sum(conf_mat, axis=1)

for i, accuracy in enumerate(class_accuracies):
    print(f'Class {i+1}: Accuracy = {accuracy * 100:.2f}%')
total_accuracy = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)

print(f'Total Accuracy: {total_accuracy * 100:.2f}%')