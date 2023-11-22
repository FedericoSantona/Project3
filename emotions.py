import os
import zipfile
from tensorflow import keras as tfk
from CNN import *
import matplotlib.pyplot as plt

"""
        Description:
        ------------
            CNN application to the emotions dataset, remeber to change the path 
            to the dataset in the beginning of the code.

        Parameters to tune:
        ------------
            I  Architecture of the CNN( Number and types of layers)
            II  Dimensions and number of the kernels (height and width) used for convolution and pooling
            III stride in the convolutional layers and pooling layers
            IV  Numbers of epochs and batches ( 2 types of batches are used, one for the GD(in the fit function) and one for data preprocessing)
            V   Learning rate and regularization parameter ( lambda)
            VI  Activation functions ( sigmoid, tanh, relu, leaky relu, softmax)
            VII  Number of neurons in the fully connected layers
        """

# Path to the zip file and extraction directory
zip_path = r'C:\Users\aalst\git_repos\FYS-STK4155-Projects\Project3\archive.zip'
extraction_path = r'C:\Users\aalst\git_repos\FYS-STK4155-Projects\Project3\dataset'

# Unzip the dataset if it hasn't already been extracted to extraction_path
if not os.listdir(extraction_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)
else:
    print("Dataset already extracted.")

# Set up directories
train_dir = os.path.join(extraction_path, 'train')
test_dir = os.path.join(extraction_path, 'test')

# Define image dimensions
img_width, img_height = 48, 48  # You can adjust this based on your specific dataset
batch_size = 26 # You can adjust this based on your specific needs


# Load images using image_dataset_from_directory
train_dataset = tfk.utils.image_dataset_from_directory(
    train_dir,
    color_mode='grayscale', # or 'rgb' for colored images
    label_mode='categorical',  # or 'binary' if you have two classes
    batch_size=batch_size,
    image_size=(img_width, img_height))


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
train_dataset = train_dataset.take(10)
test_dataset = test_dataset.take(2)


# To get the labels from the train dataset
train_labels = []
for _, labels in train_dataset:
    train_labels.extend(labels.numpy())

# Do the same for the test dataset
test_labels = []
for _, labels in test_dataset:
    test_labels.extend(labels.numpy())

# Now train_labels and test_labels contain the labels for their respective datasets
print("Train labels:", train_labels)
print("Test labels:", test_labels)


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


# Now you can use x_train, t_train, x_test, and t_test with your custom CNN

print("x train shape:", x_train.shape)
print("t train shape:", t_train.shape)
print("x test shape:", x_test.shape)
print("t test shape:", t_test.shape)

x_train = x_train.transpose(0, 3, 1, 2)
x_test = x_test.transpose(0, 3, 1, 2)

print(" After transpose")
print("x train shape:", x_train.shape)
print("t train shape:", t_train.shape)
print("x test shape:", x_test.shape)
print("t test shape:", t_test.shape)



# Define the CNN model
adam_scheduler = Adam(eta=0.1, rho=0.9, rho2=0.999)
cnn = CNN(cost_func=CostCrossEntropy, scheduler=adam_scheduler, seed=2023)

#Add layers to the CNN model


cnn.add_Convolution2DLayer(
    input_channels=1,
    feature_maps=1,
    kernel_height=6,
    kernel_width=4,
    act_func=LRELU,
)
#POOLING IS INSANELY SLOW
#cnn.add_PoolingLayer(
 #   kernel_height=2,
 #   kernel_width=2,
 #   v_stride=2,
 #   h_stride=2,
 #   pooling="max",
#)


cnn.add_Convolution2DLayer(
    input_channels=1,
    feature_maps=1,
    kernel_height=4,
    kernel_width=3,
    act_func=LRELU,
)


cnn.add_FlattenLayer()


cnn.add_FullyConnectedLayer(30, LRELU)


cnn.add_FullyConnectedLayer(20, LRELU)


cnn.add_OutputLayer(7, softmax)


scores = cnn.fit(
    x_train,
    t_train,
    lam=1e-5,
    batches=10,
    epochs=100,
    X_val=x_test,
    t_val=t_test,
)

prediction = cnn.predict(x_test)
accuracy = cnn._accuracy(t_test, prediction)

print("The accuracy is : ", accuracy)
