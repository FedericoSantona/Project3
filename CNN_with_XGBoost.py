import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
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

def tensorflow_model(x_train, t_train, x_test, t_test, batch_size=64, epochs=10, eta = 0.0001, save_results = False):
    # Set the seed
    tfk.utils.set_random_seed(seed)

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

def initialize_data() -> tuple:
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

    # # Shuffle the data
    # x_train, t_train = shuffle(x_train, t_train, random_state=seed)
    # x_test, t_test = shuffle(x_test, t_test, random_state=seed)

    return x_train, t_train, x_test, t_test

def view_image_through_cnn_layers(model, image):
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


def downscale():
    # Reshape the data for XGBoost without CNN
    downscale_factor = 6
    x_train_downscaled = np.array([cv2.resize(image, (48//downscale_factor, 48//downscale_factor), interpolation=cv2.INTER_AREA) for image in x_train])
    x_test_downscaled  = np.array([cv2.resize(image, (48//downscale_factor, 48//downscale_factor), interpolation=cv2.INTER_AREA) for image in x_test])
    x_train_downscaled = x_train_downscaled.reshape((x_train_downscaled.shape[0], x_train_downscaled.shape[1]**2))
    x_test_downscaled  = x_test_downscaled.reshape((x_test_downscaled.shape[0], x_test_downscaled.shape[1]**2))

    return x_train_downscaled, x_test_downscaled

def plot_dataset_balance(labels):
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

def main():
    # One-hot encode the labels
    # encoder = OneHotEncoder()
    # t_train_onehot = encoder.fit_transform(t_train.reshape(-1, 1)).toarray()
    # t_test_onehot = encoder.fit_transform(t_test.reshape(-1, 1)).toarray()

    # Load the CNN model and predict
    tf_model, tf_history = tensorflow_model(x_train, t_train, x_test, t_test, eta = learning_rate, batch_size=batch_size, epochs=n_epochs, save_results=False)
    cnn_predict = tf_model.predict(x_test)
    cnn_score = accuracy_score(np.argmax(t_test, axis=1), np.argmax(cnn_predict, axis=1))
    print(f"\nTensorFlow CNN accuracy: {100*cnn_score:.2f} %")

    # View an image through the CNN layers
    # view_image_through_cnn_layers(tf_model, x_test[0])

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
    l2_lambda = 0.001
    n_epochs = 2
    batch_size = 32
    target_num_images = 1000  # Set your target number of images per category
    xgboost_layer_list = ["dense_1", "dense_2", "output"]

    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    n_classes = len(labels)

    # Initialize the data
    x_train, t_train, x_test, t_test = initialize_data()
    x_train_downscaled, x_test_downscaled = downscale()     # For XGBoost without CNN

    # Run
    # plot_dataset_balance(t_train)
    main()