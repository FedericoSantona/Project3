import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras as tfk
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder

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
    # model = XGBClassifier(max_leaves=max_depth, 
    #                       learning_rate=eta, 
    #                       objective="multi:softprob", 
    #                       num_class=num_class, 
    #                       eval_metric="mlogloss", 
    #                       reg_lambda=l2_lambda, 
    #                       random_state=seed)
    # model.fit(x_train, t_train, eval_set=[(x_test, t_test)], verbose=False)

    return model

def tensorflow_model(x_train, t_train, x_test, t_test, batch_size=64, epochs=10, eta = 0.0001, save_results = False):
    # Set the seed
    tfk.utils.set_random_seed(seed)

    # Initialize CNN model
    model = tfk.Sequential([
        tfk.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        tfk.layers.MaxPooling2D((2, 2)),
        tfk.layers.Conv2D(64, (3, 3), activation="relu"),
        tfk.layers.MaxPooling2D((2, 2)),
        tfk.layers.Conv2D(64, (3, 3), activation="relu"),
        tfk.layers.Dropout(0.3),
        tfk.layers.Flatten(),
        tfk.layers.Dense(64, activation="relu", kernel_regularizer=tfk.regularizers.l2(l2_lambda), name="xgboost_input"),
        tfk.layers.Dense(7, activation="relu", kernel_regularizer=tfk.regularizers.l2(l2_lambda)),
        tfk.layers.Dense(n_classes, activation="softmax")
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

def main():
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

    # One-hot encode the labels
    encoder = OneHotEncoder()
    t_train_onehot = encoder.fit_transform(t_train.reshape(-1, 1)).toarray()
    t_test_onehot = encoder.fit_transform(t_test.reshape(-1, 1)).toarray()


    # Load the model and predict
    tf_model, tf_history = tensorflow_model(x_train, t_train, x_test, t_test, eta = learning_rate, batch_size=batch_size, epochs=n_epochs, save_results=False)
    cnn_predict = tf_model.predict(x_test)


    # Initialize the xgboost model
    xgboost_layer_model = tfk.models.Model(inputs=tf_model.input, 
                                           outputs=tf_model.get_layer("xgboost_input").output)
    xgboost_layer_output_train = xgboost_layer_model.predict(x_train)
    xgboost_layer_output_test = xgboost_layer_model.predict(x_test)
    # xgb_model = xgboost_model(xgboost_layer_output, np.argmax(t_train, axis=1), X_test_CNN, np.argmax(t_test, axis=1))
    xgb_model = xgboost_model(xgboost_layer_output_train, np.argmax(t_train, axis=1), xgboost_layer_output_test, np.argmax(t_test, axis=1))
    xgb_predict = xgb_model.predict(xgb.DMatrix(xgboost_layer_output_test))

    # Scores
    cnn_score = accuracy_score(np.argmax(t_test, axis=1), np.argmax(cnn_predict, axis=1))
    xgb_score = accuracy_score(np.argmax(t_test, axis=1), np.argmax(xgb_predict, axis=1))

    # Print the results
    print(f"\nTensorFlow CNN accuracy: {100*cnn_score:.2f} %")
    print(f"\nXGBoost accuracy: {100*xgb_score:.2f} %")

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
    learning_rate = 0.0001
    l2_lambda = 0.001
    n_epochs = 50
    batch_size = 100
    labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    n_classes = len(labels)

    main()