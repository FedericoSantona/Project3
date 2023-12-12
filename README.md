# Project3

### Files and their purpose
- CNN.py:
    An old file with a purely Python-based CNN class. It imports the CNN layer types from:
        - ActivationFunc.py
        - Convolution2DLayer.py
        - Convolution2DLayerOPT.py
        - CostFunc.py
        - FlattenLayer.py
        - Pooling2DLayer.py
        - Schedulers.py
- CNN_with_XGBoost.py:
    Uses a TensorFlow CNN model with Keras and the same architecture as the emotionsTensorFlow-files
    but with XGBoost which takes the output of the dense layers of the CNN as its input.
    It also trains a pure XGBoost model without CNN for comparison.

### TODO:
    NEXT:
    - Number of convolutional layers
    - Filter sizes
    - Pooling layer sizes
    - Striding?
    - Confusion matrix
    - ROC/AUC (maybe)
    - report
    DONE:
    - data augmentation (abandoned)
    - Grid search: batches, learning rate and lambda
