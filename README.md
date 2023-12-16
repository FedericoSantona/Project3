# Project3

### Files and their purpose
- CNN_with_XGBoost.py:
    Uses a TensorFlow CNN model with Keras and the same architecture as the emotionsTensorFlow-files
    but with XGBoost which takes the output of the dense layers of the CNN as its input.
    It also trains a pure XGBoost model without CNN for comparison.
### Grid search
- CNN_3D_gridsearch.py: script for performing grid search
- gridsearch_score_data: numerical results of the grid search
- plot_gridsearch.ipynb: plots results of the grid search

### Previous versions/ attempts at constructing a CNN
- CNN.py:
    An old file with a purely Python-based CNN class. It imports the CNN layer types from:
        - ActivationFunc.py
        - Convolution2DLayer.py
        - Convolution2DLayerOPT.py
        - CostFunc.py
        - FlattenLayer.py
        - Pooling2DLayer.py
        - Schedulers.py

### remember in report:
    - long figure captions
    - comment code clearly
    - purpose of each file
    - testing of code? sanity checks?
    - check if abstract, introduction and conclusion fulfill the requirements
    - check website citations for when they have been accessed

### TODO:
    NEXT:
    - Filter sizes, Pooling layer sizes, Striding?
    - Confusion matrix
    - ROC/AUC (maybe for underrepresented class)
    - report
    - Doc strings on functions and files. Also update list of file above.
    DONE:
    - check results of grid search
    - Number of convolutional layers
    - data augmentation (abandoned)
    - Grid search: batches, learning rate and lambda
