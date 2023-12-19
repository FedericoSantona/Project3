# Project3
by Alessia Sanfelici, Andreas Alstad, Carmen Ditscheid and Federico Santona

### Files and their purpose
- CNN_with_XGBoost.py:
    Uses a TensorFlow CNN model with Keras and the same architecture as the emotionsTensorFlow-files
    but with XGBoost which takes the output of the dense layers of the CNN as its input.
    It also trains a pure XGBoost model without CNN for comparison.
- plots.ipynb: plotting number of images per class and confusion matrix
- number_of_layers.py: search and plot for CNN with different number of layers
  
### Grid search
- CNN_3D_gridsearch.py: script for performing grid search
- gridsearch_score_data: numerical results of the grid search
- plot_gridsearch.ipynb: plots results of the grid search
  
### best_model.pth
- file containing the best model from the CNN

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
- emotions.py
- emotionsTensorFlow.ipynb
- emotionsTensorFlow.py
- emotionsTensorFlow_augmented.py
- view_image_through_CNN_layers.py
