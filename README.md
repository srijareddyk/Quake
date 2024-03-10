# Quake
Analysis of Near Fault Non- Pulse Ground Motions using Deep Learning Techniques
National University of Singapore, Singapore July 2023

This repository contains a deep learning model implemented in TensorFlow and Keras to predict the likelihood of earthquakes based on geological data. The model uses a neural network architecture to learn patterns from earthquake features and make predictions.

## Introduction

The goal of this project is to predict earthquakes using a deep learning model built with TensorFlow and Keras. The implemented neural network is trained on earthquake data, considering features like latitude, longitude, and magnitude. This report provides an overview of the code, its methodology, and the obtained results.

## Code Overview

### Data Preprocessing

- **Loading Data:** The earthquake data is loaded from the `database.csv` file using Pandas. The dataset contains information about earthquakes, including their location and magnitude.
  
- **Cleaning Data:** Rows with missing values are dropped, and a new column 'MMI' is created to map magnitudes to Modified Mercalli Intensity (MMI) levels.

### Visualization

- **Magnitude Map:** A scatter plot map is created using Plotly Express to visualize earthquake magnitudes based on their latitude and longitude. This map is saved as `magnitude_map.png`.

### Model Development

- **Neural Network Model:** A simple neural network model is created using the Sequential API from Keras. It consists of two dense layers with ReLU activation functions and a final output layer with a sigmoid activation function for binary classification.

- **Compilation:** The model is compiled using binary cross-entropy loss and the Adam optimizer.

- **Training:** The model is trained on a subset of the dataset using the `fit` method with 10 epochs and a batch size of 10.

### Evaluation

- **Accuracy Calculation:** The model's accuracy is evaluated on a separate test set using the `evaluate` method.

- **Predictions:** Predictions are made on the test set, and the results are compared with the actual labels.

- **Confusion Matrix and Classification Report:** The script outputs a confusion matrix and a classification report, providing insights into the model's performance.

### Model Save and Load

- **Saving the Model:** The trained model is saved as `earthquake_model.h5` in the `/tmp/` directory using Keras' `save` method.

- **Loading the Model:** The script demonstrates how to load the saved model using `tf.keras.models.load_model`.

## Results

- **Accuracy:** The model's accuracy on the test set is printed, giving an indication of its overall performance.

- **Confusion Matrix and Classification Report:** Detailed metrics, including precision, recall, and F1-score, are provided through the confusion matrix and classification report.

- **Visualization:** The magnitude map visualizes earthquake occurrences, providing a geographical perspective on the dataset.

## Conclusion

This code provides a basic implementation of earthquake prediction using a neural network. It demonstrates data preprocessing, model development, training, and evaluation. Further improvements could include exploring different neural network architectures or incorporating additional features for enhanced predictive capabilities.

Feel free to explore, modify, and adapt the code based on specific use cases and requirements.
