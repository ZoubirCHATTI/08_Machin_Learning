# MNIST Image Classification

## Project Description
This project implements a Deep Learning model for the classification of handwritten digits using the **MNIST (Modified National Institute of Standards and Technology)** dataset. Often referred to as the "Hello World" of Deep Learning, the MNIST dataset provides a foundational benchmark for classification algorithms in computer vision. The primary goal is to train a deep neural network (DNN) to accurately identify handwritten digits from 0 to 9.

The dataset comprises 60,000 training images and 10,000 test images, each being a $28 \times 28$ pixel grayscale image. The model's success in mapping image inputs to their corresponding numerical labels demonstrates basic proficiency in image processing and neural network construction.

## Features
- **Data Loading**: Utilizes `tensorflow.keras.datasets` to load the MNIST dataset.
- **Data Preprocessing**: Splits data into training, validation, and test sets, and flattens image data for DNN compatibility.
- **One-Hot Encoding**: Transforms categorical labels into a binary matrix format suitable for neural network training.
- **Deep Neural Network (DNN) Model**: A `Sequential` model with `Dense` layers, using `relu` activation for hidden layers and `softmax` for the output layer.
- **Model Compilation**: Configured with `adam` optimizer, `binary_crossentropy` loss function, and `accuracy` metric.
- **Early Stopping**: Implements `EarlyStopping` callback to prevent overfitting by monitoring validation loss.
- **Model Training**: Trains the DNN model on the preprocessed MNIST dataset.
- **Model Evaluation**: Assesses the model's performance on unseen test data, reporting loss and accuracy.
- **Visualization**: Plots training and validation loss and accuracy over epochs to monitor learning progress.
- **Random Assignment Comparison**: Compares the model's accuracy against a random assignment baseline.

## Installation
To run this project, you need to have Python installed along with the following libraries. You can install them using pip:

```bash
pip install numpy matplotlib tensorflow scikit-learn
```

## Usage
1. Save the provided Python script as `mnist_images_classification.py`.
2. Run the script from your terminal:

```bash
python mnist_images_classification.py
```

3. The script will:
    - Load the MNIST dataset.
    - Display an example image from the training set.
    - Train the Deep Neural Network model.
    - Evaluate the model's performance.
    - Display plots of training/validation loss and accuracy.
    - Print the test loss, test accuracy, and a comparison with random assignment accuracy.

## Results
The script will output the test loss and accuracy of the trained model. It will also generate two plots visualizing the training and validation loss, and training and validation accuracy over epochs. Finally, it will compare the model's accuracy with a random assignment baseline to highlight its effectiveness.


## Author
Zoubir CHATTI
