# Handwritten Digit Recognition with a Simple Neural Network

# First time using tensorFlow

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Loading data from mnist dataset 

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# checking whats the shape of our data loaded 
print(f"Training data shape: {x_train.shape}")
print(f"Training Label shape: {y_train.shape}")
print(f"Testing data shape : {x_test.shape}")


# Preprocessing the data (x_train and x_test)
# flattening the image grid for input to neural network

# converting 28 x 28 image grid to 784 pixels

x_train_flat = x_train.reshape(60000, 784)
x_test_flat = x_test.reshape(10000, 784)

# Normalize
# pixel values are in the range of [0,255], but neural network woeks well on small values 
# so normalising the values

x_train_norm = x_train_flat.astype('float32') / 255.0
x_test_norm = x_test_flat.astype('float32') / 255.0


# One-Hot Encoding the Labels

# Our labels are single digits (e.g., 5, 0, 4). For classification, it's
# better to convert this into a "one-hot" encoded vector. This means
# a vector of all 0s, except for a 1 at the index corresponding to the digit.
# e.g., 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
#       2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

num_classes = 10
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)


print(f"\nOriginal label for first image : {y_train[0]}")
print(f"One-hot encoded label : {y_train_cat[0]}")

# Building Model 

# We'll use the Keras Sequential API, which is perfect for building a
# simple stack of layers.

model = keras.Sequential([

# total 5 layers model :- 1 Input , 3 hidden (128, 64, 32) , output layer 

# Input Layer: A dense (fully connected) layer.
    # 'input_shape' must match our flattened image size (784 pixels).
    # '128' is the number of neurons in this layer.
    # 'relu' (Rectified Linear Unit) is a common, effective activation function.

    keras.layers.Dense(128, activation='relu', input_shape=(784,)),

    # Hidden Layers : 
    keras.layers.Dense(64, activation='relu'),

    keras.layers.Dense(32, activation='relu'),

    # Output Layers : 
    # It must have 10 neurons, one for each digit (0-9).
    # 'softmax' activation function converts the outputs into probability
    # scores for each class, ensuring they all sum to 1.

    keras.layers.Dense(10, activation='softmax')
])

# Display a summary of the model's architecture
print("\nModel Architecture:")
model.summary()

# Compile the Model
# Before training, we need to configure the learning process.

model.compile(

    # Optimizer: 'adam' is a popular and effective optimization algorithm
    # that adjusts the learning rate automatically.
    optimizer='adam',

    # Loss Function: 'categorical_crossentropy' is the standard choice for
    # multi-class classification problems that use one-hot encoding.
    loss='categorical_crossentropy',

    # Metrics: We want to monitor the 'accuracy' of the model during training.
    metrics=['accuracy']

)


# Training the model 

print("\nStarting Model Training : ")
history = model.fit(
    x_train_norm,           # The preprocessed training images
    y_train_cat,             # The one-hot encoded training labels
    epochs=10,              # Number of times to loop through the entire dataset
    batch_size=128,         # Number of samples to process in each batch
    validation_split=0.1    # Use 10% of training data for validation during training
)

print("Model Training complete !")


# Evaulating the Model 

print("\nEvaulating the Model with Test Data : ")
loss, accuracy = model.evaluate(x_test_norm, y_test_cat)
print(f'Test Loss : {loss:.4f}')
print(f"Test Accuracy : {accuracy:.2f}%")