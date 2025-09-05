import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# list of all the class names in CIFAR-10 dataset
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# loading data from CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Preprocessing data 
# Using normalization

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

# converting the labels to one-hot encoded format 

train_labels_categorical = to_categorical(train_labels)
test_labels_categorical = to_categorical(test_labels)

# Building CNN network 

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
# above code snippet explained :
# Conv2D(32,(3,3)) : means 32 feature filters, of size 3 x 3
# input_shape=(32,32,3) : means input images are 32 x 32 pixels with 3 channels (RGB)

model.add(layers.MaxPooling2D((2,2)))
# MaxPooling2D((2,2)) : reduces the spatial size by taking the maximum value from each 2 x 2 block
# if input feature map was of 32 x 32 , then after pooling it becomes 16 x 16 

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Flatten())
# Flatten layer for reducing the 3d feature maps to 1d vector for input to FC layer

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
# last layer for result / output

model.summary()

# compiling the model and Training the model
# Using categorical_crossentropy as we have multiple classes to predict
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\nStarting Model Training ")
history = model.fit(train_images,
                    train_labels_categorical,
                    epochs=15,
                    batch_size=64,
                    validation_data=(test_images, test_labels_categorical))

print("Training Finished !!")

# Evaluating the model

loss, accuracy = model.evaluate(test_images, test_labels_categorical, verbose=2)

print(f"\nModel accuracy on the test set : {accuracy*100:.2f}%")

## Visualize a Prediction
# Let's predict on the first image in the test set
test_image_to_predict = test_images[0]
img_for_prediction = np.expand_dims(test_image_to_predict, axis=0)

# Get prediction probabilities
prediction_probabilities = model.predict(img_for_prediction)
# Find the class with the highest probability
predicted_class_index = np.argmax(prediction_probabilities)
predicted_class_name = class_names[predicted_class_index]
true_class_name = class_names[test_labels[0][0]]

# Show the image and the result
plt.imshow(test_image_to_predict)
plt.title(f"True Label: {true_class_name}\nPredicted Label: {predicted_class_name}")
plt.axis("off")
plt.show()