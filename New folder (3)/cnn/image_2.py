# Unzip the dataset
import zipfile
import os

zip_file_name = 'cifar10-pngs-in-folders.zip'
output_dir = 'cifar10_dataset'

with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print(f"Dataset unzipped to '{output_dir}'")

import tensorflow as tf
from tensorflow.keras import layers
import os

# Load the CIFAR-10 dataset using image_dataset_from_directory

IMG_HEIGHT = 32
IMG_WIDTH = 32
BATCH_SIZE = 32

# Corrected root directory based on observation that `image_dataset_from_directory`
# detected only one class ('cifar10'), suggesting an extra nested 'cifar10' folder.
data_dir = os.path.join(output_dir, 'cifar10', 'cifar10')

# Set a random seed for reproducibility when splitting
seed = 123

# Create training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2, # Allocate 20% for validation/testing
    subset='training',
    seed=seed # Use a fixed seed for reproducible split
)

# Create validation/test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    interpolation='nearest',
    batch_size=BATCH_SIZE,
    shuffle=False, # No need to shuffle the test set
    validation_split=0.2, # Use the same split percentage
    subset='validation',
    seed=seed # Use the same seed to ensure no overlap with the training set
)

class_names = train_ds.class_names
print(f"Class names: {class_names}")

# Normalize image pixel values to [0, 1]
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

print("CIFAR-10 dataset loaded and normalized.")
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Removed Fashion MNIST data loading. CIFAR-10 data (train_ds, test_ds) is already loaded.
# No direct replacement for train_images/test_images as CIFAR-10 is loaded as tf.data.Dataset.

model = models.Sequential()

#Convolutional Layer
#32 filters 3x3 - kernal, activation=relu
# Updated input_shape for CIFAR-10 (32x32 RGB images)
model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
#Max Pooling Layer -> reduce size
model.add(layers.MaxPooling2D((2,2)))

#Convolutional Layer 2
model.add(layers.Conv2D(32, (2,2), activation='relu'))
#Max Pooling Layer
model.add(layers.MaxPooling2D((2,2)))

#Flatten + Dense = convert 2D feature maps -> 1D Vector
model.add(layers.Flatten())

#Fully connected Layer
model.add(layers.Dense(64, activation='relu'))

#Output Layer
model.add(layers.Dense(len(class_names), activation='softmax'))

model.summary()


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
)

model.fit(train_ds, epochs=3, validation_data=test_ds)
#
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load data
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Normalize pixels
train_images = train_images/255.0
test_images = test_images/255.0
train_images[0].shape
plt.imshow(train_images[0])
model = models.Sequential()


# Convolutional layer 1
#16 filters, 3x3 - kernel, activation=relu
model.add(layers.Conv2D(16, (3,3), strides=(1,1), activation='relu', input_shape=(28,28,1)))

#Pooling -> reduce size
model.add(layers.MaxPooling2D((2,2)))

# Convolutional layer 2
model.add(layers.Conv2D(16, (3,3), strides=(1,1), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

#Flatten + Dense
model.add(layers.Flatten())
#Fully connected layers
#We can use any activation here
model.add(layers.Dense(64, activation='relu'))

#Output layer
model.add(layers.Dense(10, activation='softmax'))
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_images,train_labels,validation_data=(test_images,test_labels),epochs=10)

