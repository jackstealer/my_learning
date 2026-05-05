import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.utils import image_dataset_from_directory,plot_model
import matplotlib.pyplot as plt
import numpy as np

DATASET_PATH="dataset"
IMG_SIZE=(128,128)
BATCH_SIZE=32
EPOCHS=20

train_ds=image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
vali_ds=image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_batches=tf.data.experimental.cardinality(vali_ds)
test_ds=vali_ds.take(val_batches//2)
vali_ds=vali_ds.skip(val_batches//2)

class_name=train_ds.class_names
print(class_name)

plt.figure(figsize=(10,10))
for images,labels in train_ds.take(1):
  for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_name[labels[i]])
    plt.axis("off")
plt.show()

import os
class_count={}
for folder_name in os.listdir(DATASET_PATH):
  class_count[folder_name]=len(os.listdir(os.path.join(DATASET_PATH,folder_name)))
print(class_count)

plt.figure(figsize=(10,5))
plt.bar(class_count.keys(),class_count.values())
plt.xlabel("Class Name")
plt.ylabel("Number of Images")
plt.title("Number of Images per Class")
plt.xticks(rotation=45)
plt.show()


normalization_layer=layers.Rescaling(1./255)
train_ds=train_ds.map(lambda x,y:(normalization_layer(x),y))
vali_ds=vali_ds.map(lambda x,y:(normalization_layer(x),y))
test_ds=test_ds.map(lambda x,y:(normalization_layer(x),y))

AUTOTUNE=tf.data.AUTOTUNE
train_ds=train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
vali_ds=vali_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds=test_ds.cache().prefetch(buffer_size=AUTOTUNE)


model=models.Sequential([
    layers.Conv2D(32,(3,3),activation="relu",input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(32,(3,3),activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32,(3,3),activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128,activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(class_name),activation="softmax")
])
model.summary()

plot_model(model,to_file='model_arch.png',show_shapes=True,show_layer_names=True)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

history=model.fit(
    train_ds,
    validation_data=vali_ds,
    epochs=EPOCHS
)
