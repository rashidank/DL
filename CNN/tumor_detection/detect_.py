import tensorflow as tf
import os
import keras
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Input
from keras.layers import concatenate
import numpy as np
from tensorflow.keras.optimizers import Adam
import keras.callbacks as kcallbacks
from keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, GlobalAveragePooling2D, Flatten, Dropout, Dense, AveragePooling2D, Attention

#variables declaration
batch_size = 64
img_height = 180
img_width = 180

data_dir=r'CNN\tumor_detection\tumordata\data'
dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,

    shuffle=True,
    image_size=(img_height,img_width),
    batch_size=batch_size
)

#splitting
def get_dataset_partitions_tf(ds, train_split=0.7, val_split=0.15, test_split=0.15, shuffle=True, shuffle_size=20000):
    assert (train_split + test_split + val_split) == 1

    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

train_ds,val_ds,test_ds=get_dataset_partitions_tf(dataset, train_split=0.7, val_split=0.15, test_split=0.15, shuffle=True, shuffle_size=10000)

#model building
model = tf.keras.Sequential([
  # data_augmentation, #Add if augmentation is needed.
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation='relu'),
  tf.keras.layers.Dense(2),
])

#model compailing  anf fitting
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', patience=10)

history = model.fit(train_ds,validation_data=val_ds,epochs=20,callbacks=[early_stop])

model.save(r"C:\Users\fayis\Documents\DL\diff_models\CNN\tumor_detection\CNN_model.h5")