import numpy as np 
import os
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D,BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
import datetime
from cfg import CFG
from metrics import Metrics
from alexnet import AlexNet


model = AlexNet((CFG.width, CFG.height, CFG.channel), 
                CFG.num_classes,
                Metrics.metrics)

train_datagen=ImageDataGenerator(rescale=1./255,
                                fill_mode='nearest',
                                rotation_range=10,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                shear_range=0.1,
                                zoom_range=0.1)

valid_datagen=ImageDataGenerator(rescale=1./255)


training_set=train_datagen.flow_from_directory(CFG.base_dir+'/train',
                                              target_size=(CFG.width,CFG.height),
                                              batch_size=CFG.batch_size,
                                              class_mode='categorical',
                                              shuffle = True)

valid_set=valid_datagen.flow_from_directory(CFG.base_dir+'/valid',
                                          target_size=(CFG.width,CFG.height),
                                          batch_size=CFG.batch_size,
                                          class_mode='categorical')
                                          
TRAINING_N = training_set.n 
VALID_N = valid_set.n

STEP_TRAIN = TRAINING_N // CFG.batch_size 
STEP_VALID = VALID_N // CFG.batch_size

os.mkdir('./logs')
os.mkdir('./logs/fit')

log_dir="./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5, min_delta=1e-3, verbose=1, min_lr=1e-7)

weights=ModelCheckpoint('model_weights.hdf5',
                       save_best_only=True,
                       monitor='val_loss',
                       verbose=1,
                       save_weights_only=False)

early_stopping=EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)

history=model.fit_generator(training_set,
                                steps_per_epoch=STEP_TRAIN,
                                validation_data=valid_set,
                                epochs=CFG.EPOCHS,
                                validation_steps=STEP_VALID,
                                callbacks=[reduce_lr,
                                            weights,
                                            early_stopping,
                                            tensorboard_callback]
                                            )


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
