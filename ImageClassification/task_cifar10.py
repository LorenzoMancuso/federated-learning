from __future__ import absolute_import, division, print_function, unicode_literals

import os
# SET CPU ONLY MODE
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
from imagenet_classes import classes as in_classes
from keras.preprocessing import image as image_utils

from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.callbacks import ModelCheckpoint, CSVLogger

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import pandas as pd

from os import listdir
from os.path import isfile, join

import math

import time
import re

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


TARGET_SIZE = (32, 32)
BATCH_SIZE = 512
EPOCHS = 100



print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.experimental.list_physical_devices('GPU'))

try: 
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
    print("Invalid device or cannot modify virtual devices once initialized. ")
    pass 



def get_last_weights(path):
    weights = [join(path, f) for f in listdir(path)
               if isfile(join(path, f)) and 'Weights' in f and 'Cifar10' in f]

    weights.sort(reverse=True)

    return weights


class CustomModelCheckpointCallback(tf.keras.callbacks.Callback):

    def __init__(self, starting_epoch = 0, path = './snapshots/'):
        self.starting_epoch = starting_epoch
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        print("logs: ", logs)
        current_epoch = epoch + 1 + self.starting_epoch

        path = self.path + "Weights-ResNet50V2-Cifar10-{epoch:02d}.hdf5".format(epoch = current_epoch)
        print(f"Saving model \"{path}\"")
        self.model.save_weights(path)

        #log
        with open(self.path + 'log.csv', 'a') as fd:
            if current_epoch == 1:
                fd.write("epoch;accuracy;loss;validation_accuracy;validation_loss\n")

            fd.write(f"{current_epoch};{logs['accuracy']};{logs['loss']};{logs.get('val_accuracy', '')};{logs.get('val_loss', '')}\n")

    """
    def on_test_batch_end(self, batch, logs=None):
        print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
    """


if __name__ == "__main__":

    """
    mirrored strategy is for multi-GPU execution
    'Data parallelism': consists in replicating the target model once on each device, 
        and using each replica to process a different fraction of the input data.
    """
    # tf.config.experimental.list_physical_devices('GPU')
    print("devices in strategy:", [f"/gpu:{i}" for i in range(len(tf.config.experimental.list_physical_devices('GPU')))])
    strategy = tf.distribute.MirroredStrategy(devices=[f"/gpu:{i}" for i in range(len(tf.config.experimental.list_physical_devices('GPU')))])
    
    with strategy.scope():

        #model = tf.keras.applications.MobileNetV2(input_shape = TARGET_SIZE + (3,), classes = 10, weights = None)
        model = tf.keras.applications.ResNet50V2(input_shape = TARGET_SIZE + (3,), classes = 10, weights = None)
        model.summary()
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

     # LOAD LAST WEIGHTS
    weights = get_last_weights('./snapshots')
    last_epoch = 0

    if len(weights) > 0:
        print(f"Loading Weights from {weights[0]} ...")
        model.load_weights(weights[0])
        print("Done.\n")

        # get last epoch number
        p = re.compile("-(\w+).hdf5")
        result = p.search(weights[0])
        last_epoch = int(result.group(1))


    checkpoint = CustomModelCheckpointCallback(starting_epoch=last_epoch)

    """
    TRAIN DATASETS
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() 

    # convert and preprocess
    y_train = keras.utils.to_categorical(y_train, 10) 
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train  /= 255
    x_test /= 255

    """
    FIT
    """
    print(f"Starting training ...")
    ts = time.time()

    model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS - last_epoch, use_multiprocessing=True, callbacks=[checkpoint])

    te = time.time()

    print(f"Epoch of train iterator finished in {te - ts} seconds ({(te - ts) / 60} minutes)")

    """
    VALIDATION
    """

    print(f"evaluate model in {val_steps} steps")

    score = model.fit(x_test, y_test, batch_size=BATCH_SIZE, epochs=EPOCHS - last_epoch, use_multiprocessing=True, callbacks=[checkpoint])
    
    print("Loss: ", score[0], "Accuracy: ", score[1])
