import logging
extra = {'actor_name':'MODEL-UTILS-CIFAR10'}

import time
import math
import warnings
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from common import Singleton
from imagenet_classes import classes as in_classes

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import re
import os
from os import listdir
from os.path import isfile, join
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras

TARGET_SIZE = (32, 32)
BATCH_SIZE = 32

class ModelUtils(metaclass = Singleton):

    def federated_aggregation(self, federated_weights: list):
        """
        ::param: federated_train_data   list containing client weights [W1, ... , Wn]
        """
        logging.info(f"Starting federated aggregation process on {len(federated_weights)} devices.", extra=extra)
                
        time_start = time.time()

        # FEDERATED AVERAGING
        averaged_weights = []
        for weights_list_tuple in zip(*federated_weights):
            averaged_weights.append(
                np.array([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)]))

        logging.info(f"[AGGR-TIME] Completed federated aggregation in {time.time() - time_start} seconds.", extra=extra)

        # set new weights
        self.current_avg_weights = averaged_weights
        self.model.set_weights(self.current_avg_weights)
        logging.info(f"Set new weights.", extra=extra)
        
        # make test on validation iterator
        logging.info(f"evaluate model")
        
        score = self.model.evaluate(self.valid_it[0], self.valid_it[1], batch_size=BATCH_SIZE)
        
        logging.info(f"Val Loss: {score[0]} , Val Accuracy: {score[1]}")
        
        self.epoch += 1

        #SAVES CHECKPOINT
        self.save_checkpoint()
        
        #SAVES LOG
        self.save_log(score[0], score[1], len(federated_weights))
        
        return averaged_weights


    def save_log(self, loss, accuracy, nodes = 0):
        #log
        with open('./snapshots/log.csv', 'a') as fd:
            if self.epoch == 0:
                fd.write("epoch;nodes;validation_accuracy;validation_loss\n")

            fd.write(f"{self.epoch};{nodes}{accuracy};{loss}\n")

        logging.info(f"Saved log on 'snapshots/log.csv'.", extra=extra)


    def save_checkpoint(self):
        self.model.save_weights("snapshots/Local-Weights-node01-MobileNetV2-{epoch:02d}.hdf5".format(epoch=self.epoch))
        logging.info("Saved checkpoint 'Local-Weights-node01-MobileNetV2-{epoch:02d}.hdf5'.".format(epoch=self.epoch), extra=extra)
    

    def get_last_weights(self, path):
        weights = [join(path, f) for f in listdir(path)
                if isfile(join(path, f)) and 'Averaged-Weights' in f]

        weights.sort(reverse=True)

        return weights


    def __init__(self):
        try:
            os.mkdir("snapshots")
        except:
            pass

        self.model = keras.applications.mobilenet_v2.MobileNetV2(input_shape = TARGET_SIZE + (3,), classes = 10, weights = None)
        #self.model = keras.applications.ResNet50V2(input_shape = TARGET_SIZE + (3,), classes = 10, weights = None)

        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        weights_checkpoints = self.get_last_weights('./snapshots')
        self.epoch = 0

        if len(weights_checkpoints) > 0:
            logging.info(f"Loading Weights from {weights_checkpoints[0]} ...", extra=extra)
            self.model.load_weights(weights_checkpoints[0])
            logging.info("Done.\n", extra=extra)

            # get last epoch number
            p = re.compile("-(\w+).hdf5")
            result = p.search(weights_checkpoints[0])
            self.epoch = int(result.group(1))
        
        self.current_avg_weights = self.model.get_weights()

        """
        VALIDATION ITERATOR
        """
        logging.info(f"Generating train iterator from tf.keras.datasets.cifar10 ...")
        ts = time.time()

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() 

        # convert and preprocess
        y_train = keras.utils.to_categorical(y_train, 10) 
        y_test = keras.utils.to_categorical(y_test, 10)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train  /= 255
        x_test /= 255
    
        self.valid_it = x_test

        te = time.time()

        logging.info(f"Train iterator finished in {te - ts} seconds ({(te - ts) / 60} minutes)\n")

