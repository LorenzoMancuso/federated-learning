from __future__ import absolute_import, division, print_function, unicode_literals

import os
# SET CPU ONLY MODE
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from alex_net import alex_net
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

VALIDATION_PATH = '/media/lore/EA72A48772A459D9/ILSVRC2012/ILSVRC2012_img_val/val/'

VALIDATION_LABELS = '../../ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'

TOTAL_VAL_IMAGES = 50000

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
try: 
    tf.config.experimental.set_memory_growth(physical_devices[0], True) 
except: 
    print("Invalid device or cannot modify virtual devices once initialized. ")
    pass 


def generate_validation_iterator():
    # IMAGES
    validation_images_path = join(VALIDATION_PATH)
    validation_x = [f for f in listdir(validation_images_path)
                    if isfile(join(validation_images_path, f))]
    validation_x.sort()

    # LABELS
    with open(VALIDATION_LABELS) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    validation_y = [in_classes[int(x.strip())][0] for x in content]

    validation_sequence = [[validation_x[i], validation_y[i]] for i in range(0, len(validation_x))]
    validation_dataframe = pd.DataFrame(validation_sequence, columns = ['x', 'y'])

    print(validation_dataframe)

    # create generator
    datagen = ImageDataGenerator()

    valid_it = datagen.flow_from_dataframe(
        dataframe=validation_dataframe,
        directory=join(VALIDATION_PATH),
        x_col='x',
        y_col='y',
        target_size=TARGET_SIZE,
        class_mode="categorical",
        color_mode= 'rgb',
        batch_size=BATCH_SIZE)

    # confirm the iterator works
    batchX, batchy = valid_it.next()

    print(f'Batch x shape={batchX.shape}')
    print(f'Batch y shape={batchy.shape}')
    print('Batch y: ', len(batchy[0]), batchy[0])

    return valid_it


def get_last_weights(path):
    weights = [join(path, f) for f in listdir(path)
               if isfile(join(path, f)) and 'Weights' in f]

    weights.sort()

    return weights


if __name__ == "__main__":

    model = keras.applications.mobilenet_v2.MobileNetV2(weights = None)
    model.summary()
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'top_k_categorical_accuracy'])
    
    """
    VALIDATION ITERATOR
    """
    print(f"Generating train iterator from {VALIDATION_PATH} ...")
    ts = time.time()

    valid_it = generate_validation_iterator()
    val_steps = math.ceil(TOTAL_VAL_IMAGES / BATCH_SIZE)

    te = time.time()

    print(f"Train iterator finished in {te - ts} seconds ({(te - ts) / 60} minutes)\n")

    """
    progresses = [
        "/home/lore/Projects/Progresses/centralized/node05_Adam_001/ImageClassification/snapshots",
        "/home/lore/Projects/Progresses/4_nodes/coordinator-progresses/Server/snapshots",
        "/home/lore/Projects/Progresses/16_nodes_async/coordinator-progresses/Server/snapshots",
        "/home/lore/Projects/Progresses/16_nodes_sync/coordinator-progresses/Server/snapshots",
        "/home/lore/Projects/Progresses/24_nodes_async/coordinator-progresses/Server/snapshots",
        "/home/lore/Projects/Progresses/24_nodes_sync/coordinator-progresses/Server/snapshots",
    ]
    """

    progresses = [
        "/home/lore/Projects/Progresses/16_nodes_sync/coordinator-progresses/Server/snapshots",
    ]

    for progress in progresses:
        
        print(f"\n{progress}")

        with open('evaluations_log.csv', 'a') as fd:
                fd.write(f"\n{progress}")
        
        weights = get_last_weights(progress)

        for weights_set in weights:
            print(f"Loading Weights from {weights_set} ...")
            model.load_weights(weights_set)
            print("Done.\n")

            with open('evaluations_log.csv', 'a') as fd:
                fd.write(f"\n{weights_set}")
            
            """
            VALIDATION
            """
            print(f"evaluate model in {val_steps} steps")
            score = model.evaluate_generator(valid_it, steps=val_steps, use_multiprocessing=True, verbose=1)
            print(score)

            with open('evaluations_log.csv', 'a') as fd:
                fd.write(f"{score}\n")
            
            print("End log writing")
                
