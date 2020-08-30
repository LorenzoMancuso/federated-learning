import logging

# create logger
logger = logging.getLogger('custom_logger')


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import datetime
import json
import paho.mqtt.client as mqtt
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from json import JSONEncoder
import numpy
import math

import time
import re
import os
from os import listdir
from os.path import isfile, join

import pickle
import zlib

MQTT_URL = '172.20.8.119'
MQTT_PORT = 1883


#IMAGENET_PATH = '/mnt/dataset/subset'
#TOTAL_IMAGES = 82000
TARGET_SIZE = (32, 32)
BATCH_SIZE = 32
EPOCHS = 1

TOTAL_CLIENTS_NUMBER = 16
CLIENT_NUMBER = 3

GPU_INDEX = 0
GPU_NAME = ''

class FederatedTask():

    def wait_for_update_from_server(self):
        while True:
            time.sleep(10)

            if self.new_weights['update'] is not None:
                #  set new weights
                self.receive_update_from_server(self.new_weights['update'])
                
                # reset
                self.new_weights['update'] = None


    def receive_update_from_server(self, weights):
        logger.info("Updated weights received")

        self.model.set_weights(weights)

        logger.info("Model weights updated successfully.")

        self.training()

        self.send_local_update_to_server()


    def training(self):
        if GPU_NAME != '':
            with tf.device(GPU_NAME):
                self.training_core(self)
        else:
            self.training_core(self)


    def training_core(self):
        time_start = time.time()

        #train_history = self.model.fit_generator(self.train_it, steps_per_epoch=math.ceil(TOTAL_IMAGES / BATCH_SIZE), epochs=EPOCHS)
        train_history = self.model.fit(self.train_it[0], self.train_it[1], batch_size=BATCH_SIZE, epochs=EPOCHS)

        logger.info(f"[TRAIN-TIME] Completed local training in {(time.time() - time_start) / 60} minutes.")

        self.epoch += 1

        #SAVES CHECKPOINT
        self.save_checkpoint()
        
        #SAVES LOG
        print(train_history.history)
        self.save_log(train_history.history['loss'][-1], train_history.history['accuracy'][-1], (time.time() - time_start))
        
        return self.model


    def save_log(self, loss, accuracy, time):
        #log
        with open('./snapshots/log.csv', 'a') as fd:
            if self.epoch == 0:
                fd.write("epoch;accuracy;loss;time\n")

            fd.write(f"{self.epoch};{accuracy};{loss};{time}\n")

        logger.info(f"Saved log on 'snapshots/log.csv'.")


    def save_checkpoint(self):
        self.model.save_weights("snapshots/Local-Weights-node01-MobileNetV2-{epoch:02d}.hdf5".format(epoch=self.epoch))
        logger.info("Saved checkpoint 'Local-Weights-node01-MobileNetV2-{epoch:02d}.hdf5'.".format(epoch=self.epoch))

    
    # UNUSED with compressed messages
    class NumpyArrayEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, numpy.ndarray):
                return obj.tolist()
            return JSONEncoder.default(self, obj)


    def send_local_update_to_server(self):

        # select training data related to selected clients
        model_weights = self.model.get_weights()

        # build message object
        send_msg = {
            'device': self.client_id,
            'data': model_weights
        }

        # publishes on MQTT topic
        compressed = zlib.compress(pickle.dumps(send_msg))
        #publication = self.client.publish("topic/fl-broadcast", json.dumps(send_msg, cls=self.NumpyArrayEncoder), qos=1)
        # send compressed message
        publication = self.client.publish("topic/fl-broadcast", compressed, qos=1)

        logger.debug(f"Result code: {publication[0]} Mid: {publication[1]}")

        while publication[0] != 0:
            self.client.connect(MQTT_URL, MQTT_PORT, 60)
            publication = self.client.publish("topic/fl-broadcast", json.dumps(send_msg, cls=self.NumpyArrayEncoder), qos=1)
            logger.debug(f"Result code: {publication[0]} Mid: {publication[1]}")


    @staticmethod
    def on_message(client, userdata, msg):
        logger.info("New model update received ")

        try:
            logger.info("Loading Weights from message ...")
            #weights = json.loads(msg.payload)
            
            # Decompress weights
            userdata['new_weights']['update'] = pickle.loads(zlib.decompress(msg.payload))
            
            logger.info("Weights loaded successfully")

            #userdata['new_weights']['update'] = weights
            

        except Exception as e:
            logger.warning(f'Error loading weights: {e}')


    @staticmethod
    def on_publish(client, userdata, mid):
        logger.info(f"published message to 'topic/fl-broadcast' with mid: {mid}")


    @staticmethod
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            logger.info("Connected to broker")
            client.subscribe("topic/fl-update")

        else:    
            logger.info("Connection failed. Retrying in 1 second...")
            time.sleep(1)
            client.connect(MQTT_URL, MQTT_PORT, 60)


    @staticmethod
    def on_subscribe(client, userdata, mid, granted_qos):
        logger.info("Subscribed to topic/fl-update")


    def get_last_weights(self, path):
        weights = [join(path, f) for f in listdir(path)
                if isfile(join(path, f)) and 'Averaged-Weights' in f]

        weights.sort(reverse=True)

        return weights

       
    def main(self):
        self.client_id = client_id
    
        try:
            os.mkdir("snapshots")
        except:
            pass

        # INIT MODEL
        self.model = keras.applications.mobilenet_v2.MobileNetV2(input_shape = TARGET_SIZE + (3,), classes = 10, weights = None)
        self.model = keras.applications.ResNet50V2(input_shape = TARGET_SIZE + (3,), classes = 10, weights = None)
        self.model.summary()
        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        weights_checkpoints = self.get_last_weights('./snapshots')
        self.epoch = 0

        if len(weights_checkpoints) > 0:
            logging.info(f"Loading Weights from {weights_checkpoints[0]} ...")
            self.model.load_weights(weights_checkpoints[0])
            logging.info("Done.\n")

            # get last epoch number
            p = re.compile("-(\w+).hdf5")
            result = p.search(weights_checkpoints[0])
            self.epoch = int(result.group(1))

        
        # create generator
        #datagen = ImageDataGenerator()
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() 

        # convert and preprocess
        y_train = keras.utils.to_categorical(y_train, 10) 
        y_test = keras.utils.to_categorical(y_test, 10)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train  /= 255
        x_test /= 255
    
        # prepare an iterators for each dataset
        section_length = math.ceil(len(x_train) / TOTAL_CLIENTS_NUMBER)

        starting_index = (section_length) * (CLIENT_NUMBER -1)

        ending_index = min(len(x_train), starting_index + section_length)
        logging.info(f"starting_index: {starting_index}, ending_index: {ending_index}")
        self.train_it = (x_train[starting_index : ending_index], y_train[starting_index : ending_index])

        # create mqtt client
        self.new_weights = {'update': None}

        self.client = mqtt.Client(userdata={'new_weights': self.new_weights})
        self.client.connect(MQTT_URL, MQTT_PORT, 60)
        # callbacks    
        self.client.on_connect = self.on_connect
        self.client.on_subscribe = self.on_subscribe
        self.client.on_publish = self.on_publish
        self.client.on_message = self.on_message

        
        self.client.loop_start()


    def __init__(self, client_id=-1):
        try:
            GPU_NAME = tf.config.experimental.list_physical_devices('GPU')[GPU_INDEX]
            
            print("GPU_INDEX: ", GPU_INDEX, "GPU_NAME: ", gpu_name)

            with tf.device(GPU_NAME):
                self.main()
                
        except:

            print("\nNO GPU DETECTED!\n")
            self.main()