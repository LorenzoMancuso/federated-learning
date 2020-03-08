import datetime
import json
import paho.mqtt.client as mqtt
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator

IMAGENET_PATH = '/media/lore/6B6223601B584A05/IMAGENET/'
TOTAL_IMAGES = 100
TARGET_SIZE = (224, 224)
BATCH_SIZE = 20
EPOCHS = 3

model = keras.applications.mobilenet_v2.MobileNetV2()


# -------- MQTT UTILS -----------------------

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe("topic/fl-broadcast")

def on_message(client, userdata, msg):
    print("received")
    if msg.payload.decode() == "Hello world!":
        print("Yes!")
        client.disconnect()

# MQTT CLIENT CONNECTION TO MESSAGE BROKER
client = mqtt.Client()
client.connect("localhost", 1883, 60)

client.on_connect = on_connect
client.on_message = on_message

# -------- END MQTT UTILS ------------------

def calculate_steps():
    steps = 1
    parts = TOTAL_IMAGES / BATCH_SIZE
    if parts > 1:
        steps = parts
        if TOTAL_IMAGES % BATCH_SIZE > 0:
            steps += 1

    return steps

def training():
    global model
    model.summary()

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # create generator
    datagen = ImageDataGenerator()

    # prepare an iterators for each dataset
    train_path = "../res/train"
    train_it = datagen.flow_from_directory(train_path,
                                           target_size=TARGET_SIZE,
                                           class_mode='categorical',
                                           color_mode='rgb',
                                           batch_size=BATCH_SIZE)

    model.fit_generator(train_it, steps_per_epoch=calculate_steps(), epochs=EPOCHS)

    return model


def device_connection_to_server():

    # select training data related to selected clients

    # build message object
    send_msg = {}
    """
    send_msg = {
        'device': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data': tf.nest.map_structure(lambda x: x.numpy().tolist(), iter(federated_train_data).next())
    }
    """

    # publishes on MQTT topic
    client.publish("topic/fl-broadcast", json.dumps(send_msg));
    
    print("\npublished message to 'topic/fl-broadcast'")
