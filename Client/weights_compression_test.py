import keras
import paho.mqtt.client as mqtt
import json
import numpy
from json import JSONEncoder
import sys
import pickle
import zlib
import base64

model = keras.applications.ResNet50V2(input_shape = (32,32) + (3,), classes = 10, weights = None)

# returns a flat list of Numpy arrays.
model_weights_original = model.get_weights()
model_weights = model.get_weights()

tot_original, tot = 0, 0

for layer in model_weights:
    tot_original += layer.nbytes
    print("before: ", layer.nbytes)
    layer = layer.astype('float16')
    tot += layer.nbytes
    print("after: ", layer.nbytes) 
    print(layer.dtype)

print(model_weights_original[0])
print(model_weights[0])
print(f"before: {int(tot_original/1024)}, after: {int(tot/1024)}")


client = mqtt.Client()
client.connect("localhost",1883,60)

idx = int(len(model_weights)/3)*2
print(len(model_weights), idx)

# build message object
send_msg = {
    'device': 5,
    'data': model_weights
}

compressed = zlib.compress(pickle.dumps(send_msg))

# publishes on MQTT topic
publication = client.publish("topic/fl-update", compressed, qos=1)
client.disconnect()

#decompress
decompressed_msg = pickle.loads(zlib.decompress(compressed))
decompressed_weights = decompressed_msg['data']
model.set_weights(decompressed_weights)

