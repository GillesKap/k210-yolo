"""
Convert YOLOV3 model to a .tflite model
"""

import cv2
import numpy as np
import yolov3_core.utils as utils
import tensorflow as tf
from yolov3_core.yolov3 import YOLOv3, decode

if __name__ == "__main__":
    input_size = 416

    input_layer = tf.keras.layers.Input([input_size, input_size, 3])
    feature_maps = YOLOv3(input_layer)

    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, i)
        bbox_tensors.append(bbox_tensor)

    # model = tf.keras.Model(input_layer, feature_maps[0])
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights(model, "./yolov3_data/yolov3.weights")
    model.summary()

    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open("yolov3.tflite", "wb") as f:
        f.write(tflite_model)
