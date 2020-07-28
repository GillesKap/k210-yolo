import tensorflow as tf
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="./ssd_mobilenet_v1/ssd_mobilenet_v1_1_default_1.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(input_details)
    print(output_details)