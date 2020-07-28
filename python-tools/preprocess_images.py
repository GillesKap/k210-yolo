"""
Converting a model to .tflite requires a set of sample images.
This script takes a number of input images and processes them as necessary
"""
import os

import cv2

if __name__ == "__main__":
    img_dir = "./yolov3_data/raw_images"
    output_dir = "./yolov3_data/processed_images"
    target_size = 300
    for f in os.listdir(img_dir):
        filename = os.path.join(img_dir, f)
        img = cv2.imread(filename)
        image_resized = cv2.resize(img, (target_size, target_size))
        cv2.imwrite(os.path.join(output_dir, f), image_resized)

