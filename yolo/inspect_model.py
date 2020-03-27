# model = "yolo_mobilev1"
# depthmul = 0.75
from tensorflow.python.keras.utils import plot_model

from models.yolonet import yolo_mobilev1
input_shape = [224, 320, 3]
yolo_model, yolo_model_warpper = yolo_mobilev1(input_shape, anchor_num=3, class_num=20, alpha=0.75)

# plot_model(yolo_model, "yolo_model.png")
# plot_model(yolo_model_warpper, "yolo_model_warpper.png")

print(yolo_model.output_shape)
print(yolo_model_warpper.output_shape)
