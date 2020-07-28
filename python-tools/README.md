## Python Tool
A collection of ad-hoc python scripts for converting TF models to TF-lite models,  
experimentation with existing tensorflow models, and making them suitable for conversion to .kmodel format 

Source for Yolov3 code:   
https://github.com/YunYang1994/TensorFlow2.0-Examples  
4-object-detection/YOLOV3 

Downloading YOLOV3 weights:  
`wget https://pjreddie.com/media/files/yolov3.weights`

For original images, download them from https://cocodataset.org/#explore

Converting .tflite to .kmodel:  
copy the .tflite file to your Maix_Toolbox directory. Copy the same images to `Maix_toolbox/images`  
run converter script 

## YOLOV3 model modification 
-- brief summary of needed modifications to be able to convert the resulting .tflite to .kmodel -- 
- Issue: Only paddings of `[[0,0],[1,1],[1,1],[0,0]]` are supported i.e. paddings where 1 pixel are added to 
each side of the image (notation indicates for each of the dimensions batch, height, width, channels how many pixels 
to add before, after)  
modification: In `yolov3_core.common.py` change  
`tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)` to   
`tf.keras.layers.ZeroPadding2D(((1, 1), (1, 1)))(input_layer)`
- Issue: layer SHAPE is not supported   
Call to function tf.shape() is not allowed. Instead, use `tensor.shape`. The only problem is that batch_size is then 
equal to None, so where necessary a batch size equal to 1 has to be hardcoded  
in `yolov3_core/yolov3.py`  
```   
    conv_shape = conv_output.get_shape()
    output_size = conv_shape[1]

    conv_shape = tf.constant((-1, output_size, output_size, 3, 5 + NUM_CLASS), dtype=tf.int32)
    conv_output = tf.reshape(conv_output, conv_shape)
     
    ... 

    # originally [batch_size, 1, 1, 3, 1]
    xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [1, 1, 1, 3, 1])

```
Note: if the use of tf.tile can be circumvened, maybe batch size does not have to be hardcoded 

- Issue: Fatal: Nullable object must have a value.  
No further information on the error is given. Removing the last part of the model, the decoding part, 'fixes' this issue   
Likely related to 
- Issue: Axis of concatenation must be 3  




