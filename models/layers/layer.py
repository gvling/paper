import tensorflow as tf
# TODO use tf
from tensorflow.keras.backend import repeat_elements

from utils.errorMsg import *

def upSampling(inputs):
    outputs = repeat_elements(inputs, 2, axis=1)
    outputs = repeat_elements(outputs, 2, axis=2)
    return outputs

def downSampling(inputs, poolSize=(2,2), strides=(2,2), name='down_sampling'):
    outputs = tf.layers.average_pooling2d(inputs, pool_size=poolSize, strides=strides, name=name)
    return outputs
