import tensorflow as tf
# TODO use tf
from tensorflow.keras.backend import repeat_elements
from tensorflow.python.ops import control_flow_ops

from utils.errorMsg import *

def getRandomInitVariable(shape, name):
    #initializer = tf.contrib.layers.variance_scaling_initializer()
    initializer = tf.initializers.he_normal()
    return tf.get_variable(
            name,
            shape,
            initializer=initializer
            )

def getConstInitVariable(shape, name, value=0.0, dtype=tf.float32):
    ## bias initial with zero
    constInit = tf.constant_initializer(value=value, dtype=dtype)
    return tf.get_variable(
            name,
            [shape],
            initializer=constInit
            )

def batchNormalization(inputs, filters, isTrain):
    beta = getConstInitVariable(filters, name='beta')
    gamma = getConstInitVariable(filters, name='gamma', value=1.0)

    batchMean, batchVar = tf.nn.moments(
        inputs,
        [0,1,2],
        name='moments'
    )

    # min(decay, (1 + num_updates) / (10 + num_updates))
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    emaApplyOp = ema.apply([batchMean, batchVar])
    emaMean = ema.average(batchMean)
    emaVar = ema.average(batchVar)

    def meanVarWithUpdate():
        with tf.control_dependencies([emaApplyOp]):
            return (
                tf.identity(batchMean),
                tf.identity(batchVar)
            )

    # if isTrain meanVarWithUpdate, else (emaMean, emaVar)
    mean, var = control_flow_ops.cond(
        isTrain,
        meanVarWithUpdate,
        lambda: (emaMean, emaVar)
    )

    return tf.nn.batch_norm_with_global_normalization(
        inputs,
        mean,
        var,
        beta,
        gamma,
        variance_epsilon=1e-3,
        scale_after_normalization=True
    )

def upSampling(inputs):
    outputs = repeat_elements(inputs, 2, axis=1)
    outputs = repeat_elements(outputs, 2, axis=2)
    return outputs

def downSampling(inputs, poolSize=(2,2), strides=(2,2), name='down_sampling'):
    outputs = tf.layers.average_pooling2d(inputs, pool_size=poolSize, strides=strides, name=name)
    return outputs

