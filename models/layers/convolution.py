import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from utils.errorMsg import *

def getRandomInitVariable(shape, name):
    # weight initial
    ## random normalize initial
    weightSize = shape[0] * shape[1] * shape[3]
    weightInit = tf.random_normal_initializer(
            stddev=(2.0/weightSize)**0.5
            )
    return tf.get_variable(
            name,
            shape,
            initializer=weightInit
            )

def getConstInitVariable(shape, name, value=0, dtype=tf.float32):
    ## bias initial with zero
    biasInit = tf.constant_initializer(value=value, dtype=dtype)
    return tf.get_variable(
            name,
            shape,
            initializer=biasInit
            )

def conv2dPreActivation(inputs, filters, kernelShape, strides=(1,1,1,1), padding='VALID',
        dataFormat='channels_last', activation='relu', batchNormalization=False):
    assertLengthError(kernelShape, 2)
    assertLengthError(strides, 4)
    assertTypeError(filters, int)
    inputShape = inputs.get_shape().as_list()

    weightShape = (*kernelShape, inputShape[-1], filters)
    W = getRandomInitVariable(weightShape, 'W')
    b = getConstInitVariable(filters, 'b')

    if(batchNormalization):
        inputs =  tf.layers.batch_normalization(inputs)
    if(activation == 'relu'):
        logit = tf.nn.relu(inputs)
    else:
        # TODO: Add more activation
        pass

    ## tf.conv2d(W) + b  ->  f(input, W) + b
    return tf.nn.bias_add(
            tf.nn.conv2d(logit, W, strides=strides, padding=padding),
            b
            )

def conv2d(inputs, filters, kernelShape, strides=(1,1), padding='VALID',
        dataFormat='channels_last', activation='relu', batchNormalization=False):
    assertLengthError(kernelShape, 2)
    assertTypeError(filters, int)

    weightShape = (*kernelShape, inputs[-1], filters)
    W = getRandomInitVariable(weightShape, 'W')
    b = getConstInitVariable(filters, 'b')

    ## logit = tf.conv2d(W) + b  ->  logit = f(input, WbatchNormalization) + b
    logit = tf.nn.bias_add(
            tf.nn.conv2d(inputs, W, strides=strides, padding=padding),
            b
            )
    if(batchNormalization):
        logit =  tf.layers.batch_normalization(logit, filters)
    if(activation == 'relu'):
        return tf.nn.relu(logit)
    else:
        # TODO: Add more activation
        pass

def octavConv2d(inputs, filters, kernelShape, alpha=0.25, strides=(1,1), padding='VALID',
        dataFormat='channels_last', activation='relu', batchNormalization=False):
    '''
    inputs: [highFrenquencyInput, lowFrenquencyInput]
    alpha: [0, 1], Ratio of low-frequency filters.
    '''
    assertLengthError(inputs, 2)
    assertLengthError(kernelShape, 2)
    assertTypeError(filters, int)
    assertRangeError(alpha, (0,1))

    # split input
    highInput, lowInput = inputs

    lowFilters = int(filters * alpha)
    highFilters = filters - lowFilters
    highWeightShape = (*kernelShape, inputs[-1], highFilters)
    lowWeightShape = (*kernelShape, inputs[-1], lowFilters)

    highW = getRandomInitVariable(highWeightShape, 'W_high')
    lowW = getRandomInitVariable(lowWeightShape, 'W_low')
    highB = getConstInitVariable(highFilters, 'b_high')
    lowB = getConstInitVariable(lowFilters, 'b_low')
#### wip
    ## logit = tf.conv2d(W) + b  ->  logit = f(input, W) + b
    logit = tf.nn.bias_add(
            tf.nn.conv2d(inputs, W, strides=strides, padding=padding),
            b
            )
    if(batchNormalization):
        logit =  tf.layers.batch_normalization(logit, filters)
    if(activation == 'relu'):
        return tf.nn.relu(logit)
    else:
        # TODO: Add more activation
        pass
