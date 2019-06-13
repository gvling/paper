import tensorflow as tf

from utils.errorMsg import *
from models.layers.layer import *



def getRandomInitVariable(shape, size, name):
    # weight initial
    ## random normalize initial
    randomInit = tf.random_normal_initializer(
            stddev=(2.0/size)**0.5
            )
    return tf.get_variable(
            name,
            shape,
            initializer=randomInit
            )

def getConstInitVariable(shape, name, value=0, dtype=tf.float32):
    ## bias initial with zero
    constInit = tf.constant_initializer(value=value, dtype=dtype)
    return tf.get_variable(
            name,
            [shape],
            initializer=constInit
            )

def conv2d(inputs, filters, kernelShape, strides=(1,1,1,1), padding='VALID',
        dataFormat='channels_last', activation='relu', batchNormalization=False):
    assertLengthError(kernelShape, 2)
    assertLengthError(strides, 4)
    assertTypeError(filters, int)
    inputShape = inputs.get_shape().as_list()

    weightShape = (kernelShape[0], kernelShape[1], inputShape[-1], filters)
    weightSize = kernelShape[0] * kernelShape[1] * inputShape[-1] * filters
    W = getRandomInitVariable(weightShape, weightSize, 'W')
    b = getConstInitVariable(filters, 'b')

    # logit = tf.conv2d(W) + b  ->  logit = f(input, WbatchNormalization) + b
    logit = tf.nn.bias_add(
            tf.nn.conv2d(inputs, W, strides=strides, padding=padding),
            b
            )
    if(batchNormalization):
        logit =  tf.layers.batch_normalization(logit)
    if(activation == 'relu'):
        return tf.nn.relu(logit)
    else:
        # TODO: Add more activation
        pass

## TODO
#def conv2dPreActivation(inputs, filters, kernelShape, strides=(1,1,1,1), padding='VALID',
#        dataFormat='channels_last', activation='relu', batchNormalization=False):
#    assertLengthError(kernelShape, 2)
#    assertLengthError(strides, 4)
#    assertTypeError(filters, int)
#    inputShape = inputs.get_shape().as_list()
#
#    weightShape = (*kernelShape, inputShape[-1], filters)
#    weightSize = kernelShape[0] * kernelShape[1] * inputShape[-1] * filters
#    W = getRandomInitVariable(weightShape, weightSize, 'W')
#    b = getConstInitVariable(filters, 'b')
#
#    if(batchNormalization):
#        inputs =  tf.layers.batch_normalization(inputs)
#    if(activation == 'relu'):
#        logit = tf.nn.relu(inputs)
#    else:
#        # TODO: Add more activation
#        pass
#
#    ## tf.conv2d(W) + b  ->  f(input, W) + b
#    return tf.nn.bias_add(
#            tf.nn.conv2d(logit, W, strides=strides, padding=padding),
#            b
#            )

def octavConv2d(inputs, filters, kernelShape, alpha=0.25, strides=(1,1,1,1), padding='SAME',
        dataFormat='channels_last', activation='relu', batchNormalization=False):
    '''
    inputs: [highFrenquencyInput, lowFrenquencyInput]
    alpha: [0, 1], Ratio of low-frequency filters.
    '''
    assertLengthError(inputs, 2)
    assertLengthError(kernelShape, 2)
    assertTypeError(filters, int)
    assertRangeError(alpha, (0,1))
    assertLengthError(strides, 4)

    # split input
    highInput, lowInput = inputs
    highInputShape = highInput.get_shape().as_list()
    lowInputShape = lowInput.get_shape().as_list()
    # split filter
    lowFilters = int(filters * alpha)
    highFilters = filters - lowFilters

    # high variable init
    highWeightShape = (*kernelShape, highInputShape[-1], highFilters)
    highWeightSize = kernelShape[0] * kernelShape[1] * highInputShape[-1] * highFilters
    highW = getRandomInitVariable(highWeightShape, highWeightSize, 'W_high')
    highB = getConstInitVariable(highFilters, 'b_high')

    # low variable init
    lowWeightShape = (*kernelShape, lowInputShape[-1], lowFilters)
    lowWeightSize = kernelShape[0] * kernelShape[1] * lowInputShape[-1] * lowFilters
    lowW = getRandomInitVariable(lowWeightShape, lowWeightSize, 'W_low')
    lowB = getConstInitVariable(lowFilters, 'b_low')

    # frequency exchange
    high2LowWeightShape = (*kernelShape, highInputShape[-1], lowFilters)
    high2LowWeightSize = kernelShape[0] * kernelShape[1] * highInputShape[-1] * lowFilters
    high2LowW = getRandomInitVariable(high2LowWeightShape, high2LowWeightSize, 'W_high_to_low')
    high2LowB = getConstInitVariable(lowFilters, 'b_high_to_low')

    low2HighWeightShape = (*kernelShape, lowInputShape[-1], highFilters)
    low2HighWeightSize = kernelShape[0] * kernelShape[1] * lowInputShape[-1] * highFilters
    low2HighW = getRandomInitVariable(low2HighWeightShape, low2HighWeightSize, 'W_low_to_high')
    low2HighB = getConstInitVariable(highFilters, 'b_low_to_high')

    # coculate logits
    high2HighLogit = tf.nn.bias_add(
            tf.nn.conv2d(highInput, highW, strides=strides, padding=padding),
            highB
            )

    # high to low down sampling
    high2LowLogit = tf.nn.bias_add(
            tf.nn.conv2d(downSampling(highInput), high2LowW, strides=strides, padding=padding),
            high2LowB
            )

    # low to high up sampling
    low2HighLogit = upSampling(tf.nn.bias_add(
        tf.nn.conv2d(lowInput, low2HighW, strides=strides, padding=padding),
        low2HighB
        ))

    low2LowLogit = tf.nn.bias_add(
            tf.nn.conv2d(lowInput, lowW, strides=strides, padding=padding),
            lowB
            )

    if(batchNormalization):
        high2HighLogit =  tf.layers.batch_normalization(high2HighLogit)
        high2LowLogit =  tf.layers.batch_normalization(high2LowLogit)
        low2HighLogit =  tf.layers.batch_normalization(low2HighLogit)
        low2LowLogit =  tf.layers.batch_normalization(low2LowLogit)

    highLogit = high2HighLogit + low2HighLogit
    lowLogit = low2LowLogit + high2LowLogit

    if(activation == 'relu'):
        return [tf.nn.relu(highLogit), tf.nn.relu(lowLogit)]
    else:
        # TODO: Add more activation
        pass
