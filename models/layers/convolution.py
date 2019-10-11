import tensorflow as tf

from utils.errorMsg import *
from utils.visualization import *
from models.layers.layer import *


def conv2d(inputs, filters, kernelShape, isTrain, strides=(1,1,1,1), padding='VALID',
        dataFormat='channels_last', activation='relu', batchNorm=False, visualization=False):

    assertLengthError(kernelShape, 2)
    assertLengthError(strides, 4)
    assertTypeError(filters, int)
    inputShape = inputs.get_shape().as_list()

    weightShape = (kernelShape[0], kernelShape[1], inputShape[-1], filters)
    weightSize = kernelShape[0] * kernelShape[1] * inputShape[-1] * filters
    W = getRandomInitVariable(weightShape, 'W')
    if(visualization):
        drawHistogram('kernel', W)

    # conv -> bn -> act
    logit = tf.nn.conv2d(inputs, W, strides=strides, padding=padding)
    if(batchNorm):
        logit = batchNormalization(logit, filters, isTrain, visualization=visualization)
    # conv -> +b -> act
    else:
        b = getConstInitVariable(filters, 'b')
        if(visualization):
            drawHistogram('bias', b)
        logit = tf.nn.bias_add(logit, b)

    if(activation == 'relu'):
        logit = tf.nn.relu(logit)
    else:
        # TODO: Add more activation
        pass

    if(visualization):
        drawHistogram('activation', logit)
    return logit

def octaveConv2d(inputs, filters, kernelShape, isTrain, alpha=0.25, strides=(1,1,1,1), padding='SAME',
        dataFormat='channels_last', activation='relu', batchNorm=False, visualization=True):
    '''
    inputs: [highFrequencyInput, lowFrequencyInput]
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
    highW = getRandomInitVariable(highWeightShape, 'W_high')

    # low variable init
    lowWeightShape = (*kernelShape, lowInputShape[-1], lowFilters)
    lowW = getRandomInitVariable(lowWeightShape, 'W_low')

    # frequency exchange
    high2LowWeightShape = (*kernelShape, highInputShape[-1], lowFilters)
    high2LowW = getRandomInitVariable(high2LowWeightShape, 'W_high_to_low')

    low2HighWeightShape = (*kernelShape, lowInputShape[-1], highFilters)
    low2HighW = getRandomInitVariable(low2HighWeightShape, 'W_low_to_high')

    if(visualization):
        drawHistogram('kernel_high_to_high', highW)
        drawHistogram('kernel_high_to_low', high2LowW)
        drawHistogram('kernel_low_to_high', low2HighW)
        drawHistogram('kernel_low_to_low', lowW)

    # coculate logits
    with tf.variable_scope('highToHighFrequencyConv'):
        high2HighLogit = tf.nn.conv2d(highInput, highW, strides=strides, padding=padding)
        highFmapShape = high2HighLogit.get_shape().as_list()

    ## high to low down sampling
    with tf.variable_scope('highToLowFrequencyConv'):
        high2LowLogit = tf.nn.conv2d(downSampling(highInput), high2LowW, strides=strides, padding=padding)

    ## low to high up sampling
    with tf.variable_scope('lowToHighFrequencyConv'):
        low2HighLogit = upSampling(tf.nn.conv2d(lowInput, low2HighW, strides=strides, padding=padding), highFmapShape)

    with tf.variable_scope('lowToLowFrequencyConv'):
        low2LowLogit = tf.nn.conv2d(lowInput, lowW, strides=strides, padding=padding)

    highLogit = high2HighLogit + low2HighLogit
    lowLogit = low2LowLogit + high2LowLogit

    # batch normalization
    if(batchNorm):
        with tf.variable_scope('highFrequencyBN'):
            highLogit =  batchNormalization(highLogit, highFilters, isTrain, visualization=visualization)
        with tf.variable_scope('lowFrequencyBN'):
            lowLogit =  batchNormalization(lowLogit, lowFilters, isTrain, visualization=visualization)
    else:
        # TODO bias dimintion 6
        highB = getConstInitVariable(highFilters, 'b_high')
        lowB = getConstInitVariable(lowFilters, 'b_low')
        if(visualization):
            drawHistogram('bias_high', highB)
            drawHistogram('bias_low', lowB)
        highLogit = tf.nn.bias_add(highLogit, highB)
        lowLogit = tf.nn.bias_add(lowLogit, lowB)

    if(activation == 'relu'):
         highLogit = tf.nn.relu(highLogit)
         lowLogit = tf.nn.relu(lowLogit)
    else:
        # TODO: Add more activation
        pass

    if(visualization):
        drawHistogram('activation_high', highLogit)
        drawHistogram('activation_low', lowLogit)

    return highLogit, lowLogit
