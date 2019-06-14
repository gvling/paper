import tensorflow as tf
from models.network import Network
from models.layers.convolution import *

class Lenet(Network):
    '''
    A Lenet with octaveConv
    '''
    def __init__(self, alpha, inputs, inputReshapeTo, labelSize, dataFormat='channels_last', visualization=False):
        super().__init__(inputs, labelSize, dataFormat, visualization)
        self.alpha = alpha
        self.inputReshapeTo = inputReshapeTo

    def inference(self, isTrain):
        with tf.variable_scope('input'):
            x = tf.reshape(self.inputs, shape=self.inputReshapeTo, name='reshape_input')
            low = tf.layers.AveragePooling2D(2,2,data_format=self.dataFormat, name='x_low_frequency')(x)

        with tf.variable_scope('conv_1'):
            conv1High, conv1Low = octavConv2d([x, low], 6, kernelShape=(5,5), alpha=self.alpha, dataFormat=self.dataFormat, isTrain=isTrain)
        with tf.variable_scope('pool_1'):
            pool1High = tf.layers.AveragePooling2D(2,2,data_format=self.dataFormat, name='pool1_high')(conv1High)
            pool1Low = tf.layers.AveragePooling2D(2,2,data_format=self.dataFormat, name='pool1_low')(conv1Low)

        with tf.variable_scope('conv_2'):
            conv2High, conv2Low = octavConv2d([pool1High, pool1Low], 16, kernelShape=(5,5), alpha=self.alpha, dataFormat=self.dataFormat, isTrain=isTrain)
        with tf.variable_scope('pool_2'):
            pool2High = tf.layers.AveragePooling2D(2,2,data_format=self.dataFormat, name='pool2_high')(conv2High)
            pool2Low = tf.layers.AveragePooling2D(2,2,data_format=self.dataFormat, name='pool2_low')(conv2Low)

        with tf.variable_scope('flat'):
            flatHigh = tf.layers.Flatten(data_format=self.dataFormat, name='flat_high')(pool2High)
            flatLow = tf.layers.Flatten(data_format=self.dataFormat, name='flat_low')(pool2Low)

        with tf.variable_scope('add'):
            # TODO: Rewrite with tensorflow
            add1 = tf.keras.layers.concatenate([flatHigh, flatLow], name='add1')

        with tf.variable_scope('fc'):
            fc1 = tf.layers.Dense(84, activation=tf.nn.relu, name='fc1')(add1)

        with tf.variable_scope('output'):
            output = tf.layers.Dense(self.labelSize, activation=tf.nn.softmax, name='softmax')(fc1)

        self.output = output
