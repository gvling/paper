import tensorflow as tf
from utils.visualization import *

from models.network import Network
from models.layers.convolution import *


class LenetKeras(Network):
    def __init__(self, inputs, inputReshapeTo, labelSize, dataFormat='channels_last', batchNorm=False, visualization=False):
        super().__init__(inputs, labelSize, dataFormat, visualization)
        self.inputReshapeTo = inputReshapeTo
        self.batchNorm = batchNorm

    def inference(self, isTrain):
        with tf.variable_scope('input'):
            x = tf.reshape(self.inputs, shape=self.inputReshapeTo)
        with tf.variable_scope('conv_1'):
            conv1 = tf.keras.layers.Conv2D(6, (5,5), padding='same')(x)
            if(self.batchNorm):
                conv1 = tf.keras.layers.BatchNormalization()(conv1)
            conv1 = tf.keras.layers.Activation('relu')(conv1)
        with tf.variable_scope('pool_1'):
            pool1 = tf.keras.layers.MaxPooling2D(2, 2, name='pool1')(conv1)

        with tf.variable_scope('conv_2'):
            conv2 = tf.keras.layers.Conv2D(16, (5,5), padding='same')(pool1)
            if(self.batchNorm):
                conv2 = tf.keras.layers.BatchNormalization()(conv2)
            conv2 = tf.keras.layers.Activation('relu')(conv2)
        with tf.variable_scope('pool_2'):
            pool2 = tf.keras.layers.MaxPooling2D(2, 2, name='pool2')(conv2)

        with tf.variable_scope('flat'):
            flat = tf.keras.layers.Flatten(name='flat')(pool2)
        with tf.variable_scope('fc'):
            fc1 = tf.keras.layers.Dense(84, activation=tf.nn.relu, name='fc1')(flat)

        with tf.variable_scope('output'):
            output = tf.keras.layers.Dense(self.labelSize, activation='softmax', name='softmax')(fc1)
        self.output = output

class LenetTF(Network):
    def __init__(self, inputs, inputReshapeTo, labelSize, dataFormat='channels_last', batchNorm=False, visualization=False):
        super().__init__(inputs, labelSize, dataFormat, visualization)
        self.inputReshapeTo = inputReshapeTo
        self.batchNorm = batchNorm

    def inference(self, isTrain):
        with tf.variable_scope('input'):
            x = tf.reshape(self.inputs, shape=self.inputReshapeTo)

        with tf.variable_scope('conv_1'):
            conv1 = tf.layers.Conv2D( 6, (5,5), padding='same', kernel_initializer=tf.initializers.random_normal())(x)
            if(self.batchNorm):
                conv1 = tf.layers.batch_normalization(conv1, training=isTrain)
            conv1 = tf.nn.relu(conv1)
        with tf.variable_scope('pool_1'):
            pool1 = tf.layers.MaxPooling2D(2,2,data_format=self.dataFormat, name='pool1')(conv1)

        with tf.variable_scope('conv_2'):
            conv2 = tf.layers.Conv2D(16, (5,5), padding='same', kernel_initializer=tf.initializers.random_normal())(pool1)
            if(self.batchNorm):
                conv2 = tf.layers.batch_normalization(conv2, training=isTrain)
            conv2 = tf.nn.relu(conv2)
        with tf.variable_scope('pool_2'):
            pool2 = tf.layers.MaxPooling2D(2,2,data_format=self.dataFormat, name='pool2')(conv2)

        with tf.variable_scope('flat'):
            flat = tf.layers.Flatten(data_format=self.dataFormat, name='flat')(pool2)
        with tf.variable_scope('fc'):
            fc1 = tf.layers.Dense(84, activation=tf.nn.relu, name='fc1')(flat)

        with tf.variable_scope('output'):
            output = tf.layers.Dense(self.labelSize, activation=tf.nn.softmax, name='softmax')(fc1)

        self.output = output

class Lenet(Network):
    def __init__(self, inputs, inputReshapeTo, labelSize, dataFormat='channels_last', batchNorm=False, visualization=False):
        super().__init__(inputs, labelSize, dataFormat, visualization)
        self.inputReshapeTo = inputReshapeTo
        self.batchNorm = batchNorm

    def inference(self, isTrain):
        with tf.variable_scope('input'):
            x = tf.reshape(self.inputs, shape=self.inputReshapeTo)

        with tf.variable_scope('conv_1'):
            conv1 = conv2d(x, 6, (5,5), padding='SAME', batchNorm=self.batchNorm, visualization=self.visualization, isTrain=isTrain)
        with tf.variable_scope('pool_1'):
            pool1 = tf.layers.MaxPooling2D(2,2,data_format=self.dataFormat, name='pool1')(conv1)

        with tf.variable_scope('conv_2'):
            conv2 = conv2d(pool1, 16, (5,5), padding='SAME', batchNorm=self.batchNorm, visualization=self.visualization, isTrain=isTrain)
        with tf.variable_scope('pool_2'):
            pool2 = tf.layers.MaxPooling2D(2,2,data_format=self.dataFormat, name='pool2')(conv2)

        with tf.variable_scope('flat'):
            flat = tf.layers.Flatten(data_format=self.dataFormat, name='flat')(pool2)
        with tf.variable_scope('fc'):
            fc1 = tf.layers.Dense(84, activation=tf.nn.relu, name='fc1')(flat)

        with tf.variable_scope('output'):
            output = tf.layers.Dense(self.labelSize, activation=tf.nn.softmax, name='softmax')(fc1)

        self.output = output

class OctLenet(Network):
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
            conv1High, conv1Low = octaveConv2d([x, low], 6, kernelShape=(5,5), alpha=self.alpha, dataFormat=self.dataFormat, isTrain=isTrain)
        with tf.variable_scope('pool_1'):
            pool1High = tf.layers.AveragePooling2D(2,2,data_format=self.dataFormat, name='pool1_high')(conv1High)
            pool1Low = tf.layers.AveragePooling2D(2,2,data_format=self.dataFormat, name='pool1_low')(conv1Low)

        with tf.variable_scope('conv_2'):
            conv2High, conv2Low = octaveConv2d([pool1High, pool1Low], 16, kernelShape=(5,5), alpha=self.alpha, dataFormat=self.dataFormat, isTrain=isTrain)
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
