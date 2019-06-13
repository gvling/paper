import tensorflow as tf
from models.network import Network
from models.layers.convolution import *

class Lenet(Network):
    '''
    A Lenet add batchNorm
    '''
    def __init__(self, x, inputShape, labelSize, dataFormat='channels_last', drawConvImg=False):
        self.drawImages = []

        with tf.variable_scope('input'):
            x = tf.reshape(x, shape=inputShape)

        with tf.variable_scope('conv_1'):
            conv1 = conv2d(x, 6, (5,5), padding='SAME', batchNormalization=True)
            if(drawConvImg):
                self.drawImages.append(tf.summary.image('conv1_kernel_1', tf.reshape(conv1, [-1, 220, 220, 1]), 60))
        with tf.variable_scope('pool_1'):
            pool1 = tf.layers.AveragePooling2D(2,2,data_format=dataFormat, name='pool1')(conv1)

        with tf.variable_scope('conv_2'):
            conv2 = conv2d(pool1, 16, (5,5), padding='SAME', batchNormalization=True)
        with tf.variable_scope('pool_2'):
            pool2 = tf.layers.AveragePooling2D(2,2,data_format=dataFormat, name='pool2')(conv2)

        with tf.variable_scope('flat'):
            flat = tf.layers.Flatten(data_format=dataFormat, name='flat')(pool2)
        with tf.variable_scope('fc'):
            fc1 = tf.layers.Dense(84, activation=tf.nn.tanh, name='fc1')(flat)

        with tf.variable_scope('output'):
            output = tf.layers.Dense(labelSize, activation=tf.nn.softmax, name='softmax')(fc1)

        self.output = output
