import tensorflow as tf
from models.layers.octaveConv import OctaveConv2D

class Lenet:
    '''
    A Lenet with octaveConv
    '''
    def __init__(self, alpha, x, inputShape, labelSize, dataFormat='channels_last', drawConvImg=False):
        self.drawImages = []

        with tf.variable_scope('input'):
            x = tf.reshape(x, shape=inputShape, name='reshape_input')
            low = tf.layers.AveragePooling2D(2,2,data_format=dataFormat, name='x_low_frequency')(x)

        with tf.variable_scope('conv_1'):
            conv1High, conv1Low = OctaveConv2D(6, alpha, kernel_size=(5,5), data_format=dataFormat, activation=tf.nn.tanh, name='octaveConv1')([x, low])
            if(drawConvImg):
                self.drawImages.append(tf.summary.image('conv1High_kernel_1', tf.reshape(conv1High, [-1, 220, 220, 1]), 60))
                self.drawImages.append(tf.summary.image('conv1Low_kernel_1', tf.reshape(conv1Low, [-1, 220, 220, 1]), 60))
            conv1High = tf.layers.BatchNormalization(name='batchNorm1_high')(conv1High)
            conv1Low = tf.layers.BatchNormalization(name='batchNorm1_low')(conv1Low)
        with tf.variable_scope('pool_1'):
            pool1High = tf.layers.AveragePooling2D(2,2,data_format=dataFormat, name='pool1_high')(conv1High)
            pool1Low = tf.layers.AveragePooling2D(2,2,data_format=dataFormat, name='pool1_low')(conv1Low)

        with tf.variable_scope('conv_2'):
            conv2High, conv2Low = OctaveConv2D(16, alpha, kernel_size=(5,5), data_format=dataFormat, activation=tf.nn.tanh, name='octaveConv2_high')([pool1High, pool1Low])
            conv2High = tf.layers.BatchNormalization(name='batchNorm2_high')(conv2High)
            conv2Low = tf.layers.BatchNormalization(name='batchNorm2_low')(conv2Low)
        with tf.variable_scope('pool_2'):
            pool2High = tf.layers.AveragePooling2D(2,2,data_format=dataFormat, name='pool2_high')(conv2High)
            pool2Low = tf.layers.AveragePooling2D(2,2,data_format=dataFormat, name='pool2_low')(conv2Low)

        with tf.variable_scope('flat'):
            flatHigh = tf.layers.Flatten(data_format=dataFormat, name='flat_high')(pool2High)
            flatLow = tf.layers.Flatten(data_format=dataFormat, name='flat_low')(pool2Low)

        with tf.variable_scope('add'):
            # TODO: Rewrite with tensorflow
            add1 = tf.keras.layers.concatenate([flatHigh, flatLow], name='add1')

        with tf.variable_scope('fc'):
            fc1 = tf.layers.Dense(84, activation=tf.nn.tanh, name='fc1')(add1)

        with tf.variable_scope('output'):
            output = tf.layers.Dense(labelSize, activation=tf.nn.softmax, name='softmax')(fc1)

        self.output = output

    def inference(self):
        return self.output

    def loss(self, y):
        with tf.variable_scope('loss'):
            xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=self.output)
            loss = tf.reduce_mean(xentropy)
        return loss

    def accuracy(self, y):
        with tf.variable_scope('accuracy'):
            output = tf.argmax(self.output, 1)
            y = tf.argmax(y, 1)
            correct = tf.equal(output, y)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        return accuracy

    def training(self, loss, optimizer):
        trainOp = optimizer.minimize(loss)
        return trainOp
