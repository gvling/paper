import tensorflow as tf
from models.layers.convolution import *

class Resnet50():
    def __init__(self, inputs, labelSize):
        self.labelSize = labelSize
        self.inputs = inputs

        ### pretreatment
        with tf.variable_scope('pretreatment'):
            x = self._pretreatment()

        with tf.variable_scope('stage1'):
            with tf.variable_scope('residualBlock1'):
                s1 = self._residualBlock(x,  64)
            with tf.variable_scope('residualBlock2'):
                s1 = self._residualBlock(s1, 64)
            with tf.variable_scope('residualBlock3'):
                s1 = self._residualBlock(s1, 64)

        with tf.variable_scope('stage2'):
            with tf.variable_scope('residualBlock1'):
                s2 = self._residualBlock(s1, 128, strides=(1,2,2,1))
            with tf.variable_scope('residualBlock2'):
                s2 = self._residualBlock(s2, 128)
            with tf.variable_scope('residualBlock3'):
                s2 = self._residualBlock(s2, 128)
            with tf.variable_scope('residualBlock4'):
                s2 = self._residualBlock(s2, 128)

        with tf.variable_scope('stage3'):
            with tf.variable_scope('residualBlock1'):
                s3 = self._residualBlock(s2, 256, strides=(1,2,2,1))
            with tf.variable_scope('residualBlock2'):
                s3 = self._residualBlock(s3, 256)
            with tf.variable_scope('residualBlock3'):
                s3 = self._residualBlock(s3, 256)
            with tf.variable_scope('residualBlock4'):
                s3 = self._residualBlock(s3, 256)
            with tf.variable_scope('residualBlock5'):
                s3 = self._residualBlock(s3, 256)
            with tf.variable_scope('residualBlock6'):
                s3 = self._residualBlock(s3, 256)

        with tf.variable_scope('stage4'):
            with tf.variable_scope('residualBlock1'):
                s4 = self._residualBlock(s3, 512, strides=(1,2,2,1))
            with tf.variable_scope('residualBlock2'):
                s4 = self._residualBlock(s4, 512)
            with tf.variable_scope('residualBlock3'):
                s4 = self._residualBlock(s4, 512)

        with tf.variable_scope('avg_polling'):
            globalPool = tf.reduce_mean(s4, [1, 2], name='global_average_pooling')
        with tf.variable_scope('softmax'):
            output = tf.layers.Dense(labelSize, activation=tf.nn.softmax, name='softmax')(globalPool)

        self.output = output

    def _pretreatment(self):
        x = conv2dPreActivation(self.inputs, 64, (7,7), strides=(1,2,2,1), padding='SAME', batchNormalization=True)
        return tf.layers.MaxPooling2D(3, 2)(x)

    def _residualBlock(self, inputs, filters, strides=(1,1,1,1)):
        with tf.variable_scope('conv_1'):
            x = conv2dPreActivation(inputs, filters, (1,1), strides=strides ,padding='SAME', batchNormalization=True)
        with tf.variable_scope('conv_2'):
            x = conv2dPreActivation(x, filters, (3,3) ,padding='SAME', batchNormalization=True)
        with tf.variable_scope('conv_3'):
            x = conv2dPreActivation(x, filters*4, (1,1) ,padding='SAME', batchNormalization=True)

        # deal to dimensions increase
        if(inputs.get_shape().as_list()[3] != filters*4):
            inputs = conv2dPreActivation(inputs, filters*4, (1,1), strides=strides, padding='SAME', batchNormalization=True)

        return x + inputs



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