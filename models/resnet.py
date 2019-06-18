import tensorflow as tf
from models.network import Network
from models.layers.convolution import *


class Resnet50(Network):
    def __init__(self, inputs, labelSize):
        super().__init__(inputs, labelSize)

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
        x = conv2d(self.inputs, 64, (7,7), strides=(1,2,2,1), padding='SAME', batchNormalization=True)
        return tf.layers.MaxPooling2D(3, 2)(x)

    def _residualBlock(self, inputs, filters, strides=(1,1,1,1)):
        with tf.variable_scope('conv_1'):
            x = conv2d(inputs, filters, (1,1), strides=strides ,padding='SAME', batchNormalization=True)
        with tf.variable_scope('conv_2'):
            x = conv2d(x, filters, (3,3) ,padding='SAME', batchNormalization=True)
        with tf.variable_scope('conv_3'):
            x = conv2d(x, filters*4, (1,1) ,padding='SAME', batchNormalization=True)

        # deal to dimensions increase
        if(inputs.get_shape().as_list()[3] != filters*4):
            with tf.variable_scope('conv_0'):
                inputs = conv2d(inputs, filters*4, (1,1), strides=strides, padding='SAME', batchNormalization=True)

        return x + inputs

class Resnet18(Network):
    def __init__(self, inputs, labelSize):
        super().__init__(inputs, labelSize)

    def inference(self, isTrain):
        self.isTrain = isTrain
        ### pretreatment
        with tf.variable_scope('pretreatment'):
            x = self._pretreatment()

        with tf.variable_scope('stage1'):
            with tf.variable_scope('residualBlock1'):
                s1 = self._residualBlock(x,  64)
            with tf.variable_scope('residualBlock2'):
                s1 = self._residualBlock(s1, 64)

        with tf.variable_scope('stage2'):
            with tf.variable_scope('residualBlock1'):
                s2 = self._residualBlock(s1, 128, strides=(1,2,2,1))
            with tf.variable_scope('residualBlock2'):
                s2 = self._residualBlock(s2, 128)

        with tf.variable_scope('stage3'):
            with tf.variable_scope('residualBlock1'):
                s3 = self._residualBlock(s2, 256, strides=(1,2,2,1))
            with tf.variable_scope('residualBlock2'):
                s3 = self._residualBlock(s3, 256)

        with tf.variable_scope('stage4'):
            with tf.variable_scope('residualBlock1'):
                s4 = self._residualBlock(s3, 512, strides=(1,2,2,1))
            with tf.variable_scope('residualBlock2'):
                s4 = self._residualBlock(s4, 512)

        with tf.variable_scope('avg_polling'):
            globalPool = tf.reduce_mean(s4, [1, 2], name='global_average_pooling')
        with tf.variable_scope('softmax'):
            output = tf.layers.Dense(self.labelSize, activation=tf.nn.softmax, name='softmax')(globalPool)

        self.output = output

    def _pretreatment(self):
        x = conv2d(self.inputs, 64, (7,7), strides=(1,2,2,1), padding='SAME', batchNorm=True, isTrain=self.isTrain)
        return tf.layers.MaxPooling2D(3, 2)(x)

    def _residualBlock(self, inputs, filters, strides=(1,1,1,1)):
        with tf.variable_scope('conv_1'):
            x = conv2d(inputs, filters, (3,3), strides=strides ,padding='SAME', batchNorm=True, isTrain=self.isTrain)
        with tf.variable_scope('conv_2'):
            x = conv2d(x, filters, (3,3) ,padding='SAME', batchNorm=True, isTrain=self.isTrain)

        # deal to dimensions increase
        if(inputs.get_shape().as_list()[3] != filters):
            with tf.variable_scope('conv_0'):
                inputs = conv2d(inputs, filters, (1,1), strides=strides, padding='SAME', batchNorm=True, isTrain=self.isTrain)

        return x + inputs

class OctResnet18(Network):
    def __init__(self, inputs, labelSize, alpha):
        super().__init__(inputs, labelSize)
        self.alpha = alpha

    def inference(self, isTrain):
        self.isTrain = isTrain
        ### pretreatment
        with tf.variable_scope('pretreatment'):
            x = self._pretreatment()
            low = tf.layers.AveragePooling2D(2,2,data_format=self.dataFormat, name='x_low_frequency')(x)

        with tf.variable_scope('stage1'):
            with tf.variable_scope('residualBlock1'):
                s1 = self._residualBlock((x, low),  64)
            with tf.variable_scope('residualBlock2'):
                s1 = self._residualBlock(s1, 64)

        with tf.variable_scope('stage2'):
            with tf.variable_scope('residualBlock1'):
                s2 = self._residualBlock(s1, 128, strides=(1,2,2,1), decreaseFeature=True)
            with tf.variable_scope('residualBlock2'):
                s2 = self._residualBlock(s2, 128)

        with tf.variable_scope('stage3'):
            with tf.variable_scope('residualBlock1'):
                s3 = self._residualBlock(s2, 256, strides=(1,2,2,1), decreaseFeature=True)
            with tf.variable_scope('residualBlock2'):
                s3 = self._residualBlock(s3, 256)

        with tf.variable_scope('stage4'):
            with tf.variable_scope('residualBlock1'):
                s4 = self._residualBlock(s3, 512, strides=(1,2,2,1), decreaseFeature=True)
            with tf.variable_scope('residualBlock2'):
                s4 = self._residualBlock(s4, 512)

        with tf.variable_scope('add'):
            # TODO: Rewrite with tensorflow
            add1 = tf.keras.layers.concatenate(s4, name='add1')

        with tf.variable_scope('avg_polling'):
            globalPool = tf.reduce_mean(add1, [1, 2], name='global_average_pooling')
        with tf.variable_scope('softmax'):
            output = tf.layers.Dense(self.labelSize, activation=tf.nn.softmax, name='softmax')(globalPool)

        self.output = output

    def _pretreatment(self):
        x = conv2d(self.inputs, 64, (7,7), strides=(1,2,2,1), padding='SAME', batchNorm=True, isTrain=self.isTrain)
        return tf.layers.MaxPooling2D(2, 2)(x)

    def _residualBlock(self, inputs, filters, strides=(1,1,1,1), decreaseFeature=False):
        with tf.variable_scope('conv1'):
            x = octaveConv2d(inputs, filters, kernelShape=(3,3), strides=strides, alpha=self.alpha, isTrain=self.isTrain)
        with tf.variable_scope('conv2'):
            x = octaveConv2d(x, filters, kernelShape=(3,3), alpha=self.alpha, isTrain=self.isTrain)

        # deal to dimensions increase
        inputHigh = inputs[0]
        inputLow = inputs[1]
        if(inputHigh.get_shape().as_list()[3]+inputLow.get_shape().as_list()[3] != filters):
            inputHigh, inputLow = octaveConv2d(inputs, filters, (1,1), strides=strides, isTrain=self.isTrain)
            inputs = (inputHigh, inputLow)

        print(x[0].shape, x[1].shape, inputs[0].shape, inputs[1].shape)
        return (x[0]+inputs[0], x[1]+inputs[1])
