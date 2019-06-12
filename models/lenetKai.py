import tensorflow as tf

class Lenet:
    '''
    A Lenet add batchNorm
    '''
    def __init__(self, x, inputShape, labelSize, dataFormat='channels_last', drawConvImg=False):
        self.drawImages = []

        with tf.variable_scope('input'):
            x = tf.reshape(x, shape=inputShape)

        with tf.variable_scope('conv_1'):
            conv1 = tf.layers.Conv2D(6, 5, data_format=dataFormat, activation=tf.nn.tanh, name='conv1')(x)
            if(drawConvImg):
                self.drawImages.append(tf.summary.image('conv1_kernel_1', tf.reshape(conv1, [-1, 220, 220, 1]), 60))
            conv1 = tf.layers.BatchNormalization(name='batchNorm1')(conv1)
        with tf.variable_scope('pool_1'):
            pool1 = tf.layers.AveragePooling2D(2,2,data_format=dataFormat, name='pool1')(conv1)

        with tf.variable_scope('conv_2'):
            conv2 = tf.layers.Conv2D(16, 5, data_format=dataFormat, activation=tf.nn.tanh, name='conv2')(pool1)
            #self.drawImages.append(tf.summary.image('conv1_kernel_2', tf.reshape(conv2, [-1, 106, 106, 16]), 10))
            conv2 = tf.layers.BatchNormalization(name='batchNorm2')(conv2)
        with tf.variable_scope('pool_2'):
            pool2 = tf.layers.AveragePooling2D(2,2,data_format=dataFormat, name='pool2')(conv2)

        with tf.variable_scope('flat'):
            flat = tf.layers.Flatten(data_format=dataFormat, name='flat')(pool2)
        with tf.variable_scope('fc'):
            fc1 = tf.layers.Dense(84, activation=tf.nn.tanh, name='fc1')(flat)

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
