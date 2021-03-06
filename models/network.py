import tensorflow as tf

class Network():
    def __init__(self, inputs, labelSize, dataFormat='channels_last', visualization=False, rescaleTo=1/255.):
        self.inputs = tf.scalar_mul(tf.convert_to_tensor(rescaleTo, dtype=tf.float32), inputs)
        self.labelSize = labelSize
        self.dataFormat = dataFormat
        self.visualization = visualization
        self.output = None

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
