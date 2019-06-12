import tensorflow as tf

class LeNet5:
    def __init__(self, imgHeight, imgWidth, channel, labelSize, dataFormat='channels_last'):
        self.imgSize = imgHeight * imgWidth
        self.imgHeight = imgHeight
        self.imgWidth = imgWidth
        self.imgChannel = channel
        self.labelSize = labelSize
        self.inputShape = [-1, imgHeight, imgWidth, channel]

        self.inputs = tf.placeholder(tf.float32, [None, self.imgSize], name='input')
        #self.outputs = tf.placeholder(tf.float32, [None, labelSize], name='output')
        self.labels = tf.placeholder(tf.float32, [None, labelSize], name='label')

        # layerの定義
        self.conv1 = tf.layers.Conv2D(6, 5, data_format=dataFormat, activation=tf.nn.tanh, name='conv1')
        self.pool1 = tf.layers.AveragePooling2D(2,2,data_format=dataFormat, name='pool1')
        self.conv2 = tf.layers.Conv2D(16, 5, data_format=dataFormat, activation=tf.nn.tanh, name='conv2')
        self.pool2 = tf.layers.AveragePooling2D(2,2,data_format=dataFormat, name='pool2')
        self.flat = tf.layers.Flatten(data_format=dataFormat, name='flat')
        self.fc1 = tf.layers.Dense(84, activation=tf.nn.tanh, name='fc1')
        self.fc2 = tf.layers.Dense(10, activation=tf.nn.softmax, name='softmax')

    def forwardPropagation(self):
        y = tf.reshape(self.inputs, self.inputShape)
        y = self.conv1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.pool2(y)
        y = self.flat(y)
        y = self.fc1(y)
        return self.fc2(y)

        #self.outputs = y
        #return self.outputs

    def coculateLoss(self):
        with tf.name_scope('crossEntropy'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.forwardPropagation()))
        return loss

    def coculateAccuracy(self):
        correctPrediction = tf.equal(tf.argmax(self.forwardPropagation(), 1), tf.argmax(self.labels, 1))
        return tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

    def backPropagation(self, lr):
        res = tf.train.AdamOptimizer(lr).minimize(self.coculateLoss())
        return res

    def drawTrainHistory(self):
        with tf.name_scope('train'):
            tf.summary.scalar('loss', self.coculateLoss())
            tf.summary.scalar('accuracy', self.coculateAccuracy())

    def drawHistogram(self):
        tf.summary.histogram('loss', self.coculateLoss())
        tf.summary.histogram('accuracy', self.coculateAccuracy())

    def drawImage(self):
        tf.summary.image('train_image', tf.reshape(self.inputs, [-1, self.imgHeight, self.imgWidth, self.imgChannel]), self.labelSize)


    def saveModel(self, modelDir, session):
        inputs = tf.placeholder(tf.float32, [None, self.imgSize], name='input')
        outputs = tf.placeholder(tf.float32, [None, self.labelSize], name='output')

        builder = tf.saved_model.builder.SavedModelBuilder(modelDir)
        builder.add_meta_graph_and_variables(session, ["tag"], signature_def_map={
                "model": tf.saved_model.signature_def_utils.predict_signature_def(
                    inputs= {"inputs": inputs},
                    outputs= {"outputs": outputs})
                })
        builder.save()

    def prediction(self, inputs):
        self.inputs = inputs
        return self.forwardPropagation()

    def estimation(self, inputs, labels):
        self.inputs = inputs
        self.labels= labels
        return self.coculateAccuracy()
