import numpy as np
import matplotlib.pyplot as plt
import re

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from utils.errorMsg import *

def createSpriteImage(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    img_h = images.shape[1]
    img_w = images.shape[2]

    # 画像数の平方根(切上)を計算(Sprite Imageの1辺の長さに使用)
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    # 全要素0の配列作成
    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j

            # 画像がある限り実行(n_plotsが割り切れないためSprite Imageは少し余る)
            if this_filter < images.shape[0]:
                # Sprite Imageの所定の配列に画像を挿入
                spriteimage[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = images[this_filter]

    return spriteimage

def projectionImage(images, metadataDir, spriteDir, labels, writer, imgHeight, imgWidth, tensorName):
    spriteImage = createSpriteImage(images)
    plt.imsave(spriteDir, spriteImage, cmap='gray')
    # prepare metadata
    with open(metadataDir, 'w') as metadataFile:
        for row in labels:
            metadataFile.write('%d\n' % row)

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = tensorName
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = 'metadata.tsv'
    # Sprite Imageパスと設定
    embedding.sprite.image_path='sprites.png'
    embedding.sprite.single_image_dim.extend([imgHeight,imgWidth])
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(writer, config)



def activationSummary(x):
    tensorName = re.sub('{}_[0-9]*/'.format('tower'), '', x.op.name)
    tf.summary.histogram(tensorName + '/activations', x)
    tf.summary.scalar(tensorName + '/sparsity', tf.nn.zero_fraction(x))

def drawSclar(name, scalars):
    assertTypeError(scalars, dict)
    drawScalars = []
    with tf.variable_scope(name):
        for k,v in scalars.items():
            drawScalars.append(
                tf.Summary.Value(tag='{}/{}'.format(name, k),
                                 simple_value=v))
    return tf.Summary(value=drawScalars)

#def drawHistogram(self, tag, values, step, bins=1000):
#    """Logs the histogram of a list/vector of values."""
#    # Convert to a numpy array
#    values = np.array(values)
#    # Create histogram using numpy
#    counts, bin_edges = np.histogram(values, bins=bins)
#
#    # Fill fields of histogram proto
#    hist = tf.HistogramProto()
#    hist.min = float(np.min(values))
#    hist.max = float(np.max(values))
#    hist.num = int(np.prod(values.shape))
#    hist.sum = float(np.sum(values))
#    hist.sum_squares = float(np.sum(values**2))
#
#    # Thus, we drop the start of the first bin
#    bin_edges = bin_edges[1:]
#
#    # Add bin edges and counts
#    for edge in bin_edges:
#        hist.bucket_limit.append(edge)
#    for c in counts:
#        hist.bucket.append(c)
#
#    # Create and write Summary
#    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
#    self.writer.add_summary(summary, step)
#    self.writer.flush()

def drawHistogram(name, values):
    tf.summary.histogram(name, values, collections=['histogram'])
