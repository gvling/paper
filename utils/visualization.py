import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

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
