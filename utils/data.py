from PIL import Image
from multiprocessing import Pool
from functools import partial
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm


def resizeImage(imgPath, size=(224,224)):
    src = Image.open(imgPath)
    return src.resize(size)

def resizeAndSaveImage(imgPath, savePath):
    imgName = imgPath.split('/')[-1]
    resizedImg = resizeImage(imgPath)
    resizedImg.save('{}{}'.format(savePath, imgName))

def resizeImageFromFolder(srcPath, savePath):
    with Pool(processes=8) as pool:
        pool.map(partial(resizeAndSaveImage, savePath=savePath), glob.glob('{}*'.format(srcPath)))
    print('Done!')
    return 1




class TFRecord:
    def __init__(self, tfrecordFilePath, labelSize, options=tf.io.TFRecordOptions(compression_type=tf.io.TFRecordCompressionType.GZIP)):
        self.filePath = tfrecordFilePath
        self.dataset = None
        self.size = 0
        self.labelSize = labelSize
        self.options = options

    def _int64Feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytesFeature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64FeaturePlaceHolder(self):
        return tf.FixedLenFeature((), tf.int64, default_value=0)

    def _bytesFeaturePlaceHolder(self):
        return tf.FixedLenFeature((), tf.string, default_value='')

    # parse feature when load TFRecord file
    def _parseFeature(self, exampleProto):
        features = {
            'index': self._int64FeaturePlaceHolder(),
            'name': self._int64FeaturePlaceHolder(),
            'label': self._int64FeaturePlaceHolder(),
            'image': self._bytesFeaturePlaceHolder()
        }
        parsedFeatures = tf.parse_single_example(exampleProto, features)
        imageDecoded = tf.image.decode_jpeg(parsedFeatures['image'])
        labelOneHot = tf.one_hot(parsedFeatures['label'], self.labelSize, axis=-1)
        return parsedFeatures['index'], imageDecoded, labelOneHot

    def getSize(self):
        # return TFRecord size
        if(self.size == 0):
           tfrSize = len(list(tf.io.tf_record_iterator(self.filePath,options=self.options)))
           self.size = tfrSize
        return self.size

    def fromCsv(self, datasetCsvPath):
        datasetFile = pd.read_csv(datasetCsvPath)
        # shuffle dataset
        datasetFile = datasetFile.sample(frac=1).reset_index(drop=True)
        # TFRecord writer
        writer = tf.io.TFRecordWriter(
            self.filePath,
            options=self.options
        )
        i = 0
        for imgPath, label in datasetFile.itertuples(index=False):
            fileName = imgPath.split('/')[-1].split('.')[0]
            image_raw = open(imgPath, 'rb').read()
            example = tf.train.Example(features=tf.train.Features(feature={
                'index': self._int64Feature(i),
                'name': self._int64Feature(int(fileName)),
                'label': self._int64Feature(int(label)),
                'image': self._bytesFeature(image_raw)
                }))
            writer.write(example.SerializeToString())
            i += 1
        print('Done!')
        writer.close()

    def toDataset(self):
        dataset = tf.data.TFRecordDataset(self.filePath, compression_type='GZIP')
        dataset = dataset.map(self._parseFeature)
        # buffer_size is dataset size
        # dataset = dataset.shuffle(buffer_size=self.getSize(), reshuffle_each_iteration=True)
        dataset = dataset.shuffle(buffer_size=self.getSize())
        self.dataset = dataset
        return dataset

    def splitDataset(self, bs=32, trainRatio=0.7):
        trainSize = int(trainRatio * self.getSize())
        testSize = self.getSize() - trainSize
        trainDataset = self.dataset.take(trainSize)
        testDataset = self.dataset.skip(trainSize)
        # set batch size
        trainDataset = trainDataset.batch(bs)
        testDataset = testDataset.batch(bs)
        return trainSize, testSize, trainDataset, testDataset
