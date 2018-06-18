import tensorflow as tf
import os
import cv2
import numpy as np
import uuid

from tensorflow.python.framework import dtypes
import libs.utils as utils

"""
    使用 Dataset api 并行读取图片数据
    参考：
        - 关于 TF Dataset api 的改进讨论：https://github.com/tensorflow/tensorflow/issues/7951
        - https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
        - https://stackoverflow.com/questions/47064693/tensorflow-data-api-prefetch
        - https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle

    TL;DR
        Dataset.shuffle() 的 buffer_size 参数影响数据的随机性， TF 会先取 buffer_size 个数据放入 catch 中，再从里面选取
        batch_size 个数据，所以使用 shuffle 有两种方法：
            1. 每次调用 Dataset api 前手动 shuffle 一下 filepaths 和 labels
            2. Dataset.shuffle() 的 buffer_size 直接设为 len(filepaths)。这种做法要保证 shuffle() 函数在 map、batch 前调用

        Dataset.prefetch() 的 buffer_size 参数可以提高数据预加载性能，但是它比 tf.FIFOQueue 简单很多。
        tf.FIFOQueue supports multiple concurrent producers and consumers
"""


# noinspection PyMethodMayBeStatic
class ImgDataset:
    """
    Use tensorflow Dataset api to load images in parallel
    """

    def __init__(self, img_dir,
                 converter,
                 batch_size,
                 num_parallel_calls=4,
                 img_channels=3,
                 shuffle=True):
        self.converter = converter
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.img_channels = img_channels
        self.img_dir = img_dir
        self.shuffle = shuffle

        labels = utils.load_labels(os.path.join(img_dir, 'labels.txt'))
        img_paths = utils.build_img_paths(img_dir, len(labels))
        self.size = len(labels)

        dataset = self._create_dataset(img_paths, labels)
        iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                   dataset.output_shapes)
        self.next_batch = iterator.get_next()
        self.init_op = iterator.make_initializer(dataset)

    def get_next_batch(self, sess):
        """return images and labels of a batch"""
        img_batch, labels, img_paths = sess.run(self.next_batch)
        img_paths = [p.decoded() for p in img_paths]
        labels = [l.decoded() for l in labels]

        encoded_label_batch = self.converter.encode_list(labels)
        sparse_label_batch = utils.sparse_tuple_from_label(encoded_label_batch)
        return img_batch, sparse_label_batch, (labels, encoded_label_batch), img_paths

    def _create_dataset(self, img_paths, labels):
        img_paths = tf.convert_to_tensor(img_paths, dtype=dtypes.string)
        labels = tf.convert_to_tensor(labels, dtype=dtypes.string)

        d = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        if self.shuffle:
            d = d.shuffle(buffer_size=self.size)

        d = d.map(self._input_parser,
                  num_parallel_calls=self.num_parallel_calls)
        d = d.batch(self.batch_size)
        # d = d.repeat(self.num_epochs)
        d = d.prefetch(buffer_size=2)
        return d

    def _input_parser(self, img_path, label):
        img_file = tf.read_file(img_path)

        img_decoded = tf.image.decode_image(img_file, channels=self.img_channels)
        if self.img_channels == 3:
            img_decoded = tf.image.rgb_to_grayscale(img_decoded)

        img_decoded = tf.cast(img_decoded, tf.float32)
        img_decoded = (img_decoded - 128.0) / 128.0

        return img_decoded, label, img_path
