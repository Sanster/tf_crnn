import os
import numpy as np
import tensorflow as tf
import cv2
from functools import reduce


def load_chars(filepath):
    if not os.path.exists(filepath):
        print("Chars file not exists. %s" % filepath)
        exit(1)

    ret = ''
    with open(filepath, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            ret += line[0]
    return ret


def load_labels(filepath, img_num=None):
    if not os.path.exists(filepath):
        print("Label file not exists. %s" % filepath)
        exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        labels = f.readlines()

    if img_num and img_num <= len(labels):
        labels = labels[0:img_num]

    # 移除换行符、首尾空格
    labels = [l[:-1].strip() for l in labels]
    return labels




# https://stackoverflow.com/questions/49063938/padding-labels-for-tensorflow-ctc-loss
def dense_to_sparse(dense_tensor, sparse_val=-1):
    """Inverse of tf.sparse_to_dense.

    Parameters:
        dense_tensor: The dense tensor. Duh.
        sparse_val: The value to "ignore": Occurrences of this value in the
                    dense tensor will not be represented in the sparse tensor.
                    NOTE: When/if later restoring this to a dense tensor, you
                    will probably want to choose this as the default value.
    Returns:
        SparseTensor equivalent to the dense input.
    """
    with tf.name_scope("dense_to_sparse"):
        sparse_inds = tf.where(tf.not_equal(dense_tensor, sparse_val),
                               name="sparse_inds")
        sparse_vals = tf.gather_nd(dense_tensor, sparse_inds,
                                   name="sparse_vals")
        dense_shape = tf.shape(dense_tensor, name="dense_shape",
                               out_type=tf.int64)
        return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)


def check_dir_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def restore_ckpt(sess, saver, checkpoint_dir):
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    try:
        saver.restore(sess, ckpt)
        print('Restore checkpoint from {}'.format(ckpt))
    except Exception as e:
        print(e)
        print("Can not restore from {}".format(checkpoint_dir))
        exit(-1)


def count_tf_params():
    """print number of trainable variables"""

    def size(v):
        return reduce(lambda x, y: x * y, v.get_shape().as_list())

    n = sum(size(v) for v in tf.trainable_variables())
    print("Tensorflow Model size: %dK" % (n / 1000,))
    return n


def get_img_paths_and_labels(img_dir):
    """label 位于文件名中"""
    img_paths = []
    labels = []

    for root, sub_folder, file_list in os.walk(img_dir):
        for idx, file_name in enumerate(sorted(file_list)):
            image_path = os.path.join(root, file_name)
            img_paths.append(image_path)

            # 00000_abcd.png
            label = file_name[:-4].split('_')[1]
            labels.append(label)

    return img_paths, labels


def get_img_paths_and_labels2(img_dir):
    """label 位于同名 txt 文件中"""
    img_paths = []
    labels = []

    def read_label(p):
        with open(p, mode='r', encoding='utf-8') as f:
            data = f.read()
        return data

    for root, sub_folder, file_list in os.walk(img_dir):
        for idx, file_name in enumerate(sorted(file_list)):
            if file_name.endswith('.jpg') and os.path.exists(os.path.join(img_dir, file_name)):
                image_path = os.path.join(root, file_name)
                img_paths.append(image_path)
                label_path = os.path.join(root, file_name[:-4] + '.txt')
                labels.append(read_label(label_path))
            else:
                print('file not found: {}'.format(file_name))

    return img_paths, labels


def get_img_paths_and_label_paths(img_dir, img_count):
    img_paths = []
    label_paths = []
    for i in range(img_count):
        base_name = "{:08d}".format(i)
        img_path = os.path.join(img_dir, base_name + ".jpg")
        label_path = os.path.join(img_dir, base_name + ".txt")
        img_paths.append(img_path)
        label_paths.append(label_path)

    return img_paths, label_paths


def build_img_paths(img_dir, img_count):
    """
    Image name should be eight length with continue num. e.g. 00000000.jpg, 00000001.jgp
    """
    img_paths = []
    for i in range(img_count):
        base_name = "{:08d}".format(i)
        img_path = os.path.join(img_dir, base_name + ".jpg")
        img_paths.append(img_path)

    return img_paths
