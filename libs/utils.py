import os
import numpy as np
import tensorflow as tf
import cv2
import logging
from functools import reduce

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)


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


def remove_symbols(labels):
    symbol_path = '../data/chars/symbol.txt'
    symbols = load_chars(symbol_path)
    out = []
    for l in labels:
        fl = list(filter(lambda x: x not in symbols, l))
        fl = ''.join(fl)
        out.append(fl)
    return out


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


def sparse_tuple_from_label(sequences, default_val=-1, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
                  encode label, e.g: [2,44,11,55]
        default_val: value should be ignored in sequences
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        seq_filtered = list(filter(lambda x: x != default_val, seq))
        indices.extend(zip([n] * len(seq_filtered), range(len(seq_filtered))))
        values.extend(seq_filtered)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)

    if len(indices) == 0:
        shape = np.asarray([len(sequences), 0], dtype=np.int64)
    else:
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def check_dir_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def restore_ckpt(sess, saver, checkpoint_dir):
    print("Restoring checkpoint from: " + checkpoint_dir)
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    try:
        saver.restore(sess, ckpt)
        logging.info('restore from ckpt {}'.format(ckpt))
    except Exception:
        raise Exception("Can not restore from {}".format(checkpoint_dir))


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
                logging.info('file not found: {}'.format(file_name))

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


def acc_under_edit_distances(edit_distances, threshold):
    """
    将 edit_distance 小于 threshold 的预测结果也认为是全对，并计算"全对率"
    """
    count = 0
    for ed in edit_distances:
        if ed < threshold:
            count += 1
    acc = count / len(edit_distances)
    result = "Accuracy edit_distance < {}: {:.03f}".format(threshold, acc)
    logging.info(result)
    return result


if __name__ == '__main__':
    # im = cv2.imdecode(np.fromfile('../data/1.jpg', dtype=np.uint8), 0)
    # cv2.imencode('.jpg', im)[1].tofile("../data/测试.jpg")

    a, b = get_img_paths_and_labels2('/home/cwq/code/Gword/output/default')
    aa = 0

    # Test load_labels(filepath, img_num)
    labels = load_labels("/home/cwq/code/ocr_end2end/output/val/labels.txt", 500)
    assert len(labels) == 500
