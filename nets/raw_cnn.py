from .net_util import *


def RawCNN(inputs, feature_maps, is_training):
    # maps: 64, k: 3x3, stride: 1, padding: 1
    # out: 16x90
    with tf.variable_scope('cnn_1'):
        conv1 = conv2d(inputs, 3, 1, feature_maps[0], is_training)
        pool1 = max_pool(conv1, 2, 2)

    # maps: 128, k: 3x3, stride: 1, padding: 1
    # out: 8x45
    with tf.variable_scope('cnn_2'):
        conv2 = conv2d(pool1, 3, feature_maps[0], feature_maps[1], is_training)
        pool2 = max_pool(conv2, 2, 2)

    # maps: 256, k: 3x3, stride: 1, padding: 1
    with tf.variable_scope('cnn_3'):
        conv3 = conv2d(pool2, 3, feature_maps[1], feature_maps[2], is_training)

    # maps: 256, k: 3x3, stride: 1, padding: 1
    # out: 4x45
    with tf.variable_scope('cnn_4'):
        conv4 = conv2d(conv3, 3, feature_maps[2], feature_maps[3], is_training)
        pool4 = max_pool(conv4, 2, 2, 2, 1)

    # maps: 512, k: 3x3, stride: 1, padding: 1, batch norm
    with tf.variable_scope('cnn_5'):
        conv5 = conv2d(pool4, 3, feature_maps[3], feature_maps[4], is_training, batch_norm=True)

    # maps: 512, k: 3x3, stride: 1, padding: 1, batch norm
    with tf.variable_scope('cnn_6'):
        conv6 = conv2d(conv5, 3, feature_maps[4], feature_maps[5], is_training, batch_norm=True)

    with tf.variable_scope('cnn_7'):
        # out: 2x45
        pool7 = max_pool(conv6, 2, 2, 2, 1)
        # out: 1x44
        conv7 = conv2d(pool7, 2, feature_maps[5], feature_maps[6], is_training, padding=False)

    return conv7
