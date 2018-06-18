import tensorflow.contrib.slim as slim


def PaperCNN(inputs, is_training):
    """
    Net structure described in crnn paper
    feature_maps = [64, 128, 256, 256, 512, 512, 512]
    """
    net = slim.conv2d(inputs, 64, 3, 1, scope='conv1')
    net = slim.max_pool2d(net, 2, 2, scope='pool1')
    net = slim.conv2d(net, 3, 128, 1, scope='conv2')
    net = slim.max_pool2d(net, 2, 2, scope='pool2')
    net = slim.conv2d(net, 3, 256, scope='conv3')
    net = slim.conv2d(net, 2, 256, scope='conv4')
    net = slim.max_pool2d(net, 2, [2, 1], scope='pool3')
    net = slim.conv2d(net, 3, 512, normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training},
                      scope='conv5')
    net = slim.conv2d(net, 3, 512, normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training},
                      scope='conv6')
    net = slim.max_pool2d(net, 2, [2, 1], scope='pool4')
    net = slim.conv2d(net, 2, 512, padding='VALID', scope='conv7')
    return net
