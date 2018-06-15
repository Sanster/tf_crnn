import tensorflow as tf


def _leaky_relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

def conv2d(x, filter_size, in_channels, out_channels, is_training, stride=1, batch_norm=False, padding=True):
    # TODO: check witch initializer to use
    kernel = tf.get_variable(name='weights',
                             shape=[filter_size, filter_size, in_channels, out_channels],
                             dtype=tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer())

    if padding:
        con2d_op = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding='SAME')
    else:
        con2d_op = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding='VALID')

    if not batch_norm:
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer())
        con2d_op = tf.nn.bias_add(con2d_op, b)

    if batch_norm:
        con2d_op = tf.layers.batch_normalization(con2d_op,
                                                 training=is_training,
                                                 momentum=0.9,
                                                 epsilon=1e-05)

    con2d_op = tf.nn.relu(con2d_op)

    return con2d_op


def max_pool(x, ksize_h, ksize_w, stride_h=2, stride_w=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize_h, ksize_w, 1],
                          strides=[1, stride_h, stride_w, 1],
                          padding='SAME',
                          name='max_pool')
