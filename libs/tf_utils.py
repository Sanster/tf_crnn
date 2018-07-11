from functools import reduce
import os

import tensorflow as tf


def add_scalar_summary(writer, tag, val, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
    writer.add_summary(summary, step)


def print_endpoints(net, inputs, is_training, img_path, CPU=True):
    cnn_output_shape = tf.shape(net.net)
    cnn_output_h = cnn_output_shape[1]
    cnn_output_w = cnn_output_shape[2]
    cnn_output_channel = cnn_output_shape[3]

    cnn_out = tf.transpose(net.net, [0, 2, 1, 3])
    cnn_out = tf.reshape(cnn_out, [-1, cnn_output_w, cnn_output_h * cnn_output_channel])

    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=1)

    conv_count = 0
    if CPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        img = sess.run(img_decoded)
        sess.run(net.net, feed_dict={inputs: [img], is_training: True})

        for k, v in net.end_points.items():
            if 'Conv' in k:
                conv_count += 1
            print("%s: %s" % (k, v.shape))

        cnn_out = sess.run(cnn_out, feed_dict={inputs: [img], is_training: True})

    def size(v):
        return reduce(lambda x, y: x * y, v.get_shape().as_list())

    print("-" * 50)

    n = sum(size(v) for v in tf.trainable_variables())
    print("Tensorflow trainable params: %.02fM (%dK)" % (n / 1000000, n / 1000))
    print("Conv layer count: %d" % conv_count)
    print("Output shape: {}".format(net.net))
    print('Cnn out reshaped for lstm: ')
    print(cnn_out.shape)
