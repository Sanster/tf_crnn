"""
https://github.com/TracyMcgrady6/Distribute_MNIST
"""

# encoding:utf-8
import math
import tempfile
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
IMAGE_PIXELS = 28
# 定义默认训练参数和数据路径
flags.DEFINE_string('data_dir', '/tmp/mnist-data', 'Directory  for storing mnist data')
flags.DEFINE_integer('hidden_units', 100, 'Number of units in the hidden layer of the NN')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps to perform')
flags.DEFINE_integer('batch_size', 100, 'Training batch size ')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
# 定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '192.168.59.160:22220', 'Comma-separated list of hostname:port pairs')

# 两个worker节点
flags.DEFINE_string('worker_hosts', '192.168.59.160:22221,192.168.59.165:22221',
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', None, 'job name: worker or ps(parameter server)')
# 设置任务的索引
flags.DEFINE_integer('task_index', None, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")

FLAGS = flags.FLAGS


def main(unused_argv):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print('task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # 创建集群
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index, config=config)

    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    if is_chief:
        print('Worker %d: Initailizing session...' % FLAGS.task_index)
    else:
        print('Worker %d: Waiting for session to be initaialized...' % FLAGS.task_index)

    with tf.device(tf.train.replica_device_setter(
            cluster=cluster
    )):
        global_step = tf.Variable(0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量

        hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                                stddev=1.0 / IMAGE_PIXELS), name='hid_w')
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')

        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32, [None, 10])

        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)

        out_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10]), name='out_w')
        out_b = tf.Variable(tf.zeros([10]), name='out_b')
        logits = tf.nn.xw_plus_b(hid, out_w, out_b)

        prediction = tf.nn.softmax(logits, name="prediction")

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits))
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)

        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        hooks = [tf.train.StopAtStepHook(last_step=FLAGS.train_steps)]

        local_step = 0
        step = 0
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               config=config,
                                               is_chief=is_chief,
                                               checkpoint_dir="/tmp/mnist",
                                               hooks=hooks) as mon_sess:
            while not mon_sess.should_stop():
                # SyncReplicasOptimizer perform *synchronous* training.
                if step == 99:
                    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
                    acc = mon_sess.run(accuracy, feed_dict=val_feed)
                    print("Val acc: %f" % acc)
                else:
                    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
                    train_feed = {x: batch_xs, y_: batch_ys}

                    # Run a training step asynchronously.
                    _, step = mon_sess.run([train_op, global_step], feed_dict=train_feed)
                    local_step += 1

                    print('%f: Worker %d: traing step %d dome (global step:%d)' % (
                        time.time(), FLAGS.task_index, local_step, step))

            print("Local run steps: %d" % local_step)


if __name__ == '__main__':
    tf.app.run()
