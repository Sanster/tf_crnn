import tensorflow as tf
import tensorflow.contrib.slim as slim

from nets.cnn.paper_cnn import PaperCNN
from nets.cnn.dense_net import DenseNet
from nets.cnn.squeeze_net import SqueezeNet


class CRNN(object):
    CTC_INVALID_INDEX = -1

    def __init__(self, FLAGS, num_classes):
        self.inputs = tf.placeholder(tf.float32,
                                     [None, 32, None, 1],
                                     name="inputs")
        # SparseTensor required by ctc_loss op
        self.labels = tf.sparse_placeholder(tf.int32, name="labels")
        # 1d array of size [batch_size]
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.FLAGS = FLAGS
        self.num_classes = num_classes

        self._build_model()
        self._build_train_op()

        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        if self.FLAGS.cnn == 'raw':
            cnn_out = PaperCNN(self.inputs, self.is_training)
        elif self.FLAGS.cnn == 'dense':
            net = DenseNet(self.inputs, self.is_training)
            cnn_out = net.net
        elif self.FLAGS.cnn == 'squeeze':
            net = SqueezeNet(self.inputs, self.is_training)
            cnn_out = net.net

        cnn_output_shape = tf.shape(cnn_out)
        batch_size = cnn_output_shape[0]
        cnn_output_h = cnn_output_shape[1]
        cnn_output_w = cnn_output_shape[2]
        cnn_output_channel = cnn_output_shape[3]

        # Get seq_len according to cnn output, so we don't need to input this as a placeholder
        self.seq_len = tf.ones([batch_size], tf.int32) * cnn_output_w

        # Reshape to the shape lstm need. [batch_size, max_time, ..]
        lstm_inputs = tf.reshape(cnn_out, [-1, cnn_output_w, cnn_output_h * cnn_output_channel])

        with tf.variable_scope('bilstm1'):
            bilstm = self._bidirectional_LSTM(lstm_inputs, self.FLAGS.num_hidden)

        with tf.variable_scope('bilstm2'):
            bilstm = self._bidirectional_LSTM(bilstm, self.num_classes)

        # ctc require time major
        self.logits = tf.transpose(bilstm, (1, 0, 2))

    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)

        # labels:   An `int32` `SparseTensor`.
        #           `labels.indices[i, :] == [b, t]` means `labels.values[i]` stores
        #           the id for (batch b, time t).
        #           `labels.values[i]` must take on values in `[0, num_labels)`.
        # inputs shape: [max_time, batch_size, num_classes]`
        self.ctc_loss = tf.nn.ctc_loss(labels=self.labels,
                                       inputs=self.logits,
                                       sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.ctc_loss)
        tf.summary.scalar('ctc_loss', self.cost)

        self.lr = tf.train.exponential_decay(self.FLAGS.lr,
                                             self.global_step,
                                             self.FLAGS.decay_steps,
                                             self.FLAGS.decay_rate,
                                             staircase=True)
        tf.summary.scalar("learning_rate", self.lr)

        if self.FLAGS.optim == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                    beta1=self.FLAGS.beta1,
                                                    beta2=self.FLAGS.beta2)
        elif self.FLAGS.optim == 'rms':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr,
                                                       epsilon=1e-8)
        elif self.FLAGS.optim == 'adadelate':
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr,
                                                        rho=0.9,
                                                        epsilon=1e-06)

        # required by batch normalize
        # add update ops(for moving_mean and moving_variance) as a dependency to the train_op
        # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.ctc_loss, global_step=self.global_step)

        # inputs shape: [max_time x batch_size x num_classes]
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.seq_len, merge_repeated=True)

        # dense_decoded shape: [batch_size, encoded_code_size(not fix)]
        # use tf.cast here to support run model on Android
        self.dense_decoded = tf.sparse_tensor_to_dense(tf.cast(self.decoded[0], tf.int32),
                                                       default_value=self.CTC_INVALID_INDEX)

        # Edit distance for wrong result
        self.edit_distances = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels)

        non_zero_indices = tf.where(tf.not_equal(self.edit_distances, 0))
        self.edit_distance = tf.reduce_mean(tf.gather(self.edit_distances, non_zero_indices))

    def _LSTM_cell(self, num_proj=None):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.FLAGS.num_hidden, num_proj=num_proj)
        if self.FLAGS.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.FLAGS.keep_prob)
        return cell

    def _paper_bidirectional_LSTM(self, inputs, num_proj):
        """
            根据 CRNN BiRnnJoin.lua 源码改写
        :param inputs: shape [batch_size, max_time, ...]
        :param num_proj: 每个 cell 输出的维度
        :return: shape [batch_size, max_time, num_proj]
        """
        (blstm_fw, blstm_bw), _ = tf.nn.bidirectional_dynamic_rnn(self._LSTM_cell(num_proj=num_proj),
                                                                  self._LSTM_cell(num_proj=num_proj),
                                                                  inputs,
                                                                  sequence_length=self.seq_len,
                                                                  dtype=tf.float32)
        return tf.add(blstm_fw, blstm_bw)

    def _bidirectional_LSTM(self, inputs, num_out):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self._LSTM_cell(),
                                                     self._LSTM_cell(),
                                                     inputs,
                                                     sequence_length=self.seq_len,
                                                     dtype=tf.float32)

        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, self.FLAGS.num_hidden * 2])

        outputs = slim.fully_connected(outputs, num_out)

        shape = tf.shape(inputs)
        outputs = tf.reshape(outputs, [shape[0], -1, num_out])

        return outputs
