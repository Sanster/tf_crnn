import tensorflow as tf
from nets.cnn.raw_cnn import RawCNN


class CRNN(object):
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

        self._build_graph()

    def _build_graph(self):
        self._build_model()
        self._build_train_op()

        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        if self.FLAGS.cnn == 'raw':
            num_feature_maps = [64, 128, 256, 256, 512, 512, 512]
        elif self.FLAGS.cnn == 'raw_light':
            num_feature_maps = [32, 64, 128, 256, 256, 256, 512]

        cnn_out = RawCNN(self.inputs, num_feature_maps, self.is_training)

        cnn_output_shape = tf.shape(cnn_out)
        batch_size = cnn_output_shape[0]
        cnn_output_h = cnn_output_shape[1]
        cnn_output_w = cnn_output_shape[2]

        self.seq_len = tf.ones([batch_size], tf.int32) * cnn_output_w

        with tf.variable_scope('bidirectional_lstm'):
            # 如果图片尺寸为 32 * 100
            # 论文中最后两个 pooling 层添加了额外的 padding，所以 cnn 的输出为 1x26
            # 目前 cnn_output shape is [-1, 1, 24, 512], lstm 的 max_time 应该等于 24
            # convert to [-1, 24, 512]
            cnn_out = tf.reshape(cnn_out, [-1, cnn_output_h * cnn_output_w, num_feature_maps[-1]])

            # 错误：因为最终的 cnn_output_h 为 1，所以转置就相当于把 feature map 的每一列组合成 feature vector 示例：(-1, 24, 512) -> (-1, 512, 24);
            # 正确： dynamic_rnn 的 inputs 维度为，[batch_size, max_time, ...], 上一步 reshape 出来的已经满足了！不需要再 transpose
            # x = tf.transpose(x, [0, 2, 1])

            blstm = cnn_out

            for i in range(self.FLAGS.lstm_layer):
                with tf.variable_scope("blstm_{}".format(i + 1)):
                    # 最后一层 lstm 应该输出分类结果数量
                    if i == (self.FLAGS.lstm_layer - 1):
                        blstm = self._bidirectional_LSTM2(blstm, self.num_classes)
                    else:
                        blstm = self._bidirectional_LSTM2(blstm, self.FLAGS.num_hidden)

            self.logits = blstm

            # ctc require time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))

    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)

        # labels:   An `int32` `SparseTensor`.
        #           `labels.indices[i, :] == [b, t]` means `labels.values[i]` stores
        #           the id for (batch b, time t).
        #           `labels.values[i]` must take on values in `[0, num_labels)`.
        # inputs shape: [max_time, batch_size, num_classes]`
        self.loss = tf.nn.ctc_loss(labels=self.labels,
                                   inputs=self.logits,
                                   sequence_length=self.seq_len)
        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)

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
            # For RMS lr should be 0.001
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr,
                                                       epsilon=1e-8)
        elif self.FLAGS.optim == 'adadelate':
            # For Adadelta lr should be 0.01
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr,
                                                        rho=0.9,
                                                        epsilon=1e-06)

        # required by tf.layers.batch_normalization()
        # add update ops(for moving_mean and moving_variance) as a dependency to the train_op
        # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

        # inputs shape: [max_time x batch_size x num_classes]
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(self.logits, self.seq_len, merge_repeated=True)

        # it's slower but you'll get better results
        # self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(inputs=self.logits,
        #                                                             sequence_length=self.seq_len,
        #                                                             merge_repeated=False)

        # dense_decoded shape: [batch_size, encoded_code_size(not fix)]
        # use tf.cast here to support run model on Android
        self.dense_decoded = tf.sparse_tensor_to_dense(tf.cast(self.decoded[0], tf.int32), default_value=-1)

        # batch labels error rate
        self.edit_distance = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels)
        self.edit_distance_mean = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))

    def _LSTM_cell(self, num_proj=None):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.FLAGS.num_hidden, num_proj=num_proj)
        if self.FLAGS.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.FLAGS.keep_prob)
        return cell

    def _bidirectional_LSTM(self, inputs, num_proj):
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

    def _bidirectional_LSTM2(self, inputs, num_out):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self._LSTM_cell(),
                                                     self._LSTM_cell(),
                                                     inputs,
                                                     sequence_length=self.seq_len,
                                                     dtype=tf.float32)

        outputs = tf.concat(outputs, 2)

        W = tf.get_variable(name='w',
                            shape=[self.FLAGS.num_hidden * 2, num_out],
                            dtype=tf.float32,
                            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(name='b',
                            shape=[num_out],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer())

        outputs = tf.reshape(outputs, [-1, self.FLAGS.num_hidden * 2])

        logits = tf.matmul(outputs, W) + b

        shape = tf.shape(inputs)
        logits = tf.reshape(logits, [shape[0], -1, num_out])

        return logits
