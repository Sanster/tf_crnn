import tensorflow as tf

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')

tf.app.flags.DEFINE_string('tag', 'default', 'sub dir created in checkpoint_dir/log_dir/result_dir')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint', 'save tensorflow checkpoint')
tf.app.flags.DEFINE_string('log_dir', './log', 'save log of tensorflwo')
tf.app.flags.DEFINE_string('result_dir', './result', 'save val/test result')
tf.app.flags.DEFINE_string('chars_file', './data/chars/chn_10.txt', 'chars file to load in data dir')

# LSTM
tf.app.flags.DEFINE_integer('lstm_layer', 2, 'LSTM layers after CNN, 1 or 2')
tf.app.flags.DEFINE_integer('num_hidden', 256, 'number of hidden units in lstm')

# CNN
tf.app.flags.DEFINE_string('cnn', 'raw', 'raw/raw_light')

tf.app.flags.DEFINE_integer('num_epochs', 2000, 'epochs num')
tf.app.flags.DEFINE_integer('step_do_val', 3000, 'steps to do val/test images and save checkpoint')
tf.app.flags.DEFINE_integer('step_write_summary', 20, 'steps to write summary')
tf.app.flags.DEFINE_integer('batch_size', 64, 'the batch_size')
tf.app.flags.DEFINE_string('decode_mode', 'ctc', 'softmax or ctc')

# Optimizer
tf.app.flags.DEFINE_string('optim', 'adadelate', 'Optimizer to use: adadelate, adam, rms')
tf.app.flags.DEFINE_float('lr', 0.01, 'inital learning rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'adam optimizer parameter: beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam optimizer parameter: beta2')

tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')

tf.app.flags.DEFINE_float('keep_prob', 1.0, 'RNN dropout keep prob')

tf.app.flags.DEFINE_string('train_dir', '../output/train', 'the train data dir')
tf.app.flags.DEFINE_integer('num_train', None, 'the image num used to train. if None: load all images in train_dir')
tf.app.flags.DEFINE_string('val_dir', '../output/val', 'the val data dir')
tf.app.flags.DEFINE_string('test_dir', '../output/val', 'the test data dir')

tf.app.flags.DEFINE_string('infer_dir', '/home/cwq/code/tensorflow_lstm_ctc_ocr/test', 'the infer data dir')
tf.app.flags.DEFINE_boolean('infer_copy_failed', False, 'copy failed image to {result_dir}/{tag}/infer/failed')
tf.app.flags.DEFINE_boolean('infer_remove_symbol', False, 'remove symbols for inder result')

tf.app.flags.DEFINE_string('mode', 'train', 'train or infer')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'num of gpus')

FLAGS = tf.app.flags.FLAGS
