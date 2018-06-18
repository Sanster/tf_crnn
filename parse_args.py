#!/usr/env/bin python3
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag', default='default', help='Subdirectory to create in checkpoint_dir/log_dir/result_dir')
    parser.add_argument('--ckpt_dir', default='./checkpoint', help='Directory to save tensorflow checkpoint')
    parser.add_argument('--log_dir', default='./log', help='Directory to save tensorboard logs')
    parser.add_argument('--result_dir', default='./result', help='Directory to save val/test result')

    parser.add_argument('--chars_file', default='./data/chars/chn.txt', help='Chars file to load')

    parser.add_argument('--epochs', type=int, default=100, help='Epochs to run')
    parser.add_argument('--val_step', type=int, default=2000, htlp='Steps to do val.test and save checkpoint')

    # Hyper parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('lr', type=float, default=0.01, help='Initial learning rate')

    parser.add_argument('--optim', default='adadelate', choices=['adadelate', 'adam', 'rms'])
    tf.app.flags.DEFINE_float('beta1', 0.9, 'adam optimizer parameter: beta1')
    tf.app.flags.DEFINE_float('beta2', 0.999, 'adam optimizer parameter: beta2')

    tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
    tf.app.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')

    tf.app.flags.DEFINE_float('keep_prob', 1.0, 'RNN dropout keep prob')

    tf.app.flags.DEFINE_string('train_dir', '../output/train', 'the train data dir')
    tf.app.flags.DEFINE_integer('num_train', None, 'the image num used to train. if None: load all images in train_dir')
    tf.app.flags.DEFINE_string('val_dir', '../output/val', 'the val data dir')
    tf.app.flags.DEFINE_string('test_dir', '../output/val', 'the test data dir')

    FLAGS = tf.app.flags.FLAGS
    help = 'Number of chars in a image, only works for chn corpus_mode')
    parser.add_argument('--img_height', type=int, default=32)
    parser.add_argument('--img_width', type=int, default=256)

    parser.add_argument('--debug', action='store_true', default=False, help="output uncroped image")
    parser.add_argument('--viz', action='store_true', default=False)

    parser.add_argument('--chars_file', type=str, default='./data/chars/chn.txt')

    parser.add_argument('--fonts_dir', type=str, default='./data/fonts/chn')
    parser.add_argument('--bg_dir', type=str, default='./data/bg')
    parser.add_argument('--corpus_dir', type=str, default='./data/corpus')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--tag', type=str, default='default', help='output images are saved under output_dir/{tag} dir')

    parser.add_argument('--line', action='store_true', default=False)
    parser.add_argument('--noise', action='store_true', default=False)

    flags, _ = parser.parse_known_args()
    flags.save_dir = os.path.join(flags.output_dir, flags.tag)

    if os.path.exists(flags.bg_dir):
        num_bg = len(os.listdir(flags.bg_dir))
    flags.num_bg = num_bg

    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)

    return flags
