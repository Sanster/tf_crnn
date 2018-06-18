import argparse
import os
import tensorflow as tf

import libs.utils as utils
from libs.infer import validation
from nets.crnn import CRNN
from libs.label_converter import LabelConverter
from libs.img_dataset import ImgDataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--tag', default='default', help='Subdirectory to create in checkpoint_dir/log_dir/result_dir')
    parser.add_argument('--ckpt_dir', default='./output/checkpoint', help='Directory to save tensorflow checkpoint')
    parser.add_argument('--result_dir', default='./output/result', help='Directory to save val/test result')
    parser.add_argument('--chars_file', default='./data/chars/chn.txt', help='Chars file to load')
    parser.add_argument('--infer_dir', default='', help='Directory store infer images and labels')

    args, _ = parser.parse_known_args()

    if not os.path.exists(args.infer_dir):
        parser.error('infer_dir not exist')

    args.ckpt_dir = os.path.join(args.ckpt_dir, args.tag)
    args.result_dir = os.path.join(args.result_dir, args.tag)

    utils.check_dir_exist(args.result_dir)

    return args


if __name__ == '__main__':
    args = parse_args()


def infer(args):
    converter = LabelConverter(chars_file=args.chars_file)
    model = CRNN(args, num_classes=converter.num_classes)
    dataset = ImgDataset(args.infer_dir, converter, args.batch_size, shuffle=False)

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        utils.restore_ckpt(sess, saver, args.checkpoint_dir)

        step = sess.run(model.global_step)
        validation(sess, model, dataset, converter, step, args.result_dir, name='infer')


def main():
    args = parse_args()
    if args.gpu == 0:
        dev = '/gpu:0'
    else:
        dev = '/cpu:0'

    with tf.device(dev):
        infer(args)


if __name__ == '__main__':
    main()
