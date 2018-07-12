import argparse
import os
import tensorflow as tf

import libs.utils as utils
from libs.infer import validation
from nets.crnn import CRNN
from libs.label_converter import LabelConverter
from libs.img_dataset import ImgDataset

from parse_args import parse_args


def infer(args):
    converter = LabelConverter(chars_file=args.chars_file)
    model = CRNN(args, num_classes=converter.num_classes)
    dataset = ImgDataset(args.infer_dir, converter, args.batch_size, shuffle=False)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.global_variables())
        utils.restore_ckpt(sess, saver, args.ckpt_dir)

        step = sess.run(model.global_step)
        validation(sess, model, dataset, converter, step, args.result_dir, name='infer', print_batch_info=True)


def main():
    args = parse_args(infer=True)
    if args.gpu:
        dev = '/gpu:0'
    else:
        dev = '/cpu:0'

    with tf.device(dev):
        infer(args)


if __name__ == '__main__':
    main()
