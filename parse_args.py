#!/usr/env/bin python3
import argparse
import os

import datetime

from libs.utils import check_dir_exist


def save_flags(args, save_dir):
    """
    Save flags into file, use date as file name
    :param args: tf.app.flags
    :param save_dir: dir to save flags file
    """
    filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, filename)
    print("Save flags to %s" % filepath)

    with open(filepath, mode="w", encoding="utf-8") as f:
        d = vars(args)
        for k, v in d.items():
            f.write("%s: %s\n" % (k, v))


def parse_args(infer=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('--restore', action='store_true', help='Whether to resotre checkpoint from ckpt_dir')
    parser.add_argument('--tag', default='default', help='Subdirectory to create in checkpoint_dir/log_dir/result_dir')
    parser.add_argument('--ckpt_dir', default='./output/checkpoint', help='Directory to save tensorflow checkpoint')
    parser.add_argument('--log_dir', default='./output/log', help='Directory to save tensorboard logs')
    parser.add_argument('--result_dir', default='./output/result', help='Directory to save val/test result')

    parser.add_argument('--chars_file', default='./data/chars/chn.txt', help='Chars file to load')

    parser.add_argument('--val_step', type=int, default=2000, help='Steps to do val.test and save checkpoint')
    parser.add_argument('--log_step', type=int, default=30, help='Steps save tensorflow summary')

    parser.add_argument('--train_dir', default='', help='Directory store training images and labels')
    parser.add_argument('--val_dir', default=None, help='Directory store validation images and labels')
    parser.add_argument('--test_dir', default=None, help='Directory store test images and labels')
    parser.add_argument('--cfg_name', default='raw', help="raw / squeeze/ dense / resnet / simple")

    # Only for inference
    parser.add_argument('--infer_dir', default='./data/demo', help='Directory store infer images and labels')
    parser.add_argument('--infer_copy_failed', action='store_true', default=False,
                        help='Copy failed image to result directory')
    parser.add_argument('--infer_batch_size', type=int, default=1)

    args, _ = parser.parse_known_args()

    if (not infer) and (not os.path.exists(args.train_dir)):
        parser.error('train_dir not exist')

    if (args.val_dir is not None) and (not os.path.exists(args.val_dir)):
        parser.error('val_dir not exist')

    if (args.test_dir is not None) and (not os.path.exists(args.test_dir)):
        parser.error('test_dir not exist')

    if infer and (not os.path.exists(args.infer_dir)):
        parser.error('infer_dir not exist')

    args.ckpt_dir = os.path.join(args.ckpt_dir, args.tag)
    args.flags_fir = os.path.join(args.ckpt_dir, "flags")
    args.log_dir = os.path.join(args.log_dir, args.tag)
    args.result_dir = os.path.join(args.result_dir, args.tag)

    check_dir_exist(args.ckpt_dir)
    check_dir_exist(args.flags_fir)
    check_dir_exist(args.log_dir)
    check_dir_exist(args.result_dir)

    save_flags(args, args.flags_fir)

    print('Use %s as base net' % args.cfg_name)

    return args


if __name__ == '__main__':
    args = parse_args()
