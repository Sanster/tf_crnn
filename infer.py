import argparse
import os
import tensorflow as tf

import libs.utils as utils
from libs.config import load_config
from libs.infer import validation
from nets.crnn import CRNN
from libs.label_converter import LabelConverter
from libs.img_dataset import ImgDataset

from parse_args import parse_args


# http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
def restore_ckpt(sess, checkpoint_dir):
    print("Restoring checkpoint from: " + checkpoint_dir)

    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt is None:
        print("Checkpoint not found")
        exit(-1)

    meta_file = ckpt + '.meta'
    try:
        print('Restore graph from {}'.format(meta_file))
        print('Restore variables from {}'.format(ckpt))
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, ckpt)
    except Exception:
        raise Exception("Can not restore from {}".format(checkpoint_dir))


def infer(args):
    converter = LabelConverter(chars_file=args.chars_file)
    dataset = ImgDataset(args.infer_dir, converter, args.infer_batch_size, shuffle=False)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        restore_ckpt(sess, args.ckpt_dir)

        # for node in sess.graph.as_graph_def().node:
        #     print(node.name)

        # https://stackoverflow.com/questions/46912721/tensorflow-restore-model-with-sparse-placeholder
        labels_placeholder = tf.SparseTensor(
            values=sess.graph.get_tensor_by_name('labels/values:0'),
            indices=sess.graph.get_tensor_by_name('labels/indices:0'),
            dense_shape=sess.graph.get_tensor_by_name('labels/shape:0')
        )

        feeds = {
            'inputs': sess.graph.get_tensor_by_name('inputs:0'),
            'is_training': sess.graph.get_tensor_by_name('is_training:0'),
            'labels': labels_placeholder
        }

        fetches = [
            sess.graph.get_tensor_by_name('SparseToDense:0'),  # dense_decoded
            sess.graph.get_tensor_by_name('Mean_1:0'),  # mean edit distance
            sess.graph.get_tensor_by_name('edit_distance:0')  # batch edit distances
        ]

        validation(sess, feeds, fetches,
                   dataset, converter, args.result_dir, name='infer',
                   print_batch_info=True, copy_failed=args.infer_copy_failed)


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
