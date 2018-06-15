import os
import time
import math
import sys
import shutil

import numpy as np
import tensorflow as tf

import libs.utils as utils
from nets.crnn import CRNN
from libs.label_converter import LabelConverter
from parse_args import FLAGS
from libs.img_dataset import ImgDataset
import libs.algorithms as algorithms

import libs.tf_utils as tf_utils

logging = utils.logging


def create_convert():
    converter = LabelConverter(chars_filepath=FLAGS.chars_file)

    logging.info(
        'Load chars file: {} num_classes: {} + 1(CTC Black)'.format(FLAGS.chars_file, converter.num_classes - 1))
    return converter, converter.num_classes


def create_img_dataset(converter, img_dir, img_count=None, shuffle=True, batch_size=FLAGS.batch_size):
    return ImgDataset(img_dir, img_count, converter, batch_size, shuffle=shuffle)


def create_model(num_classes):
    model = CRNN(FLAGS, num_classes=num_classes)
    return model


def add_train_acc_summary(sess, model, converter, writer, img_batch, label_batch, labels, global_step):
    _, _, edit_distances = do_infer_on_batch(sess, model, converter, img_batch, label_batch,
                                             labels)

    edit_distance_mean, correct_count = analyze_edit_distances(edit_distances)

    utils.tf_add_scalar_summary(writer, "train_accuracy", correct_count / len(img_batch),
                                global_step)

    utils.tf_add_scalar_summary(writer, "train_edit_distance", edit_distance_mean,
                                global_step)


def train():
    converter, num_classes = create_convert()

    tr_ds = create_img_dataset(converter, FLAGS.train_dir, FLAGS.num_train)
    val_ds = create_img_dataset(converter, FLAGS.val_dir, shuffle=False)
    test_ds = create_img_dataset(converter, FLAGS.test_dir, shuffle=False, batch_size=1)

    model = create_model(num_classes)

    # 有些 Op 不能在 GPU 上运行，自动使用 CPU 运行
    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
        train_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        num_batches = int(np.floor(tr_ds.size / FLAGS.batch_size))

        epoch_restored = 0
        batch_start = 0
        if FLAGS.restore:
            utils.restore_ckpt(sess, saver, FLAGS.checkpoint_dir)
            step_restored = sess.run(model.global_step)
            epoch_restored = math.floor(step_restored / num_batches)
            batch_start = step_restored % num_batches

            print("Restored global step: %d" % step_restored)
            print("Restored epoch: %d" % epoch_restored)
            print("Restored batch_start: %d" % batch_start)

        assert batch_start < num_batches

        logging.info('begin training...')
        for epoch in range(epoch_restored, FLAGS.num_epochs):
            sess.run(tr_ds.init_op)

            for batch in range(batch_start, num_batches):
                batch_start_time = time.time()

                img_batch, label_batch, (labels, _), _ = tr_ds.get_next_batch(sess)

                feed = {model.inputs: img_batch,
                        model.labels: label_batch,
                        model.is_training: True}

                fetches = [model.cost, model.global_step, model.train_op, model.lr]

                if batch != 0 and (batch % FLAGS.step_write_summary == 0):
                    fetches.append(model.merged_summay)
                    batch_cost, global_step, _, lr, summary_str = sess.run(fetches, feed)
                    train_writer.add_summary(summary_str, global_step)

                    add_train_acc_summary(sess, model, converter, train_writer, img_batch, label_batch, labels,
                                          global_step)
                else:
                    batch_cost, global_step, _, lr = sess.run(fetches, feed)

                print("epoch: {}, batch: {}/{}, step: {}, time: {:.02f}s, loss: {:.05}, lr: {:.05}"
                      .format(epoch, batch, num_batches, global_step, time.time() - batch_start_time,
                              batch_cost, lr))

                if global_step != 0 and (global_step % FLAGS.step_do_val == 0):
                    val_acc = do_val("val", sess, train_writer, model, val_ds, epoch, global_step)
                    test_acc = do_val("test", sess, train_writer, model, test_ds, epoch, global_step)
                    save_checkpoint(saver, sess, global_step, val_acc, test_acc)

            batch_start = 0


def save_checkpoint(saver, sess, step, val_acc, test_acc):
    logging.info("save the checkpoint {0}".format(step))
    name = os.path.join(FLAGS.checkpoint_dir, "crnn-{}-{:.03f}-{:.03f}".format(step, val_acc, test_acc))
    saver.save(sess, name)


def do_val(type, sess, writer, model, dataset, epoch, step):
    """
    :param type: val, test
    :return: accuracy
    """
    logging.info("do %s..." % type)
    accuracy, edit_distance_mean = validation(sess, model, dataset, type, global_step=step)

    utils.tf_add_scalar_summary(writer, "%s_accuracy" % type, accuracy, step)
    utils.tf_add_scalar_summary(writer, "%s_edit_distance" % type, edit_distance_mean, step)

    log = "epoch: {}/{}, %s accuracy = {:.3f}" % type
    logging.info(log.format(epoch, FLAGS.num_epochs, accuracy))
    return accuracy


def infer(img_dir):
    converter = LabelConverter(chars_filepath=FLAGS.chars_file)

    logging.info("Loading val data...")
    dataset = create_img_dataset(converter, img_dir, shuffle=False, batch_size=FLAGS.batch_size)
    logging.info("Loading finish...")

    model = CRNN(FLAGS, num_classes=converter.num_classes)

    saver = tf.train.Saver(tf.global_variables())
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        utils.restore_ckpt(sess, saver, FLAGS.checkpoint_dir)
        infer_dir_name = FLAGS.infer_dir.split('/')[-1]
        validation(sess, model, dataset, "infer", name=infer_dir_name, log_batch_acc=True)


def do_infer_on_batch(sess, model, converter, img_batch, label_batch, labels):
    # TODO: remove this, if add <space> in charset
    labels = list(map(lambda x: x.replace(" ", ""), labels))

    decoded_predict_labels = []
    decoded_target_labels = []

    val_feed = {model.inputs: img_batch,
                model.labels: label_batch,
                model.is_training: False}

    merge_repeat = False
    if FLAGS.decode_mode == 'ctc':
        invalid_index = -1
        predict_labels, edit_distances = sess.run([model.dense_decoded, model.edit_distance], val_feed)
        edit_distances = edit_distances.tolist()
    elif FLAGS.decode_mode == 'softmax':
        invalid_index = model.num_classes - 1
        merge_repeat = True
        predict_labels = sess.run(model.predict_softmax, val_feed)
        # edit_distances = cal_softmax_edit_distance(labels, predict_labels, invalid_index, converter)

    for i, p_label in enumerate(predict_labels):
        p_label = converter.decode(p_label, invalid_index=invalid_index, merge_repeat=merge_repeat)
        t_label = labels[i]

        decoded_predict_labels.append(p_label)
        decoded_target_labels.append(t_label)

    if FLAGS.decode_mode == 'softmax':
        edit_distances = cal_edit_distance(decoded_target_labels, decoded_predict_labels)

    if FLAGS.infer_remove_symbol:
        decoded_predict_labels = utils.remove_symbols(decoded_predict_labels)
        decoded_target_labels = utils.remove_symbols(decoded_target_labels)
        edit_distances = cal_edit_distance(decoded_target_labels, decoded_predict_labels)

    return decoded_predict_labels, decoded_target_labels, edit_distances


def analyze_edit_distances(edit_distances):
    error_edit_distances = list(filter(lambda x: x != 0, edit_distances))

    mean = 0
    if len(error_edit_distances) != 0:
        mean = np.mean(error_edit_distances)

    correct_count = len(edit_distances) - len(error_edit_distances)
    return mean, correct_count


def validation(sess, model, dataset, sub_dir, global_step=None, log_batch_acc=False, name=""):
    """
    :param sub_dir: val, test, infer
    :param log_batch_acc:
    :param name: saved file name: name_{acc}_{step}_{decode_mode}.txt
    :return:
    """
    dataset.init_op.run()
    num_batches = int(math.ceil(dataset.size / dataset.batch_size))

    decoded_predict_labels = []
    decoded_target_labels = []
    img_paths = []
    edit_distances = []

    for step in range(num_batches):
        img_batch, label_batch, (labels, _), img_paths_batch = dataset.get_next_batch(sess)

        batch_start_time = time.time()
        batch_decoded_p_labels, batch_decoded_t_labels, batch_edit_distances = do_infer_on_batch(sess, model,
                                                                                                 dataset.converter,
                                                                                                 img_batch,
                                                                                                 label_batch,
                                                                                                 labels)
        batch_end_time = time.time()

        batch_edit_distances_mean, batch_correct_count = analyze_edit_distances(batch_edit_distances)

        decoded_predict_labels += batch_decoded_p_labels
        decoded_target_labels += batch_decoded_t_labels
        img_paths += img_paths_batch
        edit_distances += batch_edit_distances

        if log_batch_acc:
            logging.info(
                "Batch [{}] {:.03f}s accuracy: {:.03f} ({}/{}), edit_distance: {:.03f}"
                    .format(step,
                            batch_end_time - batch_start_time,
                            batch_correct_count / dataset.batch_size,
                            batch_correct_count,
                            dataset.batch_size,
                            batch_edit_distances_mean))

    edit_distance_mean, correct_count = analyze_edit_distances(edit_distances)

    acc = correct_count / dataset.size

    acc_str = "Accuracy: {:.03f} ({}/{}), Average edit distance: {:.03f}".format(acc, correct_count, dataset.size,
                                                                                 edit_distance_mean)
    logging.info(acc_str)

    infer_result_file_path = get_save_file_path(acc, global_step, name, sub_dir)

    logging.info("Write val result to {}...".format(infer_result_file_path))
    with open(infer_result_file_path, 'w', encoding='utf-8') as f:
        for i, p_label in enumerate(decoded_predict_labels):
            t_label = decoded_target_labels[i]
            f.write("{:08d}\n".format(i))
            f.write("input:   {:17s} length: {}\n".format(t_label, len(t_label)))
            f.write("predict: {:17s} length: {}\n".format(p_label, len(p_label)))
            f.write("all match:  {}\n".format(1 if t_label == p_label else 0))
            f.write("edit distance:  {}\n".format(edit_distances[i]))
            f.write('-' * 30 + '\n')
        f.write(acc_str + "\n")
        f.write(utils.acc_under_edit_distances(edit_distances, 0.1) + "\n")
        f.write(utils.acc_under_edit_distances(edit_distances, 0.2) + "\n")

    # Save predict result as one file
    predict_result_path = os.path.join(FLAGS.result_dir, sub_dir, "predict_result.txt")
    logging.info("Write predict result to {}...".format(predict_result_path))
    with open(predict_result_path, 'w', encoding='utf-8') as f:
        for p_label in decoded_predict_labels:
            f.write("{}\n".format(p_label))

    # Copy image not all match to a dir
    if FLAGS.mode == 'infer' and FLAGS.infer_copy_failed:
        failed_infer_img_dir = infer_result_file_path[:-4] + "_failed"
        if os.path.exists(failed_infer_img_dir) and os.path.isdir(failed_infer_img_dir):
            shutil.rmtree(failed_infer_img_dir)

        utils.check_dir_exist(failed_infer_img_dir)

        failed_image_indices = []
        for i, val in enumerate(edit_distances):
            if val != 0:
                failed_image_indices.append(i)

        for i in failed_image_indices:
            img_path = img_paths[i]
            img_name = img_path.split("/")[-1]
            dst_path = os.path.join(failed_infer_img_dir, img_name)
            shutil.copyfile(img_path, dst_path)

        failed_infer_result_file_path = os.path.join(failed_infer_img_dir, "result.txt")
        with open(failed_infer_result_file_path, 'w', encoding='utf-8') as f:
            for i in failed_image_indices:
                p_label = decoded_predict_labels[i]
                t_label = decoded_target_labels[i]
                f.write("{:08d}\n".format(i))
                f.write("input:   {:17s} length: {}\n".format(t_label, len(t_label)))
                f.write("predict: {:17s} length: {}\n".format(p_label, len(p_label)))
                f.write('-' * 30 + '\n')

    return acc, edit_distance_mean


def get_save_file_path(acc, global_step, name, sub_dir):
    file_name = name + "_" if name else ""
    file_name += "{:.03f}_{}".format(acc, FLAGS.decode_mode)
    file_name += ("_%s.txt" % global_step if global_step else ".txt")
    save_dir = os.path.join(FLAGS.result_dir, sub_dir)
    utils.check_dir_exist(save_dir)
    file_path = os.path.join(save_dir, file_name)
    return file_path


def cal_edit_distance(t_labels, p_labels):
    edit_distances = []
    for i, p_label in enumerate(p_labels):
        t_label = t_labels[i]
        edit_distances.append(algorithms.edit_distance(t_label, p_label))
    return edit_distances


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    FLAGS.checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.tag)
    FLAGS.log_dir = os.path.join(FLAGS.log_dir, FLAGS.tag)
    FLAGS.result_dir = os.path.join(FLAGS.result_dir, FLAGS.tag)

    utils.check_dir_exist(FLAGS.checkpoint_dir)
    utils.check_dir_exist(FLAGS.log_dir)
    utils.check_dir_exist(FLAGS.result_dir)

    with tf.device(dev):
        if FLAGS.mode == 'train':
            flags_dir = os.path.join(FLAGS.checkpoint_dir, "flags")
            tf_utils.save_flags(FLAGS, flags_dir)
            train()

        elif FLAGS.mode == 'infer':
            infer(FLAGS.infer_dir)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
