import time
import os
import math

import numpy as np

from libs import utils
from nets.crnn import CRNN


def calculate_accuracy(predicts, labels):
    """
    :param predicts: encoded predict result
    :param labels: ground true label
    :return: accuracy
    """
    assert len(predicts) == len(labels)

    correct_count = 0
    for i, p_label in enumerate(predicts):
        if p_label == labels[i]:
            correct_count += 1

    acc = correct_count / len(predicts)
    return acc, correct_count


def calculate_edit_distance_mean(edit_distences):
    data = np.array(edit_distences)
    data = data[data != 0]
    return np.mean(data)


def validation(sess, model, dataset, converter, step, result_dir, name, print_batch_info=False):
    """
    Save file name: {acc}_{step}.txt
    :param sess: tensorflow session
    :param model: crnn network
    :param result_dir:
    :param name: val, test or infer
    :return:
    """
    sess.run(dataset.init_op)
    num_batches = int(math.floor(dataset.size / dataset.batch_size))

    predicts = []
    labels = []
    edit_distances = []

    for batch in range(num_batches):
        img_batch, label_batch, batch_labels = dataset.get_next_batch(sess)

        batch_start_time = time.time()

        feed = {model.inputs: img_batch,
                model.labels: label_batch,
                model.is_training: False}

        fetches = [
            model.dense_decoded,
            model.edit_distance,
            model.edit_distances
        ]

        batch_predicts, edit_distance, batch_edit_distances = sess.run(fetches, feed)
        batch_predicts = [converter.decode(p, CRNN.CTC_INVALID_INDEX) for p in batch_predicts]

        predicts.extend(batch_predicts)
        labels.extend(batch_labels)
        edit_distances.extend(batch_edit_distances)

        acc, correct_count = calculate_accuracy(batch_predicts, batch_labels)
        if print_batch_info:
            print("Batch [{}/{}] {:.03f}s accuracy: {:.03f} ({}/{}), edit_distance: {:.03f}"
                  .format(batch, num_batches, time.time() - batch_start_time, acc, correct_count, dataset.batch_size,
                          edit_distance))

    acc, correct_count = calculate_accuracy(predicts, labels)
    edit_distance_mean = calculate_edit_distance_mean(edit_distances)
    acc_str = "Accuracy: {:.03f} ({}/{}), Average edit distance: {:.03f}" \
        .format(acc, correct_count, dataset.size, edit_distance_mean)

    print(acc_str)

    save_dir = os.path.join(result_dir, name)
    utils.check_dir_exist(save_dir)
    file_path = os.path.join(save_dir, '%.3f_%d.txt' % (acc, step))

    print("Write result to %s" % file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        for i, p_label in enumerate(predicts):
            t_label = labels[i]
            f.write("{:08d}\n".format(i))
            f.write("input:   {:17s} length: {}\n".format(t_label, len(t_label)))
            f.write("predict: {:17s} length: {}\n".format(p_label, len(p_label)))
            f.write("all match:  {}\n".format(1 if t_label == p_label else 0))
            f.write("edit distance:  {}\n".format(edit_distances[i]))
            f.write('-' * 30 + '\n')
        f.write(acc_str + "\n")

    return acc, edit_distance_mean
