import sys, os
from . import utils

from libs.utils import load_chars


class LabelConverter:
    def __init__(self, chars_filepath):
        self.chars = ''.join(load_chars(chars_filepath))
        # char_set_length + ctc_blank
        self.num_classes = len(self.chars) + 1

        self.encode_maps = {}
        self.decode_maps = {}

        self.create_encode_decode_maps(self.chars)

    def create_encode_decode_maps(self, chars):
        for i, char in enumerate(chars):
            self.encode_maps[char] = i
            self.decode_maps[i] = char

    def encode(self, label):
        """如果 label 中有字符集中不存在的字符，则忽略"""
        encoded_label = []
        for c in label:
            if c in self.chars:
                encoded_label.append(self.encode_maps[c])
            # else:
            #     encoded_label.append(-1)

        return encoded_label

    def encode_list(self, labels):
        encoded_labels = []
        for label in labels:
            encoded_labels.append(self.encode(label))
        return encoded_labels

    def merge_repeat(self, encoded_label, invalid_index, merge_repeat):
        label_filtered = []
        for index, char_index in enumerate(encoded_label):
            if char_index != invalid_index:
                if not merge_repeat:
                    label_filtered.append(char_index)
                else:
                    if index > 0 and char_index != encoded_label[index - 1]:
                        label_filtered.append(char_index)
        return label_filtered

    def decode(self, encoded_label, invalid_index, merge_repeat):
        """
        :param invalid_index ctc空白符的索引
        :param merge_repeat 是否合并重复字符，对于 softmax 解码来说需要，对于 beam_search 解码来说不需要
        """
        label_filtered = self.merge_repeat(encoded_label, invalid_index, merge_repeat)

        label = [self.decode_maps[c] for c in label_filtered]
        return ''.join(label).strip()

    def decode_list(self, encoded_labels, invalid_index, merge_repeat):
        decoded_labels = []
        for encoded_label in encoded_labels:
            decoded_labels.append(self.decode(encoded_label, invalid_index, merge_repeat))
        return decoded_labels
