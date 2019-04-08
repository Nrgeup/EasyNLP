import numpy as np
import os
import random
import torch


def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def load_word_dict_info(word_dict_file, max_num):
    id_to_word = []
    with open(word_dict_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip()
            item_list = item.split('\t')
            word = item_list[0]
            if len(item_list) > 1:
                num = int(item_list[1])
                if num < max_num:
                    break
            id_to_word.append(word)
    print("Load word-dict with %d size and %d max_num." % (len(id_to_word), max_num))
    return id_to_word, len(id_to_word)


def load_data1(file1):
    token_stream = []
    with open(file1, 'r') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            parse_line = [int(x) for x in line]
            token_stream.append(parse_line)
    return token_stream


def prepare_data(data_path, max_num, task_type):
    print("prepare data ...")
    id_to_word, vocab_size = load_word_dict_info(data_path + 'word_to_id.txt', max_num)

    # define train / test file
    train_file_list = []
    train_label_list = []
    test_file_list = []
    test_label_list = []
    if task_type == 'yelp' or task_type == 'amazon':
        train_file_list = [
            data_path + 'sentiment.train.0', data_path + 'sentiment.train.1',
            data_path + 'sentiment.dev.0', data_path + 'sentiment.dev.1',
        ]
        train_label_list = [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ]
        test_file_list = [
            data_path + 'sentiment.test.0', data_path + 'sentiment.test.1'
        ]
        test_label_list = [
            [1, 0],
            [0, 1],
        ]
    return id_to_word, vocab_size, train_file_list, train_label_list, test_file_list, test_label_list


def pad_batch_seuqences(origin_seq, sos_id, eos_id, unk_id, max_seq_length, vocab_size):
    '''padding with 0, mask id_num > vocab_size with unk_id.'''
    max_l = 0
    for i in origin_seq:
        max_l = max(max_l, len(i))

    max_l = min(max_seq_length, max_l + 1)

    input_seq = np.zeros((len(origin_seq), max_l), dtype=int)
    target_seq = np.zeros((len(origin_seq), max_l), dtype=int)
    input_seq_length = np.zeros((len(origin_seq)), dtype=int)
    for i in range(len(origin_seq)):
        input_seq[i][0] = sos_id
        for j in range(min(max_l-1, len(origin_seq[i]))):
            this_id = origin_seq[i][j]
            if this_id >= vocab_size:
                this_id = unk_id
            input_seq[i][j + 1] = this_id
            target_seq[i][j] = this_id
        input_seq_length[i] = min(max_l, len(origin_seq[i]) + 1)
        target_seq[i][input_seq_length[i]-1] = eos_id
    return input_seq, input_seq_length, target_seq


class non_pair_data_loader():
    def __init__(self, batch_size):
        self.data_label_pairs = []
        self.sentences_batches = []
        self.labels_batches = []
        self.num_batch = 0
        self.batch_size = batch_size
        self.pointer = 0

    def create_batches(self, train_file_list, train_label_list, if_shuffle=True):
        for _index in range(len(train_file_list)):
            with open(train_file_list[_index]) as fin:
                for line in fin:
                    line = line.strip()
                    line = line.split()
                    parse_line = [int(x) for x in line]
                    self.data_label_pairs.append([parse_line, train_label_list[_index]])

        if if_shuffle:
            random.shuffle(self.data_label_pairs)

        # Split batches
        self.num_batch = int(len(self.data_label_pairs) / self.batch_size)
        for _index in range(self.num_batch):
            item_data_label_pairs = self.data_label_pairs[_index*self.batch_size:(_index+1)*self.batch_size]
            item_sentences = [_i[0] for _i in item_data_label_pairs]
            item_labels = [_i[1] for _i in item_data_label_pairs]
            # print(item_sentences)
            # print(item_labels)
            # input("--------------")
            self.sentences_batches.append(item_sentences)
            self.labels_batches.append(item_labels)
        self.pointer = 0
        print("Load data from %s !\nCreate %d batches with %d batch_size" % (
            ' '.join(train_file_list), self.num_batch, self.batch_size
        ))

    def next_batch(self):
        """take next batch by self.pointer"""
        this_batch_sentences = self.sentences_batches[self.pointer]
        this_batch_labels = self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return this_batch_sentences, this_batch_labels

    def reset_pointer(self):
        self.pointer = 0


if __name__ == '__main__':
    aa = [
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ]

    print(aa)
    print([_i[0] for _i in aa])
    print([_i[1] for _i in aa])

    myList = [[0, 1, 2, 3, 4], [0, 1, 2], [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3], [0, 1,], [0, 1, 2, 3, 4, 5,]]
    mylist2 = [5, 3, 7, 4, 2, 6]

    new_list = []
    for i in range(len(myList)):
        new_list.append([myList[i], mylist2[i]])

    myList1 = sorted(new_list, key=lambda i: len(i[0]), reverse=False)
    # myList2 = sorted(myList, key=lambda i: len(i), reverse=False)
    print(myList1)
    # print(mylist2)





