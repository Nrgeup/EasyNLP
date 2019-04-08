# coding: utf-8
# requirements: pytorch: 0.04
# Author: Ke Wang
# Contact: wangke17[AT]pku.edu.cn
import time
import argparse
import math
import os
import torch
import torch.nn as nn
from torch import optim
import numpy
import matplotlib
from matplotlib import pyplot as plt

# Import your model files.
from model import EncoderDecoder
from data import prepare_data, non_pair_data_loader, get_cuda, pad_batch_seuqences

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

######################################################################################
#  Environmental parameters
######################################################################################
parser = argparse.ArgumentParser(description="Here is your model discription.")
parser.add_argument('--device', type=str, default='cpu', help='Use cpu mode or gpu mode.')
parser.add_argument('--gpu_ids', type=str, default='0', help='Specify the ID list of the used GPU card.')
parser.add_argument('--id_pad', type=int, default=0, help='')
parser.add_argument('--id_unk', type=int, default=1, help='')
parser.add_argument('--id_bos', type=int, default=2, help='')
parser.add_argument('--id_eos', type=int, default=3, help='')


######################################################################################
#  File parameters
######################################################################################
parser.add_argument('--task', type=str, default='yelp', help='Specify datasets.')
parser.add_argument('--word_to_id_file', type=str, default='', help='')
parser.add_argument('--data_path', type=str, default='', help='')




######################################################################################
#  Model parameters
######################################################################################
parser.add_argument('--word_dict_max_num', type=int, default=5, help='')
parser.add_argument('--batch_size', type=int, default=16, help='')
parser.add_argument('--max_sequence_length', type=int, default=60)
parser.add_argument('--num_layers_AE', type=int, default=2)
parser.add_argument('--embedding_size', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--rnn_type', type=str, default='gru')
# parser.add_argument('--latent_size', type=int, default=16)
parser.add_argument('--word_dropout', type=float, default=0)
parser.add_argument('--embedding_dropout', type=float, default=0.5)
parser.add_argument('--learning_rate', type=float, default=0.001)


args = parser.parse_args()
######################################################################################
#  End of hyper parameters
######################################################################################

def preparation():
    # set gpu
    if torch.cuda.is_available():
        args.device = "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        print("Info: You are now using GPU mode: %s , with GPU id: %s" % (args.device, args.gpu_ids))
    else:
        print("Warning: You do not have a CUDA device, so you now running with CPU!")

    # set task type
    if args.task == 'yelp':
        args.data_path = '../../data/yelp/processed_files/'
    elif args.task == 'amazon':
        args.data_path = '../../data/amazon/processed_files/'
    elif args.task == 'imagecaption':
        pass
    else:
        raise TypeError('Wrong task type!')

    # prepare data
    args.id_to_word, args.vocab_size, \
    args.train_file_list, args.train_label_list, \
    args.test_file_list, args.test_label_list = \
        prepare_data(data_path=args.data_path, max_num=args.word_dict_max_num, task_type=args.task)
    return


def train_iters(ae_model, train_data_loader):
    print("Start train process.")

    ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=args.learning_rate)
    NLL_criterion = nn.NLLLoss(size_average=True, ignore_index=args.id_pad)

    for epoch in range(200):
        print('-' * 90)
        epoch_start_time = time.time()
        for it in range(train_data_loader.num_batch):
            batch_sentences, batch_labels = train_data_loader.next_batch()
            batch_input, batch_length, batch_target = pad_batch_seuqences(
                batch_sentences, args.id_bos, args.id_eos, args.id_unk,
                args.max_sequence_length, args.vocab_size
            )
            # For debug
            # print(batch_input)
            # print(batch_length)
            # print(batch_target)

            batch_input_tensor = torch.tensor(batch_input, dtype=torch.long, device=args.device)
            batch_length_tensor = torch.tensor(batch_length, dtype=torch.long, device=args.device)
            batch_target_tensor = torch.tensor(batch_target, dtype=torch.long, device=args.device)

            # Forward pass
            logp, latent = ae_model(batch_input_tensor, batch_length_tensor)

            # Loss calculation
            # flatten
            batch_target_tensor = batch_target_tensor.view(-1)
            # print(logp.size())
            # print(batch_target_tensor.size())
            # input("====+========")
            Loss_rec = NLL_criterion(logp, batch_target_tensor)

            if it % 50 == 0:
                print(
                    '| epoch {:3d} | {:5d}/{:5d} batches | rec loss {:5.4f} |'.format(
                        epoch, it, train_data_loader.num_batch, Loss_rec))

            # backward + optimization
            ae_optimizer.zero_grad()
            Loss_rec.backward()
            ae_optimizer.step()

        print(
            '| end of epoch {:3d} | time: {:5.2f}s |'.format(
                epoch, (time.time() - epoch_start_time)))

    return


if __name__ == '__main__':
    preparation()

    train_data_loader = non_pair_data_loader(args.batch_size)
    train_data_loader.create_batches(args.train_file_list, args.train_label_list, if_shuffle=True)

    # create models
    ae_model = get_cuda(EncoderDecoder(
        vocab_size=args.vocab_size, embedding_size=args.embedding_size, hidden_size=args.hidden_size,
        num_layers=args.num_layers_AE, word_dropout=args.word_dropout, embedding_dropout=args.embedding_dropout,
        sos_idx=args.id_bos, eos_idx=args.id_eos, pad_idx=args.id_pad, unk_idx=args.id_unk,
        max_sequence_length=args.max_sequence_length, rnn_type=args.rnn_type, bidirectional=True,
    ))

    train_iters(ae_model, train_data_loader)

    print("Done!")

