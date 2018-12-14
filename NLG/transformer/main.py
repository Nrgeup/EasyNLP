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
import torch.onnx
from torch import optim
import numpy
import matplotlib
from matplotlib import pyplot as plt
from torch.autograd import Variable


# Import your model files.
import data as data_help
import model as model_help



#######################################################################################
#  Hyper-parameters
######################################################################################
parser = argparse.ArgumentParser(description="Here is your model discription.")
# Add your arguments here.
# Example:
# parser.add_argument('--data', type=str, default='../../Data/wikitext-2',
#                     help='location of the data corpus')


parser.add_argument('--device', type=str, default='cpu', help='')

args = parser.parse_args()


# set gpu
if torch.cuda.is_available():
    args.device = "cuda"
    print("Info: You are now using GPU mode:", args.device)
else:
    print("Warning: You do not have a CUDA device, so you now running with CPU!")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Write your main code here.

global max_src_in_batch, max_tgt_in_batch


def run_epoch(data_iter, model, loss_compute):
    """Standard Training and Logging Function"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src.to(args.device), batch.trg.to(args.device),
                            batch.src_mask.to(args.device), batch.trg_mask.to(args.device))
        loss = loss_compute(out, batch.trg_y.to(args.device), batch.ntokens.float())
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss.cpu().numpy() / batch.ntokens.numpy(), tokens.numpy() / elapsed))
            start = time.time()
            tokens = 0
    return total_loss.cpu() / total_tokens.cpu().float()


def batch_size_fn(new, count, sofar):
    """Keep augmenting batch and calculate total number of tokens + padding."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(data_help.subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys


# Small example model.
def synthetic_data():
    print("synthetic_data task")
    # Train the simple copy task.
    V = 11
    criterion = model_help.LabelSmoothing(size=V, padding_idx=0, smoothing=0.1)
    criterion.cuda()
    model = model_help.make_model(V, V, N=2)
    model.cuda()
    model_opt = model_help.NoamOpt(model.src_embed[0].d_model, 1, 400,
                                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model.train()
        run_epoch(data_help.data_gen(V, 30, 20), model,
                  model_help.SimpleLossCompute(model.generator, criterion, args.device, model_opt))
        model.eval()
        eval_loss = run_epoch(data_help.data_gen(V, 30, 5), model,
                              model_help.SimpleLossCompute(model.generator, criterion, args.device, None))
        print("eval loss: %f" % eval_loss.numpy())

    model.eval()
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    print(greedy_decode(model, src.to(args.device), src_mask.to(args.device), max_len=10, start_symbol=1))


# A Real World Example
def german_english_translation():
    print("german_english_translation task")
    SRC, TGT, train, val, test = data_help.data_load()

    # GPUs to use
    devices = [0, 1]
    pad_idx = TGT.vocab.stoi["<blank>"]
    print("Size:", len(SRC.vocab), len(TGT.vocab))
    model = model_help.make_model(len(SRC.vocab), len(TGT.vocab), N=6).to(args.device)
    print("+===============+")
    criterion = model_help.LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1).to(args.device)
    BATCH_SIZE = 12000
    train_iter = data_help.MyIterator(train, batch_size=BATCH_SIZE, device=devices[0],
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = data_help.MyIterator(val, batch_size=BATCH_SIZE, device=devices[0],
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    model_par = nn.DataParallel(model, device_ids=devices)

    model_opt = model_help.NoamOpt(model.src_embed[0].d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model_par.train()
        run_epoch((data_help.rebatch(pad_idx, b) for b in train_iter),
                  model_par,
                  model_help.MultiGPULossCompute(model.generator, criterion,
                                      devices=devices, opt=model_opt))
        model_par.eval()
        loss = run_epoch((data_help.rebatch(pad_idx, b) for b in valid_iter),
                         model_par,
                         model_help.MultiGPULossCompute(model.generator, criterion,
                                             devices=devices, opt=None))
        print(loss)

    """Once trained we can decode the model to produce a set of translations. 
    Here we simply translate the first sentence in the validation set. This dataset 
    is pretty small so the translations with greedy search are reasonably accurate."""
    for i, batch in enumerate(valid_iter):
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask,
                            max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
        print("Translation:", end="\t")
        for i in range(1, out.size(1)):
            sym = TGT.vocab.itos[out[0, i]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        print("Target:", end="\t")
        for i in range(1, batch.trg.size(0)):
            sym = TGT.vocab.itos[batch.trg.data[i, 0]]
            if sym == "</s>": break
            print(sym, end=" ")
        print()
        break


if __name__ == '__main__':
    # synthetic_data()
    german_english_translation()




