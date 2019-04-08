import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import math
import torch.nn.utils.rnn as rnn_utils
from data import get_cuda


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers,
                 word_dropout, embedding_dropout,
                 sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length,
                 rnn_type='GRU', bidirectional=True):
        super().__init__()

        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.bidirectional = bidirectional
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length

        self.word_dropout_rate = word_dropout

        self.rnn_type = rnn_type

        # define sub-layers
        self.embedding = nn.Embedding(self.vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                               batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=bidirectional,
                               batch_first=True)

        self.hidden_size = hidden_size
        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        # self.hidden2latent = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        # self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)

        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), self.vocab_size)

    def forward(self, input_sequence, seq_lengths):
        # input_sequence:(batch_size, xx)
        # seq_lengths: (batch_size)
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(seq_lengths, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # print(input_sequence.shape)
        # Encoder
        input_embedding = self.embedding(input_sequence)  # (batch_size, length, embedding_size)

        # for debug
        # print(input_embedding.shape)
        # print(sorted_lengths.shape)
        # print(input_embedding.detach().numpy())
        # print(sorted_lengths.cpu().detach().numpy())

        packed_input = rnn_utils.pack_padded_sequence(input_embedding,
                                                      sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            latent = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            latent = hidden.squeeze()

        # z = self.hidden2latent(hidden)
        # hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = latent.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = latent.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            dropout_prob_mask = get_cuda(torch.rand(input_sequence.size()))
            # Don't replace the place that has sos or pad.
            dropout_prob_mask[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[dropout_prob_mask < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b, s, _ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(-1, self.vocab_size)  # [b*len, vocab_size]

        # Restore original order
        latent = latent[reversed_idx]

        return logp, latent

