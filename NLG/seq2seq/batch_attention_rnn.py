import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class TextDecoder(nn.Module):
    """a text decoder """
    def __init__(self, hidden_size, output_size, max_length, start_token, dropout_p=0.1, rnn_num_layers=2):
        super(TextDecoder, self).__init__()
        self.start_token = start_token
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.rnn_num_layers = rnn_num_layers

        self.emb = nn.Embedding(self.output_size, self.hidden_size)

        # print("=============")
        # print(self.embedding)
        # print("=============")
        # print(self.embedding.weight.data)
        # print("-----------------")
        # print(len(self.embedding.weight.data))
        # print("-----------------")
        # print(self.embedding.weight.data[0])
        # print("-----------------")
        # print(len(self.embedding.weight.data[0]))
        # input("++++embedding++++")

        # self.attn = Attn('dot', hidden_size)
        # self.attn = Attn('concat', hidden_size)
        self.attn = Attn('general', hidden_size)
        # self.dropout = nn.Dropout(self.dropout_p)
        self.enc = nn.GRU(self.hidden_size, hidden_size, dropout=dropout_p, batch_first=True, bidirectional=True)
        self.dec = nn.GRU(self.hidden_size * 2, hidden_size, num_layers=self.rnn_num_layers, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size * 2, output_size)

    def forward(self, input_seqs, target_seqs):
        # input_seqs: (bs, length)
        # target_seqs: length, bs
        batch_size = input_seqs.size(0)

        input_emb = self.emb(input_seqs)  # (length, bs, embd_size)

        # Encoding
        encoder_states, hidden = self.enc(input_emb)  # encoder_state: (bs, L, 2*H),  hc: (2, bs, H)
        encoder_states = encoder_states[:, :, :self.hidden_size] + encoder_states[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # print("encoder_states")
        # print(encoder_states.size()) # (bs, L, H)

        # Decoding states initialization
        decoder_input = torch.tensor([self.start_token] * batch_size).cuda()   # (bs, H)

        decoder_outputs = torch.zeros((self.max_length, batch_size, self.output_size)).cuda()
        context_vector = torch.zeros((1, batch_size, self.hidden_size)).cuda()  # (1, B, H)

        # Decoding
        for i in range(self.max_length):  # T = top_k
            decoder_input = self.emb(decoder_input).view(1, batch_size, -1)  # (1, B, hidden_size)
            # print("decoder_input", decoder_input.size())

            # Combine embedded input word and attended context, run through RNN
            cat_input = torch.cat([decoder_input, context_vector], 2)  # (1, B, 2 * hidden_size)
            logit_rnn, hidden = self.dec(cat_input, hidden)  # logit_rnn: (1, B, hidden_size) hidden: (2, B, H)
            # print("logit_rnn", logit_rnn.size())  # (1, B, hidden_size)
            # print("hidden", hidden.size())  # (2, B, H)

            # Calculate attention weights and apply to h_bag_of_word
            attn_weights, context_vector = self.attn(hidden, encoder_states.transpose(0, 1))   # (B, T)
            # print("attn_weights", attn_weights.size())  # (B, T)
            # print("context_vector", context_vector.size())  # (1, B, H)

            project_input = torch.cat([logit_rnn, context_vector], 2)  # (1, B, 2 * hidden_size)
            # print("project_input", project_input.size())
            output_logit = self.out(project_input.squeeze(0))  # (B, 2 * hidden_size) -> (B, vocab_size)

            # output = F.log_softmax(output_logit, dim=1)   # (B, vocab_size)
            # output = F.softmax(output_logit, dim=1)  # (B, vocab_size)
            # output
            # print("output", output.size())

            decoder_outputs[i] = output_logit
            decoder_input = target_seqs[i]
            # input("=====")
        return decoder_outputs   # (T, B, vocab_size)


class Attn(nn.Module):
    # luong attention
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        assert self.method == 'dot' or self.method == 'general' or self.method == 'concat'

        if self.method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.rand(hidden_size))
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            hidden state of the decoder, in shape (Layer, B, H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
            context_vector in shape (1, B, H)
        '''
        # print("hidden", hidden.size())
        hidden = torch.sum(hidden, dim=0, keepdim=True)  # (Layer, B, H) -> (1, B, H)
        # print("hidden", hidden.size())

        if self.method == 'dot':
            # (B, 1, H) * (B, H, T) -> (B, 1, T) -> (B, T)
            attn_energies = hidden.transpose(0, 1).bmm(encoder_outputs.transpose(0, 1).transpose(1, 2)).squeeze(1)
            # print("attn_energies", attn_energies.size())  # (B, T)

        if self.method == 'general':
            # (B, 1, H) * (B, H, T) -> (B, 1, T) -> (B, T)
            attn_energies = self.attn(hidden.transpose(0, 1)).bmm(encoder_outputs.transpose(0, 1).transpose(1, 2)).squeeze(1)

        if self.method == 'concat':
            max_len = encoder_outputs.size(0)
            batch_size = encoder_outputs.size(1)
            H = hidden.squeeze(0).repeat(max_len, 1, 1).transpose(0, 1)  # (B, T, H)
            # print("H", H.size())

            attn_energies = F.tanh(self.attn(torch.cat([H, encoder_outputs.transpose(0, 1)], 2)))  # (B, L, 2*H) -> (B, T, H)
            attn_energies = attn_energies.transpose(2, 1)  # (B, H, T)
            v = self.v.repeat(batch_size, 1).unsqueeze(1)  # (B, 1, H)
            attn_energies = torch.bmm(v, attn_energies).squeeze(1)  # (B, 1, H) * (B, H, T) -> (B, 1, T) -> (B, T)

        # normalize with softmax
        attn_energies = F.softmax(attn_energies)  # (B, T)
        # print("attn_energies", attn_energies.size())  # (B, T)
        # (B, 1, T) * (B, T, H) -> (B, 1, H) -> (1, B, H)
        context_vector = attn_energies.unsqueeze(1).bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)

        return attn_energies, context_vector  # (B, H); (1, B, H)

