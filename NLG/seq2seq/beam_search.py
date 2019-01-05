import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import math
from queue import PriorityQueue
import operator
import random
import heapq


class BOWDecoder(nn.Module):
    """a bag-of-word distribution decoder"""
    def __init__(self, latent_size, vocab_size):
        super(BOWDecoder, self).__init__()
        inner_size = int((latent_size + vocab_size)/2)
        self.fc1 = nn.Linear(latent_size, inner_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(inner_size, vocab_size)

    def forward(self, input):
        # input: (B * vocab_size)
        out = self.fc1(input)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out  # B * vocab_size


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

        self.temperature = 0.2

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

        self.attn = Attn('dot', hidden_size)
        # self.attn = Attn('concat', hidden_size)
        # self.attn = Attn('general', hidden_size)
        # self.dropout = nn.Dropout(self.dropout_p)
        self.enc = nn.GRU(self.hidden_size, hidden_size, dropout=dropout_p, batch_first=True, bidirectional=True)
        self.dec = nn.GRU(self.hidden_size * 2, hidden_size, num_layers=self.rnn_num_layers, dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size * 2, output_size)

    def decoder_step(self, decoder_input, hidden, context_vector, attention_states):
        """
        :param decoder_input:  (B)
        :param hidden: (B, Layers, H)
        :param context_vector: (B, H)
        :param attention_states: (B, T, H)
        :return:
        """
        decoder_input = decoder_input.unsqueeze(0)  # (B) -> （1, B）
        hidden = hidden.transpose(0, 1).contiguous() # (B, Layers, H) -> (Layers, B, H)
        context_vector = context_vector.unsqueeze(0)  # (B, H) -> （1, B, H）

        # print("decoder_input", decoder_input.size()) #
        decoder_input = self.emb(decoder_input)  # (1, B, hidden_size)
        # print("decoder_input", decoder_input.size())
        # Combine embedded input word and attended context, run through RNN
        cat_input = torch.cat([decoder_input, context_vector], 2)  # (1, B, 2 * hidden_size)
        logit_rnn, hidden = self.dec(cat_input, hidden)  # logit_rnn: (1, B, hidden_size) hidden: (2, B, H)
        # print("logit_rnn", logit_rnn.size())  # (1, B, hidden_size)
        # print("hidden", hidden.size())  # (2, B, H)

        # Calculate attention weights and apply to h_bag_of_word
        attn_weights, context_vector = self.attn(hidden, attention_states.transpose(0, 1))  # (B, T)
        # print("attn_weights", attn_weights.size())  # (B, T)
        # print("context_vector", context_vector.size())  # (1, B, H)

        project_input = torch.cat([logit_rnn, context_vector], 2)  # (1, B, 2 * hidden_size)
        # print("project_input", project_input.size())
        output_logit = self.out(project_input.squeeze(0))  # (B, 2 * hidden_size) -> (B, vocab_size)

        # output = F.log_softmax(output_logit, dim=1)   # (B, vocab_size)
        # output = F.softmax(output_logit, dim=1)  # (B, vocab_size)
        # output
        # print("output", output.size())

        hidden = hidden.transpose(0, 1)  # (Layers, B, H) -> (B, Layers, H)
        context_vector = context_vector.squeeze(0)  # (1, B, H） -> (B, H)
        return output_logit, hidden, context_vector  # (B, vocab_size)


    def forward(self, input_seqs, target_seqs, train_mode=True, teacher_forcing=True):
        # input_seqs: (bs, length)
        # target_seqs: length, bs
        batch_size = input_seqs.size(0)

        input_emb = self.emb(input_seqs)  # (length, bs, embd_size)

        # Encoding
        encoder_states, hidden = self.enc(input_emb)  # encoder_state: (bs, L, 2*H),  hc: (2, bs, H)
        encoder_states = encoder_states[:, :, :self.hidden_size] + encoder_states[:, :,
                                                                   self.hidden_size:]  # Sum bidirectional outputs
        # print("encoder_states")
        # print(encoder_states.size()) # (bs, L, H)

        # Decoding states initialization
        decoder_input = torch.tensor([self.start_token] * batch_size).cuda()  # (B)
        context_vector = torch.zeros((batch_size, self.hidden_size)).cuda()  # (B, H)
        hidden = hidden.transpose(0, 1)  # (Layers, B, H) -> (B, Layers, H)

        if train_mode:
            # Train Decoding
            decoder_outputs_logit = torch.zeros((self.max_length, batch_size, self.output_size)).cuda()
            decoder_outputs_ids = torch.zeros((self.max_length, batch_size), dtype=torch.int).cuda()

            for i in range(self.max_length):  # T = top_k
                output_logit, hidden, context_vector = self.decoder_step(decoder_input, hidden,
                                                                         context_vector, encoder_states)
                decoder_outputs_logit[i] = output_logit
                if teacher_forcing:
                    decoder_input = target_seqs[i]  # (B)
                else:
                    topv, topi = torch.max(output_logit, dim=-1)  # (B)
                    decoder_input = topi
                    decoder_outputs_ids[i] = topi

                    # text_word_weights = output_logit.div(self.temperature).exp().cpu()
                    # decoder_input = torch.multinomial(text_word_weights, 1).cuda()
                    # print("decoder_input", decoder_input.squeeze().size())
                    # decoder_outputs_ids[i] = decoder_input.squeeze()

                    # input("=====")
            if teacher_forcing:
                return decoder_outputs_logit   # (T, B, vocab_size)
            else:
                return decoder_outputs_logit, decoder_outputs_ids  # decoder_outputs_ids: (T, B)
        else:
            # beam search mode
            beam_width = 3

            partial_sequences = [TopN(beam_width) for _ in range(batch_size)]
            complete_sequences = [TopN(beam_width) for _ in range(batch_size)]

            # initial_step
            output_logit, hidden, context_vector = self.decoder_step(decoder_input, hidden,
                                                                     context_vector, encoder_states)
            output_logit = F.log_softmax(output_logit, dim=1)
            topv, topi = torch.topk(output_logit, beam_width, dim=-1)
            # print("topv", topv.size())  # (B, beam_width)
            # print("topi", topi.size())  # (B, beam_width)
            for b in range(batch_size):
                # Create first beam_size candidate hypotheses for each entry in batch
                for k in range(beam_width):
                    seq = Sequence(
                        output_ids=[topi[b][k]],
                        hidden_state=hidden[b],
                        context_vector=context_vector[b],
                        logprob=topv[b][k],
                        attention_states=encoder_states[b],
                    )
                    partial_sequences[b].push(seq)

            # Run beam search.
            for _ in range(self.max_length - 1):
                partial_sequences_list = [p.extract() for p in partial_sequences]  # (B, beam_size)
                for p in partial_sequences:
                    p.reset()

                # Keep a flattened list of parial hypotheses, to easily feed through a model as whole batch
                flattened_partial = [s for sub_partial in partial_sequences_list for s in sub_partial]  # (B*beam_size)

                decoder_input_feed = torch.tensor([c.output_ids[-1] for c in flattened_partial]).cuda().contiguous()
                # print("decoder_input_feed", decoder_input_feed.size())  # (B*beam_size)

                hidden_feed = torch.stack([c.hidden_state for c in flattened_partial])
                # print("hidden_feed", hidden_feed.size())  # (B*beam_size, Layers, H)
                context_vector_feed = torch.stack([c.context_vector for c in flattened_partial]).contiguous()
                # print("context_vector_feed", context_vector_feed.size())  # (B*beam_size, H)
                attention_states_feed = torch.stack([c.attention_states for c in flattened_partial]).contiguous()
                # print("attention_states_feed", attention_states_feed.size())  # (B*beam_size, H)

                if len(decoder_input_feed) == 0:
                    # We have run out of partial candidates; happens when beam_size=1
                    break

                # Feed current hypotheses through the model, and recieve new outputs and states
                # logprobs are needed to rank hypotheses
                new_output_logit, new_hidden, new_context_vector = self.decoder_step(decoder_input_feed, hidden_feed,
                                                                         context_vector_feed, attention_states_feed)
                new_output_logit = F.log_softmax(new_output_logit, dim=1)
                new_topv, new_topi = torch.topk(new_output_logit, beam_width, dim=-1)
                # print("new_topv", new_topv.size())  # (B*beam_size, beam_width)
                # print("new_topi", new_topi.size())  # (B*beam_size, beam_width)
                idx = 0
                for b in range(batch_size):
                    # For every entry in batch, find and trim to the most likely
                    # beam_size hypotheses
                    for partial in partial_sequences_list[b]:

                        here_hidden_state = new_hidden[idx]
                        here_context_vector = new_context_vector[idx]

                        k = 0
                        while k < beam_width:
                            here_word = new_topi[idx][k]
                            here_output_ids = partial.output_ids + [here_word]
                            here_logprob = partial.logprob + new_topv[idx][k]
                            k += 1

                            new_beam = Sequence(
                                output_ids=here_output_ids,
                                hidden_state=here_hidden_state,
                                context_vector=here_context_vector,
                                logprob=here_logprob,
                                attention_states=partial.attention_states,
                            )

                            if len(here_output_ids) == self.max_length:  # here_word.item() == self.EOS
                                complete_sequences[b].push(new_beam)
                            else:
                                partial_sequences[b].push(new_beam)
                        idx += 1

            # If we have no complete sequences then fall back to the partial sequences.
            # But never output a mixture of complete and partial sequences because a
            # partial sequence could have a higher score than all the complete
            # sequences.
            for b in range(batch_size):
                if not complete_sequences[b].size():
                    complete_sequences[b] = partial_sequences[b]
                    print("no complete sequences.")
            seqs = [[d.item() for d in complete.extract()[0].output_ids]
                    for complete in complete_sequences]
            return seqs  # decoded_batch: (B， T)


class Sequence(object):
    """Represents a complete or partial sequence."""

    def __init__(self, output_ids, hidden_state, context_vector, logprob, attention_states):
        """Initializes the Sequence.
        Args:
          output: List of word ids in the sequence.
          state: Model state after generating the previous word.
          logprob: Log-probability of the sequence.
          score: Score of the sequence.
        """
        self.output_ids = output_ids
        self.hidden_state = hidden_state
        self.context_vector = context_vector
        self.logprob = logprob
        self.score = self.eval()
        self.attention_states = attention_states

    def __cmp__(self, other):
        """Compares Sequences by score."""
        assert isinstance(other, Sequence)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Sequence)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Sequence)
        return self.score == other.score

    def eval(self,):
        return self.logprob / float(len(self.output_ids) + 1e-6)


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
        attn_energies = F.softmax(attn_energies, dim=1)  # (B, T)
        # print("attn_energies", attn_energies.size())  # (B, T)
        # (B, 1, T) * (B, T, H) -> (B, 1, H) -> (1, B, H)
        context_vector = attn_energies.unsqueeze(1).bmm(encoder_outputs.transpose(0, 1)).transpose(0, 1)

        return attn_energies, context_vector  # (B, H); (1, B, H)


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.
        The only method that can be called immediately after extract() is reset().
        Args:
          sort: Whether to return the elements in descending sorted order.
        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


if __name__ == '__main__':
    x = torch.randn(10, 5)
    print(x)
    print(x[:,-1])


    input("----------")

    y = torch.randn(10, 5)
    print("y")
    print(y)
    x = torch.randn(3, 10)
    print("x")
    print(x)
    topk, indices = torch.topk(x, k=4, dim=1)
    print("topk")
    print(topk)
    print("indices")
    print(indices)
    # z = torch.index_select(y, 0, indices[0])
    # print("z")
    # print(z)
    d = y[indices]
    print("d")
    print(d)

