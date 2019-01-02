import torch
import math
import operator
import numpy as np
from queue import PriorityQueue


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_search_decoder(_text_decoder, data_loader, generate_num=args.generated_num, max_length=args.SEQ_LENGTH):
    # _bow_decoder.train()
    _text_decoder.eval()
    with torch.no_grad():
        with open(text_eval_file, 'w') as f_text:

            # beam search decoder
            beam_width = 5
            topk = 1  # how many sentence do you want to generate
            decoded_batch = []

            decoder_input_sos = torch.tensor([[args.START_TOKEN]], device=args.device)
            # decoding goes sentence by sentence
            for _j in range(int(generate_num)):
                decoder_hidden = _text_decoder.init_hidden(batch_size=1)  # [1, B, H]: init input decoder_hidden

                # Number of sentence to generate
                endnodes = []
                number_required = min((topk + 1), topk - len(endnodes))

                # starting node -  hidden vector, previous node, word id, logp, length
                node = BeamSearchNode(decoder_hidden, None, decoder_input_sos, 0.0, 0)
                nodes = PriorityQueue()

                # start the queue
                nodes.put((-node.eval(), node))
                qsize = 1

                # start beam search
                while True:
                    # give up when decoding takes too long
                    if qsize > 2000:
                        print("Queue size larger then 2000!")
                        break
                    # fetch the best node
                    score, n = nodes.get()
                    decoder_input = n.wordid
                    decoder_hidden = n.h

                    if n.leng == max_length and n.prevNode != None:  # or n.wordid.item() == EOS_token
                        endnodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(endnodes) >= number_required:
                            break
                        else:
                            continue

                    # decode for one step using decoder
                    decoder_output, decoder_hidden = _text_decoder(
                        decoder_input, decoder_hidden
                    )

                    # Put here real beam search of top.
                    log_prob, indexes = torch.topk(decoder_output, beam_width)
                    # print(log_prob)   # tensor([[-5.2716, -5.2854, -5.3159, -5.3816, -5.5174]], device='cuda:0')
                    # print(indexes)   # tensor([[2068, 3874, 3285, 4538,  417]], device='cuda:0')

                    for new_k in range(beam_width):
                        decoded_t = indexes[0][new_k].view(1, -1)
                        # print("decoded_t", decoded_t)  # tensor([[2068]], device='cuda:0')
                        log_p = log_prob[0][new_k].item()
                        # print("log_p", log_p)  # -5.271597385406494
                        node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)

                        # put them into queue
                        nodes.put((-node.eval(), node))
                        # increase qsize
                        qsize += 1

                # choose nbest paths, back trace them
                if len(endnodes) == 0:
                    endnodes = [nodes.get() for _ in range(topk)]

                topk_value = sorted(endnodes, key=operator.itemgetter(0))[:topk]

                utterances = []
                for score, n in topk_value:
                    utterance = []
                    # print(n.wordid)
                    # back trace
                    while n.prevNode != None:
                        utterance.append(n.wordid.cpu().numpy()[0][0])
                        n = n.prevNode
                    utterance = utterance[::-1]
                    utterances.append(utterance)
                # print("utterances", utterances)
                decoded_batch.append(utterances[0])
                # input("======")

                # for print
                # if _j % 100 == 0:
                #     print(' '.join([str(_k) for _k in utterances[0]]))

            for _jj in decoded_batch:
                f_text.write("%s\n" % (' '.join([str(_k) for _k in _jj])))
            print("Example: ", ' '.join([str(_k) for _k in decoded_batch[0]]))
    return