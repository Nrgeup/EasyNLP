# Requirements:
# pip install pytorch-pretrained-bert
# More inoformation in https://github.com/huggingface/pytorch-pretrained-BERT
#
# Bert Base : L=12, H=768, A=12   Total-parameters=110M
# Bert Large: L=24, H=1024, A=16  Total-parameters=340M
# Framework:
#       Input:  Token Embedding + Segment Embedding + Position Embedding(Already)
#       Output:

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

logging.basicConfig(level=logging.INFO)


def data_process():
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenized input
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    print("tokenized_text")
    print(tokenized_text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    print("tokenized_text")
    print(tokenized_text)
    assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a',
                              'puppet', '##eer', '[SEP]']

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return tokens_tensor, segments_tensors, tokenizer


def example_get_hidden(tokens_tensor, segments_tensors):
    ''' Let's see how to use BertModel to get hidden states '''
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
        '''
        This model outputs a tuple composed of:
        1. encoded_layers: controled by the value of the output_encoded_layers argument:
            a) output_all_encoded_layers=True(Default): outputs a list of the encoded-hidden-states 
                at the end of each attention block (i.e. 12 full sequences for BERT-base, 24
                 for BERT-large), each encoded-hidden-state is a torch.FloatTensor of size 
                 [batch_size, sequence_length, hidden_size],
            b) output_all_encoded_layers=False: outputs only the encoded-hidden-states corresponding 
                to the last attention block, i.e. a single torch.FloatTensor of size [batch_size, 
                sequence_length, hidden_size],
        2. pooled_output: a torch.FloatTensor of size [batch_size, hidden_size] which is 
            the output of a classifier pretrained on top of the hidden state associated 
            to the first character of the input (CLF) to train on the Next-Sentence task (see BERT's paper).
        '''
        encoded_layers, _ = predictions
    # print("predictions")
    # print(predictions)

    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    assert len(encoded_layers) == 12
    return


def example_get_lm(tokens_tensor, segments_tensors, tokenizer):
    '''how to use BertForMaskedLM'''
    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model.eval()

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    masked_index = 8
    # confirm we were able to predict 'henson'
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    print("predicted_index")
    print(predicted_index)
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print("predicted_token")
    print(predicted_token)
    # assert predicted_token == 'henson'
    return


def main():
    tokens_tensor, segments_tensors, tokenizer = data_process()
    # example_get_hidden(tokens_tensor, segments_tensors)
    example_get_lm(tokens_tensor, segments_tensors, tokenizer)
    return

if __name__ == '__main__':
    main()


