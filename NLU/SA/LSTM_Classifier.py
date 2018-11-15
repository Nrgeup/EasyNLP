from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import time

class RNNClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, device):
        super(RNNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.initHidden()

    def initHidden(self):
        h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim, device=self.device))
        c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim, device=self.device))
        return (h0, c0)


    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        return y, lstm_out

def dis_train(input_tensor, target_tensor, _model, model_optimizer, _loss_function):
    model_optimizer.zero_grad()
    _model.zero_grad()

    _model.batch_size = len(target_tensor)
    output, lstm_out = _model(input_tensor)
    print("lstm_out", lstm_out)
    print(len(lstm_out), len(lstm_out[0]), len(lstm_out[0][0]))
    input("===========")
    loss = _loss_function(output, target_tensor)

    loss.backward()
    model_optimizer.step()

    # calc training acc
    _, predicted = torch.max(output.data, 1)
    total_acc = (predicted == target_tensor).sum()
    total = len(target_tensor)

    return loss.item(), total_acc.item() / total


def dis_train_iters(model, data_loader, _coarse_decoder, _fine_decoder, learning_rate=0.01):
    print("Start pre-training discriminator...")

    start = time.time()
    print_loss = 0.0
    print_acc = 0.0

    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    best_val_acc = None

    log_num = 1
    print('-' * 74)
    for epoch in range(50):
        epoch_start_time = time.time()
        for it in range(data_loader.num_batch):
            log_num += 1
            x_batch, y_batch = data_loader.next_batch()

            x_tensor = torch.tensor(x_batch, dtype=torch.long, device=args.device).t()
            y_tensor = torch.tensor(y_batch, dtype=torch.long, device=args.device)
            model.hidden = model.initHidden()
            loss, acc = dis_train(x_tensor, y_tensor, model, model_optimizer, loss_function)
            print_loss += loss
            print_acc += acc

        print_loss_avg = print_loss / data_loader.num_batch
        print_acc_avg = print_acc / data_loader.num_batch
        print_loss = 0.0
        print_acc = 0.0
        print(
            '| epoch {:3d} | spend time: {:5.2f}s | train loss {:5.4f} | train acc {:5.4f} |'.format(
                epoch, (time.time() - epoch_start_time), print_loss_avg, print_acc_avg))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_acc or print_acc_avg > best_val_acc:
            with open(save_path + args.save_model_dis, 'wb') as f:
                torch.save(model, f)
            best_val_acc = print_acc_avg

    print("Discriminator pre-train done, time: {:5.2f}s".format(time.time() - start))
    return



if __name__ == '__main__':
    dis_data_loader = Dis_Data_loader(args.BATCH_SIZE)
    data_loader.load_train_data(positive_file, fine_eval_file)

    discriminator = discriminator.MILClassifier(args.EMB_DIM, args.hidden_size, args.vocab_size, 2, args.BATCH_SIZE,
                                                args.device).to(args.device)
    # pre_train discriminator
    dis_train_iters(discriminator, dis_data_loader, coarse_decoder, fine_decoder)
