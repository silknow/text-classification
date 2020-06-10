import math
import torch
from tqdm import tqdm


def train_eval(model, criterion, eval_iter, rnn_out=False):
    model.eval()
    acc = 0.0
    n_total = 0
    n_correct = 0
    test_loss = 0.0

    for x, y in eval_iter:
        with torch.no_grad():
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            output = None
            if rnn_out:
                output, _ = model(x)
            else:
                output = model(x)
            loss = criterion(output, y)
            n_correct += (output.argmax(1) == y).sum().item()
            n_total += len(y)
            test_loss += loss.item() / len(y)

    test_loss /= len(eval_iter)
    acc = 100. * (n_correct / n_total)
    print(f'Test Accuracy: {acc:.2f}\tTest Loss (avg): {test_loss}')


def train(model, optim, criterion, train_iter, epochs, clip=0,
          eval_iter=None, eval_every=50):

    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        total_epoch_loss = 0.0
        for x, y in train_iter:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            # forward
            output = model(x)
            if type(output) == tuple:
                output, _ = output
            # backward
            optim.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step()
            loss_val = loss.item()
            if math.isnan(loss_val):
                print('loss = nan')
            else:
                total_epoch_loss += loss.item() / len(y)
        # display epoch stats
        total_epoch_loss /= len(train_iter)

        # eval
        if eval_iter and epoch % eval_every == 0:
            print(f'Epoch: {epoch}\tTrain Loss (avg): {total_epoch_loss}')
            train_eval(model, criterion, eval_iter)


def train_reg(model, optim, criterion, train_iter, epochs, clip=0,
              ar=0, tar=0, eval_iter=None, eval_every=50):
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        total_epoch_loss = 0.0
        for x, y in train_iter:
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            # forward
            output, rnn_out = model(x)
            # backward
            optim.zero_grad()
            loss = criterion(output, y)
            loss_val = loss.item()

            # Activation Regularization
            if ar:
                loss += ar * rnn_out.pow(2).mean()
            # Temporal Activation Regularization (slowness)
            if tar:
                loss += tar * (rnn_out[1:] - rnn_out[:-1]).pow(2).mean()

            # Backprop
            loss.backward()

            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optim.step()
            if math.isnan(loss_val):
                print('loss = nan')
            else:
                total_epoch_loss += loss_val / len(y)
        # display epoch stats
        total_epoch_loss /= len(train_iter)

        # eval
        if eval_iter and epoch % eval_every == 0:
            print(f'Epoch: {epoch}\tTrain Loss (avg): {total_epoch_loss}')
            train_eval(model, criterion, eval_iter, True)
