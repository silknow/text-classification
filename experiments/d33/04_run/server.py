#!/usr/bin/env python
# coding=utf-8

"""
Cross-Lingual Bag-of-Vectors classifier
TextCNN
BiLSTM
HAN (Sentence)
"""
import argparse
import torch
from torch.utils.data import DataLoader
from train import train, train_reg
from dnnhelper import get_vocab, get_embeddings, get_model

from csvtxtdataset import CSVDatasetMultilingualMulticlass
from torchutil import predict_sko
from evalhelper import report, save_errors, results_append
from sklearn.metrics import confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Text Classification Server')
    parser.add_argument('model_path', type=str, help='model')
    parser.add_argument('vocab', type=str, help='vocab')
    parser.add_argument('port', type=int, help='port')

    # general parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    args = parser.parse_args()
    return args


def run(args):
    set_seed(args.seed)

    # load embeddings
    embs = get_embeddings(args)

    # load vocab
    vocab = get_vocab(args)

    # load datasets
    trn, tst, labels = get_datasets(args, vocab)
    n_classes = len(labels)
    print(labels)

    # get model
    model, optim, criterion, collate_fn = get_model(args, embs, n_classes)

    print('Creating model and setting up train')
    # create loaders
    trn_loader = DataLoader(trn, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn)
    eval_loader = DataLoader(tst, batch_size=10, shuffle=False,
                             collate_fn=collate_fn)

    # train
    print(f'Epochs: {args.epochs}')

    if args.model == 'RegLSTM':
        ar = 2
        tar = 1
        train_reg(model, optim, criterion, trn_loader, args.epochs, args.clip,
                  ar, tar, eval_loader)
    else:
        train(model, optim, criterion, trn_loader, args.epochs, args.clip,
              eval_loader)

    results(args, model, trn, tst, collate_fn)


def main():
    args = parse_args()
    acc = run(args)
    return acc


if __name__ == '__main__':
    main()
