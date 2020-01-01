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
    parser = argparse.ArgumentParser(description='NN classifier')
    parser.add_argument('trn', type=str, help='train CSV')
    parser.add_argument('tst', type=str, help='test CSV')
    parser.add_argument('vocab', type=str, help='vocab')
    parser.add_argument('embeddings', type=str,
                        help='embeddings matrix (.pt)')

    parser.add_argument('--results', type=str, default=None,
                        help='Save result to file')
    parser.add_argument('--results-append', type=str, default=None,
                        help='Append result summary to file')
    parser.add_argument('--errors', type=str, default=None,
                        help='Save errors to file')
    parser.add_argument('--cm', action='store_true',
                        help='Show confusion matrix')

    parser.add_argument('--col-txt', default='text',
                        help='text column in the dataset CSV')
    parser.add_argument('--col-lang', default='lang',
                        help='language column in the dataset CSV')
    parser.add_argument('--col-tgt', default=None, required=True,
                        help='target column or columns start position')

    parser.add_argument('--model-save', type=str, help='save model path.')
    parser.add_argument('--model', type=str, default='BiLSTMpool',
                        choices=['BoV', 'TextCNN', 'TextSCNN', 'TextGCNN',
                                 'BiLSTM', 'BiLSTMpool', 'RegLSTM',
                                 'HANSentence'])
    parser.add_argument('--no-freeze', action='store_true',
                        help='Do not freeze embeddings.')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='RNN hidden size, CNN hidden fc size')

    parser.add_argument('--filters', type=int, default=300,
                        help='CNN number of filters/maps/channels out')
    parser.add_argument('--kernel-start', type=int, default=2,
                        help='CNN first kernel size (others: +1 +2)')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')

    parser.add_argument('--edrop', type=float, default=0.1,
                        help='Embedding Dropout probability')

    parser.add_argument('--wdrop', type=float, default=0.2,
                        help='WeightDrop probability')

    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                        help='weight decay applied to all weights')

    # general parameters
    parser.add_argument('--seqlen', default=300, type=int,
                        help='max text sequence length')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=500, type=int,
                        help='number of train epochs')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--clip', default=1.0, type=float,
                        help='clip gradient norm')
    parser.add_argument('--seed', default=0, type=float,
                        help='torch seed')
    args = parser.parse_args()
    return args


def get_datasets(args, vocab):
    trn = trn = labels = None
    multilabel = False
    pad = False
    if 'CNN' in args.model:
        pad = True
        if not args.seqlen > 0:
            raise ValueError('Invalid seqlen')

    trn = CSVDatasetMultilingualMulticlass(vocab, args.trn, '\t',
                                           args.col_txt, args.col_lang,
                                           args.col_tgt, args.seqlen, pad)
    labels = trn.get_labels()
    tst = CSVDatasetMultilingualMulticlass(vocab, args.tst, '\t',
                                           args.col_txt, args.col_lang,
                                           args.col_tgt, args.seqlen, pad,
                                           labels)
    print(f'Train: {args.trn} \t->\t {len(trn)}')
    print(f'Test: {args.tst} \t->\t {len(tst)}')
    print(f'Pad={pad}')
    print(f'Labels = {len(labels)}\tMultilabel={multilabel}')
    return trn, tst, labels


def set_seed(seed):
    if seed > 0:
        torch.manual_seed(seed)


def results(args, model, trn, tst, collate_fn):
    rnn_out = False
    if args.model == 'RegLSTM':
        rnn_out = True

    tst_loader = DataLoader(tst, batch_size=1, shuffle=False,
                            collate_fn=collate_fn)
    labels = tst.get_labels()

    # predict
    y_prd = predict_sko(model, rnn_out, tst_loader, 'multiclass')
    y_tst = [lbl for _, lbl in tst]
    y_tst = torch.stack(y_tst).numpy()

    # classification report
    r = report(y_tst, y_prd, labels)
    print(labels)
    print(r)
    if args.results is not None:
        with open(args.results, 'w') as fout:
            print(r, file=fout)

    # confusion
    if args.cm:
        print()
        lbl_tst = [labels[v] for v in list(y_tst)]
        lbl_prd = [labels[v] for v in list(y_prd)]
        print(labels)
        cm = confusion_matrix(lbl_tst, lbl_prd, labels=labels)
        print(cm)

    # append result summary
    if args.results_append:
        name = f'{args.model}-h:{args.hidden_size}-slen:{args.seqlen}'
        task = args.trn
        results_append(args.results_append, y_tst, y_prd, labels, name, task)

    # errors
    if args.errors:
        save_errors(args.errors + '.tst.csv', y_tst, y_prd, labels, tst)

        # predict on train
        trn_eloader = DataLoader(trn, batch_size=1, shuffle=False,
                                 collate_fn=collate_fn)
        y_prd2 = predict_sko(model, rnn_out, trn_eloader, 'multiclass')
        y_trn = [lbl for _, lbl in trn]
        y_trn = torch.stack(y_trn).numpy()
        save_errors(args.errors + '.trn.csv', y_trn, y_prd2, labels, trn)


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

    if args.model_save is not None:
        torch.save(model, args.model_save + '.pth')
        torch.save(labels, args.model_save + '.labels')
        torch.save(args, args.model_save + '.args')

    results(args, model, trn, tst, collate_fn)


def main():
    args = parse_args()
    acc = run(args)
    return acc


if __name__ == '__main__':
    main()
