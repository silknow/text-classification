#!/usr/bin/env python
# coding=utf-8

"""
Parameter tuning
"""
import random
import json
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from csvtxtdataset import CSVDatasetMultilingualMulticlass
from dnnhelper import get_vocab, get_embeddings, get_cnn_model
from torchutil import predict
from train import train

label_field = 'place'
RFILE = '/data/euprojects/silknow/tune/epochs.xling.txt'
DATA_TRN = '/data/euprojects/silknow/tasks/place/xling.trn.csv'
DATA_TST = '/data/euprojects/silknow/tasks/place/xling.tst.csv'
VOCAB = '/data/euprojects/silknow/vocab.txt'
EMBS = '/data/euprojects/silknow/embedding_matrix.pt'

text_field = 'txt'
SEED = 6492847
torch.manual_seed(SEED)
N_TRIES = 50


def get_base_args():
    args = SimpleNamespace()
    args.clip = 5.0
    args.batch_size = 64
    args.seqlen = 300
    args.seed = SEED
    args.vocab = VOCAB
    args.embeddings = EMBS
    args.no_freeze = False
    args.hidden_size = 0
    args.activation = 'gelu'
    args.wdecay = 0.0
    args.alpha_dropout = True
    args.filters = 100
    args.kernel_start = 2
    # args.lr = 0.01
    args.lr = 0.005
    args.dropout = 0.4
    return args


def get_random_params(args):
    params = {
        'epochs': random.choice([200, 250, 300, 350, 400, 500, 600]),
        'dropout': random.choice([0.2, 0.3, 0.4, 0.5]),

    }

    args.epochs = params['epochs']
    args.dropout = params['dropout']

    return args, params


# base args
args = get_base_args()
# Load vocab
vocab = get_vocab(args)
# Load Embs
embs = get_embeddings(args)


args.model = 'TextCNN'
model_name = 'TextCNN'
best_model = {'name': None, 'params': None, 'acc': 0.0}
pad = True

print(DATA_TRN)
# for ti in range(1, N_TRIES + 1):
for epochs in [200, 250, 300, 350, 400, 500, 600]:
    ti = 0
    params = {'epochs': epochs}
    args.epochs = epochs
    # args, params = get_random_params(args)

    pstr = json.dumps(params)
    print(ti)
    print(pstr)
    seqlen = args.seqlen
    trn = CSVDatasetMultilingualMulticlass(vocab, DATA_TRN, '\t',
                                           text_field, 'lang',
                                           label_field, seqlen, pad)
    labels = trn.get_labels()
    tst = CSVDatasetMultilingualMulticlass(vocab, DATA_TST, '\t',
                                           text_field, 'lang',
                                           label_field, seqlen, pad,
                                           labels)

    model, optim, criterion, collate_fn = get_cnn_model(args, embs,
                                                        len(labels))

    trn_loader = DataLoader(trn, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn)
    tst_loader = DataLoader(tst, batch_size=1, shuffle=False,
                            collate_fn=collate_fn)

    train(model, optim, criterion, trn_loader, args.epochs, args.clip,
          None)

    # predict
    y_prd = predict(model, tst_loader)
    y_tst = [lbl for _, lbl in tst]
    y_tst = torch.stack(y_tst).numpy()
    acc = (y_prd == y_tst).mean()
    print(f'{model_name} {ti}\tacc:{acc}\t{pstr}')

    if acc > best_model['acc']:
        print(f"NEW BEST: {acc} > {best_model['acc']}")
        best_model = {'name': model_name, 'params': params, 'acc': acc}

        with open(RFILE, 'a') as fout:
            print(json.dumps(best_model), file=fout)
    # blank space after try
    print('\n')
    # end of try
# end of model
print(f'\nBest for {model_name}')
print(best_model)
print('-' * 60)
