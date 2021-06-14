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
import pandas as pd
from sklearn.model_selection import train_test_split
from csvtxtdataset import CSVDatasetMultilingualMulticlass
from dnnhelper import get_vocab, get_embeddings, get_model
from torchutil import predict
from train import train

RFILE = '/data/euprojects/silknow/results/tune.txt'
DATAFILE = '/data/euprojects/silknow/tasks/material/iid.trn.csv'
VOCAB = '/data/euprojects/silknow/vocab.txt'
EMBS = '/data/euprojects/silknow/embedding_matrix.pt'

text_field = 'txt'
label_field = 'material'
SEED = 6492847
torch.manual_seed(SEED)
N_TRIES = 30


def get_base_args():
    args = SimpleNamespace()
    args.clip = 1.0
    args.batch_size = 64
    args.seed = SEED
    args.vocab = VOCAB
    args.embeddings = EMBS
    args.no_freeze = False
    return args


def get_random_params(args):
    params = {
        'epochs': random.choice([200, 300, 400]),
        'seqlen': random.choice([300, 400]),
        'hidden_size': random.choice([300, 400, 500]),
        'dropout': random.choice([0.2, 0.3, 0.4]),
        'lr': random.choice([1e-3, 3e-4, 5e-5]),
        'wdecay': random.choice([1.2e-6, 0.0])
    }

    args.epochs = params['epochs']
    args.seqlen = params['seqlen']
    args.hidden_size = params['hidden_size']
    args.dropout = params['dropout']
    args.lr = params['lr']
    args.wdecay = params['wdecay']

    return args, params


# base args
args = get_base_args()
# Load vocab
vocab = get_vocab(args)
# Load Embs
embs = get_embeddings(args)


# Load Data
types = {text_field: str, label_field: str, 'source': str,
         'file': str, 'lang': str}
df = pd.read_csv(DATAFILE, sep='\t', dtype=types)


stfy = df[label_field]
df_trn, df_tst = train_test_split(df, test_size=0.2,
                                  shuffle=True, stratify=stfy,
                                  random_state=SEED)

best_overall = {'name': None, 'params': None, 'acc': 0.0}
for model_name in ['TextSCNN', 'BiLSTMpool', 'HANSentence', 'TextCNN']:
    args.model = model_name
    best_model = {'name': None, 'params': None, 'acc': 0.0}
    pad = False
    if model_name == 'TextCNN' or model_name == 'TextSCNN':
        pad = True

    for ti in range(N_TRIES):
        args, params = get_random_params(args)
        seqlen = args.seqlen
        trn = CSVDatasetMultilingualMulticlass(vocab, df_trn, '\t',
                                               text_field, 'lang',
                                               label_field, seqlen, pad)
        labels = trn.get_labels()
        tst = CSVDatasetMultilingualMulticlass(vocab, df_tst, '\t',
                                               text_field, 'lang',
                                               label_field, seqlen, pad,
                                               labels)

        model, optim, criterion, collate_fn = get_model(args, embs,
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
        pstr = json.dumps(params)
        print(f'{model_name} {ti}\tacc:{acc}\t{pstr}')

        if acc > best_model['acc']:
            best_model = {'name': model_name, 'params': params, 'acc': acc}
        # end of try
    # end of model
    with open(RFILE, 'a') as fout:
        print(json.dumps(best_model), file=fout)
    print('-' * 60)
    print(f'Best for {model_name}')
    print(best_model)
    if best_model['acc'] > best_overall['acc']:
        best_overall = best_model

with open(RFILE, 'a') as fout:
    print(json.dumps(best_overall), file=fout)
print('\n')
print('-' * 60)
print('-' * 60)

print(best_overall)
