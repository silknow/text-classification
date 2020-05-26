# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >

import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd

from train import train
from models import sn_create_default_model
from csvtxtdataset import seq_collate_pad
from vocabulary import (VocabMultiLingual,
                        sn_create_vocab,
                        sn_vocab_from_multilingual_embeddings)
from csvtxtdataset import CSVDatasetMultilingualMulticlass
from vechelper import (sn_create_embeddings,
                       sn_load_embeddings,
                       save_vectors_multilingual,
                       export_vectors)
from torchutil import predict_sko, predict_simple
from evalhelper import report, save_classification
from sklearn.metrics import confusion_matrix


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sn_get_model(embs, n_classes):  # noqa: C901
    # create model, optimizer, loss
    model = None
    optim = None
    criterion = None
    collate_fn = None

    slen = 300
    lr = 0.005
    wdecay = 0.0

    model = sn_create_default_model(embs, slen, n_classes)
    collate_fn = seq_collate_pad

    optim = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                 lr=lr, weight_decay=wdecay)

    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    return model, optim, criterion, collate_fn


#
# SDNN API
#
def load_model(model_save_dir):
    # model
    model_file = os.path.join(model_save_dir, 'model.pth')
    model = torch.load(model_file)

    # labels
    labels_file = os.path.join(model_save_dir, 'labels.pth')
    labels = torch.load(labels_file)

    # vocab
    vocab_file = os.path.join(model_save_dir, 'vocab.txt')
    vocab = VocabMultiLingual(sos=None, eos=None, unk=None)
    vocab.load(vocab_file)

    return model, labels, vocab


def save_model(model_save_dir, model, labels, vocab):
    """Save SilkNOW text classification model."""
    # create dir
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # model
    model_file = os.path.join(model_save_dir, 'model.pth')
    torch.save(model, model_file)

    # labels
    labels_file = os.path.join(model_save_dir, 'labels.pth')
    torch.save(labels, labels_file)

    # vocab
    vocab_file = os.path.join(model_save_dir, 'vocab.txt')
    vocab.save(vocab_file)

    return None


def create_vocab_embs(train_file_path, embeddings_dir):
    # map pretrained embeddings files
    all_vec_files = [x for x in os.listdir(embeddings_dir)
                     if x.endswith('.vec')]
    embeddings_files = {x.split('.')[1]: os.path.join(embeddings_dir, x)
                        for x in all_vec_files}

    # create vocab
    vocab_text_path = train_file_path
    vocab = sn_create_vocab(embeddings_files, vocab_text_path)

    # create embeddings
    embs = sn_create_embeddings(embeddings_files, vocab, vector_size=300)

    return vocab, embs


def create_embeddings(train_file_path, embeddings_dir, save_path):
    vocab, embs = create_vocab_embs(train_file_path, embeddings_dir)
    vec_size = embs.shape[1]
    data = export_vectors(vocab, embs)
    save_vectors_multilingual(save_path, data, vec_size)


def load_vocab_embs(embeddings_path):
    vocab = sn_vocab_from_multilingual_embeddings(embeddings_path)
    embs = sn_load_embeddings(embeddings_path, vocab)

    return vocab, embs


#
# SNTXTCLASSIFY Commands
#
def train_model(train_file_path,
                embeddings_path,
                col_tgt,
                model_save_dir,
                all_embeddings=False):
    """Train a SilkNOW text classification model."""
    #
    # embeddings and vocab
    #
    vocab = None
    embs = None

    if os.path.isdir(embeddings_path):
        vocab, embs = create_vocab_embs(train_file_path, embeddings_path)
    elif os.path.isfile(embeddings_path):
        vocab, embs = load_vocab_embs(embeddings_path)
    else:
        print('Embs not found')
        sys.exit(1)
    assert vocab is not None
    assert embs is not None

    # load dataset
    pad = True
    slen = 300
    trn = CSVDatasetMultilingualMulticlass(vocab, train_file_path, '\t',
                                           'txt', 'lang', col_tgt,
                                           slen, pad)
    #
    # train model
    #
    batch_size = 64
    epochs = 300
    clip = 1.0
    labels = trn.get_labels()
    n_classes = len(labels)
    model, optim, criterion, collate_fn = sn_get_model(embs, n_classes)

    trn_loader = DataLoader(trn, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_fn)

    train(model, optim, criterion, trn_loader, epochs, clip, None)

    #
    # Save
    #
    if model_save_dir is not None:
        save_model(model_save_dir, model, labels, vocab)

    return None


def eval_model(model_load_path, test_file_path, col_tgt, confusion=False):
    """Evaluate a SilkNOW text classification model."""

    # Load model
    model, labels, vocab = load_model(model_load_path)

    # Load data
    slen = 300
    pad = True
    collate_fn = seq_collate_pad
    tst = CSVDatasetMultilingualMulticlass(vocab, test_file_path, '\t',
                                           'txt', 'lang', col_tgt, slen, pad,
                                           labels)
    tst_loader = DataLoader(tst, batch_size=1, shuffle=False,
                            collate_fn=collate_fn)

    # predict
    rnn_out = False
    y_prd = predict_sko(model, rnn_out, tst_loader, 'multiclass')
    y_tst = [lbl for _, lbl in tst]
    y_tst = torch.stack(y_tst).numpy()

    # classification report
    r = report(y_tst, y_prd, labels)
    print(labels)
    print(r)

    # confusion
    if confusion:
        print()
        lbl_tst = [labels[v] for v in list(y_tst)]
        lbl_prd = [labels[v] for v in list(y_prd)]
        print(labels)
        cm = confusion_matrix(lbl_tst, lbl_prd, labels=labels)
        print(cm)


def classify_csv(model_load_path, test_file_path, dest_file_path):
    # Load model
    model, labels, vocab = load_model(model_load_path)

    # Load data
    slen = 300
    pad = True
    collate_fn = seq_collate_pad
    tst = CSVDatasetMultilingualMulticlass(vocab, test_file_path, '\t',
                                           'txt', 'lang', None, slen, pad,
                                           labels)
    tst_loader = DataLoader(tst, batch_size=1, shuffle=False,
                            collate_fn=collate_fn)

    # predict
    rnn_out = False
    y_prd = predict_sko(model, rnn_out, tst_loader, 'multiclass')

    # save
    save_classification(dest_file_path, y_prd, labels, tst)


def classify_text(model, labels, vocab, text_dict_list, add_index=False):
    # Load data
    slen = 300
    collate_fn = seq_collate_pad
    pad = True

    df = pd.DataFrame(text_dict_list)
    tst = CSVDatasetMultilingualMulticlass(vocab, df, None, 'txt', 'lang',
                                           None, slen, pad, labels)

    tst_loader = DataLoader(tst, batch_size=1, shuffle=False,
                            collate_fn=collate_fn)

    # predict
    y_prd = predict_simple(model, tst_loader)

    # create list of dicts to return
    rows = []
    for ii, p in enumerate(y_prd):
        class_idx, score = p
        label = labels[class_idx]
        row = {'label': label, 'score': score}
        if add_index:
            row['index'] = ii
        rows.append(row)

    return rows
