# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >
"""
CSV Text Classification Datasets (pytorch)
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_sequence


class CSVDatasetMultilingualMultilabel(Dataset):
    """Multilingual Multilabel text classification.
    """

    def __init__(self, vocab, csv_file, sep, text_field, lang_field,
                 label_start, seqlen):
        # vocab
        self.vocab = vocab
        self.seqlen = seqlen

        assert type(label_start) == int
        assert label_start > 0
        assert seqlen > 0

        # csv
        df = pd.read_csv(csv_file, sep=sep)
        self.labels = df.keys().tolist()[label_start:]

        # read/load text and labels
        self.data = []
        for idx in range(len(df)):
            # vectorize text
            text = df.iloc[idx][text_field]
            lang = df.iloc[idx][lang_field]
            text = self.vocab.sentence_to_multilingual_sequence(text, lang)
            text = self.vocab.sequence2ids(text)
            text = text[:seqlen]  # truncate text to seqlen
            text = torch.LongTensor(text)

            # labels
            labels = df.iloc[idx, label_start:]
            labels = torch.FloatTensor(labels)

            # add
            self.data.append((text, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_labels(self):
        return self.labels


class CSVDatasetMultilingualMulticlass(Dataset):
    """Multilingual Multiclass text classification.
    """

    def __init__(self, vocab, csv_file, sep, text_field, lang_field,
                 label_field, seqlen, pad=False, labels=None):
        # vocab
        self.vocab = vocab
        self.seqlen = seqlen

        assert seqlen > 0

        df = None
        if type(csv_file) == str:
            types = {text_field: str, label_field: str, 'source': str,
                     'file': str}
            df = pd.read_csv(csv_file, sep=sep, dtype=types)
        elif type(csv_file) == pd.core.frame.DataFrame:
            df = csv_file
        else:
            raise ValueError('csv_file must be str or DataFrame')

        assert df is not None

        self.df = df

        # read labels
        if labels:
            self.labels = labels
        else:
            self.labels = sorted(df[label_field].unique().tolist())

        self.data = []
        for idx in range(len(df)):
            # vectorize text
            text = df.iloc[idx][text_field]
            lang = df.iloc[idx][lang_field]
            text = self.vocab.sentence_to_multilingual_sequence(text, lang)
            text = self.vocab.sequence2ids(text)
            assert len(text) > 0
            text = text[:seqlen]  # truncate text to seqlen
            if pad:
                text = ([0] * (seqlen - len(text))) + text
            text = torch.LongTensor(text)

            # labels
            label = df.iloc[idx][label_field]
            if label not in self.labels:
                continue
            label = self.labels.index(label)
            label = torch.tensor(label)

            # add
            self.data.append((text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_labels(self):
        return self.labels

    def reverse(self, idx):
        text, label = self.__getitem__(idx)
        text = self.vocab.ids2sequence(text.numpy().tolist())
        text = self.vocab.sentence_to_monolingual(text)
        label = self.labels[label]
        return text, label


def seq_collate_pad(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    # sort by length
    idx = np.flip(np.argsort([len(item) for item in data])).tolist()
    data = [data[i] for i in idx]
    targets = torch.stack([targets[i] for i in idx])
    # pad
    data = pad_sequence(data, batch_first=True)
    return [data, targets]


def seq_collate_pack(batch):
    # batch contains a list of tuples of structure (sequence, target)
    data = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    # sort by length
    idx = np.flip(np.argsort([len(item) for item in data])).tolist()
    data = [data[i] for i in idx]
    targets = torch.stack([targets[i] for i in idx])
    # pack
    data = pack_sequence(data)
    return [data, targets]
