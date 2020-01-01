#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >
import os
import pandas as pd
from vocabulary import VocabMultiLingual
# import unicodedata

ORIGINAL = '/data/euprojects/silknow/dataset.prp.csv'
SPLIT2 = '/data/euprojects/silknow/tasks/'
SRC_DST = [(ORIGINAL, SPLIT2), ]

# TASKS_SINGLE = ['timespan', 'place', 'material', 'technique']
TASKS_SINGLE = ['timespan', 'place', 'material', 'technique', 'dataset']

MIN_COUNT = 120
# MIN_COUNT_TOKENS = 15
MIN_COUNT_TOKENS = 15
# TASKS_MULTI = ['material']
TASKS_MULTI = []
ALL = TASKS_SINGLE + TASKS_MULTI
data_types = {k: 'str' for k in ALL}

VOCAB = '/data/euprojects/silknow/vocab.txt'
vocab = VocabMultiLingual(sos=None, eos=None, unk=None)
vocab.load(VOCAB)


for src, dst_base in SRC_DST:
    df = pd.read_csv(src, sep='\t', dtype=data_types)
    df.txt = df.txt.str.normalize('NFKC')
    print(f'{src}\t->\t{dst_base}')

    for task in TASKS_SINGLE:
        print(task)
        # DROP OTHER COLUMNS
        # drop columns that are not used for this task
        others = [t for t in ALL if t != task]
        others = [t for t in others if t != 'dataset']
        dft = df.drop(columns=others)

        # DROP NULL
        # drop null values for this task
        dft = dft.dropna(axis='rows', subset=[task])

        # DROP DUPLICATES
        # if same txt, label keep one
        b = len(dft)
        dft = dft.drop_duplicates(subset=['txt', task])
        a = len(dft)
        print(f'drop DUP examples: before: {b}\tafter: {a}')
        # if same txt different label, drop both
        b = len(dft)
        dft = dft.drop_duplicates(subset='txt', keep=False)
        a = len(dft)
        print(f'drop BAD examples: before: {b}\tafter: {a}')
        print(dft.dataset.value_counts().index.tolist())

        # MIN COUNT TOKENS
        b = len(dft)

        def len_ft(x):
            return len(x.txt.split())

        def len_f(x):
            s = vocab.sentence_to_multilingual_sequence(x.txt, x.lang)
            s = vocab.sequence2ids(s)
            if len(s) == 0:
                print(x.txt)
                print(s)
                print(x.lang)
            return len(s)

        number_of_tokens = dft.apply(func=len_ft, axis=1).values
        dft = dft.assign(number_of_tokens=number_of_tokens)
        dft = dft[dft['number_of_tokens'] >= MIN_COUNT_TOKENS]
        dft = dft.drop(columns=['number_of_tokens'])
        a = len(dft)
        print(f'tokens min count={MIN_COUNT_TOKENS}: before: {b}\tafter: {a}')
        print(dft.dataset.value_counts().index.tolist())

        number_of_tokens = dft.apply(func=len_f, axis=1).values
        dft = dft.assign(number_of_tokens=number_of_tokens)
        dft = dft[dft['number_of_tokens'] >= MIN_COUNT_TOKENS]
        dft = dft.drop(columns=['number_of_tokens'])
        a = len(dft)
        print(f'tokens min count={MIN_COUNT_TOKENS}: before: {b}\tafter: {a}')
        print(dft.dataset.value_counts().index.tolist())

        # MIN COUNT LABELS
        # drop labels below a min freq
        above = [k for k, v in dft[task].value_counts().items()
                 if v >= MIN_COUNT]
        b = len(dft)
        dft = dft[dft[task].isin(above)]
        a = len(dft)
        print(above)
        print(f'label min count={MIN_COUNT}: before: {b}\tafter: {a}')
        print(dft.dataset.value_counts().index.tolist())

        # write data
        fname = task + '.csv'
        dst = os.path.join(dst_base, task)
        if not os.path.isdir(dst):
            os.makedirs(dst)
        dst = os.path.join(dst, fname)
        dft.to_csv(dst, na_rep='null', sep='\t', index=False)
        print(dft[task].value_counts())
        print(len(dft))
        print('')
    print('')
    print('-' * 79)
    print('')

    """
    for task in TASKS_MULTI:
        print(task)
        # drop columns that are not used for this task
        others = [t for t in ALL if t != task]
        dft = df.drop(columns=others)
        # drop null values for this task
        dft = dft.dropna(axis='rows', subset=[task])
        # split the multitask field, drop below count
        df2 = dft[task].str.get_dummies(',')
        cols = list(df2.columns)
        for c in cols:
            if df2[c].sum() < MIN_COUNT:
                df2 = df2.drop(columns=[c])
        cols = list(df2.columns)
        print(cols)
        dft = dft.join(df2)
        dft = dft.drop(columns=[task])
        # write data
        fname = task + '.csv'
        dst = os.path.join(SPLIT2, task)
        dst = os.path.join(dst, fname)
        dft.to_csv(dst, na_rep='null', sep='\t', quoting=csv.QUOTE_NONE,
                   index_label='index')
        print('')
    """
