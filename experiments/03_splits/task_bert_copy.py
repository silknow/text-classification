#!/usr/bin/env python
# -*- coding: utf-8 -*-
# License: MIT
# author: Luis Rei < me@luisrei.com >
import os
import pandas as pd

RAW = '/data/euprojects/silknow/dataset.mapped.csv'
SPLIT2 = '/data/euprojects/silknow/tasks/'
SPLIT2BERT = '/data/euprojects/silknow/tasks_bert/'

TASKS_SINGLE = ['timespan', 'place', 'material', 'technique']
# TASKS_SINGLE = ['timespan', 'place', 'material']

MIN_COUNT = 120
# MIN_COUNT_TOKENS = 15
MIN_COUNT_TOKENS = 0  # DISABLED - done in extract now
# TASKS_MULTI = ['material']
TASKS_MULTI = []
ALL = TASKS_SINGLE + TASKS_MULTI
DATA_TYPES = {k: 'str' for k in ALL}

df = pd.read_csv(RAW, sep='\t', dtype=DATA_TYPES)
print(f'{RAW}\t->\t{SPLIT2BERT}')

for task in TASKS_SINGLE:
    print(task)
    # DROP OTHER COLUMNS
    # drop columns that are not used for this task
    others = [t for t in ALL if t != task]
    dft = df.drop(columns=others)

    # READ IDs FROM TASK_SPLIT
    fname = task + '.csv'
    idx_file = os.path.join(SPLIT2, task)
    idx_file = os.path.join(idx_file, fname)
    df_idx = pd.read_csv(idx_file, sep='\t')
    ids = df_idx.ecode.tolist()
    print(f'TASK_SPLIT LEN={len(df_idx)}')
    assert len(dft) == dft.ecode.nunique()  # no duplicate codes
    idx = dft['ecode'].isin(ids)
    dft = dft[idx]
    assert(len(dft) == len(df_idx))  # dft and df_idx should now be same size

    # write file
    fname = task + '.csv'
    dst = os.path.join(SPLIT2BERT, task)
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
