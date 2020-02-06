#!/usr/bin/env python

import pandas as pd


SRC = '/data/euprojects/silknow/dataset.mapped.csv'
TASKS_S = ['place', 'timespan', 'material', 'technique']
COMMON_COLS = ['dataset', 'filename', 'lang', 'txt']

usecols = COMMON_COLS + TASKS_S
data_types = {k: 'str' for k in usecols}
df = pd.read_csv(SRC, sep='\t', dtype=data_types, usecols=usecols)

datasets = df.dataset.unique()

for task in TASKS_S:
    tdf = df[['dataset', task]]
    tdf = tdf[tdf[task].notnull()]
    print('#' * 50)
    print(task.upper())
    print('#' * 50)
    for ds in datasets:
        print(ds)
        sdf = tdf[tdf.dataset == ds]
        values = list(sdf[task].unique())
        for v in sorted(values):
            print(v)
        print(len(sdf))
        print('-' * 20)
        print('')
