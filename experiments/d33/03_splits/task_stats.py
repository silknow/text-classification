#!/usr/bin/env python

import os
import pandas as pd


BASE = '/data/euprojects/silknow/tasks/'
TASKS_S = ['place', 'timespan', 'material', 'technique']
COMMON_COLS = ['dataset', 'filename', 'lang', 'txt']
SCENARIOS = ['iid', 'group', 'xling', 'alt', 'final']

for scenario in SCENARIOS:
    print('#' * 50)
    print(scenario.upper())
    print('#' * 50)
    for task in TASKS_S:
        print(task)
        base_task = os.path.join(BASE, task)
        usecols = COMMON_COLS + [task]
        data_types = {k: 'str' for k in usecols}

        fname = scenario + '.trn.csv'
        if scenario == 'alt':
            fname = 'xling.alt.csv'
        src = os.path.join(base_task, fname)
        df_trn = pd.read_csv(src, sep='\t', dtype=data_types, usecols=usecols)
        print(len(df_trn))
        # print(df_trn[task].value_counts())

        fname = scenario + '.tst.csv'
        if scenario == 'alt':
            fname = 'xling.tst.csv'

        src = os.path.join(base_task, fname)
        df_tst = pd.read_csv(src, sep='\t', dtype=data_types, usecols=usecols)
        print(len(df_tst))

        print('-' * 20)
    print('')
