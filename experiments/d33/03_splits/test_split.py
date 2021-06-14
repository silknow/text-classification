#!/usr/bin/env python
"""
Here we create 1 train set and 4 testsets:
1 - Cross-Lingual Test: JOCONDE (FRENCH)
        * This tests how well our model handles unseen languages
2 - GROUP Test: 'IMATEX' (SPANISH) and 'RISD' (ENGLISH)
        * This test how well our model handles unseen groups (e.g. museums)
3 - Stratified Test (Subset of VAM+CERES)
        * This tests how well our model generalizes to unseen examples in
        the same group (e.g. close to IID setting)
4 - Full: a test set made of all of the previous 3

"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 136790160
BASE = '/data/euprojects/silknow/tasks/'
BASE_BERT = '/data/euprojects/silknow/tasks_bert/'
EXT = '.csv'
TASKS_S = ['place', 'timespan', 'material', 'technique']
# TASKS_S = ['place', 'timespan', 'material']
TASKS_M = {}

ALL_SETS = ['vam', 'met', 'mfa', 'joconde', 'ceres', 'imatex']

COMMON_COLS = ['dataset', 'filename', 'lang', 'txt']
MIN_COUNT_TRAIN = 80


def train_for_task_idx(df, task):
    idx = df['dataset'].isin(['vam', ])
    return idx


def alt_train_for_task_idx(df, task):
    idx = df['dataset'].isin(['mfa', ])
    return idx


def test_for_task_idx(df, task):
    idx = df['dataset'].isin(['ceres', ])
    return idx


def val_for_task_idx(df, task):
    idx = df['dataset'].isin(['mfa', 'joconde', ])
    return idx


def make_xling(df, task, base_task):
    print(f'TASK={task}\tSPLIT=XLING')

    # TRAIN DATA
    df_trn = None
    idx = train_for_task_idx(df, task)
    if len(idx) == 0:
        print('No examples for XLING {task} train')
        return
    else:
        df_trn = df[idx]

    # ALT TRN
    df_alt = None
    idx = alt_train_for_task_idx(df, task)
    if len(idx) == 0:
        print('No examples for XLING ALT {task}')
        return
    else:
        df_alt = df[idx]

    # TEST DATA
    df_tst = None
    idx = test_for_task_idx(df, task)
    if len(idx) == 0:
        print('No examples for XLING ALT {task}')
        return
    else:
        df_tst = df[idx]

    # remove classes with train < MIN_COUNT_TRAIN
    above = [k for k, v in df_trn[task].value_counts().items()
             if v >= MIN_COUNT_TRAIN]
    df_trn = df_trn[df_trn[task].isin(above)]
    df_alt = df_alt[df_alt[task].isin(above)]
    df_tst = df_tst[df_tst[task].isin(above)]

    # Create the files
    dst = os.path.join(base_task, 'xling.tst.csv')
    df_tst.to_csv(dst, na_rep='null', sep='\t', index=False)
    dst = os.path.join(base_task, 'xling.trn.csv')
    df_trn.to_csv(dst, na_rep='null', sep='\t', index=False)
    dst = os.path.join(base_task, 'xling.alt.csv')
    df_alt.to_csv(dst, na_rep='null', sep='\t', index=False)
    print(f'\tTrain={len(df_trn)}\talt={len(df_alt)}')
    print(df_trn[task].value_counts())
    print(f'\tTest={len(df_tst)}')
    print(df_tst[task].value_counts())
    print()


def make_group(df, task, base_task):
    print(f'TASK={task}\tSPLIT=GROUP')
    df_trn = df[df.dataset == 'vam']
    df_tst = df[df.dataset == 'mfa']

    # remove classes with train < MIN_COUNT_TRAIN
    above = [k for k, v in df_trn[task].value_counts().items()
             if v >= MIN_COUNT_TRAIN]
    df_trn = df_trn[df_trn[task].isin(above)]
    df_tst = df_tst[df_tst[task].isin(above)]

    dst = os.path.join(base_task, 'group.tst.csv')
    df_tst.to_csv(dst, na_rep='null', sep='\t', index=False)
    dst = os.path.join(base_task, 'group.trn.csv')
    df_trn.to_csv(dst, na_rep='null', sep='\t', index=False)

    print(f'\tTrain={len(df_trn)}')
    print(df_trn[task].value_counts())
    print(f'\tTest={len(df_tst)}')
    print(df_tst[task].value_counts())
    print()


def make_iid(df, task, base_task):
    df_iid = df[df.dataset == 'vam']
    stfy = None
    if task in TASKS_S:
        stfy = df_iid[task]
    df_trn, df_tst = train_test_split(df_iid, test_size=0.2,
                                      shuffle=True, stratify=stfy,
                                      random_state=SEED)
    dst = os.path.join(base_task, 'iid.tst.csv')
    df_tst.to_csv(dst, na_rep='null', sep='\t', index=False)
    dst = os.path.join(base_task, 'iid.trn.csv')
    df_trn.to_csv(dst, na_rep='null', sep='\t', index='index')
    print(f'#IID: Train={len(df_trn)}\tTest={len(df_tst)}')
    print('')

    #
    # Final
    #
    stfy = None
    stfy = df[task]
    df_trn, df_tst = train_test_split(df, test_size=0.2,
                                      shuffle=True, stratify=stfy,
                                      random_state=SEED)
    dst = os.path.join(base_task, 'final.tst.csv')
    df_tst.to_csv(dst, na_rep='null', sep='\t', index=False)
    dst = os.path.join(base_task, 'final.trn.csv')
    df_trn.to_csv(dst, na_rep='null', sep='\t', index='index')
    print(f'#Final: Train={len(df_trn)}\tTest={len(df_tst)}')
    print('')


def main():
    for base_dir in [BASE, BASE_BERT]:
        for task in TASKS_S + list(TASKS_M.keys()):
            print(task)
            base_task = os.path.join(base_dir, task)
            # load data
            src = os.path.join(base_task, task + EXT)
            task_cols = [task]
            if task in TASKS_M:
                task_cols = TASKS_M[task]
            data_types = {k: 'str' for k in COMMON_COLS + task_cols}
            usecols = COMMON_COLS + task_cols
            df = pd.read_csv(src, sep='\t', dtype=data_types, usecols=usecols)

            # idx = df['dataset'].isin(TRAIN_SUBSET + TEST_SUBSET_1)
            # df = df[idx]

            # Create test sets:
            # (1) Cross Lingual
            make_xling(df, task, base_task)

            #
            # (2) Group test
            #
            make_group(df, task, base_task)

            #
            # (3) IID test
            #
            make_iid(df, task, base_task)

        print('-' * 79)
        print('')

    # dataset prediction
    base_dir = BASE
    task = 'dataset'
    print(task)
    base_task = os.path.join(base_dir, task)
    # load data
    src = os.path.join(base_task, task + EXT)
    data_types = {k: 'str' for k in COMMON_COLS}
    usecols = COMMON_COLS
    df = pd.read_csv(src, sep='\t', dtype=data_types, usecols=usecols)

    stfy = df[task]
    df_trn, df_tst = train_test_split(df, test_size=0.2,
                                      shuffle=True, stratify=stfy,
                                      random_state=SEED)
    dst = os.path.join(base_task, 'iid.tst.csv')
    df_tst.to_csv(dst, na_rep='null', sep='\t', index=False)
    dst = os.path.join(base_task, 'iid.trn.csv')
    df_trn.to_csv(dst, na_rep='null', sep='\t', index='index')
    print(f'#DATASET#: Train={len(df_trn)}\tTest={len(df_tst)}')


if __name__ == '__main__':
    main()
