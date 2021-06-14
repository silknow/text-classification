# coding: utf-8
import sys
import os
import shutil
import json
import csv
import numpy as np
from collections import defaultdict, OrderedDict, Counter
from sklearn.model_selection import StratifiedShuffleSplit


def read_label_map(description2label_txt):
    """Reads a label map text file into a dictionary (map)
    """
    description2label = {}
    f = open(description2label_txt, 'rb')
    s = f.read().decode('ISO-8859-1')
    lines = s.splitlines()
    lines = lines[1:]  # skip header

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # convert line to fields
        fields = line.split(';')
        if len(fields) < 2:
            continue

        # read description
        description = fields[0].strip().lower()

        # read label
        label = fields[1].strip()
        if label.lower() == 'nan':
            continue
        label = label.replace(' ', '_').replace('/', '').replace('-', '')

        # add to dict
        description2label[description] = label.strip().lower()

    return description2label


def convert_labels(data_in, label_map):
    """Convert labels in data=[(txt, lbl)] according to the map
    Drop if label does not exist in map
    """
    data = []
    missing = []

    for (txt, lbl) in data_in:
        if lbl not in label_map:
            missing.append(lbl)
        else:
            # convert label
            lbl = label_map[lbl]
            # add to data
            data.append((txt, lbl))

    return data, missing


def remove_short_texts(data, min_tokens):
    data = [(txt, lbl) for (txt, lbl) in data if len(txt.split()) > min_tokens]
    return data


def remove_duplicates(data):
    # keep only 1 example for pairs with same text and label
    dedup = set()
    data_dd = []
    for entry in data:
        if entry in dedup:
            continue
        dedup.add(entry)
        data_dd.append(entry)
    data = data_dd

    # now handle pairs with same text but different label by removing both
    dedup = Counter([txt for (txt, _) in data]).most_common()
    dedup = set(txt for (txt, cnt) in dedup if cnt > 1)
    data = [(txt, lbl) for (txt, lbl) in data if txt not in dedup]

    return data


def ex_p_class(data):
    """Creates a dictionary of label -> num examples
    """
    epc = defaultdict(int)

    for (_, lbl) in data:
        epc[lbl] += 1

    return OrderedDict(epc)


def ex_p_class_print(data, fname=None, dst_dir=None):
    if not data:
        return

    stats = ex_p_class(data)
    fout = sys.stdout

    if fname:
        p = os.path.join(dst_dir, fname)
        fout = open(p, 'w')

    for k, v in sorted(stats.items(), key=lambda kv: kv[1], reverse=True):
        print('{}\t{}'.format(k, v), file=fout)


def ex_p_class_min(data, min_ex=20):
    """Filters data for examples with labels that are present in at least
    `min_ex` examples."""
    stats = ex_p_class(data)
    stats = stats.items()
    labels_selected = [lbl for (lbl, cnt) in stats if cnt >= min_ex]
    data = [(txt, lbl) for (txt, lbl) in data if lbl in labels_selected]
    return data


def count_unique_labels(data):
    return len(set(lbl for (_, lbl) in data))


def save_data_txt(data, dst_dir, name):
    f_txt = open(os.path.join(dst_dir, name + '.txt'), 'w', encoding='utf8')
    f_lbl = open(os.path.join(dst_dir, name + '.labels'), 'w', encoding='utf8')

    for (txt, lbl) in data:
        print(txt, file=f_txt)
        print(lbl, file=f_lbl)
    f_txt.close()
    f_lbl.close()


def save_data_tsv(data, dst_dir, name):
    output_file = os.path.join(dst_dir, name + '.tsv')
    with open(output_file, 'w', encoding='utf8') as fout:
        writer = csv.writer(fout, delimiter='\t', quotechar=None)
        for (txt, lbl) in data:
            writer.writerow([txt, lbl])
    return


def save_data_fasttext(data, dst_dir, name):
    output_file = os.path.join(dst_dir, name + '.ft')
    with open(output_file, 'w', encoding='utf8') as fout:
        for (txt, lbl) in data:
            line = '__' + lbl + '__ ' + txt
            print(line, file=fout)
    return


def save_data(data, dst_dir, name):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    save_data_txt(data, dst_dir, name)
    save_data_tsv(data, dst_dir, name)
    # save_data_fasttext(data, dst_dir, name)
    return


def split_data(data, tst_size=0.4, seed=0):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=tst_size,
                                 random_state=seed)
    X, y = map(list, zip(*data))
    X = np.asarray(X)
    y = np.asarray(y)

    for train_index, test_index in sss.split(X, y):
        train_index = list(train_index.astype(int))
        X_train = X[train_index]
        y_train = y[train_index]

        test_index = test_index.astype(int)
        X_test = X[test_index]
        y_test = y[test_index]

        data_trn = list(zip(X_train, y_train))
        data_tst = list(zip(X_test, y_test))

        return data_trn, data_tst


def split_save_data(data, dst_dir, task_name, tst_size=0.4, seed=0):
    if not data:
        return
    dst_dir = os.path.join(dst_dir, task_name)

    # split into train and test
    data_trn, data_tst = split_data(data, tst_size, seed)

    # write train/test files
    save_data(data_trn, dst_dir, 'train')
    save_data(data_tst, dst_dir, 'test')


def load_data_txt(src_dir, name):
    texts = []
    labels = []

    with open(os.path.join(src_dir, name + '.txt'), 'r') as fin:
        for line in fin:
            texts.append(line.strip())

    with open(os.path.join(src_dir, name + '.labels'), 'r') as fin:
        for line in fin:
            labels.append(line.strip())

    return texts, labels


def delete_data(src_dir, task_name):
    p = os.path.join(src_dir, task_name)
    try:
        shutil.rmtree(p)
    except OSError:
        pass
