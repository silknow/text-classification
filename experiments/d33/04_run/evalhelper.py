"""
Model evaluation helper functions
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    """Compute the Hamming score (a.k.a. label-based accuracy)
    for the multi-label case: http://stackoverflow.com/q/32239577/395857
    """
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) / \
                float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def report(y_true, y_pred, target_names):
    report_pred = ''
    labels = list(set(list(y_true)))

    report_pred = classification_report(y_true, y_pred, labels=labels,
                                        target_names=target_names)

    acc_pred = accuracy_score(y_true, y_pred)
    report_pred = report_pred + '\naccuracy:\t{}'.format(acc_pred)

    return report_pred


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_errors(dst, y_true, y_pred, target_names, dataset):
    y_true = list(y_true)
    y_pred = list(y_pred)

    rows = []
    for ii, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue
        t = target_names[t]
        p = target_names[p]
        text, label = dataset.reverse(ii)
        assert label == t
        source = dataset.df.iloc[ii]['dataset']
        filename = dataset.df.iloc[ii]['filename']
        rows.append({'filename': filename,
                     'dataset': source, 'input': text, 'pred': p, 'true': t})
    df = pd.DataFrame(rows)
    df.to_csv(dst, sep='\t')


def results_append(dst, y_true, y_pred, labels, name, task):
    acc = accuracy_score(y_true, y_pred)

    r = report(y_true, y_pred, labels)
    lbl_true = [labels[v] for v in list(y_true)]
    lbl_pred = [labels[v] for v in list(y_pred)]
    pr, rc, f1, sp = precision_recall_fscore_support(lbl_true, lbl_pred,
                                                     beta=1.0,
                                                     labels=labels,
                                                     average='micro')
    res = {}
    if os.path.isfile(dst):
        res = json.load(open(dst))

    if not res.get(name):
        res[name] = {}
    res[name][task] = {
        'accuracy': acc,
        'f1': f1,
        'precision': pr,
        'recall': rc,
        'report': r
    }
    with open(dst, 'w') as outfile:
        json.dump(res, outfile)
