#!/usr/bin/env python

"""Merge text predictions into dev and test sets."""

import argparse
import pandas as pd
from sklearn.metrics import classification_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="merge predictions into dataset files"
    )
    parser.add_argument(
        "--dataset", type=str, help="dataset CSV file", required=True
    )

    parser.add_argument(
        "--field-target", type=str, help="target column", required=True
    )

    parser.add_argument(
        "--field-pred",
        type=str,
        help="predicted column",
        required=True,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # read base file
    usecols = [args.field_pred, args.field_target]
    df = pd.read_csv(args.dataset, delimiter="\t", usecols=usecols)

    y_pred = df[args.field_pred].tolist()
    y_true = df[args.field_target].tolist()

    report = classification_report(y_true, y_pred)
    n = len([x for x in y_pred if not pd.isna(x)])
    p = (n / len(y_true)) * 100

    print(report)
    print(f"not nan = {n} {p:.1f}")


if __name__ == "__main__":
    main()
