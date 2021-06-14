#!/usr/bin/env python

"""Merge text predictions into dev and test sets."""

import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="merge predictions into dataset files"
    )
    parser.add_argument(
        "--dataset-file", type=str, help="dataset CSV file", required=True
    )

    parser.add_argument(
        "--predictions-file", type=str, help="dataset CSV file", required=True
    )

    parser.add_argument(
        "--output-file", type=str, help="dataset CSV file", required=True
    )

    parser.add_argument(
        "--field-pred-id", type=str, help="id column", default="id"
    )
    parser.add_argument(
        "--field-data-id", type=str, help="id column", default="obj"
    )

    parser.add_argument(
        "--field-pred", type=str, help="predicted column", required=True
    )

    parser.add_argument(
        "--field-target", type=str, help="new field name", required=True
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # read base dataset file
    base = pd.read_csv(args.dataset_file, delimiter="\t")
    print(f"base = {len(base)}")

    # read predictions
    cols = [args.field_pred_id, args.field_pred]
    preds = pd.read_csv(args.predictions_file, delimiter="\t", usecols=cols)
    print(f"preds = {len(preds)}")

    # renamed predictions, rename id
    preds.rename(
        columns={
            args.field_pred_id: args.field_data_id,
            args.field_pred: args.field_target,
        },
        inplace=True,
    )

    # merge
    res = pd.merge(base, preds, how="left", on=args.field_data_id)

    # NULL -> String + Save
    res.to_csv(args.output_file, index=False, na_rep="NULL", sep="\t")


if __name__ == "__main__":
    main()
