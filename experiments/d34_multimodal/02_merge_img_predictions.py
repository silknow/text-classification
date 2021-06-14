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
        "--prefix-pred",
        type=str,
        help="add a prefix to the predicted column",
        default=None,
    )

    parser.add_argument(
        "--field-target", type=str, help="new field name", required=True
    )

    args = parser.parse_args()
    return args


def fix_label(x, p):
    """Prefix label."""
    if pd.isnull(x):
        return x
    if not x:
        return x
    if x == "NULL":
        return x
    return p + x.strip()


def main():
    args = parse_args()
    print(
        f"{args.dataset_file} + {args.predictions_file} => {args.output_file}"
    )
    # read base dataset file
    base = pd.read_csv(args.dataset_file, delimiter="\t")
    print(f"base = {len(base)}")

    # read predictions
    cols = [args.field_pred_id, args.field_pred]
    preds = pd.read_csv(args.predictions_file)
    for col in preds.columns:
        preds.rename(columns={col: col.strip()}, inplace=True)
    preds = preds[cols]
    print(f"preds = {len(preds)}")

    # renamed predictions, rename id
    preds.rename(
        columns={
            args.field_pred_id: args.field_data_id,
            args.field_pred: args.field_target,
        },
        inplace=True,
    )
    preds[args.field_target] = preds[args.field_target].apply(
        lambda x: x.strip()
    )

    if args.prefix_pred is not None:
        preds[args.field_target] = preds[args.field_target].apply(
            lambda x: fix_label(x, args.prefix_pred)
        )

    # merge
    res = pd.merge(base, preds, how="left", on=args.field_data_id)

    # NULL -> String + Save
    res.to_csv(args.output_file, index=False, na_rep="NULL", sep="\t")


if __name__ == "__main__":
    main()
