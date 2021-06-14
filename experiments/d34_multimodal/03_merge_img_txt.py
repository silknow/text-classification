#!/usr/bin/env python

"""Merge text predictions into dev and test sets."""

import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="merge predictions into dataset files"
    )
    parser.add_argument(
        "--dataset-base", type=str, help="dataset CSV file", required=True
    )
    parser.add_argument(
        "--dataset-img", type=str, help="dataset CSV file", required=True
    )
    parser.add_argument(
        "--dataset-txt", type=str, help="dataset CSV file", required=True
    )

    parser.add_argument(
        "--output-file", type=str, help="dataset CSV file", required=True
    )

    parser.add_argument("--field-id", type=str, help="id column", default="obj")

    parser.add_argument(
        "--field-img",
        type=str,
        help="predicted column",
        default="img_prediction",
    )
    parser.add_argument(
        "--field-txt",
        type=str,
        help="predicted column",
        default="text_prediction",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    msg = f"{args.dataset_base}"
    msg += f" + {args.dataset_img} + {args.dataset_txt}"
    msg += f" => {args.output_file}"
    print(msg)

    # read base file
    df_base = pd.read_csv(args.dataset_base, delimiter="\t")

    # read img file
    usecols = [args.field_id, args.field_img]
    df_img = pd.read_csv(args.dataset_img, delimiter="\t", usecols=usecols)

    # read txt file
    usecols = [args.field_id, args.field_txt]
    df_txt = pd.read_csv(args.dataset_txt, delimiter="\t", usecols=usecols)

    # merge
    df = pd.merge(df_base, df_img, how="left", on=args.field_id)
    df = pd.merge(df, df_txt, how="left", on=args.field_id)

    # NULL -> String + Save
    df.to_csv(args.output_file, index=False, na_rep="NULL", sep="\t")


if __name__ == "__main__":
    main()
