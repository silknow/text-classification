#!/usr/bin/env python

"""Merge text predictions into dev and test sets."""

import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="merge predictions into dataset files"
    )
    parser.add_argument(
        "--dataset", type=str, help="dataset CSV file", required=True
    )
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


def only_one_exists(df, args):
    total = len(df)
    one_exists = 0
    df_one = df
    df_one = df_one[df_one[args.field_img].notna()]
    df_one = df_one[df_one[args.field_txt].isna()]
    one_exists += len(df_one)

    df_one = df
    df_one = df_one[df_one[args.field_img].isna()]
    df_one = df_one[df_one[args.field_txt].notna()]
    one_exists += len(df_one)
    one_exists /= total
    print(f"only one exists: {one_exists:.2f}")
    return one_exists


def if_both_exist_agree(df, args):
    # same prediction
    # discard any null
    df_same = df
    df_same = df_same[df_same[args.field_img].notna()]
    df_same = df_same[df_same[args.field_txt].notna()]
    total = len(df_same)
    df_same = df_same[df_same[args.field_img] == df_same[args.field_txt]]

    agree = len(df_same) / total
    print(f"if both exist, agree ratio:{agree:.2f}")
    return agree


def main():
    args = parse_args()
    # read base file
    usecols = [args.field_img, args.field_txt]
    df = pd.read_csv(args.dataset, delimiter="\t", usecols=usecols)

    # existance overlap
    only_one_exists(df, args)
    if_both_exist_agree(df, args)


if __name__ == "__main__":
    main()
