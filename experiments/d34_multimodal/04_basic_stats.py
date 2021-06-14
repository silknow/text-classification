#!/usr/bin/env python

"""Merge text predictions into dev and test sets."""

import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="basic stats")
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


def has_img(df, args):
    total = len(df)
    df_img = df
    df_img = df_img[df_img[args.field_img].notna()]
    has_image = len(df_img)
    has_image = (has_image / total) * 100
    print(f"has image: {has_image:.1f}%%")
    return


def has_txt(df, args):
    total = len(df)
    df_txt = df
    df_txt = df_txt[df_txt[args.field_txt].notna()]
    has_txt = len(df_txt)
    has_txt = (has_txt / total) * 100
    print(f"has txt: {has_txt:.1f}%%")
    return


def has_img_and_txt(df, args):
    total = len(df)
    df_both = df
    df_both = df_both[df_both[args.field_txt].notna()]
    df_both = df_both[df_both[args.field_img].notna()]
    has_both = len(df_both)
    has_both = (has_both / total) * 100
    print(f"has both: {has_both:.1f}%%")
    return


def no_img_no_txt(df, args):
    total = len(df)
    df_neither = df
    df_neither = df_neither[df_neither[args.field_txt].isna()]
    df_neither = df_neither[df_neither[args.field_img].isna()]
    has_neither = len(df_neither)
    has_neither = (has_neither / total) * 100
    print(f"has neither: {has_neither:.1f}%%")
    return


def main():
    args = parse_args()
    # read base file
    usecols = [args.field_img, args.field_txt]
    df = pd.read_csv(args.dataset, delimiter="\t", usecols=usecols)

    # existance overlap
    has_img(df, args)
    has_txt(df, args)
    has_img_and_txt(df, args)
    no_img_no_txt(df, args)


if __name__ == "__main__":
    main()
