#!/usr/bin/env python
"""
Extract the text from the test files, add to the train text.
"""

import pandas as pd

FILES = [
    "data/text_train.tsv",
    "data/test_data/place.tsv",
    "data/test_data/time-span.tsv",
    "data/test_data/material.tsv",
    "data/test_data/technique.tsv",
]

txts = []
langs = []

for fpath in FILES:
    print(fpath)
    df = pd.read_csv(fpath, sep="\t", usecols=["txt", "lang"])

    txts += df["txt"].tolist()
    langs += df["lang"].tolist()

df = pd.DataFrame(list(zip(txts, langs)), columns=["txt", "lang"])
df.drop_duplicates(subset="txt", inplace=True)

df.to_csv("data/text.tsv", sep="\t", index=False)
