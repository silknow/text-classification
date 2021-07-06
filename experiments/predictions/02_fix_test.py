#!/usr/bin/env python
"""
Fix test files
"""
import ast
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import langid


RANDOM_NUMBER = 621323849
RANDOM_NUMBER2 = 581085259
FILES = [
    "data/test_data/place.csv",
    "data/test_data/material.csv",
    "data/test_data/technique.csv",
    "data/test_data/time-span.csv",
]
COLS = [
    "obj",
    "text",
    "museum",
]
MIN_CHARS = 15


# MAP from museum to language code
# 'unk' means language identification is required
# None means there is no text
LANG_MAP = {
    "artic": "en",
    "cer": "es",
    "europeana": "unk",
    "garin": "es",
    "imatex": "unk",
    "joconde": "fr",
    "mad": np.nan,
    "met": "en",
    "mfa": "en",
    "mobilier": "fr",
    "mtmad": "fr",
    "paris-musees": "fr",
    "risd": np.nan,
    "smithsonian": "en",
    "unipa": "it",
    "vam": "en",
    "venezia": "it",
    "versailles": "fr",
}


def rec_is_nan(x):
    """Check if a record has a NaN equivalent value.

    I don't think this first block of checks is necessary in the latest
    version of the data export.
    """
    if pd.isna(x):
        return True

    if not x:
        return True

    if type(x) == list:
        if len(x) == 0:
            return True

    if type(x) == str:
        x = x.strip()
        if not x:
            return True
        elif x == "[nan]":
            return True
        elif x == "nan":
            return True
        elif x == "[]":
            return True

    if type(x) == int:
        return True
    if type(x) == float:
        return True

    return False


def converter_func_text(x):
    """Converts text to a single string."""
    if rec_is_nan(x):
        return np.nan

    return str(x)


def classify_lang(text):
    """Classify language or NaN if no text."""
    if pd.isna(text):
        return np.nan
    return langid.classify(text)[0]


for fname in FILES:
    # Load data from file
    df = pd.read_csv(fname, usecols=COLS)

    # Handle lists
    print("LISTS")
    df["text"] = df["text"].map(converter_func_text)

    # Normalize whitespaces
    df["text"] = df["text"].map(lambda t: " ".join(str(t).split()).strip())

    # fix museum name
    df["museum"] = df["museum"].map(lambda t: str(t).split("/")[-1])

    # add language
    print("Lang")
    df = df.assign(lang=np.nan)
    for museum in LANG_MAP:
        mask = df["museum"] == museum
        lang = LANG_MAP[museum]
        if lang == np.nan:
            continue
        elif lang == "unk":
            if museum == "imatex":
                imatex = df[df.museum == museum]["text"]
                print(imatex)
                imatex = [classify_lang(x) for x in imatex]
                acceptable = {"ca", "it", "es", "en", "fr"}
                # default to "ca"
                imatex = [x if x in acceptable else "ca" for x in imatex]
                df.loc[df.museum == museum, "lang"] = imatex
                n = len(imatex)
                print(f"\t{museum}: {lang} - {n}")
            else:
                text = df[df.museum == museum]["text"].tolist()
                print(text)
                langs = [classify_lang(x) for x in text]
                print(langs)
                df.loc[df.museum == lang, "lang"] = langs
        else:
            df.loc[df.museum == museum, "lang"] = lang
            n = len(df[df["museum"] == museum])
            print(f"\t{museum}: {lang} - {n}")

    # Remove Unknown lang
    print("Known languaguages only")
    acceptable = {"ca", "it", "es", "en", "fr", "de"}
    langs = df["lang"].tolist()
    langs = [x if x in acceptable else np.nan for x in langs]
    df["lang"] = langs

    # rename text
    print("Rename text -> txt")
    df.rename({"text": "txt"}, axis=1, inplace=True)

    print(f"Min Chars = {MIN_CHARS}")
    mask = df["txt"].map(len) < MIN_CHARS
    df.loc[mask, "txt"] = np.nan

    df = df[df["txt"].notna()].copy()
    df = df[df["lang"].notna()].copy()

    print(f"DF len = {len(df)}")

    # Write processed file to disk
    print("\nWrite files to disk")
    fname = fname.replace("csv", "tsv")
    # save files
    df.to_csv(fname, sep="\t", index=False, na_rep="NULL")

print()
