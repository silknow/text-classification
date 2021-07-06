#!/usr/bin/env python
"""Convert EURECOM data dump file into a train sets.

- Extract all text to create a embeddings

"""
import ast
import os
import numpy as np
import pandas as pd
from collections import defaultdict
import langid


RANDOM_NUMBER = 621323849
RANDOM_NUMBER2 = 581085259
FNAME = "data/total_post.csv"
COLS = [
    "obj",
    "museum",
    "text",
    "place_country_code",
    "time_label",
    "technique_group",
    "material_group",
    "category_group",
]

LABEL_COLS = [
    "place_country_code",
    "time_label",
    "technique_group",
    "material_group",
]
MIN_CHARS = 25
MIN_LABEL_COUNT = 120


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


def converter_func_labels(x):
    """Converts list string to list. Handles some empty value.

    Use for:
        - technique
        - place
        - material
    """
    if rec_is_nan(x):
        return np.nan

    try:
        x = ast.literal_eval(x)
        if type(x) == list:
            if len(x) == 0:
                return np.nan
            elif len(x) != 1:
                return np.nan
            else:
                return x[0].strip()
    except Exception as ex:
        raise (ex)


def converter_func_time(x):
    """Converts list string to list. Handles some empty value.

    Use for:
        - technique
        - place
        - material
    """
    if rec_is_nan(x):
        return np.nan

    try:
        x = ast.literal_eval(x)
        if type(x) == list:
            if len(x) == 0:
                return np.nan
            elif len(x) == 2:
                if x[0].strip() == x[1].strip():
                    return x[0].strip()
                return np.nan
            elif len(x) == 1:
                return x[0].strip()
            else:
                return np.nan
    except:
        raise


def converter_func_img(x):
    """Convert list of strings to a single space delimited string."""
    if rec_is_nan(x):
        return np.nan

    try:
        x = ast.literal_eval(x)
        if type(x) == list:
            if len(x) == 0:
                return np.nan

            x = [u.strip() for u in x]
            return " ".join(x)
    except:
        raise


def converter_func_text(x):
    """Converts text to a single string."""
    if rec_is_nan(x):
        return np.nan

    if type(x) == list:
        return " ".join(x).strip()

    try:
        x = ast.literal_eval(x)
        if type(x) == list:
            return " ".join(x).strip()
        if type(x) == str:
            return x.strip()
        raise TypeError

    except:
        raise


def classify_lang(text):
    """Classify language or NaN if no text."""
    if pd.isna(text):
        return np.nan
    return langid.classify(text)[0]


def labels_for_col(df, col):
    dfc = df.copy()
    # Must Have Text
    dfc = dfc[dfc["txt"].notna()].copy()
    dfc = dfc[dfc["lang"].notna()].copy()
    # Must be Fabrics
    dfc = dfc[dfc["category_group"] == "fabrics"]

    above = [
        k for k, v in df[col].value_counts().items() if v >= MIN_LABEL_COUNT
    ]
    return above


# Load data from file
df = pd.read_csv(FNAME, usecols=COLS)

# Handle lists
print("LISTS")
df["text"] = df["text"].map(converter_func_text)

df["category_group"] = df["category_group"].map(converter_func_img)

df["time_label"] = df["time_label"].map(converter_func_time)
for col in ["technique_group", "material_group", "place_country_code"]:
    df[col] = df[col].map(converter_func_labels)

# Normalize whitespaces
df["text"] = df["text"].map(lambda t: " ".join(str(t).split()))

# Drop duplicates
print("Duplicate OBJ code")
ori = len(df)
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["obj"])
fin = len(df)
diff = ori - fin
print(f"{ori} - {diff} = {fin}")
print()


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

# rename text
print("Rename text -> txt")
df.rename({"text": "txt"}, axis=1, inplace=True)


# Labels
print("Labels")
for col in LABEL_COLS:
    labels = labels_for_col(df, col)
    print(f"{col}: {labels}")
    mask = ~df[col].isin(labels)
    df.loc[mask, col] = np.nan
    print(df[col].value_counts(dropna=False))

print("Saving Museum-Label counts")
for col in LABEL_COLS:
    dfc = df.groupby(["museum", col]).size().unstack(fill_value=0)
    dfc.loc["Total"] = dfc.sum(numeric_only=True, axis=0)
    dfc.loc[:, "Total"] = dfc.sum(numeric_only=True, axis=1)
    f_name = f"analysis/museum_count_{col}.tsv"
    dfc.to_csv(f_name, sep="\t")


# Min Chars
print(f"Min Chars = {MIN_CHARS}")
mask = df["txt"].map(len) < MIN_CHARS
df.loc[mask, "txt"] = np.nan

# Remove Unknown lang
print("Known languaguages only")
acceptable = {"ca", "it", "es", "en", "fr", "de"}
langs = df["lang"].tolist()
langs = [x if x in acceptable else np.nan for x in langs]
df["lang"] = langs

df = df[df["txt"].notna()].copy()
df = df[df["lang"].notna()].copy()

print(f"DF len = {len(df)}")


# Write processed file to disk
print("\nWrite files to disk")
f_name = "data/processed.tsv"
# save files
df.to_csv(f_name, sep="\t", index=False, na_rep="NULL")


# extract text
print("Extract text")
columns = ["lang", "txt"]
dfc = df[columns].copy()
# drop dups
dfc = dfc.drop_duplicates()
print(len(dfc))
f_name = "data/text_train.tsv"
dfc.to_csv(f_name, sep="\t", index=False, na_rep="NULL")
print(f"Text samples: {len(dfc)}")

print()
