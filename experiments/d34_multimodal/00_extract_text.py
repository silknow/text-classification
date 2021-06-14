"""Convert EURECOM data dump file into a train sets.

This exact code is only suitable for text only as it will drop other labels
and will drop duplicate objects that can have different images.
"""
import ast
import os
import numpy as np
import pandas as pd
import langid
from sklearn.model_selection import train_test_split


FNAME = "data/total_post.csv"
COLS = ["obj", "text", "museum", "type_a_group"]
ALL_COLS = COLS

MIN_CHARS = 50
MIN_TOKENS = 15


def converter_func(x):
    if not x:
        return np.nan
    elif not x.strip():
        return np.nan
    elif x.strip() == "[nan]":
        return np.nan

    try:
        x = ast.literal_eval(x)
        if type(x) == list:
            if len(x) == 0:
                return np.nan
            elif len(x) != 1:
                return "OTHER"
            else:
                return x[0].strip().split("/")[-1]
            return x
    except:
        return x.strip()


def merge_text(x):
    if not x:
        return np.nan
    if pd.isna(x):
        return np.nan

    try:
        x = ast.literal_eval(x)
        if type(x) == list:
            if len(x) == 0:
                return np.nan
            x = [u.strip() for u in x]
    except:
        pass

    if type(x) == str:
        x = x.strip()
        if not x:
            return np.nan
        elif x == "nan":
            return np.nan
        elif x.strip() == "[nan]":
            return np.nan
        else:
            return x.strip()

    if type(x) == list:
        return " ".join(x).strip()
    elif type(x) == int:
        return np.nan
    elif type(x) == float:
        return np.nan
    else:
        raise TypeError


# Load data from file
df = pd.read_csv(FNAME, usecols=ALL_COLS)

# Handle lists
print("LISTS")
col = "text"
print("\t list=text")
df[col] = df[col].map(merge_text)

col = "obj"
print(f"\t list={col}")
df[col] = df[col].map(converter_func)
col = "type_a_group"
print(f"\t list={col}")
df[col] = df[col].map(converter_func)

col = "museum"
print(f"\t list={col}")
df[col] = df[col].map(converter_func)

# Drop duplicates
print("Duplicates")
ori = len(df)
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["obj"])
fin = len(df)
diff = ori - fin
print(f"{ori} - {diff} = {fin}")
print()

# Handle text
print("No Text")
ori = len(df)
# Remove no text
df = df[df.text.notna()]
fin = len(df)
diff = ori - fin
print(f"{ori} - {diff} = {fin}")
print()

# Fix whitespaces
df["text"] = df["text"].map(lambda t: " ".join(str(t).split()))

print("Short Text: chars")
ori = len(df)
mask = df["text"].map(len) >= MIN_CHARS
df = df[mask]

fin = len(df)
diff = ori - fin
print(f"{ori} - {diff} = {fin}")
print()

print("Short Text: tokens")
ori = len(df)
mask = df["text"].map(lambda x: len(x.split())) >= MIN_TOKENS
df = df[mask]

fin = len(df)
diff = ori - fin
print(f"{ori} - {diff} = {fin}")
print()

print("Drop All Entries with Duplicate Text")
ori = len(df)
df.drop_duplicates(["text"], keep=False, inplace=True)
fin = len(df)
diff = ori - fin
print(f"{ori} - {diff} = {fin}")
print()


# add language, filter languages from imatex
print("Lang")
df = df.assign(lang=np.nan)

imatex = df[df.museum == "http://data.silknow.org/graph/imatex"]["text"]
imatex = [langid.classify(x)[0] for x in imatex]
acceptable = {"ca", "it", "es", "en", "fr"}
imatex = [x if x in acceptable else "ca" for x in imatex]
df.loc[df.museum == "http://data.silknow.org/graph/imatex", "lang"] = imatex
# df = df[df["lang"].notna()]

df = df.assign(lang=np.nan)
mask = df["museum"] == "vam"
df.loc[mask, "lang"] = "en"
mask = df["museum"] == "mfa"
df.loc[mask, "lang"] = "en"
mask = df["museum"] == "met"
df.loc[mask, "lang"] = "en"
mask = df["museum"] == "garin"
df.loc[mask, "lang"] = "es"
mask = df["museum"] == "cer"
df.loc[mask, "lang"] = "es"
mask = df["museum"] == "mtmad"
df.loc[mask, "lang"] = "fr"
mask = df["museum"] == "joconde"
df.loc[mask, "lang"] = "fr"
mask = df["museum"] == "unipa"
df.loc[mask, "lang"] = "it"

imatex = df[df.museum == "imatex"]["text"]
imatex = [langid.classify(x)[0] for x in imatex]
acceptable = {"ca", "it", "es", "en", "fr"}
imatex = [x if x in acceptable else "ca" for x in imatex]
df.loc[df.museum == "imatex", "lang"] = imatex
df = df[df["lang"].notna()]

# rename text
df.rename({"text": "txt"}, axis=1, inplace=True)

print()
print(df["museum"].value_counts())


# Write files to disk
print("\nWrite files to disk")
f_name = "data/text_index.tsv"
columns = ["obj", "lang", "txt"]
# select columns
dfc = df[columns].copy()

# remove empty text rows
dfc = dfc[dfc["txt"].notna()].copy()

# drop dups
dfc = dfc.drop_duplicates()
print(len(dfc))

# save files
dfc.to_csv(f_name, sep="\t", index=False, na_rep="NULL")
print(f"Text samples: {len(dfc)}")
print()
