#!/usr/bin/env python
"""Convert EURECOM data dump file into a train sets.

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
MIN_LABEL_COUNT = 120


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


def labels_for_col(df, col):
    dfc = df.copy()
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

df["category_group"] = df["category_group"].map(converter_func_img)

df["time_label"] = df["time_label"].map(converter_func_time)
for col in ["technique_group", "material_group", "place_country_code"]:
    df[col] = df[col].map(converter_func_labels)


# Drop duplicates
print("Duplicate OBJ code")
ori = len(df)
df = df.drop_duplicates()
df = df.drop_duplicates(subset=["obj"])
fin = len(df)
diff = ori - fin
print(f"{ori} - {diff} = {fin}")
print()

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

print(f"DF len = {len(df)}")


# Write processed file to disk
print("\nWrite files to disk")
f_name = "data/processed.tsv"
# save files
df.to_csv(f_name, sep="\t", index=False, na_rep="NULL")

print()
