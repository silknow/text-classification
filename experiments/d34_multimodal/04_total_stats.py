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


RANDOM_NUMBER = 621323849
RANDOM_NUMBER2 = 581085259
FNAME = "data/total_post.csv"
COLS = [
    "museum",
    "text",
    "place_country_code",
    "time_label",
    "technique_group",
    "material_group",
    "img",
    "type_a_group",
]
ALL_COLS = ["obj"] + COLS

LABEL_COLS = [
    "place_country_code",
    "time_label",
    "technique_group",
    "material_group",
]
MIN_CHARS = 10
MIN_LABEL_COUNT = 50


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
                return x[0].strip()
            return x
    except:
        return x.strip()


def converter_func_img(x):
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

            x = [u.strip() for u in x]
            return ",".join(x)
    except:
        return x.strip()


def merge_text(x):
    if not x:
        return np.nan
    if pd.isna(x):
        return np.nan

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
for col in ALL_COLS:
    if col == "text":
        print("\tlist=text")
        df[col] = df[col].map(merge_text)
    elif col == "img":
        print("\tlist=img")
        df[col] = df[col].map(converter_func_img)
    else:
        print(f"\tlist={col}")
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

# Drop non-fabric
print("Non-Fabric")
ori = len(df)
df = df[
    df["type_a_group"] == "http://data.silknow.org/vocabulary/facet/fabrics"
]
fin = len(df)
diff = ori - fin
print(f"{ori} - {diff} = {fin}")
print()

"""
# Handle text
print("No Text")
ori = len(df)
# Remove no text
df = df[df.text.notna()]
fin = len(df)
diff = ori - fin
print(f"{ori} - {diff} = {fin}")
print()
"""

# Fix whitespaces
# df["text"] = df["text"].map(lambda t: " ".join(str(t).split()))

"""
print("Short Text")
ori = len(df)
mask = df["text"].map(len) >= MIN_CHARS
df = df[mask]

fin = len(df)
diff = ori - fin
print(f"{ori} - {diff} = {fin}")
print()
"""

# add language, filter languages from imatex
print("Lang")
df = df.assign(lang=np.nan)
mask = df["museum"] == "http://data.silknow.org/graph/vam"
df.loc[mask, "lang"] = "en"
mask = df["museum"] == "http://data.silknow.org/graph/mfa"
df.loc[mask, "lang"] = "en"
mask = df["museum"] == "http://data.silknow.org/graph/met"
df.loc[mask, "lang"] = "en"
mask = df["museum"] == "http://data.silknow.org/graph/garin"
df.loc[mask, "lang"] = "es"
mask = df["museum"] == "http://data.silknow.org/graph/cer"
df.loc[mask, "lang"] = "es"
mask = df["museum"] == "http://data.silknow.org/graph/mtmad"
df.loc[mask, "lang"] = "fr"
mask = df["museum"] == "http://data.silknow.org/graph/joconde"
df.loc[mask, "lang"] = "fr"
mask = df["museum"] == "http://data.silknow.org/graph/unipa"
df.loc[mask, "lang"] = "it"

imatex = df[df.museum == "http://data.silknow.org/graph/imatex"]["text"]
imatex = [langid.classify(x)[0] if not pd.isna(x) else "ca" for x in imatex]
acceptable = {"ca", "it", "es", "en", "fr"}
imatex = [x if x in acceptable else "ca" for x in imatex]
df.loc[df.museum == "http://data.silknow.org/graph/imatex", "lang"] = imatex
# df = df[df["lang"].notna()]

# rename text
df.rename({"text": "txt"}, axis=1, inplace=True)

# rename other for technique
df.replace(
    {
        "technique_group": {
            "http://data.silknow.org/vocabulary/facet/other_technique": "OTHER"
        }
    },
    inplace=True,
)


# Count labels, mark small labels as OTHER
print("Label count")
for col in LABEL_COLS:
    print(col)
    # drop labels below a min freq
    below = [
        k for k, v in df[col].value_counts().items() if v < MIN_LABEL_COUNT
    ]
    # print(below)
    mask = df[col].isin(below)
    df.loc[mask, col] = "OTHER"

    print(df[col].value_counts(dropna=False))
    print()


print()
print(df["museum"].value_counts())


# for the text (to create embeddings)
# for each property
for col in LABEL_COLS:
    print(f"Generating {col}")

    columns = ["obj", "museum", "lang", "txt", "img"] + LABEL_COLS
    # select columns
    dfc = df[columns].copy()

    # remove empty label rows
    dfc = dfc[dfc[col].notna()].copy()

    # remove OTHER label
    mask = dfc[col] == "OTHER"
    dfc.loc[mask, col] = np.nan
    dfc = dfc[dfc[col].notna()].copy()

    total = len(dfc)
    # percentage
    text_n = len(dfc[dfc["txt"].notna()])
    text_p = text_n / total
    text_p = int(text_p * 100)

    img_n = len(dfc[dfc["img"].notna()])
    img_p = img_n / total
    img_p = int(img_p * 100)

    both = dfc[dfc["img"].notna()]
    both = both[both["txt"].notna()]
    both_n = len(both)
    both_p = both_n / total
    both_p = int(both_p * 100)

    # print
    print(
        f"{col} ({len(dfc)}): text: {text_n} ({text_p}) img: {img_n} ({img_p})"
    )
    print("\n")
