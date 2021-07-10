#!/usr/bin/env python

import os
import pandas as pd

COLS = [
    "obj",
    "museum",
    "place_country_code",
    "time_label",
    "technique_group",
    "material_group",
]

LABEL_COLS = [
    "place_country_code",
    "time_label",
    "technique_group",
    "material_group",
]

df = pd.read_csv("data/processed.tsv", sep="\t", usecols=COLS)

for label in LABEL_COLS:
    fname = label + ".tsv"
    fpath = os.path.join("data/train_data", fname)
    df_t = df[df[label].notna()]
    df_t.to_csv(fpath, sep="\t", na_rep="NULL", index=False)
