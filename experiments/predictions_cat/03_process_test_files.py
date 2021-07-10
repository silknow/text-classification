#!/usr/bin/env python

import numpy as np
import pandas as pd

INPUT_FILES = [
    "data/test_data/time_label.csv",
    "data/test_data/place_country_code.csv",
    "data/test_data/material_group.csv",
    "data/test_data/technique_group.csv",
]
LABEL_COLS = [
    "time_label",
    "place_country_code",
    "material_group",
    "technique_group",
]
TRAIN_FILE = "data/processed.tsv"


# Target Label, File
for label, ifile in zip(LABEL_COLS, INPUT_FILES):
    print(label)
    print(ifile)

    # get the corresponding training data
    dfo = pd.read_csv(TRAIN_FILE, sep="\t")
    dfo = dfo[dfo[label].notna()]
    museums = dfo["museum"].unique().tolist()
    print(museums)

    # load target file
    df = pd.read_csv(ifile)

    # fix museum string
    df["museum"] = df["museum"].map(lambda s: s.split("/")[-1].strip())
    # NULL unseen musem
    mask = ~df["museum"].isin(museums)
    df.loc[mask, "museum"] = np.nan

    # process for each "other" label process the data to remove unseen labels
    for other in LABEL_COLS:
        if other == label:
            continue
        # get train labels
        labels = dfo[dfo[other].notna()][other].unique().tolist()
        print(labels)
        # NULL unseen labels
        mask = ~df[other].isin(labels)
        df.loc[mask, other] = np.nan
    # write output
    dst = ifile.split(".")[0] + ".tsv"
    df.to_csv(dst, sep="\t", na_rep="NULL", index=False)
