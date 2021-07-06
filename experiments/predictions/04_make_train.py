#!/usr/bin/env python
"""
Split train data per task.
"""
import pandas as pd

TASKS = [
    "material_group",
    "place_country_code",
    "time_label",
    "technique_group",
]
COLS = ["obj", "museum", "txt", "lang"]

full_df = pd.read_csv("data/processed.tsv", sep="\t")

for task in TASKS:
    cols = COLS + [task]
    df = full_df[cols]
    df = df[df[task].notna()]
    df = df[df["txt"].notna()]
    dst = f"data/train_data/{task}.tsv"
    df.to_csv(dst, sep="\t", index=False)
