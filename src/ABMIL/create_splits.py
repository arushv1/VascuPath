import os
import csv
import json
import argparse
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


# =============================================================================
# SECTION 1: LABEL FILE PARSING
# =============================================================================
COMPARISONS = {
    "control_vs_rhi": {
        "class_0": "Control",       # label in spreadsheet
        "class_1": "RHI",
        "name": "Control vs RHI",
        "description": "Effect of head impacts without CTE pathology",
    },
    "rhi_vs_low": {
        "class_0": "RHI",
        "class_1": "Low CTE",
        "name": "RHI vs Low CTE",
        "description": "Onset of early CTE changes",
    },
    "low_vs_high": {
        "class_0": "Low CTE",
        "class_1": "High CTE",
        "name": "Low CTE vs High CTE",
        "description": "Disease severity differentiation",
    },
    "control_vs_CTE": {
        "class_0": "Control",
        "class_1": ["Low CTE", "High CTE"],
        "name": "No CTE vs CTE",
        "description": "Any CTE pathology vs none",
    },
}

# Label File Parsing
def load_label_file(label_file: str, comparison_key, sheet_name=0):
    df = pd.read_excel(label_file, sheet_name=sheet_name, engine="openpyxl")
    
    comp = COMPARISONS[comparison_key]

    df.columns = [
        "case_id", "age", "block_id", "stain", "description",
        "scanner", "mag", "filename", "path_group", 
        *[f"extra_{i}" for i in range(len(df.columns) - 9)]
    ]

    class_0 = comp["class_0"] if isinstance(comp["class_0"], list) else [comp["class_0"]]
    class_1 = comp["class_1"] if isinstance(comp["class_1"], list) else [comp["class_1"]]

    mask = df["path_group"].isin(class_0 + class_1)
    df = df[mask].copy()

    label_map = {**{c: 0 for c in class_0}, **{c: 1 for c in class_1}}
    df["label"] = df["path_group"].map(label_map)
    df["svs_stem"] = df["filename"].apply(lambda x: str(x).replace(".svs", ""))

    slide_labels = {}
    for _, row in df.iterrows():
        slide_labels[row["svs_stem"]] = (row["label"], row["case_id"])

    return slide_labels, comp