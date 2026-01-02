# src/preprocessing/load_data.py

import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def load_diadata_long_format(
    file_paths,
    chunksize=1_000_000
):
    """
    Load DiaData CSVs in chunks and return
    per-subject longitudinal glucose series.

    Returns
    -------
    dict:
        { PtID : DataFrame(ts, glucose) }
    """

    subject_data = defaultdict(list)

    for file_path in file_paths:
        print(f"Reading {file_path} in chunks...")

        for chunk in tqdm(
            pd.read_csv(
                file_path,
                usecols=["PtID", "ts", "GlucoseCGM"],
                chunksize=chunksize
            ),
            desc="Chunks"
        ):
            # Rename for consistency
            chunk = chunk.rename(
                columns={"GlucoseCGM": "glucose"}
            )

            # Drop missing values
            chunk = chunk.dropna(subset=["PtID", "ts", "glucose"])

            # Group by patient
            for pid, group in chunk.groupby("PtID"):
                subject_data[pid].append(
                    group[["ts", "glucose"]]
                )

    # Concatenate per subject safely
    for pid in list(subject_data.keys()):
        subject_data[pid] = (
            pd.concat(subject_data[pid])
              .sort_values("ts")
              .reset_index(drop=True)
        )

    return subject_data
