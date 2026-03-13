"""
Data loading and metadata parsing for the MLEnd Hums & Whistles dataset.
"""

import os
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .features import load_and_preprocess, extract_mfcc_features, extract_mfcc_extended


def parse_filename(filename: str) -> dict | None:
    """Extract metadata from a dataset filename.

    Expected format: ``S{interpreter}_{type}_{instance}_{song}.wav``

    Returns
    -------
    dict or None
        Keys: filename, interpreter, type, instance, song.
    """
    match = re.match(r"S(\d+)_(hum|whistle)_(\d+)_(\w+)\.wav", filename)
    if match:
        return {
            "filename": filename,
            "interpreter": int(match.group(1)),
            "type": match.group(2),
            "instance": int(match.group(3)),
            "song": match.group(4),
        }
    return None


def load_metadata(audio_dir: str) -> pd.DataFrame:
    """Scan a directory for .wav files and build a metadata DataFrame."""
    all_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    metadata = [parse_filename(f) for f in all_files]
    return pd.DataFrame([m for m in metadata if m is not None])


def extract_all_features(
    df: pd.DataFrame,
    audio_dir: str,
    extended: bool = False,
    verbose: bool = True,
):
    """Extract MFCC features for every file in the metadata DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'filename', 'song', 'interpreter' columns.
    audio_dir : str
        Directory containing .wav files.
    extended : bool
        If True, extract 78-dim extended MFCCs; otherwise 26-dim basic MFCCs.
    verbose : bool
        Print progress every 100 files.

    Returns
    -------
    X : np.ndarray
        Feature matrix, shape ``(n_samples, n_features)``.
    y : np.ndarray
        Integer-encoded labels.
    groups : np.ndarray
        Interpreter IDs (for GroupShuffleSplit).
    label_encoder : LabelEncoder
        Fitted encoder for label ↔ string mapping.
    """
    extract_fn = extract_mfcc_extended if extended else extract_mfcc_features
    features, labels, groups = [], [], []

    for idx, row in df.iterrows():
        if verbose and idx % 100 == 0:
            print(f"  Processing file {idx + 1}/{len(df)}")
        file_path = os.path.join(audio_dir, row["filename"])
        y_audio, sr = load_and_preprocess(file_path)
        features.append(extract_fn(y_audio, sr))
        labels.append(row["song"])
        groups.append(row["interpreter"])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    return np.array(features), y, np.array(groups), label_encoder
