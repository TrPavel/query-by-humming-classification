"""
Feature extraction utilities for audio classification.

Provides functions to load, preprocess, and extract features from audio files
for Query-by-Humming (QbH) classification tasks.
"""

import numpy as np
import librosa


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
SAMPLE_RATE = 22050
DURATION = 10  # seconds
N_MFCC = 13


# ──────────────────────────────────────────────
# Audio loading
# ──────────────────────────────────────────────
def load_and_preprocess(file_path: str, sr: int = SAMPLE_RATE, duration: int = DURATION):
    """Load an audio file and normalise to a fixed length.

    Pads short recordings with silence or truncates long ones
    so that every sample has exactly ``sr * duration`` samples.

    Parameters
    ----------
    file_path : str
        Path to .wav file.
    sr : int
        Target sample rate.
    duration : int
        Target duration in seconds.

    Returns
    -------
    y : np.ndarray
        Audio time-series of shape ``(sr * duration,)``.
    sr : int
        Sample rate.
    """
    y, sr = librosa.load(file_path, sr=sr)
    target_length = sr * duration
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        y = y[:target_length]
    return y, sr


# ──────────────────────────────────────────────
# Feature extractors
# ──────────────────────────────────────────────
def extract_mfcc_features(y, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC):
    """Extract basic MFCC statistics (mean + std) → 26-dim vector."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])


def extract_mfcc_extended(y, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC):
    """Extract extended MFCC features (MFCC + Δ + ΔΔ, mean + std) → 78-dim vector.

    Includes first- and second-order delta coefficients to capture
    temporal dynamics that static MFCCs miss.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    return np.concatenate([
        np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
        np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1),
        np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1),
    ])


def extract_mel_spectrogram(y, sr: int = SAMPLE_RATE, n_mels: int = 128):
    """Compute a log-scaled Mel-spectrogram suitable for CNN input.

    Returns
    -------
    np.ndarray
        Mel-spectrogram in dB scale, shape ``(n_mels, time_frames)``.
    """
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel_spec, ref=np.max)
