"""Query-by-Humming classification utilities."""

from .features import (
    load_and_preprocess,
    extract_mfcc_features,
    extract_mfcc_extended,
    extract_mel_spectrogram,
    SAMPLE_RATE,
    DURATION,
    N_MFCC,
)
from .data import parse_filename, load_metadata, extract_all_features
from .plotting import (
    plot_waveform_and_spectrogram,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_cnn_training,
)
