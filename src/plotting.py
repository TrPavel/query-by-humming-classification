"""
Plotting utilities for the Query-by-Humming project.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display
from sklearn.metrics import confusion_matrix


def plot_waveform_and_spectrogram(y, sr, title: str = "", save_path: str | None = None):
    """Side-by-side waveform and Mel-spectrogram for a single recording."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(np.linspace(0, len(y) / sr, len(y)), y)
    axes[0].set_title(f"Waveform: {title}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    img = librosa.display.specshow(mel_spec_db, x_axis="time", y_axis="mel", sr=sr, ax=axes[1])
    axes[1].set_title("Mel-spectrogram")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_class_distribution(df, save_path: str | None = None):
    """Bar charts for song and type distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df["song"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue")
    axes[0].set_title("Samples per Song")
    axes[0].set_xlabel("Song")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=45)

    df["type"].value_counts().plot(kind="bar", ax=axes[1], color=["coral", "lightgreen"])
    axes[1].set_title("Hum vs Whistle")
    axes[1].set_xlabel("Type")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_true, y_pred, class_names, title: str = "Confusion Matrix", save_path: str | None = None
):
    """Annotated heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cnn_training(history, save_path: str | None = None):
    """Accuracy and loss curves from a Keras training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("CNN Accuracy over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("CNN Loss over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
