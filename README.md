# Song Recognition from Humming and Whistling

Audio classification pipeline that identifies songs from hummed or whistled recordings. Built as part of the Machine Learning module (ECS7020P) at Queen Mary University of London.

Given a ~10-second recording of someone humming or whistling, the model predicts which of 8 target songs it corresponds to — a classic **Query-by-Humming (QbH)** problem.

## Results

| Approach | Features | Val Accuracy |
|----------|----------|:------------:|
| Random guess (baseline) | — | 12.5% |
| SVM (RBF) | 26-dim MFCC | 22.8% |
| kNN (k=5) | 26-dim MFCC | 21.5% |
| Random Forest | 26-dim MFCC | 22.8% |
| SVM (RBF, C=10) | 78-dim extended MFCC | 30.4% |
| **Random Forest** | **78-dim extended MFCC** | **31.0%** |
| CNN (3-block, 102K params) | Mel-spectrogram | 25.3% |

**Best model: Random Forest with extended MFCC features — 2.5× better than random guessing.**

The CNN reached **25.3% validation accuracy** but was outperformed by classical models. With only 800 samples, the CNN struggled to learn robust mel-spectrogram representations — consistent with QbH literature where deep learning models require significantly larger datasets to surpass engineered features.

## Dataset

The project uses the **MLEnd Hums and Whistles II** dataset:

- **800 recordings** (humming + whistling) from **187 unique interpreters**
- **8 song classes** — fragments from animated movie soundtracks
- Balanced: 100 recordings per song (50 hums + 50 whistles)
- Format: WAV, ~10–20 seconds each

The dataset is not included in this repository. Download it from [MLEndHW_Sample](https://github.com/thekmannn/MLEndHW_Sample) and place the .wav files in a `data/` directory.

## Project Structure

```
├── README.md
├── notebooks/
│   └── analysis.ipynb          # Full analysis notebook with outputs
├── src/
│   ├── __init__.py
│   ├── features.py             # Audio loading + MFCC / Mel-spectrogram extraction
│   ├── data.py                 # Metadata parsing + batch feature extraction
│   └── plotting.py             # Visualisation utilities
├── figures/
│   ├── waveform_spectrogram.png
│   ├── class_distribution.png
│   ├── confusion_matrix_baseline.png
│   ├── confusion_matrix_best.png
│   └── cnn_training.png
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Approach

### Feature Extraction

Two feature representations were explored:

**Basic MFCC (26-dim):** Mean and standard deviation of 13 MFCCs across each recording. Captures the average spectral envelope but loses all temporal dynamics.

**Extended MFCC (78-dim):** Adds first-order (Δ) and second-order (ΔΔ) delta coefficients, providing information about how the spectral shape changes over time — critical for distinguishing melodies.

**Mel-spectrogram (128 × 431):** Full time-frequency representation used as 2D input for the CNN. Preserves temporal structure but requires a much larger model to process.

### Models

**Classical ML (scikit-learn):** SVM with RBF kernel, Random Forest, and k-Nearest Neighbours were trained on MFCC feature vectors. SVM performed best after hyperparameter tuning (C=10) on extended features.

**CNN (TensorFlow/Keras):** Three convolutional blocks (32 → 64 → 128 filters) with batch normalisation, GlobalAveragePooling, and dropout. Despite ~102K parameters, the model struggled to compete with classical approaches — validation accuracy reached **25.3%**, below the best classical model, consistent with QbH literature where neural approaches require significantly more training data.

### Key Design Decisions

- **GroupShuffleSplit** for train/val split — ensures no interpreter appears in both sets, preventing data leakage through voice identity.
- **StandardScaler** fitted on training set only, applied to validation set.
- **Early stopping** (patience=10) for CNN training to mitigate overfitting.

## Setup

```bash
# Clone the repo
git clone https://github.com/TrPavel/query-by-humming-classification.git
cd query-by-humming-classification

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Place .wav files into data/ directory (see Dataset section above)
```

## Usage

### Using the src modules directly

```python
from src.features import load_and_preprocess, extract_mfcc_extended
from src.data import load_metadata, extract_all_features

# Load and extract features from a single file
y, sr = load_and_preprocess("data/S001_hum_1_Happy.wav")
features = extract_mfcc_extended(y, sr)  # → (78,)

# Batch extraction
df = load_metadata("data/")
X, y, groups, encoder = extract_all_features(df, "data/", extended=True)
```

### Running the notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

The notebook contains the full analysis pipeline with all outputs preserved.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![librosa](https://img.shields.io/badge/librosa-audio-blue)

## Acknowledgements

- **Dataset:** [MLEnd Hums and Whistles II](https://github.com/thekmannn/MLEndHW_Sample) — sample dataset for MLEnd coursework.
- **Course:** ECS7020P Machine Learning, Queen Mary University of London (2025–26).

## License

[MIT](LICENSE)
