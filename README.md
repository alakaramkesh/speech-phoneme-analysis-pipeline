# Acoustic and Neural Representations in a Phonetically Aligned Speech Corpus

Research Project — Advanced Statistics
M1 Computational Linguistics · Université Paris · 2025–2026

## Overview

This project compares classical acoustic phonetic features with neural speech representations extracted from modern pre-trained speech models.
The analyses were conducted on the Russian–French Interference Corpus and focus on:

* phonological structure,
* speaker variability,
* L1/L2 pronunciation differences,
* phoneme clustering,
* and representation similarity.

The project combines:

* acoustic phonetics,
* neural speech representations,
* statistical modelling,
* and reproducible machine learning pipelines.

The complete workflow was implemented using DVC and Pixi.

---

# Repository Structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── statistics/
│
├── src/
│   ├── parse_corpus.py
│   ├── extract_acoustics.py
│   ├── extract_neural_whisper.py
│   ├── extract_neural_xlsr.py
│   ├── normalization.py
│   ├── descriptive_statistics.py
│   ├── rsm_analysis.py
│   ├── mixed_models.py
│   ├── clustering_analysis.py
│   └── section*_analysis.py
│
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── pixi.toml
└── README.md
```

---

# Corpus

The project uses the Russian–French Interference Corpus available on ORTOLANG:

[https://www.ortolang.fr/market/corpora/ru-fr_interference](https://www.ortolang.fr/market/corpora/ru-fr_interference)

The corpus contains:

* native French speakers (L1),
* Russian learners of French (L2),
* repeated sentence productions,
* TextGrid alignments,
* and speaker metadata.

---

# Pipeline Overview

The project is organised as a reproducible DVC pipeline.

## Stage 1 — Corpus Parsing

Input:

* TextGrid files
* metadata.csv

Output:

* phoneme_table.csv

This stage extracts phoneme-level information:

* phoneme labels,
* onset/offset boundaries,
* duration,
* speaker metadata,
* repetition information.

Run:

```bash
pixi run python src/parse_corpus.py
```

---

## Stage 2 — Acoustic Feature Extraction

Input:

* phoneme_table.csv
* WAV files

Output:

* features_acoustic.csv

Extracted acoustic features:

* duration,
* F1,
* F2,
* F3,
* f0,
* spectral centre of gravity (SCG).

Run:

```bash
pixi run python src/extract_acoustics.py
```

---

## Stage 3 — Whisper Representations

Model:

* openai/whisper-medium

Layers analysed:

* layer 4,
* layer 20.

Output:

* features_whisper_layer4.npz
* features_whisper_layer20.npz

Run:

```bash
pixi run python src/extract_neural_whisper.py
```

---

## Stage 4 — XLS-R Representations

Model:

* facebook/wav2vec2-large-xlsr-53

Layers analysed:

* layer 4,
* layer 10,
* layer 20.

Output:

* features_xlsr_layer4.npz
* features_xlsr_layer10.npz
* features_xlsr_layer20.npz

Run:

```bash
pixi run python src/extract_neural_xlsr.py
```

---

## Stage 5 — Normalisation and Dimensionality Reduction

This stage includes:

* Lobanov normalisation,
* PCA reduction,
* UMAP reduction.

Output:

* normalised acoustic features,
* PCA embeddings,
* UMAP projections.

---

## Stage 6 — Statistical Analysis

The analysis stage includes:

* descriptive statistics,
* Mantel tests,
* mixed-effects models,
* ROPE analysis,
* hierarchical clustering,
* phoneme classification.

Outputs are stored in:

```text
results/
```

---

# Main Findings

## Acoustic Features

* Vowels such as /a/ and /ɑ/ showed the largest inter-speaker variability.
* Lobanov normalisation reduced speaker-dependent variability effectively.
* Acoustic vowel charts broadly preserved the expected French vowel structure.

## Whisper

* Whisper embeddings produced relatively weak phoneme separation.
* Cluster boundaries were less distinct.
* Whisper appeared less sensitive to detailed phonological structure.

## XLS-R

* XLS-R consistently produced clearer phoneme clusters.
* XLS-R achieved the highest phoneme classification accuracy (~95%).
* XLS-R preserved acoustic-phonological structure much more strongly than Whisper.

---

# Reproducibility

The project uses:

* DVC for pipeline reproducibility,
* Pixi for dependency management,
* parameter versioning through params.yaml.

To reproduce the full pipeline:

```bash
pixi run dvc repro
```

To check pipeline status:

```bash
pixi run dvc status
```

---

# Environment Setup

## Clone the Repository

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

## Install Pixi

[https://pixi.sh/latest/](https://pixi.sh/latest/)

## Install Dependencies

```bash
pixi install
```

---

# Libraries and Tools

Main libraries used in this project:

* transformers
* torch
* librosa
* soundfile
* parselmouth
* pandas
* numpy
* scipy
* scikit-learn
* matplotlib
* seaborn
* statsmodels
* umap-learn
* dvc

---

# Figures and Results

Generated outputs include:

* vowel charts,
* PCA projections,
* UMAP projections,
* representational similarity
