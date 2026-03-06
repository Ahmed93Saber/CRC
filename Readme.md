# CRC ABMIL: Whole Slide Image CRC grade Classification 

This repository contains a deep learning pipeline for analyzing Whole Slide Images (WSIs), specifically aimed at Colorectal Cancer (CRC) binary classification tasks (e.g., predicting PD-L1 status). It leverages **Attention-Based Multiple Instance Learning (ABMIL)** on pre-extracted patch features to aggregate bag-level predictions and provide interpretability through attention heatmaps.

## 🚀 Key Features

* **Multiple Instance Learning (MIL):** Implements an ABMIL slide encoder (`src/trident_load.py`) with optional sigmoid-gating and multi-head attention capabilities.
* **Hyperparameter Optimization:** Fully integrated with `optuna` for automated tuning of learning rate, weight decay, batch size, attention heads, and hidden dimensions.
* **Robust Evaluation:** Uses Stratified K-Fold Cross-Validation (`src/train.py`) alongside an array of metrics including AUC, F1-Score (Weighted/Binary), Precision, Recall, and Balanced Accuracy.
* **Interpretability:** Includes a pipeline to generate and visualize attention heatmaps overlaid on the original WSIs using the `trident` library.
* **Efficient Data Loading:** Utilizes `h5py` to load pre-computed patch features (e.g., Conch v15 features) directly from `.h5` files, drastically speeding up the training process.

## 📁 Repository Structure

```text
.
├── main.py                     # Entry point for training and Optuna optimization
├── src/                        # Core source code directory
│   ├── datasets.py             # PyTorch H5Dataset for loading patch features and labels
│   ├── engine.py               # Training and evaluation loops (train_one_epoch, evaluate_model)
│   ├── models.py               # BinaryClassificationModel wrapping the ABMIL encoder
│   ├── train.py                # Cross-validation orchestrator and Optuna trial reporting
│   ├── trident_load.py         # ABMIL architecture and BaseSlideEncoder definitions
│   └── utils.py                # Utilities for early stopping, metrics, seeding, and device config
├── dataframes/                 # Directory for metadata CSVs (e.g., metadata.csv)
├── artifacts_max/              # Directory where Optuna trials, models, and confusion matrices are saved
├── heatmaps.ipynb              # Jupyter notebook for running inference and generating heatmaps
└── playground.ipynb            # Notebook for random scripts :D 