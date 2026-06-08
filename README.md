# Synthetic Jet XGBoost Modeling

This repository contains Python scripts for modeling synthetic jet data using XGBoost. The project is divided into two main approaches:

## 1. Full Profile Model (`full_profile_model/`)
This model uses the full flux profiles for training and prediction.
- `data_generation.py`: Script to generate synthetic jet data.
- `model_training.py`: XGBoost training script using the full profiles.
- `model_testing.py`: Script to test the trained model.
- `diagnostic_plots.py`: Diagnostic and visualization tools.

## 2. SED 30-Frequency Model (`sed_30freq_model/`)
This model uses a 30-frequency Spectral Energy Distribution (SED) approach.
- `data_generation.py`: Data generation specific to the 30-frequency bands.
- `model_training.py`: XGBoost training script for SED-based data.
- `model_testing.py`: Testing and performance evaluation.
- `model_validation.py`: Additional validation tools for the SED model.

## Requirements
- `numpy`
- `xgboost`
- `h5py`
- `matplotlib`
- `scikit-learn`
- `joblib`
- `scipy`
