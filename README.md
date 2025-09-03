# Celestial Object Classifier

## Introduction

This project aims to classify celestial objects from the Sloan Digital Sky Survey (SDSS) dataset. Four different machine learning models have been trained and evaluated for this classification task:

*   K-Nearest Neighbors (KNN)
*   Decision Tree
*   Neural Network
*   Random Forest

The trained models are saved and can be used for inference on new data. The `load_and_run_all_models.ipynb` notebook provides a demonstration of how to load and use these models.

## Dataset

The dataset used in this project is from the Sloan Digital Sky Survey (SDSS). The SDSS is a major multi-spectral imaging and spectroscopic redshift survey using a dedicated 2.5-m wide-angle optical telescope at Apache Point Observatory in New Mexico, United States.

Due to different experimentation and feature selection techniques used by team members, each model was trained on a different set of features, which are stored in separate CSV files.

## Models Used

### 1. K-Nearest Neighbors (KNN)

*   **Features Used:** The KNN model was trained on the 'ugriz' features, which represent the response of the object to five different filters.
*   **Model File:** `models/final_knn_solana.pkl`
*   **Data Files:** `data/KNN_X_ugriz.csv`, `data/KNN_y.csv`

### 2. Decision Tree

*   **Features Used:** The Decision Tree model uses features selected through Recursive Feature Elimination (RFE) to identify the most significant features for classification.
*   **Model File:** `models/final_decision_tree_model.pkl`
*   **Data Files:** `data/DT_X_rfe_selected.csv`, `data/DT_y.csv`

### 3. Neural Network

*   **Features Used:** The Neural Network model was trained on the 'solana' features.
*   **Model File:** `models/final_neural_network_solana.keras`
*   **Data Files:** `data/NN_X_solana.csv`, `data/NN_y.csv`

### 4. Random Forest

*   **Features Used:** The Random Forest model was trained on a specific set of features.
*   **Model File:** `models/final_random_forest_model.pkl`
*   **Data Files:** `data/RF_X_features.csv`, `data/RF_y.csv`

## File Structure

The project is organized into the following directories:

```
├── data/                # Contains the CSV files for each model
├── models/              # Contains the saved model files
├── notebooks/           # Contains the Jupyter notebooks for model training
├── load_and_run_all_models.ipynb  # Notebook to load and run all models
└── README.md
```

## How to Run

To see the models in action, you can run the `load_and_run_all_models.ipynb` notebook. This notebook demonstrates how to:

1.  Load the pre-trained models.
2.  Load the corresponding data for each model.
3.  Make predictions using the loaded models.

Please ensure you have the required libraries installed to run the notebook.
