# Stars, Galaxy, and Quasars Classification

A comprehensive machine learning project for classifying celestial objects (Stars, Galaxies, and Quasars) using data from the **Sloan Digital Sky Survey (SDSS)**. This project implements and compares four different machine learning models to achieve optimal classification accuracy.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Models Implemented](#models-implemented)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Model Performance](#model-performance)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This project aims to classify celestial objects into three categories:
- **Galaxy (0)**: Extended astronomical objects with billions of stars
- **Quasar (1)**: Quasi-stellar radio sources, extremely distant and luminous objects
- **Star (2)**: Individual massive balls of plasma held together by gravity

The dataset contains photometric data from the **Sloan Digital Sky Survey (SDSS)**, which includes magnitude measurements across multiple filter bands (u, g, r, i, z) and spatial coordinates. Multiple machine learning algorithms are employed and compared to determine the most effective approach for this multi-class classification task.

### Problem Statement
Automatically classify celestial objects from observational data without manual inspection, enabling scalable processing of large astronomical datasets.

### Dataset Characteristics
- **Total Samples**: Thousands of labeled celestial observations
- **Classes**: 3 (Galaxy, Quasar, Star)
- **Features**: ~16 astronomical attributes including:
  - **Positional Data**: Right Ascension (ra), Declination (dec)
  - **Photometric Data**: Magnitudes in u, g, r, i, z bands
  - **Observational Metadata**: run, rerun, camcol, field, plate, fiberid, specobjid
  - **Physical Data**: Redshift

---

## 📊 Dataset Description

### Data Source
**Sloan Digital Sky Survey (SDSS)** - A comprehensive astronomical survey that has mapped millions of celestial objects.

### Features Included

| Feature | Type | Description |
|---------|------|-------------|
| ra | Continuous | Right Ascension (angular position) |
| dec | Continuous | Declination (angular position) |
| u, g, r, i, z | Continuous | Magnitudes in different filter bands |
| redshift | Continuous | Cosmological redshift value |
| run | Categorical | Observing run identifier |
| rerun | Categorical | Processing run identifier |
| camcol | Categorical | Camera column (1-6) |
| field | Categorical | Field identifier |
| plate | Categorical | Spectroscopic plate identifier |
| fiberid | Categorical | Fiber identifier |
| specobjid | Categorical | Spectroscopic object identifier |
| objid | Categorical | Unique object identifier |
| class | Target | Class label (GALAXY, QSO, STAR) |

### Data Quality
- **Missing Values**: None (clean dataset)
- **Class Distribution**: Imbalanced - Galaxies are the majority class
- **Data Preprocessing**: Feature scaling and selection performed per model requirements

---

## 🤖 Models Implemented

### 1. **Decision Tree Classifier**
**File**: [notebooks/decision_tree.ipynb](notebooks/decision_tree.ipynb)

- **Feature Selection**: RFE (Recursive Feature Elimination) selected top 10 features
- **Purpose**: Interpretable tree-based classification with clear decision rules
- **Preprocessing**:
  - Removed ID columns (objid, specobjid)
  - One-hot encoding of categorical features
  - RFE feature selection (10 features)
- **Output Files**:
  - `data/DT_X_rfe_selected.csv` - Selected features
  - `data/DT_y.csv` - Target labels

**Strengths**:
- Highly interpretable (decision rules can be visualized)
- Fast training and prediction
- Handles non-linear relationships
- No feature scaling required

**Weaknesses**:
- Prone to overfitting without proper pruning
- Sensitive to small data variations

---

### 2. **K-Nearest Neighbors (KNN)**
**File**: [notebooks/knn.ipynb](notebooks/knn.ipynb)

- **Feature Set**: UGRIZ magnitudes (u, g, r, i, z bands)
- **Approach**: Instance-based learning using distance metrics
- **Preprocessing**:
  - Feature scaling (essential for distance-based algorithms)
  - Focus on photometric features only
- **Output Files**:
  - `data/KNN_X_ugriz.csv` - UGRIZ magnitude features
  - `data/KNN_y.csv` - Target labels

**Hyperparameters**:
- k = optimal value (determined via cross-validation)
- Distance metric = Euclidean

**Strengths**:
- Simple and effective for this dataset
- Non-parametric approach (no training phase)
- Naturally handles multi-class classification

**Weaknesses**:
- Computationally expensive for large datasets
- Sensitive to feature scaling and dimensionality
- Memory-intensive (stores all training data)

---

### 3. **Random Forest Classifier**
**File**: [notebooks/random_fotrst.ipynb](notebooks/random_fotrst.ipynb)

- **Ensemble Method**: Multiple decision trees with voting mechanism
- **Feature Set**: All available features
- **Preprocessing**:
  - One-hot encoding of categorical features
  - Feature importance analysis
- **Output Files**:
  - `data/RF_X_features.csv` - All selected features
  - `data/RF_y.csv` - Target labels

**Model Architecture**:
- Number of trees = configurable (e.g., 100)
- Bootstrap sampling for diversity
- Majority voting for final predictions

**Strengths**:
- Reduces overfitting compared to single decision tree
- Feature importance ranking
- Handles non-linear relationships well
- Robust to outliers
- Excellent performance on imbalanced datasets

**Weaknesses**:
- Less interpretable than single decision tree
- Higher computational cost
- Longer training time

---

### 4. **Neural Network (Deep Learning)**
**File**: [notebooks/neural_network.ipynb](notebooks/neural_network.ipynb)

- **Framework**: TensorFlow/Keras
- **Architecture**: Dense neural network with multiple layers
- **Feature Set**: Comprehensive feature set (SDSS photometry)
- **Preprocessing**:
  - Standardization (mean=0, std=1)
  - One-hot encoding of target classes
- **Output Files**:
  - `data/NN_X_solana.csv` - Preprocessed features
  - `data/NN_y.csv` - Target labels
  - `models/final_neural_network_solana.keras` - Trained model

**Network Architecture**:
```
Input Layer (n_features)
    ↓
Dense Layer (128 units, ReLU activation)
    ↓
Dropout (0.3)
    ↓
Dense Layer (64 units, ReLU activation)
    ↓
Dropout (0.3)
    ↓
Dense Layer (32 units, ReLU activation)
    ↓
Output Layer (3 units, Softmax activation)
```

**Hyperparameters**:
- Loss Function: Categorical Crossentropy
- Optimizer: Adam
- Batch Size: 32
- Epochs: 100+ (with early stopping)

**Strengths**:
- Learns complex non-linear patterns
- Excellent for large datasets
- Can achieve high accuracy with proper tuning
- Flexible architecture

**Weaknesses**:
- Requires more data for optimal performance
- Longer training time
- Less interpretable ("black box")
- Hyperparameter tuning complexity

---

## 📁 Project Structure

```
Stars-Galaxy-Quasars-Classification/
│
├── README.md                              # Project documentation (this file)
│
├── load_and_run_all_models.ipynb          # Master notebook to run all models
│
├── notebooks/                             # Individual model notebooks
│   ├── decision_tree.ipynb                # Decision Tree implementation
│   ├── knn.ipynb                          # K-Nearest Neighbors implementation
│   ├── neural_network.ipynb               # Neural Network implementation
│   └── random_fotrst.ipynb                # Random Forest implementation
│
├── data/                                  # Processed datasets
│   ├── DT_X_rfe_selected.csv             # Decision Tree features (RFE selected)
│   ├── DT_y.csv                          # Decision Tree labels
│   ├── KNN_X_ugriz.csv                   # KNN features (UGRIZ magnitudes)
│   ├── KNN_y.csv                         # KNN labels
│   ├── NN_X_solana.csv                   # Neural Network features
│   ├── NN_y.csv                          # Neural Network labels
│   ├── RF_X_features.csv                 # Random Forest features
│   └── RF_y.csv                          # Random Forest labels
│
└── models/                                # Saved trained models
    └── final_neural_network_solana.keras # Pre-trained neural network
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/krishparmar22242/Stars-Galaxy-Quasars-Classification.git
cd Stars-Galaxy-Quasars-Classification
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Libraries
```bash
pip install -r requirements.txt
```

Or manually install key dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras joblib
```

### Key Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=1.0.0 | Data manipulation and analysis |
| numpy | >=1.18.0 | Numerical computations |
| scikit-learn | >=0.24.0 | ML algorithms (Decision Tree, KNN, Random Forest) |
| tensorflow | >=2.4.0 | Neural Network framework |
| keras | >=2.4.0 | Neural Network API (part of TensorFlow) |
| matplotlib | >=3.3.0 | Data visualization |
| seaborn | >=0.11.0 | Statistical data visualization |
| joblib | >=1.0.0 | Model serialization |

### Step 4: Launch Jupyter Notebook
```bash
jupyter notebook
```

---

## 💻 Usage Guide

### Option 1: Run All Models at Once
Open and execute `load_and_run_all_models.ipynb` in Jupyter Notebook. This master notebook:
- Loads all preprocessed datasets
- Runs inference on random samples from each model
- Displays predictions and actual labels for comparison

```python
# The notebook will:
# 1. Load Decision Tree model and make predictions
# 2. Load KNN model and make predictions
# 3. Load Random Forest model and make predictions
# 4. Load Neural Network model and make predictions
```

### Option 2: Run Individual Model Notebooks
Each model has its own dedicated notebook:

#### Decision Tree
```bash
# Open notebooks/decision_tree.ipynb
# Run cells to:
# 1. Load and explore SDSS dataset
# 2. Preprocess and encode features
# 3. Apply RFE for feature selection
# 4. Train Decision Tree model
# 5. Evaluate with confusion matrix and classification report
```

#### K-Nearest Neighbors
```bash
# Open notebooks/knn.ipynb
# Run cells to:
# 1. Load SDSS dataset
# 2. Select UGRIZ magnitude features
# 3. Apply feature scaling
# 4. Train KNN with optimal k value
# 5. Visualize decision boundaries (2D projections)
```

#### Random Forest
```bash
# Open notebooks/random_fotrst.ipynb
# Run cells to:
# 1. Load and preprocess data
# 2. Train Random Forest ensemble
# 3. Analyze feature importance
# 4. Generate confusion matrix and metrics
```

#### Neural Network
```bash
# Open notebooks/neural_network.ipynb
# Run cells to:
# 1. Load SDSS dataset
# 2. Standardize features
# 3. Encode target classes (one-hot)
# 4. Build and train neural network
# 5. Plot training history and confusion matrix
# 6. Generate classification report
```

### Making Predictions on New Data
```python
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load a specific model
dt_model = joblib.load('models/final_decision_tree_model.pkl')
knn_model = joblib.load('models/final_knn_solana.pkl')
rf_model = joblib.load('models/final_random_forest_model.pkl')
nn_model = load_model('models/final_neural_network_solana.keras')

# Prepare your data
X_new = pd.read_csv('your_data.csv')

# Make predictions
dt_predictions = dt_model.predict(X_new)
knn_predictions = knn_model.predict(X_new)
rf_predictions = rf_model.predict(X_new)
nn_predictions = np.argmax(nn_model.predict(X_new), axis=1)

# Class mapping
class_mapping = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}
```

---

## 📈 Model Performance

### Performance Metrics
The models are evaluated using:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives among predicted positives
- **Recall**: True positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Breakdown of correct/incorrect predictions per class
- **ROC-AUC**: Area under the receiver operating characteristic curve (for binary problems)

### Expected Results
Each model generates:
1. **Classification Report** - Precision, Recall, F1-Score per class
2. **Confusion Matrix Visualization** - Shows misclassification patterns
3. **Feature Importance** (for tree-based models) - Which features matter most
4. **Training History** (for Neural Network) - Accuracy and loss curves

### Model Comparison
| Model | Interpretability | Training Speed | Prediction Speed | Accuracy | Scalability |
|-------|-----------------|-----------------|------------------|----------|-------------|
| Decision Tree | Excellent | Fast | Very Fast | Moderate | Good |
| KNN | Good | None | Slow | Moderate-High | Poor |
| Random Forest | Good | Moderate | Moderate | High | Excellent |
| Neural Network | Poor | Slow | Fast | Very High | Excellent |

---

## 🎨 Key Features

### 1. **Multi-Model Approach**
- Compare 4 different algorithms on the same dataset
- Understand trade-offs between interpretability and accuracy

### 2. **Comprehensive Data Preprocessing**
- Feature scaling and normalization
- One-hot encoding for categorical variables
- Feature selection using RFE
- Handling of imbalanced classes

### 3. **Detailed Model Evaluation**
- Confusion matrices with visualizations
- Classification reports with per-class metrics
- Feature importance analysis
- Training history plots for neural network

### 4. **Master Notebook**
- Central hub to run all models and compare results
- Consistent output format across models
- Easy-to-understand prediction examples

### 5. **Saved Models**
- Pre-trained models included for quick inference
- Ready for deployment without retraining

---

## 🛠️ Technologies Used

### Core Libraries
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow/Keras**: Deep learning framework
- **Matplotlib & Seaborn**: Data visualization

### Development Tools
- **Jupyter Notebook**: Interactive development environment
- **Git**: Version control
- **Python 3.7+**: Programming language

### Computational Aspects
- Handles datasets with thousands of samples
- Efficient feature engineering pipelines
- Optimized model training and inference

---

## 📊 Dataset Information

### Original Data Source
**Sloan Digital Sky Survey (SDSS)**
- URL: https://www.sdss.org/
- Free public astronomical database
- Contains data from millions of celestial objects
- Includes photometric and spectroscopic observations

### Astronomical Context
**Features Explained**:
- **Magnitudes (u, g, r, i, z)**: Brightness measurements in different wavelengths
- **Redshift**: Indicates distance and recession velocity
- **Coordinates (ra, dec)**: Position in the sky
- **Spectroscopic Data**: Detailed light spectrum analysis

### Class Definitions
1. **Galaxy** - Distant collections of billions of stars
2. **Quasar** - Extremely luminous active galactic nuclei
3. **Star** - Individual luminous spheres of plasma in our galaxy

---

## 🔍 Results & Insights

### Analysis Highlights
- **Class Distribution**: Imbalanced with Galaxies being the majority class
- **Feature Importance**: Color indices (u-g, g-r, etc.) are highly discriminative
- **Model Strengths**:
  - Random Forest: Best overall balanced performance
  - Neural Network: Highest accuracy with proper tuning
  - Decision Tree: Most interpretable results
  - KNN: Good accuracy with simpler training

### Visualization Outputs
Each notebook generates:
1. Density plots showing class separation
2. Confusion matrices for prediction analysis
3. Feature importance rankings
4. Training history curves
5. ROC/AUC curves (where applicable)

---

## 🤝 Contributing

Contributions are welcome! Here's how to help:

1. **Fork** the repository
2. **Create** a new branch (`git checkout -b feature/improvement`)
3. **Make** your changes
4. **Commit** with descriptive messages (`git commit -am 'Add feature'`)
5. **Push** to the branch (`git push origin feature/improvement`)
6. **Open** a Pull Request

### Ideas for Improvement
- Hyperparameter optimization using GridSearchCV
- Cross-validation for more robust evaluation
- Additional models (SVM, Gradient Boosting, XGBoost)
- Web deployment with Flask/FastAPI
- Real-time prediction API
- Extended feature engineering
- Handling of edge cases and outliers
- Performance benchmarking

---

## 📝 License

This project is open source and available under the **MIT License**.

---

## 👨‍💻 Author

**Krish Parmar**
- GitHub: [@krishparmar22242](https://github.com/krishparmar22242)
- Project: Stars-Galaxy-Quasars-Classification

---

## 📚 References & Resources

### Astronomical Concepts
- [SDSS Data Release Documentation](https://www.sdss.org/dr16/)
- [Sloan Digital Sky Survey Guide](https://www.sdss.org/guides/)
- [Photometric Classification](https://en.wikipedia.org/wiki/Photometric_redshift)

### Machine Learning
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide)
- [Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [K-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Random Forest](https://en.wikipedia.org/wiki/Random_forest)

### Tutorials
- [Classification Metrics Explained](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Feature Selection Methods](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Neural Network Fundamentals](https://keras.io/api/layers/)

---

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end machine learning pipeline development
- Multiple algorithm implementation and comparison
- Data preprocessing and feature engineering
- Model evaluation and interpretation
- Deep learning with TensorFlow/Keras
- Classification on multi-class problems
- Handling imbalanced datasets
- Feature importance analysis
- Production-ready model saving and loading

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: ModuleNotFoundError for tensorflow
```bash
Solution: pip install tensorflow --upgrade
```

**Issue**: Memory error with large datasets
```bash
Solution: Use data batching or reduce dataset size for initial testing
```

**Issue**: Slow KNN prediction
```bash
Solution: This is normal; use Random Forest or Neural Network for faster predictions
```

**Issue**: Neural Network not converging
```bash
Solution: 
- Normalize features
- Adjust learning rate
- Increase epochs
- Check for data quality issues
```

---

## ⭐ Acknowledgments

- Sloan Digital Sky Survey for providing the dataset
- Scikit-learn community for excellent ML tools
- TensorFlow/Keras teams for deep learning framework
- Open source community for continuous support

---

**Last Updated**: December 2024
**Project Status**: Active & Maintained

For questions or issues, please open a GitHub Issue in the repository.
