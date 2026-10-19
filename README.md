# 🌱 AgriSmart: Intelligent Crop Recommendation System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive, end-to-end precision agriculture web application built with Streamlit and scikit-learn. **AgriSmart** helps farmers, agronomists, and agricultural researchers select optimal crops based on soil nutrient profiles and environmental parameters. The system integrates supervised ensemble learning, unsupervised soil-climate clustering, and symbolic concept/rule learning algorithms.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Machine Learning Algorithms & Benchmarks](#-machine-learning-algorithms--benchmarks)
- [Dataset Description](#-dataset-description)
- [Project Directory Structure](#-project-directory-structure)
- [Local Installation & Setup](#-local-installation--setup)
- [Running the Streamlit Application](#-running-the-streamlit-application)
- [Running Unit Tests](#-running-unit-tests)
- [Model Training & Clustering Scripts](#-model-training--clustering-scripts)
- [Deploying to Streamlit Community Cloud](#-deploying-to-streamlit-community-cloud)
- [Contributing & License](#-contributing--license)

---

## 🌾 Overview

Precision agriculture uses field data to optimize crop yield, reduce fertilizer overuse, and conserve water resources. **AgriSmart** evaluates seven primary soil and meteorological factors:
- **Soil Nutrients:** Nitrogen ($N$), Phosphorus ($P$), and Potassium ($K$)
- **Meteorological Factors:** Temperature (°C), Relative Humidity (%), Soil pH, and Rainfall (mm)

The system predicts the most suitable crop out of 22 different crop categories with over **99.5% accuracy** using a trained Random Forest ensemble classifier.

---

## ✨ Key Features

The Streamlit web application includes six dedicated modules accessible from the sidebar navigation:

1. **🚜 Crop Recommendation**:
   - Interactive sliders to input Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.
   - Real-time prediction with prediction confidence badge and visual crop iconography.
   - Interactive probability breakdown of the top recommended candidate crops.

2. **📊 Model Comparison & Benchmarks**:
   - Empirical comparison of 4 Machine Learning classifiers across Accuracy, Weighted Precision, Recall, and F1-score.
   - Interactive Confusion Matrix heatmaps for all 22 crop classes.
   - Model parameters, ensemble characteristics, and performance analysis.

3. **🗺️ Soil & Climate Clustering**:
   - Unsupervised K-Means clustering ($k=4$) on standardized soil and environmental conditions.
   - 2D Principal Component Analysis (PCA) projection scatter plot.
   - Cluster profile breakdown (mean nutrient levels in original units and dominant crops per cluster).

4. **🧪 Concept Learning Lab**:
   - Interactive symbolic concept learning using the **Find-S** algorithm (maximally specific hypothesis).
   - **Candidate Elimination** algorithm tracking the version space boundary ($S$ and $G$ sets).
   - Step-by-step hypothesis specialization/generalization and version space collapse detection on conflicting data.

5. **📜 Sequential Covering (Rule Induction)**:
   - Learn human-interpretable `IF-THEN` decision rules directly from discretized agricultural data.
   - Multi-class rule induction covering positive examples while penalizing false positives.
   - Rule performance metrics: positive/negative coverage, precision percentage, and coverage fraction.

6. **🧠 Algorithm Explanations**:
   - Detailed educational guide explaining the mechanics of Bagging, Random Forests, AdaBoost, K-Means clustering, and Inductive Logic / Version Spaces.

---

## 🤖 Machine Learning Algorithms & Benchmarks

The models were trained on 80% of the dataset (1,760 samples) with stratified sampling and evaluated on a holdout test set of 20% (440 samples):

| Algorithm | Base Estimator | Estimators / Settings | Test Accuracy | Macro / Weighted F1 | Purpose |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | Decision Tree | 100 Trees (Subsampling) | **99.55%** | **0.995** | Primary production predictor |
| **Bagging Classifier** | Decision Tree | 100 Bootstrap Trees | **99.09%** | **0.991** | Ensemble variance reduction |
| **AdaBoost Classifier** | Decision Stump | 100 Stumps ($d=1$) | **31.36%** | **0.252** | Sequential boosting benchmark |
| **Decision Stump** | Single Split ($d=1$) | 1 Stump | **9.09%** | **0.050** | Baseline weak learner |

### Unsupervised Clustering & PCA
- **StandardScaler**: Normalizes all 7 features to zero mean and unit variance.
- **K-Means ($k=4$)**: Identifies 4 agro-ecological soil/climate clusters with a **Silhouette Score of 0.2553**.
- **PCA (2D)**: Projects the 7-dimensional space into 2 principal components capturing **46.07%** total variance (PC1: 27.59%, PC2: 18.48%).

---

## 📂 Dataset Description

The primary dataset is located at [`data/Crop_recommendation.csv`](data/Crop_recommendation.csv):
- **Instances**: 2,200 records (100 samples per crop class)
- **Target Variable**: `label` (22 distinct crops):
  *apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon*
- **Feature Attributes**:
  1. `N` (Nitrogen ratio in soil, 0 – 140)
  2. `P` (Phosphorus ratio in soil, 5 – 145)
  3. `K` (Potassium ratio in soil, 5 – 205)
  4. `temperature` (Temperature in °C, 8.8 – 43.7)
  5. `humidity` (Relative humidity in %, 14.3 – 99.9)
  6. `ph` (Soil pH value, 3.5 – 9.9)
  7. `rainfall` (Precipitation in mm, 20.2 – 298.6)

An additional derived dataset [`data/clustered_crops.csv`](data/clustered_crops.csv) contains precomputed K-Means cluster IDs and 2D PCA projection coordinates.

---

## 📁 Project Directory Structure

```text
SmartCropRecommendation/
│
├── app.py                      # Main Streamlit web application entry point
├── requirements.txt            # Python package dependencies
├── .gitignore                  # Git ignore rules for cache, venv, and local configs
├── README.md                   # Project documentation and deployment guide
│
├── data/                       # Datasets
│   ├── Crop_recommendation.csv # Raw 2,200-sample agricultural dataset
│   └── clustered_crops.csv     # Clustered dataset with PCA coordinates
│
├── models/                     # Trained models, scalers, and metric summaries
│   ├── random_forest.joblib    # Serialized Random Forest Classifier
│   ├── bagging_classifier.joblib
│   ├── adaboost_classifier.joblib
│   ├── decision_stump.joblib
│   ├── kmeans.joblib           # Serialized K-Means clustering model (k=4)
│   ├── pca.joblib              # 2D PCA projection transformer
│   ├── scaler.joblib           # StandardScaler model
│   ├── classes.json            # Target crop labels
│   ├── features.json           # Ordered list of feature names
│   ├── model_metrics.json      # Comprehensive evaluation metrics & confusion matrices
│   └── clustering_metrics.json # Silhouette score, cluster profiles, and PCA variance
│
├── src/                        # Core algorithms and training scripts
│   ├── __init__.py             # Package marker
│   ├── concept_learning.py     # Find-S and Candidate Elimination implementations
│   ├── sequential_covering.py  # Sequential Covering rule induction algorithm
│   ├── train.py                # Model training, validation, and artifact export script
│   └── clustering.py           # K-Means clustering and PCA export script
│
└── tests/                      # Unit test suite
    ├── __init__.py             # Package marker
    ├── test_concept_learning.py    # Unit tests for Find-S and Candidate Elimination
    └── test_sequential_covering.py # Unit tests for Rule Induction and matching
```

---

## ⚙️ Local Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Git installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/SmartCropRecommendation.git
cd SmartCropRecommendation
```

### 2. Create and Activate a Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **On Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Streamlit Application

Launch the local Streamlit development server:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at:
`http://localhost:8501`

---

## 🧪 Running Unit Tests

Execute the comprehensive test suite to verify algorithm implementations:

```bash
python -m unittest discover -s tests -v
```

All 13 test cases validate:
- Find-S hypothesis generalization and wildcard propagation.
- Candidate Elimination version space convergence and conflict detection.
- Sequential Covering rule learning, coverage metrics, and precision calculations.

---

## 🔄 Model Training & Clustering Scripts (Optional)

All trained model weights and evaluation metrics are pre-saved in `models/`. If you wish to retrain or update the models:

1. **Retrain Supervised Classifiers**:
   ```bash
   python src/train.py
   ```
   *Generates `random_forest.joblib`, `bagging_classifier.joblib`, `adaboost_classifier.joblib`, `decision_stump.joblib`, and `model_metrics.json`.*

2. **Recompute K-Means Clustering & PCA**:
   ```bash
   python src/clustering.py
   ```
   *Generates `scaler.joblib`, `kmeans.joblib`, `pca.joblib`, `clustering_metrics.json`, and `data/clustered_crops.csv`.*

---

## ☁️ Deploying to Streamlit Community Cloud

Follow these steps to deploy **AgriSmart** on Streamlit Community Cloud for free:

### Step 1: Push Repository to GitHub
1. Initialize git in your local directory (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit: AgriSmart Crop Recommendation System"
   ```
2. Create a new repository on [GitHub](https://github.com/new) named `SmartCropRecommendation`.
3. Link your local repo and push:
   ```bash
   git branch -M main
   git remote add origin https://github.com/<your-username>/SmartCropRecommendation.git
   git push -u origin main
   ```

### Step 2: Connect to Streamlit Community Cloud
1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using your GitHub account.
2. Click **"New app"** or **"Create app"**.
3. Select your repository: `<your-username>/SmartCropRecommendation`.
4. Set the **Branch** to `main`.
5. Set the **Main file path** to `app.py`.
6. Click **"Deploy!"**.

Streamlit Cloud will read `requirements.txt`, install dependencies, and launch your live application with a public URL.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
