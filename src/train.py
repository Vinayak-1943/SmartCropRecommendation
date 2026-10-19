"""
Smart Crop Recommendation System - Model Training & Evaluation
===============================================================
This script trains and evaluates four Machine Learning (ML-II) algorithms:
1. Decision Stump (DecisionTreeClassifier with max_depth=1)
2. Bagging Classifier (Bagging with Decision Trees as base estimators)
3. Random Forest Classifier
4. AdaBoost Classifier (AdaBoost with Decision Stumps as base estimators)

It evaluates test accuracy, precision, recall, F1-score, and confusion matrices,
saves the trained models as joblib files, and exports metrics and features for the Streamlit app.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_dataset(data_path="data/Crop_recommendation.csv"):
    """
    Load dataset from CSV, validate expected columns, and extract features and target.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    print(f"[1/5] Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)

    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target_col = 'label'

    # Validate columns
    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset.")

    X = df[feature_cols]
    y = df[target_col]

    print(f"      Loaded {len(df)} samples across {len(feature_cols)} features.")
    print(f"      Target classes ({y.nunique()}): {sorted(y.unique())}\n")

    return X, y, feature_cols


def prepare_data_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training (80%) and testing (20%) sets with stratification.
    """
    print(f"[2/5] Splitting data ({int((1-test_size)*100)}% train, {int(test_size*100)}% test, stratify=y)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"      Training set size : {X_train.shape[0]} samples")
    print(f"      Testing set size  : {X_test.shape[0]} samples\n")
    return X_train, X_test, y_train, y_test


def get_model_definitions():
    """
    Define and return the four ML-II models along with their save filenames.
    """
    models = {
        "Decision Stump": {
            "model": DecisionTreeClassifier(max_depth=1, random_state=42),
            "filename": "decision_stump.joblib",
            "description": "Decision Tree with max_depth=1 (Weak Learner)"
        },
        "Bagging Classifier": {
            "model": BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42),
                n_estimators=100,
                random_state=42
            ),
            "filename": "bagging_classifier.joblib",
            "description": "Bagging ensemble of 100 Decision Trees"
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            "filename": "random_forest.joblib",
            "description": "Random Forest ensemble of 100 Trees with feature subsampling"
        },
        "AdaBoost Classifier": {
            "model": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                n_estimators=100,
                random_state=42
            ),
            "filename": "adaboost_classifier.joblib",
            "description": "AdaBoost ensemble using 100 Decision Stumps"
        }
    }
    return models


def train_and_evaluate_models(models, X_train, X_test, y_train, y_test, class_labels):
    """
    Train each model, compute evaluation metrics, and store results.
    """
    print("[3/5] Training and evaluating models...")
    metrics_report = {}

    for name, config in models.items():
        print(f"      -> Training {name} ({config['description']})...")
        model = config["model"]
        
        # Train model
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=class_labels)

        metrics_report[name] = {
            "accuracy": float(acc),
            "precision_weighted": float(prec),
            "recall_weighted": float(rec),
            "f1_weighted": float(f1),
            "confusion_matrix": cm.tolist(),
            "filename": config["filename"],
            "description": config["description"]
        }
        print(f"         Done! Accuracy: {acc*100:.2f}% | F1-Score: {f1:.4f}")

    print()
    return metrics_report


def display_comparison_table(metrics_report):
    """
    Print an ASCII table comparing all four models.
    """
    print("[4/5] Model Performance Comparison Table:")
    print("=" * 78)
    print(f"{'Model Name':<24} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 78)

    for name, m in metrics_report.items():
        print(
            f"{name:<24} | "
            f"{m['accuracy']*100:>8.2f}% | "
            f"{m['precision_weighted']:>10.4f} | "
            f"{m['recall_weighted']:>10.4f} | "
            f"{m['f1_weighted']:>10.4f}"
        )
    print("=" * 78 + "\n")


def save_artifacts(models, metrics_report, feature_cols, class_labels, output_dir="models"):
    """
    Save trained model joblib files, feature list, class labels, and evaluation metrics JSON.
    """
    print(f"[5/5] Saving model artifacts to '{output_dir}/'...")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save trained models
    for name, config in models.items():
        save_path = os.path.join(output_dir, config["filename"])
        joblib.dump(config["model"], save_path)
        print(f"      Saved model  : {save_path}")

    # 2. Save feature names for inference
    features_path = os.path.join(output_dir, "features.json")
    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"      Saved features: {features_path}")

    # 3. Save class labels
    classes_path = os.path.join(output_dir, "classes.json")
    with open(classes_path, "w") as f:
        json.dump(class_labels, f, indent=2)
    print(f"      Saved classes : {classes_path}")

    # 4. Save evaluation metrics JSON
    metrics_path = os.path.join(output_dir, "model_metrics.json")
    full_output = {
        "metadata": {
            "dataset": "Crop_recommendation.csv",
            "feature_order": feature_cols,
            "classes": class_labels,
            "train_test_split": {"train_ratio": 0.8, "test_ratio": 0.2, "random_state": 42, "stratified": True}
        },
        "models": metrics_report
    }
    with open(metrics_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"      Saved metrics : {metrics_path}\n")


def main():
    # 1. Load dataset
    X, y, feature_cols = load_dataset("data/Crop_recommendation.csv")
    class_labels = sorted(y.unique().tolist())

    # 2. Split dataset
    X_train, X_test, y_train, y_test = prepare_data_split(X, y, test_size=0.2, random_state=42)

    # 3. Define models
    models = get_model_definitions()

    # 4. Train and evaluate
    metrics_report = train_and_evaluate_models(models, X_train, X_test, y_train, y_test, class_labels)

    # 5. Display comparison
    display_comparison_table(metrics_report)

    # 6. Save artifacts
    save_artifacts(models, metrics_report, feature_cols, class_labels, output_dir="models")

    print("All models successfully trained, evaluated, and saved!")


if __name__ == "__main__":
    main()
