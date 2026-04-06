"""
Model Training Pipeline
========================
Trains 5 ML models (KNN, Naive Bayes, Decision Tree, SVM, ANN),
compares performance, saves best model, and generates SHAP explanations.
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

# Five models as specified in problem statement
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import shap

# ---- Features used for training ----
FEATURE_COLS = [
    "age", "semester", "is_orphan", "has_guardian",
    "is_first_gen_learner", "is_hostel_resident",
    "income_category", "has_scholarship", "scholarship_dependent",
    "attendance_pct", "internal_score", "backlogs",
    "fee_paid_ratio", "missed_fee_payments",
    "library_visits_per_month", "counselor_visits",
]

TARGET_COL = "dropped_out"


def load_and_preprocess(csv_path: str):
    """Load CSV data, encode categoricals, scale features."""
    df = pd.read_csv(csv_path)

    # Encode gender
    le = LabelEncoder()
    df["gender_encoded"] = le.fit_transform(df["gender"])

    feature_cols = FEATURE_COLS + ["gender_encoded"]

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols, le


def get_models():
    """Return dict of all 5 models to train."""
    return {
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=10, random_state=42
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        ),
    }


def evaluate_model(model, X_test, y_test):
    """Compute all evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_prob), 4)

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    return metrics


def train_all_models(X_train, X_test, y_train, y_test):
    """Train all 5 models and return results."""
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")

        metrics = evaluate_model(model, X_test, y_test)
        metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_f1_std"] = round(cv_scores.std(), 4)

        results[name] = {
            "model": model,
            "metrics": metrics,
        }
        print(f"    → Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | AUC: {metrics.get('roc_auc', 'N/A')}")

    return results


def select_best_model(results: dict) -> tuple:
    """Select best model based on F1 score (primary) and AUC (secondary)."""
    best_name = None
    best_f1 = -1

    for name, res in results.items():
        f1 = res["metrics"]["f1_score"]
        if f1 > best_f1:
            best_f1 = f1
            best_name = name

    return best_name, results[best_name]["model"]


def generate_shap_values(model, X_test, feature_names, model_name):
    """Generate SHAP explanations for the best model."""
    print(f"\n  Generating SHAP explanations for {model_name}...")

    try:
        if model_name == "Decision Tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Use small background + small test subset for speed
            bg = X_test[:50]
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            shap_values = explainer.shap_values(X_test[:30], nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
    except Exception as e:
        print(f"  SHAP failed ({e}), using permutation importance fallback...")
        from sklearn.inspection import permutation_importance
        result = permutation_importance(model, X_test, np.zeros(len(X_test)), n_repeats=10, random_state=42)
        importances = result.importances_mean
        feature_importance = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1], reverse=True,
        )
        return None, feature_importance

    # Ensure 2D: (n_samples, n_features)
    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 1]  # take class 1
    
    mean_shap = np.abs(shap_arr).mean(axis=0).flatten()
    feature_importance = sorted(
        zip(feature_names, [float(v) for v in mean_shap]),
        key=lambda x: x[1],
        reverse=True,
    )

    return shap_arr, feature_importance


def save_artifacts(best_model, best_name, scaler, label_encoder,
                   feature_cols, results, feature_importance):
    """Save trained model, scaler, and metadata."""
    os.makedirs("models", exist_ok=True)

    # Save model
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Save scaler
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save label encoder
    with open("models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Save metadata
    all_metrics = {}
    for name, res in results.items():
        m = res["metrics"].copy()
        m.pop("confusion_matrix", None)  # Remove non-serializable
        all_metrics[name] = m

    metadata = {
        "best_model_name": best_name,
        "feature_columns": feature_cols,
        "all_model_metrics": all_metrics,
        "feature_importance": [
            {"feature": feat, "importance": round(float(imp), 4)}
            for feat, imp in feature_importance
        ],
    }

    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Artifacts saved to models/")


def main():
    print("=" * 60)
    print("DROPOUT PREDICTION — MODEL TRAINING PIPELINE")
    print("=" * 60)

    csv_path = "data/student_data.csv"
    if not os.path.exists(csv_path):
        print("  Data not found. Run data_generator.py first.")
        return

    print("\n[1/4] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, feature_cols, le = load_and_preprocess(csv_path)
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

    print("\n[2/4] Training 5 models...")
    results = train_all_models(X_train, X_test, y_train, y_test)

    print("\n[3/4] Selecting best model...")
    best_name, best_model = select_best_model(results)
    best_metrics = results[best_name]["metrics"]
    print(f"  Best model: {best_name}")
    print(f"  F1: {best_metrics['f1_score']} | AUC: {best_metrics.get('roc_auc', 'N/A')} | Acc: {best_metrics['accuracy']}")

    print("\n[4/4] Generating SHAP explanations...")
    shap_values, feature_importance = generate_shap_values(
        best_model, X_test, feature_cols, best_name
    )

    print("\n  Top 5 features driving dropout prediction:")
    for feat, imp in feature_importance[:5]:
        print(f"    {feat}: {imp:.4f}")

    save_artifacts(best_model, best_name, scaler, le, feature_cols, results, feature_importance)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
