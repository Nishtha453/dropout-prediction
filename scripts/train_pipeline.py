# ML training pipeline for dropout prediction
# trains 5 models, picks best one, generates SHAP explanations

import numpy as np
import pandas as pd
import pickle
import json
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import shap

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

# risk tier thresholds - configurable
TIER_THRESHOLDS = {"low": 0.25, "medium": 0.50, "high": 0.75}


def load_and_prepare_data():
    df = pd.read_csv(DATA_DIR / "ml_ready_dataset.csv")
    profiles = pd.read_csv(DATA_DIR / "student_profiles.csv")
    
    # features (excluding semesters_completed - it leaks the target)
    feature_cols = [
        "age", "is_orphan", "has_guardian", "is_first_gen",
        "is_hosteler", "on_scholarship", "distance_from_home_km",
        "prev_academic_score", "avg_attendance", "min_attendance",
        "latest_attendance", "avg_ia_score", "min_ia_score_overall",
        "total_subjects_failed", "max_cumulative_backlog",
        "fee_defaults_count", "avg_fee_delay", "max_fee_delay",
        "avg_library_visits", "extracurricular_rate",
        "total_counselor_visits",
    ]
    
    # encode categoricals
    le_gender = LabelEncoder()
    df["gender_enc"] = le_gender.fit_transform(df["gender"])
    feature_cols.append("gender_enc")
    
    le_income = LabelEncoder()
    df["income_enc"] = le_income.fit_transform(df["income_category"])
    feature_cols.append("income_enc")
    
    le_branch = LabelEncoder()
    df["branch_enc"] = le_branch.fit_transform(df["branch"])
    feature_cols.append("branch_enc")
    
    bool_cols = ["is_orphan", "has_guardian", "is_first_gen", "is_hosteler", "on_scholarship"]
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    X = df[feature_cols].values
    y = df["dropped_out"].astype(int).values
    
    metadata = {
        "feature_cols": feature_cols,
        "encoders": {"gender": le_gender, "income": le_income, "branch": le_branch},
        "df": df,
    }
    return X, y, metadata


def get_models():
    return {
        "KNN": KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=20, min_samples_leaf=10,
            class_weight="balanced", random_state=42
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, class_weight="balanced", random_state=42
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), activation="relu",
            solver="adam", max_iter=500, early_stopping=True,
            validation_fraction=0.15, random_state=42
        ),
    }


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = get_models()
    results = {}
    trained_models = {}
    
    print("=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nTraining: {name}")
        print("-" * 40)
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1")
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
        }
        results[name] = metrics
        
        print(f"  Accuracy:  {metrics['accuracy']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")
        print(f"  F1 Score:  {metrics['f1_score']}")
        print(f"  ROC-AUC:   {metrics['roc_auc']}")
        print(f"  CV F1:     {metrics['cv_f1_mean']} +/- {metrics['cv_f1_std']}")
    
    # pick best by f1 (we care about catching dropouts)
    best_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = trained_models[best_name]
    
    print(f"\n{'=' * 60}")
    print(f"BEST MODEL: {best_name} (F1: {results[best_name]['f1_score']})")
    print(f"{'=' * 60}")
    
    # save models
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_DIR / "all_models.pkl", "wb") as f:
        pickle.dump(trained_models, f)
    with open(MODEL_DIR / "model_results.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)
    
    return best_model, scaler, trained_models, results, best_name, X_train_scaled, X_test_scaled, y_test


def generate_shap_explanations(model, X_scaled, feature_names, model_name):
    print("\nGenerating SHAP explanations...")
    
    if model_name == "Decision Tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    else:
        # kernel explainer for non-tree models
        background = shap.kmeans(X_scaled, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_scaled[:200])  # limit for speed
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, [round(float(np.mean(v)), 4) if np.ndim(v) > 0 else round(float(v), 4) for v in mean_abs_shap]))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    print("Top risk factors (SHAP):")
    for feat, val in list(importance.items())[:10]:
        print(f"  {feat}: {val}")
    
    with open(MODEL_DIR / "shap_importance.json", "w") as f:
        json.dump(importance, f, indent=2)
    
    return shap_values, importance


def assign_risk_tier(prob):
    if prob < TIER_THRESHOLDS["low"]:
        return "Low"
    elif prob < TIER_THRESHOLDS["medium"]:
        return "Medium"
    elif prob < TIER_THRESHOLDS["high"]:
        return "High"
    return "Critical"


def assign_counseling_track(row, risk_tier, top_factors):
    # route to the right counseling track based on vulnerability profile
    is_orphan = row.get("is_orphan", 0)
    has_guardian = row.get("has_guardian", 1)
    income_cat = row.get("income_category", "MIG")
    age = row.get("age", 18)
    
    # welfare track: orphans, no guardian, financially distressed
    welfare_signals = (
        bool(is_orphan) or
        not bool(has_guardian) or
        (income_cat == "BPL" and "fee_defaults_count" in top_factors) or
        (income_cat == "BPL" and "max_fee_delay" in top_factors)
    )
    
    # career track: older students disengaging
    career_signals = (
        age >= 20 and
        ("extracurricular_rate" in top_factors or
         "avg_library_visits" in top_factors or
         "semesters_completed" in top_factors)
    )
    
    if risk_tier == "Low":
        return "Monitoring"
    elif welfare_signals and risk_tier in ["High", "Critical"]:
        return "Welfare"
    elif career_signals and risk_tier in ["Medium", "High", "Critical"]:
        return "Career Guidance"
    else:
        return "Academic"


def generate_risk_profiles(model, scaler, metadata):
    df = metadata["df"]
    feature_cols = metadata["feature_cols"]
    
    X_all = df[feature_cols].values
    X_scaled = scaler.transform(X_all)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    with open(MODEL_DIR / "shap_importance.json") as f:
        global_importance = json.load(f)
    top_factors = list(global_importance.keys())[:5]
    
    profiles = []
    for idx, row in df.iterrows():
        prob = float(probs[idx])
        tier = assign_risk_tier(prob)
        track = assign_counseling_track(row, tier, top_factors)
        
        # per-student top factors (using global importance for now)
        # TODO: compute per-student shap values for more detail
        student_factors = []
        for feat in top_factors[:3]:
            val = row.get(feat, "N/A")
            student_factors.append({"factor": feat, "value": str(val)})
        
        profiles.append({
            "student_id": row["student_id"],
            "dropout_probability": round(prob, 4),
            "risk_tier": tier,
            "counseling_track": track,
            "top_risk_factors": student_factors,
            "age": int(row["age"]),
            "branch": row["branch"],
            "is_orphan": bool(row["is_orphan"]),
            "has_guardian": bool(row["has_guardian"]),
            "is_first_gen": bool(row["is_first_gen"]),
            "income_category": row["income_category"],
            "avg_attendance": round(float(row["avg_attendance"]), 1),
            "avg_ia_score": round(float(row["avg_ia_score"]), 1),
            "fee_defaults_count": int(row["fee_defaults_count"]),
            "semesters_completed": int(row["semesters_completed"]),
        })
    
    profiles_df = pd.DataFrame(profiles)
    profiles_df.to_csv(DATA_DIR / "risk_profiles.csv", index=False)
    
    tier_dist = profiles_df["risk_tier"].value_counts().to_dict()
    track_dist = profiles_df["counseling_track"].value_counts().to_dict()
    
    summary = {
        "total_students": len(profiles_df),
        "tier_distribution": tier_dist,
        "track_distribution": track_dist,
        "avg_dropout_prob": round(profiles_df["dropout_probability"].mean(), 4),
        "critical_count": int(tier_dist.get("Critical", 0)),
        "high_count": int(tier_dist.get("High", 0)),
    }
    with open(DATA_DIR / "risk_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nRisk Profile Summary:")
    print(f"  Tier Distribution: {tier_dist}")
    print(f"  Track Distribution: {track_dist}")
    
    return profiles_df


def main():
    X, y, metadata = load_and_prepare_data()
    print(f"Dataset: {X.shape[0]} students, {X.shape[1]} features")
    print(f"Dropout rate: {y.mean():.1%}")
    
    best_model, scaler, all_models, results, best_name, X_train, X_test, y_test = \
        train_and_evaluate(X, y)
    
    shap_values, importance = generate_shap_explanations(
        best_model, X_test, metadata["feature_cols"], best_name
    )
    
    profiles = generate_risk_profiles(best_model, scaler, metadata)
    print("\nPipeline complete. Artifacts saved to /models and /data")


if __name__ == "__main__":
    main()
