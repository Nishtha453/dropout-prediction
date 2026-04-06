"""
Predictor
=========
Loads trained model and generates predictions + per-student SHAP explanations.
"""

import pickle
import json
import numpy as np
import pandas as pd
import shap
import warnings
warnings.filterwarnings("ignore")


class DropoutPredictor:
    """Wraps the trained model for inference and explanation."""

    def __init__(self, model_dir: str = "models"):
        with open(f"{model_dir}/best_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open(f"{model_dir}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open(f"{model_dir}/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        with open(f"{model_dir}/metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.feature_cols = self.metadata["feature_columns"]
        self.model_name = self.metadata["best_model_name"]

    def predict_single(self, student_dict: dict) -> dict:
        """
        Predict dropout probability for a single student.
        
        Args:
            student_dict: dict with student features
            
        Returns:
            dict with dropout_probability, risk_tier, prediction
        """
        features = self._extract_features(student_dict)
        features_scaled = self.scaler.transform([features])

        prob = self.model.predict_proba(features_scaled)[0][1]
        prediction = int(prob > 0.5)

        return {
            "dropout_probability": round(float(prob), 4),
            "prediction": prediction,
            "prediction_label": "At Risk" if prediction else "Safe",
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for an entire DataFrame of students."""
        # Encode gender if present
        if "gender" in df.columns:
            df = df.copy()
            df["gender_encoded"] = self.label_encoder.transform(df["gender"])

        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)

        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs > 0.5).astype(int)

        df["dropout_probability"] = np.round(probs, 4)
        df["prediction"] = preds
        df["prediction_label"] = np.where(preds == 1, "At Risk", "Safe")

        return df

    def explain_student(self, student_dict: dict, top_n: int = 5) -> list:
        """
        Generate SHAP-based explanation for a single student's prediction.
        
        Returns:
            List of (feature_name, shap_value, feature_value, direction) tuples
        """
        features = self._extract_features(student_dict)
        features_scaled = self.scaler.transform([features])

        # Use KernelExplainer for model-agnostic SHAP
        # Use a small background dataset
        background = np.zeros((1, len(self.feature_cols)))  # baseline
        try:
            if self.model_name == "Decision Tree":
                explainer = shap.TreeExplainer(self.model)
            else:
                explainer = shap.KernelExplainer(
                    self.model.predict_proba, background
                )
            
            sv = explainer.shap_values(features_scaled)
            
            # Handle different SHAP output formats
            if isinstance(sv, list):
                sv = sv[1]  # class 1 (dropout)
            
            shap_vals = sv[0]  # first (only) sample
        except Exception:
            # Fallback: use feature importance from metadata
            importance = {
                item["feature"]: item["importance"]
                for item in self.metadata["feature_importance"]
            }
            explanations = []
            for i, feat in enumerate(self.feature_cols):
                val = features[i]
                imp = importance.get(feat, 0)
                direction = "increases" if imp > 0 else "decreases"
                explanations.append((feat, imp, val, direction))
            explanations.sort(key=lambda x: abs(x[1]), reverse=True)
            return explanations[:top_n]

        explanations = []
        for i, feat in enumerate(self.feature_cols):
            val = features[i]
            sv_val = float(shap_vals[i])
            direction = "increases" if sv_val > 0 else "decreases"
            explanations.append((feat, sv_val, val, direction))

        explanations.sort(key=lambda x: abs(x[1]), reverse=True)
        return explanations[:top_n]

    def _extract_features(self, student_dict: dict) -> list:
        """Extract feature vector from student dict, handling gender encoding."""
        features = []
        for col in self.feature_cols:
            if col == "gender_encoded":
                gender = student_dict.get("gender", "Male")
                val = self.label_encoder.transform([gender])[0]
            else:
                val = student_dict.get(col, 0)
            features.append(float(val))
        return features
