# FastAPI backend for dropout prediction system

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io

app = FastAPI(
    title="Dropout Prediction & Counseling System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# loaded at startup
model = None
scaler = None
model_results = None
shap_importance = None
risk_profiles = None
risk_summary = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler, model_results, shap_importance, risk_profiles, risk_summary
    try:
        with open(MODEL_DIR / "best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODEL_DIR / "model_results.json") as f:
            model_results = json.load(f)
        with open(MODEL_DIR / "shap_importance.json") as f:
            shap_importance = json.load(f)
        risk_profiles = pd.read_csv(DATA_DIR / "risk_profiles.csv")
        with open(DATA_DIR / "risk_summary.json") as f:
            risk_summary = json.load(f)
        print("All artifacts loaded.")
    except FileNotFoundError as e:
        print(f"Warning: {e}. Run the training pipeline first.")


class StudentInput(BaseModel):
    age: int = 18
    gender: str = "M"
    branch: str = "Computer Science"
    is_orphan: bool = False
    has_guardian: bool = True
    is_first_gen: bool = False
    income_category: str = "MIG"
    is_hosteler: bool = False
    on_scholarship: bool = False
    distance_from_home_km: float = 20.0
    prev_academic_score: float = 65.0
    avg_attendance: float = 75.0
    min_attendance: float = 60.0
    latest_attendance: float = 70.0
    avg_ia_score: float = 25.0
    min_ia_score_overall: float = 15.0
    total_subjects_failed: int = 0
    max_cumulative_backlog: int = 0
    fee_defaults_count: int = 0
    avg_fee_delay: float = 0.0
    max_fee_delay: float = 0.0
    avg_library_visits: float = 5.0
    extracurricular_rate: float = 0.5
    total_counselor_visits: int = 0
    semesters_completed: int = 1


class PredictionResponse(BaseModel):
    student_id: Optional[str] = None
    dropout_probability: float
    risk_tier: str
    counseling_track: str
    top_risk_factors: list
    recommended_actions: list


class DashboardStats(BaseModel):
    total_students: int
    tier_distribution: dict
    track_distribution: dict
    avg_dropout_prob: float
    critical_count: int
    high_count: int


TIER_THRESHOLDS = {"low": 0.25, "medium": 0.50, "high": 0.75}

def get_risk_tier(prob):
    if prob < TIER_THRESHOLDS["low"]:
        return "Low"
    elif prob < TIER_THRESHOLDS["medium"]:
        return "Medium"
    elif prob < TIER_THRESHOLDS["high"]:
        return "High"
    return "Critical"


def get_counseling_track(student, tier, top_factors):
    if tier == "Low":
        return "Monitoring"
    welfare = (
        student.get("is_orphan", False) or
        not student.get("has_guardian", True) or
        (student.get("income_category") == "BPL" and
         any(f in top_factors for f in ["fee_defaults_count", "max_fee_delay"]))
    )
    career = (
        student.get("age", 18) >= 20 and
        any(f in top_factors for f in ["extracurricular_rate", "avg_library_visits"])
    )
    if welfare and tier in ["High", "Critical"]:
        return "Welfare"
    if career and tier in ["Medium", "High", "Critical"]:
        return "Career Guidance"
    return "Academic"


def get_recommended_actions(track, tier, student):
    actions = []
    
    if track == "Academic":
        actions.append("Assign subject mentor for weak areas")
        if student.get("avg_attendance", 100) < 60:
            actions.append("Attendance improvement counseling - weekly check-ins")
        if student.get("total_subjects_failed", 0) > 2:
            actions.append("Remedial classes for backlog subjects")
        actions.append("Peer study group matching")
    
    elif track == "Welfare":
        if student.get("is_orphan", False):
            actions.append("Assign dedicated welfare officer as point of contact")
            actions.append("Schedule empathetic check-in (non-clinical outreach)")
        if not student.get("has_guardian", True):
            actions.append("Assign peer mentor from senior batch")
        if student.get("income_category") in ["BPL", "LIG"]:
            actions.append("Review financial aid and scholarship eligibility")
            actions.append("Connect with fee waiver or emergency fund")
        if student.get("fee_defaults_count", 0) > 0:
            actions.append("Alert hostel warden for support coordination")
        actions.append("Emotional support referral if distress indicators persist")
    
    elif track == "Career Guidance":
        actions.append("Generate skill-interest profile assessment")
        actions.append("Connect with career advisor")
        actions.append("Curate free learning resources and vocational pathways")
        actions.append("Share relevant government scheme links")
        if student.get("is_first_gen", False):
            actions.append("First-gen learner career mentorship program")
    
    else:  # Monitoring
        actions.append("Continue standard academic monitoring")
        actions.append("Flag for review next semester if indicators change")
    
    if tier == "Critical":
        actions.insert(0, "URGENT: Schedule immediate intervention meeting")
    
    return actions


def encode_student_features(student):
    gender_map = {"F": 0, "M": 1}
    income_map = {"BPL": 0, "HIG": 1, "LIG": 2, "MIG": 3}
    branch_map = {
        "Chemical": 0, "Civil": 1, "Computer Science": 2,
        "Electrical": 3, "Electronics": 4, "Mechanical": 5
    }
    features = [
        student.get("age", 18),
        int(student.get("is_orphan", False)),
        int(student.get("has_guardian", True)),
        int(student.get("is_first_gen", False)),
        int(student.get("is_hosteler", False)),
        int(student.get("on_scholarship", False)),
        student.get("distance_from_home_km", 20),
        student.get("prev_academic_score", 65),
        student.get("avg_attendance", 75),
        student.get("min_attendance", 60),
        student.get("latest_attendance", 70),
        student.get("avg_ia_score", 25),
        student.get("min_ia_score_overall", 15),
        student.get("total_subjects_failed", 0),
        student.get("max_cumulative_backlog", 0),
        student.get("fee_defaults_count", 0),
        student.get("avg_fee_delay", 0),
        student.get("max_fee_delay", 0),
        student.get("avg_library_visits", 5),
        student.get("extracurricular_rate", 0.5),
        student.get("total_counselor_visits", 0),
        gender_map.get(student.get("gender", "M"), 1),
        income_map.get(student.get("income_category", "MIG"), 3),
        branch_map.get(student.get("branch", "Computer Science"), 2),
    ]
    return np.array(features).reshape(1, -1)


@app.get("/")
def root():
    return {"service": "Dropout Prediction System", "status": "running", "model_loaded": model is not None}


@app.get("/api/stats", response_model=DashboardStats)
def get_dashboard_stats():
    if risk_summary is None:
        raise HTTPException(500, "Risk summary not loaded")
    return risk_summary


@app.get("/api/model-performance")
def get_model_performance():
    if model_results is None:
        raise HTTPException(500, "Model results not loaded")
    return model_results


@app.get("/api/shap-importance")
def get_shap_importance():
    if shap_importance is None:
        raise HTTPException(500, "SHAP data not loaded")
    return shap_importance


@app.get("/api/students")
def get_students(
    tier: Optional[str] = Query(None),
    track: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    if risk_profiles is None:
        raise HTTPException(500, "Risk profiles not loaded")
    
    df = risk_profiles.copy()
    if tier:
        df = df[df["risk_tier"] == tier]
    if track:
        df = df[df["counseling_track"] == track]
    if search:
        df = df[df["student_id"].str.contains(search, case=False)]
    
    total = len(df)
    df = df.iloc[offset:offset + limit]
    return {"total": total, "students": df.to_dict(orient="records"), "limit": limit, "offset": offset}


@app.get("/api/student/{student_id}")
def get_student_detail(student_id: str):
    if risk_profiles is None:
        raise HTTPException(500, "Risk profiles not loaded")
    
    student = risk_profiles[risk_profiles["student_id"] == student_id]
    if student.empty:
        raise HTTPException(404, f"Student {student_id} not found")
    
    row = student.iloc[0].to_dict()
    row["recommended_actions"] = get_recommended_actions(row["counseling_track"], row["risk_tier"], row)
    return row


@app.post("/api/predict", response_model=PredictionResponse)
def predict_single(student: StudentInput):
    if model is None or scaler is None:
        raise HTTPException(500, "Model not loaded")
    
    student_dict = student.dict()
    features = encode_student_features(student_dict)
    features_scaled = scaler.transform(features)
    
    prob = float(model.predict_proba(features_scaled)[0][1])
    tier = get_risk_tier(prob)
    top_factors = list(shap_importance.keys())[:5] if shap_importance else []
    track = get_counseling_track(student_dict, tier, top_factors)
    actions = get_recommended_actions(track, tier, student_dict)
    
    return PredictionResponse(
        dropout_probability=round(prob, 4),
        risk_tier=tier,
        counseling_track=track,
        top_risk_factors=[{"factor": f, "importance": shap_importance.get(f, 0)} for f in top_factors[:5]],
        recommended_actions=actions,
    )


@app.post("/api/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    if model is None or scaler is None:
        raise HTTPException(500, "Model not loaded")
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files accepted")
    
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    
    results = []
    for _, row in df.iterrows():
        student_dict = row.to_dict()
        features = encode_student_features(student_dict)
        features_scaled = scaler.transform(features)
        prob = float(model.predict_proba(features_scaled)[0][1])
        tier = get_risk_tier(prob)
        top_factors = list(shap_importance.keys())[:5] if shap_importance else []
        track = get_counseling_track(student_dict, tier, top_factors)
        results.append({
            "student_id": student_dict.get("student_id", ""),
            "dropout_probability": round(prob, 4),
            "risk_tier": tier,
            "counseling_track": track,
        })
    
    return {"total": len(results), "predictions": results}


@app.get("/api/alerts")
def get_alerts(min_tier: str = Query("High")):
    if risk_profiles is None:
        raise HTTPException(500, "Risk profiles not loaded")
    
    tier_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    min_level = tier_order.get(min_tier, 2)
    
    df = risk_profiles.copy()
    df["tier_level"] = df["risk_tier"].map(tier_order)
    alerts = df[df["tier_level"] >= min_level].sort_values("dropout_probability", ascending=False)
    
    alert_list = []
    for _, row in alerts.iterrows():
        row_dict = row.to_dict()
        row_dict["recommended_actions"] = get_recommended_actions(row["counseling_track"], row["risk_tier"], row_dict)
        alert_list.append(row_dict)
    
    return {"total_alerts": len(alert_list), "alerts": alert_list}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
