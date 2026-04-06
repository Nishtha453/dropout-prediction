"""
AI-Based Dropout Prediction & Counseling System
=================================================
Streamlit Dashboard for administrators, counselors, and mentors.

Features:
  - Bulk CSV upload for batch predictions
  - Risk tier distribution (color-coded)
  - Individual student drilldown with SHAP explanations
  - Three-track counseling routing
  - Alert generation for assigned staff
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# Local imports
from utils.predictor import DropoutPredictor
from utils.counseling_router import (
    classify_risk_tier, get_tier_color, route_counseling_track
)

# ---- Page Config ----
st.set_page_config(
    page_title="Dropout Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .tier-low { color: #2ecc71; font-weight: bold; }
    .tier-medium { color: #f39c12; font-weight: bold; }
    .tier-high { color: #e67e22; font-weight: bold; }
    .tier-critical { color: #e74c3c; font-weight: bold; }
    .track-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---- Initialize ----
@st.cache_resource
def load_predictor():
    """Load trained model (cached across reruns)."""
    return DropoutPredictor(model_dir="models")


def load_default_data():
    """Load the generated dataset as default."""
    return pd.read_csv("data/student_data.csv")


# ---- Sidebar ----
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=60)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Dashboard", "🔍 Student Lookup", "📁 Batch Upload", "📈 Model Performance", "⚙️ Settings"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Info**")

# Load model metadata
if os.path.exists("models/metadata.json"):
    with open("models/metadata.json") as f:
        meta = json.load(f)
    st.sidebar.success(f"Model: {meta['best_model_name']}")
    best_metrics = meta["all_model_metrics"][meta["best_model_name"]]
    st.sidebar.metric("F1 Score", f"{best_metrics['f1_score']:.2%}")
else:
    st.sidebar.warning("No trained model found. Run model_training.py first.")
    meta = None


# ---- Helper Functions ----
def add_risk_columns(df: pd.DataFrame, predictor) -> pd.DataFrame:
    """Add prediction, risk tier, and counseling track to dataframe."""
    df = predictor.predict_batch(df)
    df["risk_tier"] = df["dropout_probability"].apply(classify_risk_tier)
    
    # Add counseling routing
    tracks, assignees, priorities = [], [], []
    for _, row in df.iterrows():
        student = row.to_dict()
        student["risk_tier"] = classify_risk_tier(student["dropout_probability"])
        routing = route_counseling_track(student)
        tracks.append(routing["track"])
        assignees.append(routing["assigned_to"])
        priorities.append(routing["priority"])
    
    df["counseling_track"] = tracks
    df["assigned_to"] = assignees
    df["priority"] = priorities
    
    return df


def render_tier_badge(tier):
    """Render colored tier badge."""
    color = get_tier_color(tier)
    return f'<span style="color:{color};font-weight:bold;">● {tier}</span>'


# ===========================================================
# PAGE: DASHBOARD
# ===========================================================
if page == "📊 Dashboard":
    st.markdown('<p class="main-header">🎓 AI Dropout Prediction & Counseling System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Government of Rajasthan — Directorate of Technical Education</p>', unsafe_allow_html=True)

    if meta is None:
        st.error("⚠️ Model not trained yet. Please run `python model_training.py` first.")
        st.stop()

    predictor = load_predictor()
    df = load_default_data()
    df = add_risk_columns(df, predictor)

    # ---- KPI Row ----
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total = len(df)
    at_risk = (df["prediction"] == 1).sum()
    critical = (df["risk_tier"] == "Critical").sum()
    orphan_risk = df[(df["is_orphan"] == 1) & (df["prediction"] == 1)].shape[0]
    welfare_routed = (df["counseling_track"] == "Welfare").sum()

    col1.metric("Total Students", f"{total:,}")
    col2.metric("At Risk", f"{at_risk}", delta=f"{at_risk/total*100:.1f}%", delta_color="inverse")
    col3.metric("Critical Tier", f"{critical}", delta_color="inverse")
    col4.metric("Orphans At Risk", f"{orphan_risk}", delta_color="inverse")
    col5.metric("Welfare Track", f"{welfare_routed}")

    st.markdown("---")

    # ---- Charts Row ----
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Risk Tier Distribution")
        tier_counts = df["risk_tier"].value_counts().reindex(["Low", "Medium", "High", "Critical"], fill_value=0)
        fig_tier = go.Figure(data=[
            go.Bar(
                x=tier_counts.index,
                y=tier_counts.values,
                marker_color=["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"],
                text=tier_counts.values,
                textposition="auto",
            )
        ])
        fig_tier.update_layout(
            height=350,
            margin=dict(t=20, b=20),
            xaxis_title="Risk Tier",
            yaxis_title="Student Count",
        )
        st.plotly_chart(fig_tier, use_container_width=True)

    with chart_col2:
        st.subheader("Counseling Track Assignment")
        track_counts = df["counseling_track"].value_counts()
        fig_track = px.pie(
            values=track_counts.values,
            names=track_counts.index,
            color_discrete_sequence=["#3498db", "#e74c3c", "#2ecc71", "#95a5a6"],
            hole=0.4,
        )
        fig_track.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_track, use_container_width=True)

    # ---- Scatter: Attendance vs Score ----
    st.subheader("Attendance vs Internal Score (colored by Risk Tier)")
    fig_scatter = px.scatter(
        df, x="attendance_pct", y="internal_score",
        color="risk_tier",
        color_discrete_map={
            "Low": "#2ecc71", "Medium": "#f39c12",
            "High": "#e67e22", "Critical": "#e74c3c"
        },
        hover_data=["student_id", "backlogs", "counseling_track"],
        opacity=0.7,
        height=400,
    )
    fig_scatter.update_layout(
        xaxis_title="Attendance %",
        yaxis_title="Internal Assessment Score",
        margin=dict(t=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ---- Critical Students Table ----
    st.subheader("🚨 Critical & High Risk Students")
    critical_df = df[df["risk_tier"].isin(["Critical", "High"])].sort_values(
        "dropout_probability", ascending=False
    )
    
    display_cols = [
        "student_id", "risk_tier", "dropout_probability", "counseling_track",
        "assigned_to", "priority", "attendance_pct", "internal_score",
        "backlogs", "is_orphan", "has_guardian", "missed_fee_payments",
    ]
    
    st.dataframe(
        critical_df[display_cols].head(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            "dropout_probability": st.column_config.ProgressColumn(
                "Dropout Prob", format="%.2f", min_value=0, max_value=1,
            ),
        },
    )


# ===========================================================
# PAGE: STUDENT LOOKUP
# ===========================================================
elif page == "🔍 Student Lookup":
    st.header("🔍 Individual Student Analysis")

    if meta is None:
        st.error("⚠️ Model not trained. Run model_training.py first.")
        st.stop()

    predictor = load_predictor()
    df = load_default_data()

    # Student selector
    student_id = st.selectbox(
        "Select Student",
        df["student_id"].tolist(),
        index=0,
    )

    student = df[df["student_id"] == student_id].iloc[0].to_dict()

    # Predict
    pred = predictor.predict_single(student)
    prob = pred["dropout_probability"]
    tier = classify_risk_tier(prob)
    student["risk_tier"] = tier
    routing = route_counseling_track(student)

    # ---- Student Profile ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Profile")
        st.write(f"**ID:** {student['student_id']}")
        st.write(f"**Age:** {student['age']} | **Gender:** {student['gender']}")
        st.write(f"**Semester:** {student['semester']}")
        st.write(f"**Orphan:** {'Yes' if student['is_orphan'] else 'No'}")
        st.write(f"**Guardian:** {'Yes' if student['has_guardian'] else 'No'}")
        st.write(f"**First-Gen Learner:** {'Yes' if student['is_first_gen_learner'] else 'No'}")
        st.write(f"**Income Category:** {['','BPL','LIG','MIG','HIG'][student['income_category']]}")

    with col2:
        st.subheader("Academic")
        st.metric("Attendance", f"{student['attendance_pct']}%")
        st.metric("Internal Score", f"{student['internal_score']}")
        st.metric("Backlogs", f"{student['backlogs']}")
        st.metric("Library Visits/Month", f"{student['library_visits_per_month']}")

    with col3:
        st.subheader("Financial")
        st.metric("Fee Paid", f"{student['fee_paid_ratio']*100:.0f}%")
        st.metric("Missed Payments", f"{student['missed_fee_payments']}")
        st.metric("Scholarship", "Yes" if student['has_scholarship'] else "No")
        st.metric("Scholarship Dependent", "Yes" if student['scholarship_dependent'] else "No")

    st.markdown("---")

    # ---- Prediction Result ----
    pred_col1, pred_col2 = st.columns(2)

    with pred_col1:
        st.subheader("Prediction")
        tier_color = get_tier_color(tier)
        st.markdown(
            f"<h1 style='color:{tier_color};'>{tier} Risk — {prob*100:.1f}%</h1>",
            unsafe_allow_html=True,
        )

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Dropout Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": tier_color},
                "steps": [
                    {"range": [0, 20], "color": "#d5f5e3"},
                    {"range": [20, 40], "color": "#fdebd0"},
                    {"range": [40, 60], "color": "#fadbd8"},
                    {"range": [60, 100], "color": "#f1948a"},
                ],
            },
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with pred_col2:
        st.subheader(f"{routing['track_icon']} Counseling: {routing['track']} Track")
        st.write(f"**Assigned to:** {routing['assigned_to']}")
        st.write(f"**Priority:** {routing['priority']}")
        st.write(f"**Alert Frequency:** {routing['alert_frequency']}")
        st.markdown("**Recommended Actions:**")
        for action in routing["recommended_actions"]:
            st.write(f"  - {action}")

    # ---- SHAP Explanation ----
    st.markdown("---")
    st.subheader("🔬 Why This Prediction? (SHAP Explanation)")
    
    with st.spinner("Computing SHAP explanation..."):
        explanations = predictor.explain_student(student, top_n=8)

    if explanations:
        exp_df = pd.DataFrame(explanations, columns=["Feature", "SHAP Value", "Student Value", "Direction"])
        
        fig_shap = go.Figure(go.Bar(
            y=[e[0] for e in explanations][::-1],
            x=[e[1] for e in explanations][::-1],
            orientation="h",
            marker_color=["#e74c3c" if e[1] > 0 else "#2ecc71" for e in explanations][::-1],
            text=[f"{e[2]}" for e in explanations][::-1],
            textposition="auto",
        ))
        fig_shap.update_layout(
            height=350,
            xaxis_title="SHAP Value (impact on dropout prediction)",
            yaxis_title="Feature",
            margin=dict(t=20, l=150),
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        
        st.caption(
            "Red bars = feature pushes prediction toward dropout. "
            "Green bars = feature pushes prediction toward safe. "
            "Numbers on bars show the student's actual value for that feature."
        )


# ===========================================================
# PAGE: BATCH UPLOAD
# ===========================================================
elif page == "📁 Batch Upload":
    st.header("📁 Batch CSV Upload")
    st.markdown("Upload a CSV file with student data to generate predictions for the entire cohort.")

    if meta is None:
        st.error("⚠️ Model not trained. Run model_training.py first.")
        st.stop()

    predictor = load_predictor()

    # Show expected format
    with st.expander("📋 Expected CSV Format"):
        st.markdown("""
        Your CSV should contain these columns:
        
        | Column | Type | Description |
        |--------|------|-------------|
        | student_id | string | Unique student identifier |
        | age | int | Student age (17-22) |
        | gender | string | Male / Female |
        | semester | int | Current semester (1-8) |
        | is_orphan | int | 0 or 1 |
        | has_guardian | int | 0 or 1 |
        | is_first_gen_learner | int | 0 or 1 |
        | is_hostel_resident | int | 0 or 1 |
        | income_category | int | 1=BPL, 2=LIG, 3=MIG, 4=HIG |
        | has_scholarship | int | 0 or 1 |
        | scholarship_dependent | int | 0 or 1 |
        | attendance_pct | float | Attendance percentage |
        | internal_score | float | Internal assessment score |
        | backlogs | int | Number of failed subjects |
        | fee_paid_ratio | float | 0.0 to 1.0 |
        | missed_fee_payments | int | Number of missed payments |
        | library_visits_per_month | int | Monthly library visits |
        | counselor_visits | int | Number of counselor visits |
        """)

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {len(df_upload)} student records.")

        with st.spinner("Running predictions..."):
            df_result = add_risk_columns(df_upload, predictor)

        # Summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total", len(df_result))
        col2.metric("At Risk", (df_result["prediction"] == 1).sum())
        col3.metric("Critical", (df_result["risk_tier"] == "Critical").sum())
        col4.metric("Welfare Track", (df_result["counseling_track"] == "Welfare").sum())

        # Results table
        st.dataframe(
            df_result[[
                "student_id", "risk_tier", "dropout_probability",
                "counseling_track", "assigned_to", "priority",
                "attendance_pct", "internal_score", "is_orphan",
            ]].sort_values("dropout_probability", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        # Download results
        csv_out = df_result.to_csv(index=False)
        st.download_button(
            "⬇️ Download Results CSV",
            csv_out,
            file_name="dropout_predictions.csv",
            mime="text/csv",
        )


# ===========================================================
# PAGE: MODEL PERFORMANCE
# ===========================================================
elif page == "📈 Model Performance":
    st.header("📈 Model Comparison & Performance")

    if meta is None:
        st.error("⚠️ No trained model found.")
        st.stop()

    # ---- Model Comparison Table ----
    st.subheader("Five-Model Comparison")
    
    metrics_df = pd.DataFrame(meta["all_model_metrics"]).T
    metrics_df.index.name = "Model"
    metrics_df = metrics_df.reset_index()
    
    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "accuracy": st.column_config.NumberColumn(format="%.4f"),
            "precision": st.column_config.NumberColumn(format="%.4f"),
            "recall": st.column_config.NumberColumn(format="%.4f"),
            "f1_score": st.column_config.NumberColumn(format="%.4f"),
            "roc_auc": st.column_config.NumberColumn(format="%.4f"),
            "cv_f1_mean": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.info(f"✅ **Best Model: {meta['best_model_name']}** (selected by F1 score)")

    # ---- Bar chart comparison ----
    comparison_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    available_metrics = [m for m in comparison_metrics if m in metrics_df.columns]
    
    fig_compare = go.Figure()
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    
    for i, metric in enumerate(available_metrics):
        fig_compare.add_trace(go.Bar(
            name=metric.replace("_", " ").title(),
            x=metrics_df["Model"],
            y=metrics_df[metric],
            marker_color=colors[i],
        ))
    
    fig_compare.update_layout(
        barmode="group",
        height=400,
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        margin=dict(t=30),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # ---- Feature Importance ----
    st.subheader("Feature Importance (SHAP-based)")
    
    fi = meta["feature_importance"]
    fig_fi = go.Figure(go.Bar(
        y=[f["feature"] for f in fi][::-1],
        x=[f["importance"] for f in fi][::-1],
        orientation="h",
        marker_color="#667eea",
    ))
    fig_fi.update_layout(
        height=450,
        xaxis_title="Mean |SHAP Value|",
        margin=dict(t=20, l=200),
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ===========================================================
# PAGE: SETTINGS
# ===========================================================
elif page == "⚙️ Settings":
    st.header("⚙️ System Settings")

    st.subheader("Risk Tier Thresholds")
    st.markdown("Configure the probability cutoffs for each risk tier.")

    col1, col2, col3 = st.columns(3)
    with col1:
        low_thresh = st.slider("Low → Medium threshold", 0.0, 1.0, 0.20, 0.05)
    with col2:
        med_thresh = st.slider("Medium → High threshold", 0.0, 1.0, 0.40, 0.05)
    with col3:
        high_thresh = st.slider("High → Critical threshold", 0.0, 1.0, 0.60, 0.05)

    st.markdown("---")
    
    st.subheader("Alert Configuration")
    alert_freq = st.selectbox("Default Alert Frequency", ["Weekly", "Bi-weekly", "Monthly"])
    enable_email = st.checkbox("Enable Email Alerts", value=False)
    enable_sms = st.checkbox("Enable SMS Alerts", value=False)

    st.markdown("---")
    
    st.subheader("System Architecture")
    st.markdown("""
    **Data Flow:**
    1. **Data Fusion Layer** — Ingests attendance, scores, fees, and background data into unified profiles
    2. **Prediction Layer** — 5 ML models trained and compared; best model selected automatically
    3. **Risk Tiering** — Students classified into Low / Medium / High / Critical tiers
    4. **Counseling Router** — Three-track routing based on academic risk + vulnerability profile
    5. **Dashboard & Alerts** — Real-time visualization and automated staff notifications
    
    **Models Used:** KNN, Naive Bayes, Decision Tree, SVM, ANN (MLP)
    
    **Explainability:** SHAP (SHapley Additive exPlanations) for per-student feature attribution
    
    **Tech Stack:** Python, scikit-learn, SHAP, Streamlit, Plotly, Pandas
    """)
