# streamlit dashboard for dropout prediction system

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import io

# page config

st.set_page_config(
    page_title="Dropout Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# custom styling

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .stApp {
        font-family: 'DM Sans', sans-serif;
    }
    
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #e2e8f0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .tier-critical { color: #dc2626; }
    .tier-high { color: #ea580c; }
    .tier-medium { color: #d97706; }
    .tier-low { color: #16a34a; }
    
    .track-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .track-welfare { background: #fef2f2; color: #dc2626; }
    .track-academic { background: #eff6ff; color: #2563eb; }
    .track-career { background: #f0fdf4; color: #16a34a; }
    .track-monitoring { background: #f8fafc; color: #64748b; }
    
    .student-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid;
    }
    .student-card.critical { border-left-color: #dc2626; }
    .student-card.high { border-left-color: #ea580c; }
    .student-card.medium { border-left-color: #d97706; }
    .student-card.low { border-left-color: #16a34a; }
    
    div[data-testid="stSidebar"] {
        background: #0f172a;
    }
    div[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# load data

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


def setup_if_needed():
    """Auto-generate data and train models if files don't exist (for cloud deployment)"""
    if not (DATA_DIR / "risk_profiles.csv").exists() or not (MODEL_DIR / "best_model.pkl").exists():
        import subprocess, sys
        st.info("First run detected. Generating data and training models... (this takes ~2 minutes)")
        
        DATA_DIR.mkdir(exist_ok=True)
        MODEL_DIR.mkdir(exist_ok=True)
        
        # run data generation
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "scripts" / "generate_data.py")],
            capture_output=True, text=True, cwd=str(BASE_DIR)
        )
        if result.returncode != 0:
            st.error(f"Data generation failed: {result.stderr}")
            st.stop()
        
        # run training pipeline
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "scripts" / "train_pipeline.py")],
            capture_output=True, text=True, cwd=str(BASE_DIR)
        )
        if result.returncode != 0:
            st.error(f"Training failed: {result.stderr}")
            st.stop()
        
        st.success("Setup complete! Data generated and models trained.")
        st.cache_data.clear()
        st.cache_resource.clear()

setup_if_needed()


@st.cache_data
def load_data():
    risk_profiles = pd.read_csv(DATA_DIR / "risk_profiles.csv")
    student_profiles = pd.read_csv(DATA_DIR / "student_profiles.csv")
    semester_records = pd.read_csv(DATA_DIR / "semester_records.csv")
    ml_data = pd.read_csv(DATA_DIR / "ml_ready_dataset.csv")
    
    with open(DATA_DIR / "risk_summary.json") as f:
        risk_summary = json.load(f)
    with open(MODEL_DIR / "model_results.json") as f:
        model_results = json.load(f)
    with open(MODEL_DIR / "shap_importance.json") as f:
        shap_importance = json.load(f)
    
    return risk_profiles, student_profiles, semester_records, ml_data, risk_summary, model_results, shap_importance


@st.cache_resource
def load_model():
    with open(MODEL_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


try:
    risk_profiles, student_profiles, semester_records, ml_data, risk_summary, model_results, shap_importance = load_data()
    model, scaler = load_model()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Failed to load data: {e}")
    st.stop()


# sidebar

with st.sidebar:
    st.markdown("## 🎓 Navigation")
    st.markdown("---")
    
    page = st.radio(
        "Go to",
        ["📊 Dashboard", "🔍 Student Explorer", "🤖 Model Performance",
         "📈 Analytics", "⚡ Live Predictor", "📤 Batch Upload"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Filters")
    
    tier_filter = st.multiselect(
        "Risk Tier",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"]
    )
    
    track_filter = st.multiselect(
        "Counseling Track",
        ["Welfare", "Academic", "Career Guidance", "Monitoring"],
        default=["Welfare", "Academic", "Career Guidance", "Monitoring"]
    )
    
    branch_filter = st.multiselect(
        "Branch",
        sorted(risk_profiles["branch"].unique()),
        default=sorted(risk_profiles["branch"].unique())
    )
    
    st.markdown("---")
    st.markdown(
        f"<small>Best model: **{model_results['best_model']}**<br>"
        f"F1: {model_results['results'][model_results['best_model']]['f1_score']}</small>",
        unsafe_allow_html=True
    )


# Apply filters
filtered = risk_profiles[
    (risk_profiles["risk_tier"].isin(tier_filter)) &
    (risk_profiles["counseling_track"].isin(track_filter)) &
    (risk_profiles["branch"].isin(branch_filter))
]


# colors

TIER_COLORS = {"Critical": "#dc2626", "High": "#ea580c", "Medium": "#d97706", "Low": "#16a34a"}
TRACK_COLORS = {"Welfare": "#dc2626", "Academic": "#2563eb", "Career Guidance": "#16a34a", "Monitoring": "#64748b"}


# helper functions

def get_recommended_actions(track, tier, student):
    actions = []
    if track == "Academic":
        actions = ["Assign subject mentor", "Attendance improvement counseling",
                   "Remedial classes for backlog subjects", "Peer study group matching"]
    elif track == "Welfare":
        if student.get("is_orphan", False):
            actions.append("Assign dedicated welfare officer")
            actions.append("Schedule empathetic check-in")
        actions.extend(["Review financial aid eligibility", "Peer mentor from senior batch",
                       "Emotional support referral if distress persists"])
    elif track == "Career Guidance":
        actions = ["Skill-interest profile assessment", "Connect with career advisor",
                   "Curate vocational pathways", "Government scheme links"]
    else:
        actions = ["Continue standard monitoring", "Review next semester"]
    
    if tier == "Critical":
        actions.insert(0, "URGENT: Immediate intervention meeting")
    return actions


# DASHBOARD PAGE

if page == "📊 Dashboard":
    st.markdown('<p class="main-header">Dropout Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Government of Rajasthan · Directorate of Technical Education</p>', unsafe_allow_html=True)
    
    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{risk_summary['total_students']}</div>
            <div class="metric-label">Total Students</div>
        </div>""", unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value tier-critical">{risk_summary.get('critical_count', 0)}</div>
            <div class="metric-label">Critical Risk</div>
        </div>""", unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value tier-high">{risk_summary.get('high_count', 0)}</div>
            <div class="metric-label">High Risk</div>
        </div>""", unsafe_allow_html=True)
    
    with c4:
        welfare_count = len(risk_profiles[risk_profiles["counseling_track"] == "Welfare"])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #7c3aed;">{welfare_count}</div>
            <div class="metric-label">Welfare Track</div>
        </div>""", unsafe_allow_html=True)
    
    with c5:
        avg_prob = risk_summary.get("avg_dropout_prob", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_prob:.1%}</div>
            <div class="metric-label">Avg Dropout Risk</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        tier_counts = filtered["risk_tier"].value_counts().reindex(
            ["Critical", "High", "Medium", "Low"], fill_value=0
        )
        fig_tier = go.Figure(data=[go.Bar(
            x=tier_counts.index,
            y=tier_counts.values,
            marker_color=[TIER_COLORS[t] for t in tier_counts.index],
            text=tier_counts.values,
            textposition="outside",
        )])
        fig_tier.update_layout(
            title="Risk Tier Distribution",
            xaxis_title="", yaxis_title="Students",
            height=380, margin=dict(t=50, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig_tier, use_container_width=True)
    
    with col2:
        track_counts = filtered["counseling_track"].value_counts()
        fig_track = go.Figure(data=[go.Pie(
            labels=track_counts.index,
            values=track_counts.values,
            marker_colors=[TRACK_COLORS.get(t, "#94a3b8") for t in track_counts.index],
            hole=0.4,
            textinfo="label+percent",
        )])
        fig_track.update_layout(
            title="Counseling Track Distribution",
            height=380, margin=dict(t=50, b=30),
            font=dict(family="DM Sans"),
        )
        st.plotly_chart(fig_track, use_container_width=True)
    
    # Vulnerability Heatmap
    st.markdown("### Vulnerability × Risk Matrix")
    
    vuln_data = filtered.copy()
    vuln_data["vulnerability"] = "Standard"
    vuln_data.loc[vuln_data["is_orphan"] == True, "vulnerability"] = "Orphan"
    vuln_data.loc[vuln_data["is_first_gen"] == True, "vulnerability"] = "First-Gen"
    vuln_data.loc[
        (vuln_data["is_orphan"] == True) & (vuln_data["is_first_gen"] == True),
        "vulnerability"
    ] = "Orphan + First-Gen"
    
    heatmap_data = pd.crosstab(
        vuln_data["vulnerability"],
        vuln_data["risk_tier"],
    ).reindex(columns=["Critical", "High", "Medium", "Low"], fill_value=0)
    
    fig_heat = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="OrRd",
        text_auto=True,
        aspect="auto",
    )
    fig_heat.update_layout(
        height=300, margin=dict(t=30, b=30),
        font=dict(family="DM Sans"),
        xaxis_title="Risk Tier", yaxis_title="",
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Priority Alerts
    st.markdown("### Priority Alerts - Critical & High Risk")
    
    alerts = filtered[filtered["risk_tier"].isin(["Critical", "High"])].sort_values(
        "dropout_probability", ascending=False
    ).head(10)
    
    for _, row in alerts.iterrows():
        tier_class = row["risk_tier"].lower()
        track_class = row["counseling_track"].lower().replace(" ", "-")
        
        vuln_tags = []
        if row.get("is_orphan"):
            vuln_tags.append("🏠 Orphan")
        if row.get("is_first_gen"):
            vuln_tags.append("📚 First-Gen")
        if not row.get("has_guardian"):
            vuln_tags.append("👤 No Guardian")
        vuln_str = " · ".join(vuln_tags) if vuln_tags else ""
        
        st.markdown(f"""
        <div class="student-card {tier_class}">
            <strong>{row['student_id']}</strong> · {row['branch']} · Age {row['age']}
            {f' · {vuln_str}' if vuln_str else ''}
            <br>
            <span class="tier-{tier_class}">■ {row['risk_tier']}</span> · 
            Dropout Prob: <strong>{row['dropout_probability']:.1%}</strong> · 
            Track: <span class="track-badge track-{track_class}">{row['counseling_track']}</span>
            <br>
            <small>Attendance: {row['avg_attendance']:.0f}% · IA Score: {row['avg_ia_score']:.1f}/40 · Fee Defaults: {row['fee_defaults_count']}</small>
        </div>
        """, unsafe_allow_html=True)


# STUDENT EXPLORER PAGE

elif page == "🔍 Student Explorer":
    st.markdown('<p class="main-header">Student Explorer</p>', unsafe_allow_html=True)
    
    search = st.text_input("🔍 Search by Student ID", placeholder="e.g., STU00042")
    
    if search:
        match = risk_profiles[risk_profiles["student_id"].str.contains(search, case=False)]
    else:
        match = filtered
    
    st.markdown(f"**Showing {len(match)} students**")
    
    # Student table
    display_cols = ["student_id", "risk_tier", "counseling_track", "dropout_probability",
                    "branch", "age", "is_orphan", "is_first_gen", "avg_attendance",
                    "avg_ia_score", "fee_defaults_count"]
    
    st.dataframe(
        match[display_cols].sort_values("dropout_probability", ascending=False),
        use_container_width=True,
        height=400,
        column_config={
            "dropout_probability": st.column_config.ProgressColumn(
                "Dropout Risk", format="%.1f%%", min_value=0, max_value=1,
            ),
            "avg_attendance": st.column_config.NumberColumn("Avg Attendance %", format="%.1f"),
            "avg_ia_score": st.column_config.NumberColumn("Avg IA Score", format="%.1f"),
        }
    )
    
    # Student Detail View
    st.markdown("---")
    st.markdown("### Student Detail View")
    
    selected_id = st.selectbox(
        "Select student",
        match["student_id"].tolist()[:100],
    )
    
    if selected_id:
        student = risk_profiles[risk_profiles["student_id"] == selected_id].iloc[0]
        profile = student_profiles[student_profiles["student_id"] == selected_id].iloc[0]
        sem_data = semester_records[semester_records["student_id"] == selected_id]
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("#### Profile")
            st.markdown(f"""
            - **ID:** {student['student_id']}
            - **Age:** {student['age']}
            - **Branch:** {student['branch']}
            - **Income:** {student['income_category']}
            - **Orphan:** {'Yes' if student['is_orphan'] else 'No'}
            - **Guardian:** {'Yes' if student['has_guardian'] else 'No'}
            - **First-Gen:** {'Yes' if student['is_first_gen'] else 'No'}
            """)
        
        with c2:
            st.markdown("#### Risk Assessment")
            tier_color = TIER_COLORS.get(student['risk_tier'], '#666')
            st.markdown(f"""
            - **Dropout Probability:** <span style="color:{tier_color};font-size:1.5rem;font-weight:700">{student['dropout_probability']:.1%}</span>
            - **Risk Tier:** <span style="color:{tier_color}">{student['risk_tier']}</span>
            - **Counseling Track:** {student['counseling_track']}
            - **Semesters Completed:** {student['semesters_completed']}
            """, unsafe_allow_html=True)
        
        with c3:
            st.markdown("#### Recommended Actions")
            actions = get_recommended_actions(
                student["counseling_track"], student["risk_tier"], student.to_dict()
            )
            for action in actions:
                st.markdown(f"- {action}")
        
        # Semester trend chart
        if not sem_data.empty:
            st.markdown("#### Semester Trends")
            fig_trend = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Attendance %", "Avg IA Score"),
            )
            fig_trend.add_trace(
                go.Scatter(x=sem_data["semester"], y=sem_data["attendance_pct"],
                          mode="lines+markers", name="Attendance",
                          line=dict(color="#2563eb", width=2)),
                row=1, col=1
            )
            fig_trend.add_trace(
                go.Scatter(x=sem_data["semester"], y=sem_data["avg_ia_score"],
                          mode="lines+markers", name="IA Score",
                          line=dict(color="#16a34a", width=2)),
                row=1, col=2
            )
            fig_trend.update_layout(height=300, showlegend=False, font=dict(family="DM Sans"))
            st.plotly_chart(fig_trend, use_container_width=True)


# MODEL PERFORMANCE PAGE

elif page == "🤖 Model Performance":
    st.markdown('<p class="main-header">Model Performance Comparison</p>', unsafe_allow_html=True)
    
    results = model_results["results"]
    best = model_results["best_model"]
    
    # Model comparison table
    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "Model"
    metrics_df = metrics_df.reset_index()
    
    st.markdown(f"**Best Model: {best}** (selected by F1 Score — prioritizes catching dropouts)")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Radar chart
    fig_radar = go.Figure()
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    
    for model_name, metrics in results.items():
        fig_radar.add_trace(go.Scatterpolar(
            r=[metrics[m] for m in metrics_to_plot],
            theta=[m.replace("_", " ").title() for m in metrics_to_plot],
            name=model_name,
            fill="toself",
            opacity=0.6,
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Model Comparison Radar",
        height=500,
        font=dict(family="DM Sans"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # SHAP Feature Importance
    st.markdown("### SHAP Feature Importance")
    st.markdown("*Which factors most influence dropout predictions?*")
    
    top_features = dict(list(shap_importance.items())[:12])
    fig_shap = go.Figure(data=[go.Bar(
        y=list(top_features.keys())[::-1],
        x=list(top_features.values())[::-1],
        orientation="h",
        marker_color="#6366f1",
    )])
    fig_shap.update_layout(
        height=450, margin=dict(l=200),
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="",
        font=dict(family="DM Sans"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_shap, use_container_width=True)


# ANALYTICS PAGE

elif page == "📈 Analytics":
    st.markdown('<p class="main-header">Cohort Analytics</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Dropout rate by branch
        branch_dropout = ml_data.groupby("branch")["dropped_out"].mean().sort_values(ascending=False)
        fig_branch = go.Figure(data=[go.Bar(
            x=branch_dropout.index,
            y=branch_dropout.values,
            marker_color="#6366f1",
            text=[f"{v:.1%}" for v in branch_dropout.values],
            textposition="outside",
        )])
        fig_branch.update_layout(
            title="Dropout Rate by Branch",
            yaxis_title="Dropout Rate", yaxis_tickformat=".0%",
            height=400, font=dict(family="DM Sans"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_branch, use_container_width=True)
    
    with col2:
        # Dropout rate by income
        income_dropout = ml_data.groupby("income_category")["dropped_out"].mean()
        income_order = ["BPL", "LIG", "MIG", "HIG"]
        income_dropout = income_dropout.reindex(income_order)
        
        fig_income = go.Figure(data=[go.Bar(
            x=income_dropout.index,
            y=income_dropout.values,
            marker_color=["#dc2626", "#ea580c", "#d97706", "#16a34a"],
            text=[f"{v:.1%}" for v in income_dropout.values],
            textposition="outside",
        )])
        fig_income.update_layout(
            title="Dropout Rate by Income Category",
            yaxis_title="Dropout Rate", yaxis_tickformat=".0%",
            height=400, font=dict(family="DM Sans"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_income, use_container_width=True)
    
    # Attendance vs Dropout Probability scatter
    st.markdown("### Attendance vs Dropout Risk")
    fig_scatter = px.scatter(
        filtered,
        x="avg_attendance",
        y="dropout_probability",
        color="risk_tier",
        color_discrete_map=TIER_COLORS,
        hover_data=["student_id", "branch", "counseling_track"],
        opacity=0.6,
    )
    fig_scatter.update_layout(
        height=500,
        xaxis_title="Average Attendance %",
        yaxis_title="Dropout Probability",
        font=dict(family="DM Sans"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Vulnerability analysis
    col1, col2 = st.columns(2)
    
    with col1:
        orphan_rate = ml_data[ml_data["is_orphan"] == True]["dropped_out"].mean()
        non_orphan_rate = ml_data[ml_data["is_orphan"] == False]["dropped_out"].mean()
        
        fig_orphan = go.Figure(data=[go.Bar(
            x=["Orphan Students", "Non-Orphan Students"],
            y=[orphan_rate, non_orphan_rate],
            marker_color=["#dc2626", "#64748b"],
            text=[f"{v:.1%}" for v in [orphan_rate, non_orphan_rate]],
            textposition="outside",
        )])
        fig_orphan.update_layout(
            title="Orphan vs Non-Orphan Dropout Rates",
            yaxis_tickformat=".0%", height=350,
            font=dict(family="DM Sans"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_orphan, use_container_width=True)
    
    with col2:
        fg_rate = ml_data[ml_data["is_first_gen"] == True]["dropped_out"].mean()
        nfg_rate = ml_data[ml_data["is_first_gen"] == False]["dropped_out"].mean()
        
        fig_fg = go.Figure(data=[go.Bar(
            x=["First-Gen Learners", "Non-First-Gen"],
            y=[fg_rate, nfg_rate],
            marker_color=["#7c3aed", "#64748b"],
            text=[f"{v:.1%}" for v in [fg_rate, nfg_rate]],
            textposition="outside",
        )])
        fig_fg.update_layout(
            title="First-Gen vs Non-First-Gen Dropout Rates",
            yaxis_tickformat=".0%", height=350,
            font=dict(family="DM Sans"),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_fg, use_container_width=True)


# LIVE PREDICTOR PAGE

elif page == "⚡ Live Predictor":
    st.markdown('<p class="main-header">Live Risk Predictor</p>', unsafe_allow_html=True)
    st.markdown("Enter student details to get an instant dropout risk assessment.")
    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown("**Demographics**")
            age = st.number_input("Age", 15, 25, 18)
            gender = st.selectbox("Gender", ["M", "F"])
            branch = st.selectbox("Branch", sorted(risk_profiles["branch"].unique()))
        
        with c2:
            st.markdown("**Vulnerability Profile**")
            is_orphan = st.checkbox("Orphan")
            has_guardian = st.checkbox("Has Guardian", value=True)
            is_first_gen = st.checkbox("First-Generation Learner")
            income = st.selectbox("Income Category", ["BPL", "LIG", "MIG", "HIG"])
            is_hosteler = st.checkbox("Hosteler")
            on_scholarship = st.checkbox("On Scholarship")
        
        with c3:
            st.markdown("**Academic Indicators**")
            avg_attendance = st.slider("Avg Attendance %", 0, 100, 75)
            avg_ia = st.slider("Avg IA Score (out of 40)", 0, 40, 25)
            subjects_failed = st.number_input("Total Subjects Failed", 0, 20, 0)
            backlog = st.number_input("Max Cumulative Backlog", 0, 20, 0)
            fee_defaults = st.number_input("Fee Defaults", 0, 10, 0)
            semesters = st.number_input("Semesters Completed", 1, 6, 1)
        
        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)
    
    if submitted:
        # Encode features
        gender_map = {"F": 0, "M": 1}
        income_map = {"BPL": 0, "HIG": 1, "LIG": 2, "MIG": 3}
        branch_map = {b: i for i, b in enumerate(sorted(risk_profiles["branch"].unique()))}
        
        features = np.array([[
            age, int(is_orphan), int(has_guardian), int(is_first_gen),
            int(is_hosteler), int(on_scholarship), 50.0, 65.0,
            avg_attendance, avg_attendance * 0.7, avg_attendance * 0.9,
            avg_ia, avg_ia * 0.6, subjects_failed, backlog,
            fee_defaults, fee_defaults * 15, fee_defaults * 30,
            5.0, 0.5, 0, semesters,
            gender_map.get(gender, 1),
            income_map.get(income, 3),
            branch_map.get(branch, 2),
        ]])
        
        features_scaled = scaler.transform(features)
        prob = float(model.predict_proba(features_scaled)[0][1])
        
        tier = "Low" if prob < 0.25 else "Medium" if prob < 0.5 else "High" if prob < 0.75 else "Critical"
        tier_color = TIER_COLORS[tier]
        
        student_dict = {
            "is_orphan": is_orphan, "has_guardian": has_guardian,
            "is_first_gen": is_first_gen, "income_category": income,
            "age": age, "avg_attendance": avg_attendance,
            "total_subjects_failed": subjects_failed,
        }
        
        top_factors = list(shap_importance.keys())[:5]
        track_val = "Monitoring"
        if tier != "Low":
            welfare = is_orphan or not has_guardian or (income == "BPL" and fee_defaults > 0)
            career = age >= 20
            if welfare and tier in ["High", "Critical"]:
                track_val = "Welfare"
            elif career and tier in ["Medium", "High", "Critical"]:
                track_val = "Career Guidance"
            else:
                track_val = "Academic"
        
        actions = get_recommended_actions(track_val, tier, student_dict)
        
        st.markdown("---")
        st.markdown("### Prediction Result")
        
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.metric("Dropout Probability", f"{prob:.1%}")
        with rc2:
            st.markdown(f"**Risk Tier:** <span style='color:{tier_color};font-size:1.5rem'>{tier}</span>", unsafe_allow_html=True)
        with rc3:
            st.markdown(f"**Counseling Track:** {track_val}")
        
        st.markdown("**Recommended Actions:**")
        for a in actions:
            st.markdown(f"- {a}")


# BATCH UPLOAD PAGE

elif page == "📤 Batch Upload":
    st.markdown('<p class="main-header">Batch CSV Upload</p>', unsafe_allow_html=True)
    st.markdown("Upload a CSV file with student data to get bulk predictions.")
    
    st.markdown("**Required columns:** age, gender, branch, is_orphan, has_guardian, "
                "is_first_gen, income_category, avg_attendance, avg_ia_score, "
                "total_subjects_failed, fee_defaults_count, semesters_completed")
    
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    
    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"**Loaded {len(df)} records**")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("🚀 Run Predictions", use_container_width=True):
            with st.spinner("Processing..."):
                # Simplified batch prediction
                results = []
                for _, row in df.iterrows():
                    student_dict = row.to_dict()
                    gender_map = {"F": 0, "M": 1}
                    income_map = {"BPL": 0, "HIG": 1, "LIG": 2, "MIG": 3}
                    branch_map = {b: i for i, b in enumerate(sorted(risk_profiles["branch"].unique()))}
                    
                    features = np.array([[
                        student_dict.get("age", 18),
                        int(student_dict.get("is_orphan", False)),
                        int(student_dict.get("has_guardian", True)),
                        int(student_dict.get("is_first_gen", False)),
                        int(student_dict.get("is_hosteler", False)),
                        int(student_dict.get("on_scholarship", False)),
                        student_dict.get("distance_from_home_km", 50),
                        student_dict.get("prev_academic_score", 65),
                        student_dict.get("avg_attendance", 75),
                        student_dict.get("avg_attendance", 75) * 0.7,
                        student_dict.get("avg_attendance", 75) * 0.9,
                        student_dict.get("avg_ia_score", 25),
                        student_dict.get("avg_ia_score", 25) * 0.6,
                        student_dict.get("total_subjects_failed", 0),
                        student_dict.get("max_cumulative_backlog", 0),
                        student_dict.get("fee_defaults_count", 0),
                        student_dict.get("avg_fee_delay", 0),
                        student_dict.get("max_fee_delay", 0),
                        student_dict.get("avg_library_visits", 5),
                        student_dict.get("extracurricular_rate", 0.5),
                        student_dict.get("total_counselor_visits", 0),
                        student_dict.get("semesters_completed", 1),
                        gender_map.get(student_dict.get("gender", "M"), 1),
                        income_map.get(student_dict.get("income_category", "MIG"), 3),
                        branch_map.get(student_dict.get("branch", "Computer Science"), 2),
                    ]])
                    
                    features_scaled = scaler.transform(features)
                    prob = float(model.predict_proba(features_scaled)[0][1])
                    tier = "Low" if prob < 0.25 else "Medium" if prob < 0.5 else "High" if prob < 0.75 else "Critical"
                    
                    results.append({
                        "student_id": student_dict.get("student_id", ""),
                        "dropout_probability": round(prob, 4),
                        "risk_tier": tier,
                    })
                
                results_df = pd.DataFrame(results)
                st.success(f"Processed {len(results_df)} students!")
                st.dataframe(results_df, use_container_width=True)
                
                csv_out = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results CSV",
                    csv_out,
                    "predictions.csv",
                    "text/csv",
                    use_container_width=True,
                )