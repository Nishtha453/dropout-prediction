"""
Counseling Router
==================
Routes at-risk students to one of three counseling tracks:
  1. Academic Track  → subject mentors, remediation
  2. Welfare Track   → welfare officers, financial aid, emotional support
  3. Career Guidance → career advisors, skill profiling, vocational pathways

Routing logic depends on BOTH risk tier AND vulnerability profile.
"""


# ---- Risk Tier Thresholds (configurable) ----
DEFAULT_THRESHOLDS = {
    "low": 0.20,       # below 20% → Low risk
    "medium": 0.40,    # 20-40%    → Medium risk
    "high": 0.60,      # 40-60%    → High risk
    # above 60%        → Critical risk
}


def classify_risk_tier(dropout_probability: float, thresholds: dict = None) -> str:
    """Assign student to risk tier based on dropout probability."""
    t = thresholds or DEFAULT_THRESHOLDS

    if dropout_probability < t["low"]:
        return "Low"
    elif dropout_probability < t["medium"]:
        return "Medium"
    elif dropout_probability < t["high"]:
        return "High"
    else:
        return "Critical"


def get_tier_color(tier: str) -> str:
    """Color code for dashboard display."""
    return {
        "Low": "#2ecc71",       # green
        "Medium": "#f39c12",    # amber
        "High": "#e67e22",      # orange
        "Critical": "#e74c3c",  # red
    }.get(tier, "#95a5a6")


def route_counseling_track(student: dict) -> dict:
    """
    Determine which counseling track a student should be routed to.
    
    Returns:
        dict with track name, assigned_to, recommended_actions, priority
    """
    tier = student.get("risk_tier", "Low")
    
    # Extract vulnerability signals
    is_orphan = student.get("is_orphan", 0)
    has_guardian = student.get("has_guardian", 1)
    is_first_gen = student.get("is_first_gen_learner", 0)
    income_cat = student.get("income_category", 3)
    scholarship_dep = student.get("scholarship_dependent", 0)
    missed_fees = student.get("missed_fee_payments", 0)
    attendance = student.get("attendance_pct", 75)
    internal_score = student.get("internal_score", 50)
    backlogs = student.get("backlogs", 0)
    age = student.get("age", 18)
    semester = student.get("semester", 1)
    
    # ---- Decision Logic ----
    
    # WELFARE TRACK: Guardian-less, orphaned, or severe financial distress
    is_welfare_case = (
        (is_orphan == 1) or
        (has_guardian == 0) or
        (income_cat == 1 and missed_fees >= 3) or
        (scholarship_dep == 1 and missed_fees >= 2)
    )
    
    # CAREER TRACK: Older students (sem 5+) disengaging due to lack of direction
    is_career_case = (
        semester >= 5 and
        age >= 20 and
        attendance < 65 and
        internal_score >= 35  # not failing academically per se
    )
    
    # Welfare takes priority over career for truly vulnerable students
    if is_welfare_case and tier in ("High", "Critical"):
        return _welfare_track(student)
    elif is_career_case and tier in ("Medium", "High", "Critical"):
        return _career_track(student)
    elif tier in ("Medium", "High", "Critical"):
        return _academic_track(student)
    else:
        return _no_intervention(student)


def _academic_track(student: dict) -> dict:
    """Route to subject mentors with specific remediation."""
    actions = ["Assign subject mentor for weak areas"]

    if student.get("attendance_pct", 75) < 60:
        actions.append("Implement attendance recovery plan")
    if student.get("backlogs", 0) >= 2:
        actions.append("Schedule backlog clearance tutoring sessions")
    if student.get("internal_score", 50) < 40:
        actions.append("Provide supplementary study materials")
    
    actions.append("Schedule bi-weekly progress review with mentor")

    return {
        "track": "Academic",
        "track_icon": "📚",
        "assigned_to": "Subject Mentor",
        "recommended_actions": actions,
        "priority": "Standard" if student.get("risk_tier") == "Medium" else "Urgent",
        "alert_frequency": "Weekly",
    }


def _welfare_track(student: dict) -> dict:
    """Route to welfare officer with vulnerability-aware actions."""
    actions = []

    if student.get("is_orphan", 0):
        actions.append("Assign dedicated welfare officer as primary contact")
        actions.append("Initiate empathetic check-in (non-clinical outreach)")
        if student.get("attendance_pct", 75) < 50:
            actions.append("Assess emotional distress indicators — escalate to helpline if persistent")
    
    if not student.get("has_guardian", 1):
        actions.append("Assign peer mentor from senior batch")
        actions.append("Connect with hostel warden for regular check-ins")
    
    if student.get("missed_fee_payments", 0) >= 2:
        actions.append("Initiate financial aid review")
        actions.append("Check eligibility for emergency scholarship")
    
    if student.get("is_first_gen_learner", 0):
        actions.append("Enroll in first-generation learner support group")
    
    actions.append("Schedule weekly welfare check-in")

    return {
        "track": "Welfare",
        "track_icon": "🛡️",
        "assigned_to": "Welfare Officer / Hostel Warden",
        "recommended_actions": actions,
        "priority": "Urgent",
        "alert_frequency": "Bi-weekly",
    }


def _career_track(student: dict) -> dict:
    """Route to career advisor for directionless older students."""
    actions = [
        "Conduct skill-interest profiling assessment",
        "Generate personalized career pathway report",
        "Share curated free learning resources (NPTEL, Coursera, SWAYAM)",
        "Connect with industry mentorship program",
    ]

    if student.get("semester", 1) >= 7:
        actions.append("Prioritize placement preparation support")
        actions.append("Link to relevant government employment schemes")
    
    if student.get("is_orphan", 0) or not student.get("has_guardian", 1):
        actions.append("Connect with NGO career support network")
    
    actions.append("Schedule monthly career guidance session")

    return {
        "track": "Career Guidance",
        "track_icon": "🎯",
        "assigned_to": "Career Advisor",
        "recommended_actions": actions,
        "priority": "Standard",
        "alert_frequency": "Monthly",
    }


def _no_intervention(student: dict) -> dict:
    """Low-risk student — monitoring only."""
    return {
        "track": "Monitoring",
        "track_icon": "✅",
        "assigned_to": "Class Coordinator",
        "recommended_actions": ["Continue regular monitoring", "No intervention required"],
        "priority": "Low",
        "alert_frequency": "Monthly",
    }
