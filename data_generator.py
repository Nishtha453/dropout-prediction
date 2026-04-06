"""
Synthetic Student Data Generator
================================
Generates realistic student data for dropout prediction.
Encodes plausible correlations:
- Low attendance + low grades → higher dropout
- No guardian + missed fees → higher dropout
- First-gen learner + financial stress → higher dropout
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_student_data(n_students: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic student records with vulnerability flags
    and realistic dropout correlations.
    """
    students = []

    for i in range(n_students):
        student_id = f"STU{i+1:04d}"

        # ---- Demographics ----
        age = np.random.choice(range(17, 23), p=[0.15, 0.30, 0.25, 0.15, 0.10, 0.05])
        gender = np.random.choice(["Male", "Female"], p=[0.65, 0.35])

        # ---- Vulnerability flags ----
        is_orphan = np.random.choice([0, 1], p=[0.92, 0.08])
        has_guardian = 0 if is_orphan else np.random.choice([0, 1], p=[0.05, 0.95])
        is_first_gen = np.random.choice([0, 1], p=[0.55, 0.45])
        is_hostel_resident = 1 if is_orphan else np.random.choice([0, 1], p=[0.60, 0.40])

        # Income category: 1=BPL, 2=LIG, 3=MIG, 4=HIG
        if is_orphan or (is_first_gen and np.random.random() < 0.7):
            income_category = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        else:
            income_category = np.random.choice([1, 2, 3, 4], p=[0.15, 0.25, 0.35, 0.25])

        has_scholarship = 1 if (income_category <= 2 and np.random.random() < 0.6) else 0
        scholarship_dependent = 1 if (has_scholarship and income_category == 1) else 0

        # ---- Academic features ----
        # Base attendance influenced by vulnerability
        base_attendance = np.random.normal(75, 12)
        if is_orphan:
            base_attendance -= np.random.uniform(5, 15)
        if is_first_gen:
            base_attendance -= np.random.uniform(2, 8)
        attendance_pct = np.clip(base_attendance, 10, 100)

        # Internal assessment scores (out of 100)
        base_score = np.random.normal(55, 18)
        if attendance_pct < 50:
            base_score -= np.random.uniform(10, 20)
        if is_first_gen:
            base_score -= np.random.uniform(3, 8)
        internal_score = np.clip(base_score, 5, 100)

        # Number of backlogs (failed subjects)
        if internal_score < 35:
            backlogs = np.random.choice([2, 3, 4, 5], p=[0.3, 0.3, 0.25, 0.15])
        elif internal_score < 50:
            backlogs = np.random.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.25, 0.15])
        else:
            backlogs = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])

        # ---- Financial features ----
        # Fee payment ratio (0 to 1, where 1 = fully paid)
        if income_category == 1:
            fee_paid_ratio = np.clip(np.random.normal(0.55, 0.25), 0, 1)
        elif income_category == 2:
            fee_paid_ratio = np.clip(np.random.normal(0.70, 0.20), 0, 1)
        else:
            fee_paid_ratio = np.clip(np.random.normal(0.90, 0.10), 0, 1)

        if has_scholarship:
            fee_paid_ratio = min(1.0, fee_paid_ratio + 0.2)

        missed_fee_payments = int(np.clip((1 - fee_paid_ratio) * 6, 0, 6))  # out of 6 installments

        # ---- Engagement features ----
        # Library visits per month
        library_visits = max(0, int(np.random.normal(
            8 if internal_score > 60 else 4 if internal_score > 40 else 2, 3
        )))

        # Counselor visits (0-5)
        counselor_visits = np.random.choice(
            [0, 1, 2, 3, 4, 5],
            p=[0.40, 0.25, 0.15, 0.10, 0.06, 0.04]
        )

        # Semester number (1-8)
        semester = np.random.choice(range(1, 9))

        # ---- Compute dropout probability ----
        # This is the "ground truth" generation logic
        dropout_score = 0.0

        # Academic risk
        if attendance_pct < 40:
            dropout_score += 0.30
        elif attendance_pct < 60:
            dropout_score += 0.15
        elif attendance_pct < 75:
            dropout_score += 0.05

        if internal_score < 30:
            dropout_score += 0.25
        elif internal_score < 45:
            dropout_score += 0.12

        dropout_score += backlogs * 0.06

        # Financial risk
        if missed_fee_payments >= 4:
            dropout_score += 0.20
        elif missed_fee_payments >= 2:
            dropout_score += 0.10

        # Vulnerability risk
        if is_orphan and not has_guardian:
            dropout_score += 0.15
        if is_first_gen:
            dropout_score += 0.05
        if scholarship_dependent and fee_paid_ratio < 0.5:
            dropout_score += 0.10
        if income_category == 1:
            dropout_score += 0.08

        # Later semesters with backlogs = higher risk
        if semester >= 5 and backlogs >= 3:
            dropout_score += 0.10

        # Add noise
        dropout_score += np.random.normal(0, 0.08)
        dropout_score = np.clip(dropout_score, 0, 1)

        # Binary label
        dropped_out = 1 if dropout_score > 0.40 else 0

        students.append({
            "student_id": student_id,
            "age": age,
            "gender": gender,
            "semester": semester,
            "is_orphan": is_orphan,
            "has_guardian": has_guardian,
            "is_first_gen_learner": is_first_gen,
            "is_hostel_resident": is_hostel_resident,
            "income_category": income_category,
            "has_scholarship": has_scholarship,
            "scholarship_dependent": scholarship_dependent,
            "attendance_pct": round(attendance_pct, 1),
            "internal_score": round(internal_score, 1),
            "backlogs": backlogs,
            "fee_paid_ratio": round(fee_paid_ratio, 2),
            "missed_fee_payments": missed_fee_payments,
            "library_visits_per_month": library_visits,
            "counselor_visits": counselor_visits,
            "dropout_probability": round(dropout_score, 4),
            "dropped_out": dropped_out,
        })

    df = pd.DataFrame(students)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_student_data(2000)
    df.to_csv("data/student_data.csv", index=False)

    print(f"Generated {len(df)} student records")
    print(f"Dropout rate: {df['dropped_out'].mean()*100:.1f}%")
    print(f"Orphan count: {df['is_orphan'].sum()}")
    print(f"First-gen learners: {df['is_first_gen_learner'].sum()}")
    print(f"\nSample:\n{df.head()}")
