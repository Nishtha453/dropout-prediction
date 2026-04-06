# generate synthetic student data for dropout prediction
# correlations between vulnerability factors are built in

import numpy as np
import pandas as pd
from pathlib import Path
import json

np.random.seed(42)

NUM_STUDENTS = 2000
SEMESTERS = 6  # max sems for 3yr diploma
OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_student_profiles(n):
    students = []
    for i in range(n):
        student_id = f"STU{i+1:05d}"
        
        age = np.random.choice(range(16, 23), p=[0.05, 0.25, 0.30, 0.20, 0.10, 0.06, 0.04])
        gender = np.random.choice(["M", "F"], p=[0.65, 0.35])
        
        # vulnerability flags - correlated with each other
        is_orphan = np.random.random() < 0.06
        has_guardian = not is_orphan and (np.random.random() < 0.92)
        is_first_gen = np.random.random() < 0.40
        
        # income depends on first-gen status
        if is_first_gen:
            income_cat = np.random.choice(
                ["BPL", "LIG", "MIG", "HIG"],
                p=[0.45, 0.35, 0.15, 0.05]
            )
        else:
            income_cat = np.random.choice(
                ["BPL", "LIG", "MIG", "HIG"],
                p=[0.10, 0.25, 0.40, 0.25]
            )
        
        # orphans always in hostel
        is_hosteler = True if is_orphan else np.random.random() < 0.45
        
        # scholarship more likely for lower income
        schol_probs = {"BPL": 0.80, "LIG": 0.55, "MIG": 0.20, "HIG": 0.05}
        on_scholarship = np.random.random() < schol_probs[income_cat]
        
        # hostelers travel further from home
        if is_hosteler:
            distance_km = np.random.lognormal(mean=4.5, sigma=0.8)
        else:
            distance_km = np.random.lognormal(mean=2.5, sigma=0.7)
        distance_km = min(distance_km, 500)
        
        # prev academic score (10th/12th %)
        if is_first_gen:
            prev_score = np.clip(np.random.normal(58, 12), 33, 95)
        else:
            prev_score = np.clip(np.random.normal(68, 10), 33, 98)
        
        # TODO: maybe add more branches later
        branch = np.random.choice([
            "Computer Science", "Mechanical", "Electrical",
            "Civil", "Electronics", "Chemical"
        ], p=[0.25, 0.20, 0.18, 0.15, 0.12, 0.10])
        
        students.append({
            "student_id": student_id,
            "age": age,
            "gender": gender,
            "branch": branch,
            "is_orphan": is_orphan,
            "has_guardian": has_guardian,
            "is_first_gen": is_first_gen,
            "income_category": income_cat,
            "is_hosteler": is_hosteler,
            "on_scholarship": on_scholarship,
            "distance_from_home_km": round(distance_km, 1),
            "prev_academic_score": round(prev_score, 1),
        })
    
    return pd.DataFrame(students)


def compute_base_risk(row):
    risk = 0.08  # baseline

    if row["is_orphan"]:
        risk += 0.25
    if not row["has_guardian"]:
        risk += 0.15
    if row["is_first_gen"]:
        risk += 0.10
    if row["income_category"] == "BPL":
        risk += 0.12
    elif row["income_category"] == "LIG":
        risk += 0.06
    if row["on_scholarship"]:
        risk -= 0.05  # helps retention
    if row["prev_academic_score"] < 50:
        risk += 0.15
    elif row["prev_academic_score"] < 60:
        risk += 0.08
    if row["age"] >= 21:
        risk += 0.07
    
    # might need to tune these thresholds with real data
    return np.clip(risk, 0.02, 0.85)


def generate_semester_records(profiles):
    records = []
    
    for _, student in profiles.iterrows():
        base_risk = compute_base_risk(student)
        dropped = False
        cumulative_backlog = 0
        
        for sem in range(1, SEMESTERS + 1):
            if dropped:
                break
            
            # attendance degrades over semesters for at-risk students
            mean_att = 85 - (base_risk * 40) + np.random.normal(0, 5)
            mean_att -= (sem - 1) * base_risk * 5
            attendance_pct = np.clip(mean_att + np.random.normal(0, 8), 10, 100)
            
            # IA scores (out of 40)
            num_subjects = np.random.choice([5, 6, 7])
            subject_scores = []
            for _ in range(num_subjects):
                mean_score = 28 - (base_risk * 15) + np.random.normal(0, 3)
                mean_score -= (sem - 1) * base_risk * 2
                score = np.clip(mean_score + np.random.normal(0, 5), 2, 40)
                subject_scores.append(round(score, 1))
            
            avg_ia_score = np.mean(subject_scores)
            min_ia_score = min(subject_scores)
            subjects_below_pass = sum(1 for s in subject_scores if s < 14)  # 35% of 40
            cumulative_backlog += subjects_below_pass
            
            # fee payment
            fee_delay_prob = base_risk * 0.6 + (0.1 if student["income_category"] in ["BPL", "LIG"] else 0)
            fee_on_time = np.random.random() > fee_delay_prob
            fee_delay_days = 0 if fee_on_time else int(np.random.exponential(30) + 5)
            fee_defaulted = fee_delay_days > 60
            
            # behavioral stuff
            lib_visits = max(0, int(np.random.normal(8 - base_risk * 10, 3)))
            extra_curricular = np.random.random() > (0.3 + base_risk * 0.4)
            
            # counselor visits mostly for orphans/guardianless
            counselor_visits = 0
            if student["is_orphan"] or not student["has_guardian"]:
                if base_risk > 0.3:
                    counselor_visits = np.random.poisson(1.5)
            
            records.append({
                "student_id": student["student_id"],
                "semester": sem,
                "attendance_pct": round(attendance_pct, 1),
                "num_subjects": num_subjects,
                "avg_ia_score": round(avg_ia_score, 1),
                "min_ia_score": round(min_ia_score, 1),
                "subjects_failed": subjects_below_pass,
                "cumulative_backlog": cumulative_backlog,
                "fee_paid_on_time": fee_on_time,
                "fee_delay_days": fee_delay_days,
                "fee_defaulted": fee_defaulted,
                "library_visits_monthly": lib_visits,
                "extracurricular_active": extra_curricular,
                "counselor_visits": counselor_visits,
            })
            
            # does the student drop this sem?
            sem_risk = base_risk
            if attendance_pct < 50:
                sem_risk += 0.20
            if avg_ia_score < 15:
                sem_risk += 0.15
            if fee_defaulted:
                sem_risk += 0.15
            if cumulative_backlog >= 4:
                sem_risk += 0.10
            
            sem_risk = np.clip(sem_risk, 0, 0.95)
            if np.random.random() < sem_risk * 0.4:  # scaled down
                dropped = True
    
    records_df = pd.DataFrame(records)
    
    # if student didn't finish all 6 sems -> dropout
    student_max_sem = records_df.groupby("student_id")["semester"].max().reset_index()
    student_max_sem.columns = ["student_id", "last_semester"]
    student_max_sem["dropped_out"] = student_max_sem["last_semester"] < SEMESTERS
    
    records_df = records_df.merge(
        student_max_sem[["student_id", "dropped_out"]],
        on="student_id"
    )
    return records_df


def main():
    print("Generating student profiles...")
    profiles = generate_student_profiles(NUM_STUDENTS)
    profiles.to_csv(OUTPUT_DIR / "student_profiles.csv", index=False)
    print(f"  {len(profiles)} profiles saved")
    
    print("Generating semester records...")
    records = generate_semester_records(profiles)
    records.to_csv(OUTPUT_DIR / "semester_records.csv", index=False)
    print(f"  {len(records)} semester records saved")
    
    # merge into one ML-ready dataset
    print("Creating ML-ready dataset...")
    
    agg = records.groupby("student_id").agg(
        avg_attendance=("attendance_pct", "mean"),
        min_attendance=("attendance_pct", "min"),
        latest_attendance=("attendance_pct", "last"),
        avg_ia_score=("avg_ia_score", "mean"),
        min_ia_score_overall=("min_ia_score", "min"),
        total_subjects_failed=("subjects_failed", "sum"),
        max_cumulative_backlog=("cumulative_backlog", "max"),
        fee_defaults_count=("fee_defaulted", "sum"),
        avg_fee_delay=("fee_delay_days", "mean"),
        max_fee_delay=("fee_delay_days", "max"),
        avg_library_visits=("library_visits_monthly", "mean"),
        extracurricular_rate=("extracurricular_active", "mean"),
        total_counselor_visits=("counselor_visits", "sum"),
        semesters_completed=("semester", "max"),
        dropped_out=("dropped_out", "first"),
    ).reset_index()
    
    ml_data = profiles.merge(agg, on="student_id")
    ml_data.to_csv(OUTPUT_DIR / "ml_ready_dataset.csv", index=False)
    print(f"  {len(ml_data)} students in ML dataset")
    
    dropout_rate = ml_data["dropped_out"].mean()
    orphan_dropout = ml_data[ml_data["is_orphan"]]["dropped_out"].mean()
    firstgen_dropout = ml_data[ml_data["is_first_gen"]]["dropped_out"].mean()
    
    stats = {
        "total_students": len(ml_data),
        "overall_dropout_rate": round(dropout_rate, 3),
        "orphan_dropout_rate": round(orphan_dropout, 3),
        "first_gen_dropout_rate": round(firstgen_dropout, 3),
        "total_semester_records": len(records),
        "features": list(ml_data.columns),
    }
    with open(OUTPUT_DIR / "data_summary.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDataset Summary:")
    print(f"  Overall dropout rate: {dropout_rate:.1%}")
    print(f"  Orphan dropout rate:  {orphan_dropout:.1%}")
    print(f"  First-gen dropout:    {firstgen_dropout:.1%}")
    print("Done")


if __name__ == "__main__":
    main()
