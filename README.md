# AI-Based Dropout Prediction and Counseling System

Coursework project for AI course - SIH25102 problem statement.

Predicts student dropout risk using ML and routes at-risk students to the right counseling track (academic, welfare, or career guidance) based on their vulnerability profile.

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # windows
# source venv/bin/activate  # mac/linux

pip install -r requirements.txt
```

## Running

```bash
# 1. generate synthetic data
python scripts/generate_data.py

# 2. train models
python scripts/train_pipeline.py

# 3. launch dashboard
streamlit run dashboard/app.py

# 4. (optional) start api
uvicorn api.main:app --reload
```

## Project structure

```
dropout-prediction/
├── api/
│   └── main.py              # FastAPI backend
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── data/                     # generated after running scripts
├── models/                   # saved after training
├── notebooks/
│   └── EDA.ipynb             # exploratory data analysis
├── scripts/
│   ├── generate_data.py      # synthetic data generation
│   └── train_pipeline.py     # ML training pipeline
├── requirements.txt
└── README.md
```

## Models compared

- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Support Vector Machine
- Neural Network (MLP)

Best model selected by F1 score. SHAP used for explainability.

## Counseling tracks

| Track | Who gets routed here | Alert goes to |
|-------|---------------------|---------------|
| Academic | Low grades / attendance | Subject mentors |
| Welfare | Orphans, no guardian, BPL | Welfare officer |
| Career Guidance | Older students, disengaging | Career advisor |
| Monitoring | Low risk | Standard review |

## Tech stack

Python, scikit-learn, SHAP, Streamlit, Plotly, FastAPI, Pandas
