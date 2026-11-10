import numpy as np
import joblib
import pandas as pd
import csv
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Student Performance Predictor")

MODEL_PATH = "model/model.joblib"
LOG_PATH = "prediction_logs.csv"

model = joblib.load(MODEL_PATH)


class StudentInput(BaseModel):
    Hours_Studied: float = Field(ge=0, le=24)
    Attendance: float = Field(ge=0, le=100)
    Sleep_Hours: float = Field(ge=0, le=24)
    Previous_Scores: float = Field(ge=0, le=100)
    Tutoring_Sessions: int = Field(ge=0)
    Physical_Activity: float = Field(ge=0)

    Extracurricular_Activities: str
    Internet_Access: str
    School_Type: str
    Learning_Disabilities: str
    Gender: str
    Parental_Involvement: str
    Access_to_Resources: str
    Motivation_Level: str
    Family_Income: str
    Teacher_Quality: str
    Peer_Influence: str
    Distance_from_Home: str
    Parental_Education_Level: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/", response_class=HTMLResponse)
def home():
    path = Path("static/index.html")
    if path.exists():
        return path.read_text()
    return "<h1>Student Performance Predictor</h1><p>Use <a href='/docs'>API documentation</a>.</p>"


@app.post("/predict")
def predict(student: StudentInput):
    data = student.model_dump()
    input_df = pd.DataFrame([data])

    prediction = float(model.predict(input_df)[0])
    
    
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]
    
    transformed_input = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()
    coefficients = regressor.coef_
    
    contributions = transformed_input[0] * coefficients
    
    explanations = [
      {
        "feature": str(feature_names[i]).split("__")[-1],
        "contribution": round(float(contributions[i]), 3)
      }
      for i in range(len(feature_names))
    ]
    
    explanations.sort(
      key=lambda x: abs(x["contribution"]),
      reverse=True
    )
    
    explanations = explanations[:8]
    

    file_exists = Path(LOG_PATH).exists()

    with open(LOG_PATH, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

    if prediction >= 90:
        tip = "Excellent performance. Maintain your study routine."
    elif prediction >= 75:
        tip = "Good performance. Focus on consistent revision."
    elif prediction >= 50:
        tip = "More practice and focused study may improve your score."
    else:
        tip = "Consider seeking academic support and improving study habits."

    return {
    "predicted_exam_score": round(prediction, 2),
    "tip": tip,
    "explanations": explanations
    }