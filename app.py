
import joblib
import pandas as pd
import csv
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Literal


app = FastAPI(title="Student Performance Predictor")

MODEL_PATH = "model/model.joblib"
LOG_PATH = "prediction_logs.csv"

# Load model once at application startup
model = joblib.load(MODEL_PATH)


class StudentInput(BaseModel):
    Hours_Studied: float = Field(ge=0, le=24)
    Attendance: float = Field(ge=0, le=100)
    Sleep_Hours: float = Field(ge=0, le=24)
    Previous_Scores: float = Field(ge=0, le=100)
    Tutoring_Sessions: int = Field(ge=0)
    Physical_Activity: float = Field(ge=0)

    Extracurricular_Activities: Literal["No", "Yes"]
    Internet_Access: Literal["No", "Yes"]
    School_Type: Literal["Private", "Public"]
    Learning_Disabilities: Literal["No", "Yes"]
    Gender: Literal["Female", "Male"]
    Parental_Involvement: Literal["High", "Low", "Medium"]
    Access_to_Resources: Literal["High", "Low", "Medium"]
    Motivation_Level: Literal["High", "Low", "Medium"]
    Family_Income: Literal["High", "Low", "Medium"]
    Teacher_Quality: Literal["High", "Low", "Medium"]
    Peer_Influence: Literal["Negative", "Neutral", "Positive"]
    Distance_from_Home: Literal["Far", "Moderate", "Near"]
    Parental_Education_Level: Literal["College", "High School", "Postgraduate"]


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

    # Generate prediction
    prediction = float(model.predict(input_df)[0])

    # Feature contribution explanations
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

    # Log prediction inputs
    file_exists = Path(LOG_PATH).exists()

    with open(LOG_PATH, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

    # Score-based performance tip
    if prediction >= 90:
        tip = "Excellent performance. Maintain your study routine."
    elif prediction >= 75:
        tip = "Good performance. Focus on consistent revision."
    elif prediction >= 50:
        tip = "More practice and focused study may improve your score."
    else:
        tip = "Consider seeking academic support and improving study habits."

    # Personalized rule-based recommendations
    
    # Personalized rule-based recommendations
    recommendations = []

    # 1. Attendance — strongest numeric model coefficient
    if data["Attendance"] < 70:
        recommendations.append(
            "Improve class attendance to strengthen concept understanding and avoid missing lessons."
        )

    # 2. Study hours — second strongest numeric model coefficient
    if data["Hours_Studied"] < 16:
        recommendations.append(
            "Increase weekly study time gradually and follow a structured revision schedule."
        )

    # 3. Sleep
    if data["Sleep_Hours"] < 6:
        recommendations.append(
            "Aim for at least 7 hours of sleep to support concentration and learning."
        )

    # 4. Previous academic performance
    if data["Previous_Scores"] < 63:
        recommendations.append(
            "Strengthen foundational concepts through topic-wise revision and regular practice."
        )

    # 5. Motivation
    if data["Motivation_Level"] == "Low":
        recommendations.append(
            "Set small, achievable daily learning goals to build consistency and motivation."
        )

    # 6. Access to resources
    if data["Access_to_Resources"] == "Low":
        recommendations.append(
            "Use library materials, online learning resources, and class notes to support your studies."
        )

    # 7. Additional academic support
    if data["Previous_Scores"] < 63 and data["Tutoring_Sessions"] == 0:
        recommendations.append(
            "Consider seeking academic guidance for subjects or topics you find difficult."
        )

    # Contextual recommendations when fewer than three weaknesses are found
    if len(recommendations) < 3:
        if data["Attendance"] >= 70:
            recommendations.append(
                "Maintain consistent attendance and participate actively in class."
            )

        if data["Hours_Studied"] >= 16:
            recommendations.append(
                "Continue your study routine and use practice questions to reinforce learning."
            )

        if data["Previous_Scores"] >= 63:
            recommendations.append(
                "Track your academic progress and set measurable improvement goals."
            )

        if data["Sleep_Hours"] >= 6:
            recommendations.append(
                "Maintain a balanced routine with sufficient rest and regular physical activity."
            )
            
    tips = recommendations[:3]

    
    return {
        "predicted_exam_score": round(prediction, 2),
        "tip": tip,
        "tips": tips,
        "explanations": explanations
    }