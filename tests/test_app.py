from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_homepage():
    response = client.get("/")
    assert response.status_code == 200
    assert "Student Performance Predictor" in response.text


def test_prediction():
    payload = {
        "Hours_Studied": 20,
        "Attendance": 90,
        "Sleep_Hours": 8,
        "Previous_Scores": 80,
        "Tutoring_Sessions": 2,
        "Physical_Activity": 3,
        "Extracurricular_Activities": "Yes",
        "Internet_Access": "Yes",
        "School_Type": "Public",
        "Learning_Disabilities": "No",
        "Gender": "Female",
        "Parental_Involvement": "High",
        "Access_to_Resources": "High",
        "Motivation_Level": "High",
        "Family_Income": "Medium",
        "Teacher_Quality": "High",
        "Peer_Influence": "Positive",
        "Distance_from_Home": "Near",
        "Parental_Education_Level": "College"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_exam_score" in response.json()
    assert "tip" in response.json()


def test_invalid_prediction():
    payload = {
        "Hours_Studied": 30,
        "Attendance": 90
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422