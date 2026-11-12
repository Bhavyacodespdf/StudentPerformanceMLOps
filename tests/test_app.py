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
    body = response.json()
    assert "predicted_exam_score" in body
    assert "tip" in body
    assert "explanations" in body
    assert isinstance(body["explanations"], list)
    assert len(body["explanations"]) > 0
    assert "feature" in body["explanations"][0]
    assert "contribution" in body["explanations"][0]


def test_invalid_prediction():
    payload = {
        "Hours_Studied": 30,
        "Attendance": 90
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    

def test_stronger_student_gets_higher_prediction():
    strong_payload = {
        "Hours_Studied": 20,
        "Attendance": 95,
        "Sleep_Hours": 8,
        "Previous_Scores": 90,
        "Tutoring_Sessions": 3,
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

    weak_payload = strong_payload.copy()
    weak_payload.update({
        "Hours_Studied": 2,
        "Attendance": 50,
        "Previous_Scores": 40
    })

    strong_response = client.post("/predict", json=strong_payload)
    weak_response = client.post("/predict", json=weak_payload)

    assert strong_response.status_code == 200
    assert weak_response.status_code == 200

    assert (
        strong_response.json()["predicted_exam_score"]
        > weak_response.json()["predicted_exam_score"]
    )
    
def test_invalid_categorical_input():
    payload = {
        "Hours_Studied": 20,
        "Attendance": 90,
        "Sleep_Hours": 8,
        "Previous_Scores": 80,
        "Tutoring_Sessions": 2,
        "Physical_Activity": 3,
        "Extracurricular_Activities": "Yes",
        "Internet_Access": "Yes",
        "School_Type": "Invalid",
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

    assert response.status_code == 422
