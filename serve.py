# Serves static/ alongside the API without editing app.py. Run: uvicorn serve:app --reload
from fastapi.staticfiles import StaticFiles

from app import app

app.mount("/static", StaticFiles(directory="static", html=True), name="static")
