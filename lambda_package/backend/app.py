# pyrefly: ignore [missing-import]
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    # Replace with your actual prediction function
    return {"message": "Prediction endpoint reached"}