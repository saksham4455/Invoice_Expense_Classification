from pathlib import Path
import re
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from utils import preprocess_text

repo_root = Path(__file__).parent
artifacts = repo_root / "artifacts"
model_path = artifacts / "model.joblib"
vectorizer_path = artifacts / "vectorizer.joblib"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

app = FastAPI(title="Invoice Expense Classifier", version="1.0.0")

class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    cleaned = preprocess_text(request.text)
    features = vectorizer.transform([cleaned])
    probs = model.predict_proba(features)[0]
    best = int(probs.argmax())
    return {"category": model.classes_[best], "confidence": float(round(float(probs[best]), 4))}
