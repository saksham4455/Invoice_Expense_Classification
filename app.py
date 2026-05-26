from pathlib import Path
import re
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources
nltk_packages = ["stopwords", "wordnet", "omw-1.4"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

repo_root = Path(__file__).parent
artifacts = repo_root / "artifacts"
model_path = artifacts / "model.joblib"
vectorizer_path = artifacts / "vectorizer.joblib"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

app = FastAPI(title="Invoice Expense Classifier", version="1.0.0")

class PredictRequest(BaseModel):
    text: str

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            tokens.append(lemmatizer.lemmatize(token))
    return " ".join(tokens)


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    cleaned = preprocess_text(request.text)
    features = vectorizer.transform([cleaned])
    probs = model.predict_proba(features)[0]
    best = int(probs.argmax())
    return {"category": model.classes_[best], "confidence": float(round(float(probs[best]), 4))}
