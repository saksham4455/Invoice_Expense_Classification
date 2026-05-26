# Invoice Expense Classification

## Setup

```bash
pip install fastapi uvicorn scikit-learn pandas joblib nltk pytest requests
```

## Train

```bash
python train.py --data data/training_data.csv --output-dir artifacts
```

## Run API

```bash
uvicorn app:app --reload --port 8000
```

## Predict

```bash
curl -X POST "http://127.0.0.1:8000/predict"   -H "Content-Type: application/json"   -d '{"text": "AWS monthly cloud hosting bill"}'
```

## Docker

```bash
docker build -t invoice-classifier .
docker run -p 8000:8000 invoice-classifier
```

## Tests

```bash
pytest -q
```