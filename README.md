# Invoice Expense Classification

## Setup

```bash
pip install fastapi uvicorn scikit-learn pandas joblib nltk pytest requests
```

## Train

Preferred: run the training notebook which produces the model and vectorizer in `artifacts/`:

```bash
# Start the notebook UI and run the cells
jupyter notebook invoice_expense_classification.ipynb

# Or run headless (executes all cells and writes an output notebook)
jupyter nbconvert --to notebook --execute invoice_expense_classification.ipynb --output artifacts/training_run.ipynb
```

If you have a training script instead, it should create `artifacts/model.joblib` and `artifacts/vectorizer.joblib`:

```bash
python train.py --data data/training_data.csv --output-dir artifacts
```

Ensure the `artifacts/` directory exists (it is gitignored) before running training.

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