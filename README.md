# Invoice Expense Classification

A small FastAPI service that classifies invoice text into expense categories.

**Repository layout**
- `app.py` — FastAPI app exposing `POST /predict`.
- `invoice_expense_classification.ipynb` — training notebook (preferred for training).
- `artifacts/` — saved model and vectorizer (gitignored).
- `data/training_data.csv` — sample training data.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Train (create artifacts)

Preferred: run the notebook which produces `artifacts/model.joblib` and `artifacts/vectorizer.joblib`.

```bash
# Start the notebook UI and run the cells interactively
jupyter notebook invoice_expense_classification.ipynb

# Or run headless (executes all cells and writes an output notebook)
jupyter nbconvert --to notebook --execute invoice_expense_classification.ipynb --output artifacts/training_run.ipynb
```

Note: `artifacts/` is gitignored. Create it locally (the notebook writes outputs there) before starting the API.

## Run API (development)

Ensure `artifacts/model.joblib` and `artifacts/vectorizer.joblib` exist, then:

```bash
uvicorn app:app --reload --port 8000
```

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{"text": "AWS monthly cloud hosting bill"}'
```

Response includes `category` and `confidence`.

## Docker

Build and run the container (artifacts are not included in the image):

```bash
docker build -t invoice-classifier .
# create artifacts locally first, or mount them into the container
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts invoice-classifier
```

## Tests

Run unit tests locally:

```bash
pytest -q
```

Current tests cover preprocessing. If you add model artifacts, consider adding an API smoke test.

## Continuous Integration (suggested)

Add a GitHub Actions workflow to run `pytest` on pushes and pull requests. Example workflow path: `.github/workflows/python.yml`.

## Notes

- `artifacts/` is intentionally ignored to avoid committing large model files. Add your trained `model.joblib` and `vectorizer.joblib` to that folder locally before running the API.
- Training is performed in `invoice_expense_classification.ipynb` — keep the notebook as the canonical training flow.
