import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from utils import preprocess_text


def load_data(path):
    df = pd.read_csv(path)
    if 'text' not in df.columns or 'category' not in df.columns:
        raise ValueError("training CSV must contain 'text' and 'category' columns")
    return df['text'].astype(str).tolist(), df['category'].astype(str).tolist()


def train(data_path, output_dir):
    texts, labels = load_data(data_path)
    cleaned = [preprocess_text(t) for t in texts]
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X = vec.fit_transform(cleaned)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)

    os.makedirs(output_dir, exist_ok=True)
    vec_path = os.path.join(output_dir, 'vectorizer.joblib')
    model_path = os.path.join(output_dir, 'model.joblib')
    joblib.dump(vec, vec_path)
    joblib.dump(clf, model_path)
    print(f"Saved vectorizer to {vec_path}")
    print(f"Saved model to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to training CSV')
    parser.add_argument('--output-dir', default='artifacts', help='Directory to save artifacts')
    args = parser.parse_args()
    train(args.data, args.output_dir)


if __name__ == '__main__':
    main()
