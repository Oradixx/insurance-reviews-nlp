"""Train the star-rating model (TF-IDF + logistic regression), track the run with MLflow and
save the model only if it passes the quality gate.

Same model, split and settings as section 2 of the notebook, packaged as one scikit-learn
Pipeline so the API receives raw text and applies the exact same vectoriser.

Usage (from the repository root):
    python mlops/train.py                       # data/processed/reviews_clean.csv
    python mlops/train.py --data other.csv --min-macro-f1 0.40
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs

Exit code 1 when the model misses the gate: CI and scripts can stop there.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent

PARAMS = {
    'text_column': 'avis_cor_en',     # spell-checked review, translated to English
    'label_column': 'note',           # 1 to 5 stars
    'test_size': 0.2,
    'random_state': 42,
    'max_features': 5000,
    'stop_words': 'english',
    'max_iter': 1000,
}


def build_pipeline(p=PARAMS):
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=p['max_features'], stop_words=p['stop_words'])),
        ('clf', LogisticRegression(max_iter=p['max_iter'])),
    ])


def load(path, p=PARAMS):
    df = pd.read_csv(path, usecols=[p['text_column'], p['label_column']])
    df = df.dropna(subset=[p['label_column']])
    X = df[p['text_column']].fillna('')
    y = df[p['label_column']].astype(int)
    return X, y


def train(data, out_dir, min_macro_f1, tracking_uri, experiment='star-rating'):
    X, y = load(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=PARAMS['test_size'], random_state=PARAMS['random_state'])

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(PARAMS | {'n_train': len(X_train), 'n_test': len(X_test),
                                    'min_macro_f1': min_macro_f1})
        model = build_pipeline().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'macro_f1': f1_score(y_test, y_pred, average='macro'),
            'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
        } | {f'f1_{c}_star': report[str(c)]['f1-score'] for c in sorted(y.unique())}
        mlflow.log_metrics(metrics)

        passed = metrics['macro_f1'] >= min_macro_f1
        mlflow.set_tag('quality_gate', 'passed' if passed else 'failed')
        print(classification_report(y_test, y_pred, zero_division=0))
        print(f"macro F1 {metrics['macro_f1']:.3f} (gate {min_macro_f1}): {'passed' if passed else 'FAILED'}")
        if not passed:
            return run.info.run_id, metrics, False

        # Only a model that passed the gate replaces the one the API serves
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, out_dir / 'rating_model.joblib')
        meta = {
            'run_id': run.info.run_id,
            'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'sklearn_version': sklearn.__version__,
            'params': PARAMS,
            'metrics': {k: round(v, 4) for k, v in metrics.items()},
        }
        (out_dir / 'model_meta.json').write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(out_dir / 'model_meta.json'))
        # The pipeline is logged too, so any run can be reloaded from MLflow later
        mlflow.sklearn.log_model(model, name='model')
        return run.info.run_id, metrics, True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', type=Path, default=ROOT / 'data' / 'processed' / 'reviews_clean.csv')
    ap.add_argument('--out', type=Path, default=ROOT / 'models')
    ap.add_argument('--min-macro-f1', type=float, default=0.40,
                    help='quality gate on the test set (the notebook model scores 0.43)')
    ap.add_argument('--tracking-uri', default=f"sqlite:///{ROOT / 'mlflow.db'}",
                    help='MLflow backend (runs in mlflow.db, artifacts in mlruns/)')
    args = ap.parse_args()

    _, _, passed = train(args.data, args.out, args.min_macro_f1, args.tracking_uri)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
