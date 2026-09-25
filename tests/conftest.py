"""Synthetic reviews: the real dataset is not redistributable, so CI trains on fake data
with a clear link between words and stars (enough to test the pipeline, not the model)."""
import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'mlops'))

WORDS = {
    1: ['scandal', 'refused', 'never', 'thieves', 'avoid', 'worst'],
    2: ['slow', 'disappointed', 'expensive', 'waiting', 'unanswered'],
    3: ['average', 'okay', 'correct', 'but', 'could'],
    4: ['good', 'quick', 'helpful', 'fair', 'recommend'],
    5: ['excellent', 'perfect', 'wonderful', 'thank', 'fast'],
}
FILLER = ['insurance', 'contract', 'claim', 'advisor', 'price', 'car', 'home', 'service']


def make_reviews(n=600, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        star = i % 5 + 1
        words = rng.choices(WORDS[star], k=4) + rng.choices(FILLER, k=6)
        rng.shuffle(words)
        rows.append({'note': float(star), 'avis_cor_en': ' '.join(words), 'avis_cor': 'x'})
    return pd.DataFrame(rows)


@pytest.fixture(scope='session')
def reviews_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp('data') / 'reviews_clean.csv'
    make_reviews().to_csv(path, index=False)
    return path


@pytest.fixture(scope='session')
def trained_model_dir(reviews_csv, tmp_path_factory):
    import train
    out = tmp_path_factory.mktemp('models')
    tracking = f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"
    _, _, passed = train.train(reviews_csv, out, min_macro_f1=0.5, tracking_uri=tracking)
    assert passed
    return out
