"""REST API serving the star-rating model.

    uvicorn mlops.api:app --port 8000       # from the repository root, after mlops/train.py
    curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
         -d '{"text": "Claim settled in two days, very helpful advisor"}'

Interactive docs at http://localhost:8000/docs. MODEL_DIR (default: models/) points to the
folder written by the training script.
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.environ.get('MODEL_DIR', Path(__file__).resolve().parent.parent / 'models'))
state = {}


@asynccontextmanager
async def lifespan(app):
    # Load once at startup: a missing model is a deployment error, better seen immediately
    state['model'] = joblib.load(MODEL_DIR / 'rating_model.joblib')
    state['meta'] = json.loads((MODEL_DIR / 'model_meta.json').read_text())
    # A pickled scikit-learn model is only safe to load with the version that saved it
    if state['meta']['sklearn_version'] != sklearn.__version__:
        raise RuntimeError(f"model saved with scikit-learn {state['meta']['sklearn_version']}, "
                           f"API runs {sklearn.__version__}: align the versions or retrain")
    yield
    state.clear()


app = FastAPI(title='Insurance review rating API', version='1.0.0', lifespan=lifespan)


class Review(BaseModel):
    text: str = Field(min_length=3, max_length=5000, description='review text, in English')


class Reviews(BaseModel):
    texts: list[Review] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    rating: int
    probabilities: dict[str, float]


def predict_many(texts):
    model = state['model']
    probas = model.predict_proba(texts)
    classes = [int(c) for c in model.classes_]
    return [Prediction(rating=classes[p.argmax()],
                       probabilities={str(c): round(float(v), 4) for c, v in zip(classes, p)})
            for p in probas]


@app.get('/health')
def health():
    return {'status': 'ok', 'model_run_id': state['meta']['run_id']}


@app.get('/model')
def model_info():
    """Version, training date, parameters and test metrics of the served model."""
    return state['meta']


@app.post('/predict', response_model=Prediction)
def predict(review: Review):
    if not review.text.strip():
        raise HTTPException(422, 'empty review')
    return predict_many([review.text])[0]


@app.post('/predict/batch', response_model=list[Prediction])
def predict_batch(reviews: Reviews):
    return predict_many([r.text for r in reviews.texts])
