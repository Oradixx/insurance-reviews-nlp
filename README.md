# Insurance Reviews NLP

Predicting star ratings, finding topics and answering questions on **~24,000 French insurance customer reviews**, with a Streamlit dashboard powered by Hugging Face models and **Llama 3 running locally**.

School project (NLP course, ESILV, March 2026), made by two students.

[![CI](https://github.com/Oradixx/insurance-reviews-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/Oradixx/insurance-reviews-nlp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Llama%203-000000)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)

## What it does

**Notebook** ([`notebooks/insurance_reviews_nlp.ipynb`](notebooks/insurance_reviews_nlp.ipynb)):

1. **Cleaning and translation**: regex cleaning, French spelling correction (`pyspellchecker` with a word cache), then translation of every review to English (`deep-translator`, 15 threads), so English pre-trained models can be used.
2. **Exploration**: rating distribution, most reviewed insurers, word cloud.
3. **Baseline**: TF-IDF + logistic regression, explained with **SHAP**.
4. **Topic modeling**: LDA with 5 topics.
5. **Deep learning**: `paraphrase-multilingual-MiniLM-L12-v2` sentence embeddings + a small PyTorch classifier (2 hidden layers, dropout).
6. **Error analysis** of the worst predictions (3 stars or more off).
7. **Word2Vec** trained on the corpus, **semantic search** with cosine similarity, and an export for the TensorFlow Embedding Projector.

**Streamlit app** ([`app/app.py`](app/app.py)), 4 pages:

| Page | How it works |
|---|---|
| Global dashboard | Dataset statistics |
| Prediction | Star rating with multilingual BERT (`nlptown/bert-base-multilingual-uncased-sentiment`) + main subject (Pricing, Coverage, Claims…) with zero-shot `facebook/bart-large-mnli` |
| Summary by insurer | Llama 3 summarises the insurer's reviews in two sentences |
| RAG & QA | The question is embedded with MiniLM, the 5 closest English reviews are retrieved by cosine similarity, and Llama 3 answers only from them (sources shown) |

[`app/app_lite.py`](app/app_lite.py) is the same app without Ollama: BART-XSum for summaries and Flan-T5 for answers. We kept Llama 3 for the main app because Flan-T5 hallucinated more in our tests.

## Results

Rating prediction (1 to 5 stars), test set of 4,814 reviews:

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + logistic regression | 0.52 | 0.43 |
| MiniLM embeddings + neural network | **0.54** | **0.46** |

- The embedding model is only slightly better than the baseline.
- Most errors are off by one star: **89 %** of predictions are within ±1 star of the true rating (computed from the confusion matrix below).
- 2- and 3-star reviews are the hardest (F1 around 0.25): customers often describe a bad event but a correct service, or the opposite.

<p align="center">
  <img src="docs/images/confusion_matrix_nn.png" width="48%" alt="Confusion matrix of the neural network">
  <img src="docs/images/shap_tfidf.png" width="48%" alt="SHAP summary plot of the TF-IDF model for the 1-star class">
</p>

On the right, SHAP for the 1-star class of the baseline: *satisfied*, *good* and *thank* push away from 1 star; *avoid* and *flee* push towards it. *Flee* is a translation of the French *à fuir* ("stay away").

## Serving the model (MLOps)

Added after the course (September 2026): the TF-IDF baseline goes from notebook cell to a served, versioned model.

```
reviews_clean.csv ──► mlops/train.py ──► quality gate ──► models/ ──► Docker image ──► FastAPI /predict
                         │ (macro F1 ≥ 0.40)
                         └──► MLflow: parameters, metrics, model of every run
```

- **Training** ([`mlops/train.py`](mlops/train.py)): the vectoriser and the classifier are one scikit-learn `Pipeline`, so the API
  applies exactly the same preprocessing. Each run is tracked in **MLflow** (parameters, accuracy, macro F1, F1 per rating, model).
  It reproduces the notebook exactly: accuracy 0.52, macro F1 0.43.
- **Quality gate**: below 0.40 macro F1 the model is not saved and the script exits with an error.
- **API** ([`mlops/api.py`](mlops/api.py)): `POST /predict` (rating + probabilities), `POST /predict/batch`, `GET /model`
  (version, training date, test metrics), `GET /health`. Input is validated (length, empty text), and the API refuses to start
  if the model was saved with another scikit-learn version, because a pickled model is only safe with the version that saved it.
- **Docker**: slim image with only the serving dependencies, non-root user, health check.
- **CI** ([`ci.yml`](.github/workflows/ci.yml)): the reviews cannot be published, so CI trains on **synthetic reviews**, runs the
  tests (training, gate, reproducibility, API), builds the image and queries the running container.
- **Model card**: [`mlops/MODEL_CARD.md`](mlops/MODEL_CARD.md) (data, metrics, intended use, limits).

```bash
pip install -r mlops/requirements.txt
python mlops/train.py                     # needs data/processed/reviews_clean.csv (from the notebook)
mlflow ui --backend-store-uri sqlite:///mlflow.db     # runs at http://localhost:5000

docker build -f mlops/Dockerfile -t reviews-rating-api .
docker run --rm -p 8000:8000 reviews-rating-api       # docs at http://localhost:8000/docs
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{"text": "They refused my claim and never answered my emails."}'
```

Without the dataset, `python mlops/make_sample.py data/processed/sample_reviews.csv` writes synthetic reviews to try the whole chain
(add `--data data/processed/sample_reviews.csv` to the training command).

## Project structure

```
├── notebooks/
│   └── insurance_reviews_nlp.ipynb   # full analysis, with the outputs of the original run
├── app/
│   ├── app.py                        # Streamlit app (Hugging Face + Llama 3 via Ollama)
│   ├── app_lite.py                   # same app, Hugging Face models only
│   └── common.py                     # data loading, English filter, shared pages
├── data/
│   ├── raw/                          # course dataset (not included)
│   └── processed/                    # created by the notebook (not included)
├── mlops/
│   ├── train.py                      # training + MLflow tracking + quality gate
│   ├── api.py                        # FastAPI service
│   ├── Dockerfile                    # API image
│   └── MODEL_CARD.md
├── tests/                            # pytest on synthetic reviews (run in CI)
├── docs/images/                      # figures used in this README
└── requirements.txt
```

## Run it

```bash
git clone https://github.com/Oradixx/insurance-reviews-nlp.git
cd insurance-reviews-nlp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. **Data**: the dataset was provided by the course and is not redistributed here (see [`data/README.md`](data/README.md)). Put the `avis_*_traduit*.xlsx` files in `data/raw/`.
2. **Notebook**: run `notebooks/insurance_reviews_nlp.ipynb`. It writes `reviews_clean.csv` and `review_embeddings.pkl` to `data/processed/`. A GPU is recommended: we used a Colab T4. On Colab, set `DATA_DIR` in the first cell.
3. **App**:
   ```bash
   # main app: needs Ollama (https://ollama.com)
   ollama pull llama3
   streamlit run app/app.py

   # or without Ollama
   streamlit run app/app_lite.py
   ```

## Limitations

- **Translation quality**: some reviews stayed in French or mixed languages (translation errors fall back to the original text). One LDA topic is made of French stop words. The app filters those reviews out with a simple word-based check (`is_valid_english`).
- **Spelling correction is not used downstream**: the translation step takes the original review (`avis`), not the corrected one (`avis_cor`), so `avis_cor_en` is a translation of the raw text.
- **Single train/test split** and no hyperparameter search; the neural network runs for 10 epochs.
- **"Keywords" in the app** are a simple heuristic (longer words of the review), not a model explanation; the SHAP analysis is in the notebook.
- The app's star prediction uses a pre-trained multilingual BERT, not the classifier trained in the notebook.

## Authors

- **Clément Vurpillot** — [@Oradixx](https://github.com/Oradixx)
- **Noé Spychala**

Project from the NLP course at ESILV (Data & AI major).
