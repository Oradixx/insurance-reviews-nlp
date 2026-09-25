# Model card: insurance review star rating

| | |
|---|---|
| **Model** | scikit-learn Pipeline: TF-IDF (5,000 terms, English stop words removed) + multinomial logistic regression |
| **Task** | predict the star rating (1 to 5) of an insurance customer review |
| **Input** | review text in **English** (the training reviews are French reviews machine-translated to English) |
| **Output** | predicted rating and the probability of each rating |
| **Code** | [`train.py`](train.py) (training, MLflow tracking, quality gate), [`api.py`](api.py) (FastAPI) |

## Training data

About 24,000 French reviews of insurance companies provided for the NLP course (24,069 after removing empty and duplicate
reviews), translated to English. Not redistributed: they contain real reviews with the authors' usernames. For the same
reason, the **trained model is not published**; anyone with the data can rebuild it with `python mlops/train.py`.

Split: 80 % train / 20 % test (4,814 reviews), `random_state=42`, same as the notebook.

## Evaluation (test set, 4,814 reviews)

| Rating | 1 | 2 | 3 | 4 | 5 | **Accuracy** | **Macro F1** |
|---|---|---|---|---|---|---|---|
| F1 | 0.72 | 0.26 | 0.23 | 0.42 | 0.55 | **0.52** | **0.43** |

The script reproduces the notebook result exactly. Quality gate: a model below 0.40 macro F1 is not saved and the script
exits with an error, so it never replaces the served model.

## Intended use and limits

- A **triage signal** for review monitoring (for example flagging likely 1-star reviews), not an automatic decision.
- **Middle ratings are unreliable** (F1 about 0.25 for 2 and 3 stars): customers often describe a bad event and a correct service
  in the same review.
- **English only**: the vocabulary was learned on English translations, so French input gives unreliable predictions.
- Trained on one period and one set of insurers: new products or wording (drift) will lower accuracy. The test metrics are
  stored with the model (`GET /model`) to compare against later.
- Single split, no hyperparameter search. The notebook's MiniLM + neural network model is slightly better
  (0.54 accuracy, 0.46 macro F1) but needs a sentence-transformer at inference; the linear model is served because it is
  small, fast on CPU and explainable (SHAP analysis in the notebook).
