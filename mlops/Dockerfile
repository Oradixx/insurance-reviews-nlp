# API image. Build from the repository root after training (the model is not in git):
#   python mlops/train.py
#   docker build -f mlops/Dockerfile -t reviews-rating-api .
#   docker run --rm -p 8000:8000 reviews-rating-api
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY mlops/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY mlops/api.py mlops/api.py
COPY models/rating_model.joblib models/model_meta.json models/

# Non-root user
RUN useradd --create-home --uid 10001 api
USER api

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]
