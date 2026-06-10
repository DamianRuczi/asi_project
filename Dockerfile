FROM python:3.11-slim

WORKDIR /app

# Najpierw zależności — cache warstw przy kolejnych przebudowach
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Kod aplikacji
COPY app/ app/

# Artefakty modelu NIE są częścią obrazu — docker-compose montuje
# data/06_models/ jako wolumen. Obraz zostaje mały, a model można
# podmienić (po retreningu) bez przebudowy obrazu.
ENV MODEL_PATH=/app/models/model.pkl \
    FEATURES_PATH=/app/models/model_features.json \
    PREDICTIONS_LOG=/app/logs/predictions.jsonl

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
