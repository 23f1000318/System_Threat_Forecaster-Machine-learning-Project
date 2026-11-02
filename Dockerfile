FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model/model.joblib

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py train.py ./
COPY model ./model

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
