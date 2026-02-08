FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY frontend/ ./frontend/

WORKDIR /app/api

ENV PORT=8000

CMD uvicorn main:app --host 0.0.0.0 --port $PORT
