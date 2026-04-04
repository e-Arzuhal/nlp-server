FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model at build time so startup is fast
RUN python -c "from transformers import pipeline; \
    pipeline('ner', model='savasy/bert-base-turkish-ner-cased', \
    aggregation_strategy='simple')"

COPY . .

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
