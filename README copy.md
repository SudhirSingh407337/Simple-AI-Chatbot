# AI Chatbot & Data Query Interface (MVP)

Minimal FastAPI demo that loads E-commerce Dataset.csv into an in-memory SQLite table and exposes a chat hint endpoint plus a SQL query endpoint.

## Setup

1) Create a virtual environment (optional)

2) Install dependencies:

```
pip install -r requirements.txt
```

3) Run the API:

```
uvicorn app.main:app --reload
```

## Example requests

- Health:

```
curl http://127.0.0.1:8000/health
```

- Dataset info:

```
curl http://127.0.0.1:8000/dataset
```

- Query:

```
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"sql":"SELECT * FROM ecommerce LIMIT 5"}'
```

- Chat:

```
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Show me top countries"}'
```
