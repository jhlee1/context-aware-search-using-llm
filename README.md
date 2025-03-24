# This is an example project for a context-aware searching by building RAG using LangChain and ChromaDB

## Setup

My python version is 3.13.2

### Install dependencies with requirements.txt

```bash
pip install -r requirements.txt
```

### Install dependencies manually

```bash
python3 -m venv venv
source venv/bin/activate
pip install slack-sdk langchain chromadb sentence-transformers pydantic python-dotenv fastapi uvicorn pydantic
```

## How to run without API server

```bash
python3 main.py ingest --channels <channel_id>
python3 main.py search --query <query>
```

### Examples

```bash
python3 main.py ingest --channels C0600000000
python3 main.py search --query "I'm having trouble with the app"
```

## How to run the API server

1. Normal

```bash
python3 api_server.py
```

2. With hot reloading

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

3. With hot reloading and debug

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload --debug
```

## Example API calls

1. Ingest data

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"channels": ["C01234ABCDE", "C04321EDCBA"], "limit": 500}'
```

2. Ingest data (sync)

```bash
curl -X POST http://localhost:8000/ingest/sync \
  -H "Content-Type: application/json" \
  -d '{"channels": ["C01234ABCDE"]}'
```

3. Search

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "I'm having trouble with the app"}'
```

4. Get status

```bash
curl http://localhost:8000/status
```

## Dockerfile for Cuda

```Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

CMD ["python3", "api_server.py"]
```
