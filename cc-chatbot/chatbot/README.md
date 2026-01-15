# Minimal CC Chatbot Frontend

## Quickstart

Launch ollama server, then run:

```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

Click on the link that uvicorn prints in your terminal to open the frontend.

## Configuration

Configuration is done via environment variables in `api.py`. Defaults:

```bash
OLLAMA_URL=http://localhost:11434 # /api/generate is appended
INFERENCE_MODEL=tinyllama
STREAMING=1
````
