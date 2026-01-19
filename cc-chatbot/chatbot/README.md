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

## Manual / Development

Make sure ollama is running, then open 2 terminal windows.

In the first, launch llama stack configured to talk to ollama:

```bash
OLLAMA_URL=http://localhost:11434/v1 uv run --with llama-stack==0.4.1 llama stack run starte
```

In the second, launch the cc chatbot:

```bash
cd cc-chatbot/chatbot
OLLAMA_URL=http://localhost:8321 uvicorn api:app --reload
```

## Building the vector store

Make sure ollama is running, then open 2 terminal windows.

In the first, launch llama stack configured to talk to ollama:

```bash
OLLAMA_URL=http://localhost:11434/v1 uv run --with llama-stack==0.4.1 llama stack run starte
```

In the second, run cc-vec:

```bash
uv run cc-vec index --url-patterns "%commoncrawl.org" --limit 1000 --vector-store-name 'commoncrawl-org-v1' --chunk-size 800 --overlap 400
```
