# Minimal Llama Chatbot Frontend

This directory contains a minimal web-based chatbot UI and a FastAPI backend that proxies requests to an Ollama (Llama) backend.

## Usage

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Run the backend**

```bash
uvicorn api:app --reload
```

3. **Open the frontend**

Open `index.html` in your browser (or serve it with any static file server).

## Configuration

- The backend expects an Ollama server running at `http://localhost:11434` by default.
- You can override the Ollama URL and model with environment variables:
  - `OLLAMA_URL` (e.g. `http://localhost:11434/api/generate`)
  - `INFERENCE_MODEL` (e.g. `tinyllama`)

## Notes
- The backend is intentionally minimal and does not persist chat history.
- The frontend is pure HTML/JS, no frameworks or build step required.
