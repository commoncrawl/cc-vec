# CC chatbot docker setup

## Quickstart

To run with default configuration (internal ollama, tinyllama baked into image, chatbot on http://localhost:8008), run:

`docker-compose up --build`

## Configuration

Copy `.env.sample` to `.env` and modify as needed to customize configuration.

Alternatively, set environment variables directly in your shell before running the `docker-compose up` command. For example:
```bash
OLLAMA_URL=http://host.docker.internal:11434 PREFETCH_MODEL=0 INFERENCE_MODEL=llama3.2:3B  docker-compose up --build
```