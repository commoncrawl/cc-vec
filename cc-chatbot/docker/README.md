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


## Populating a vector store

Spin up a llama-stack instance where you want the vector store to live:

```bash
uv run --with llama-stack==0.4.1 llama stack run starter
```

Wait until you see the `Uvicorn running on <url>` message. Then, test everything works:

```bash
# Set environment variables        
export OPENAI_BASE_URL=http://localhost:8321/v1
export OPENAI_API_KEY=none # Llama Stack doesn't require a real key
export OPENAI_EMBEDDING_MODEL=sentence-transformers/nomic-ai/nomic-embed-text-v1.5
export OPENAI_EMBEDDING_DIMENSIONS=768

# Set your Athena credentials
export ATHENA_OUTPUT_BUCKET=s3://cc-vec-damian-01/test-results
export AWS_PROFILE=cc-volunteers
export AWS_DEFAULT_REGION=us-east-1

# Use cc-vec with local models
uv run cc-vec index --url-patterns "%commoncrawl.org" --limit 10
```

If it succeeds run again with `--limit 1000` to index everything 


> Note: if running debug locally, llama stack needs additional pip packages `sentence-transformers einops`
