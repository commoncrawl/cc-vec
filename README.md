# CCVec - Common Crawl to Vector Stores

Search, analyze, and index Common Crawl data into vector stores for RAG applications. Three surfaces available:
* CLI
* Python library
* MCP server

## Quick Start

**Environment variables:**

- **`ATHENA_OUTPUT_BUCKET`** - Required S3 bucket for Athena query results (needed for reliable queries to Common Crawl metadata)
- **`AWS_ACCESS_KEY_ID`** - Required for Athena/S3 access (needed to run Athena queries)
- **`AWS_SECRET_ACCESS_KEY`** - Required for Athena/S3 access (needed to run Athena queries)
- **`AWS_SESSION_TOKEN`** - Optional for Athena/S3 access (needed to run Athena queries). This is required for temporary credentials
- **`OPENAI_API_KEY`** - Required for vector operations (index, query, list)
- `OPENAI_BASE_URL` - Optional custom OpenAI endpoint (e.g., `http://localhost:8321/v1` for Llama Stack)
- `OPENAI_VERIFY_SSL` - Verify SSL certificates (default: `true`). Set to `false` for self-signed certs or local development. ⚠️ Use only with trusted endpoints.
- `OPENAI_EMBEDDING_MODEL` - Embedding model to use (e.g., `text-embedding-3-small`, `nomic-embed-text`)
- `OPENAI_EMBEDDING_DIMENSIONS` - Embedding dimensions (optional, model-specific)
- `AWS_DEFAULT_REGION` - AWS region (defaults to us-west-2)
- `LOG_LEVEL` - Logging level (defaults to INFO)

**Note:** Uses SQL wildcards (`%`) not glob patterns (`*`) for URL matching.

## 1. ⌨️ Command Line

```bash
# Search Common Crawl index
uv run cc-vec search --url-patterns "%.github.io" --limit 10

# Get statistics
uv run cc-vec stats --url-patterns "%.edu"

# Fetch and process content (returns clean text)
uv run cc-vec fetch --url-patterns "%.example.com" --limit 5

# Advanced filtering - multiple filters can be combined
uv run cc-vec fetch --url-patterns "%.github.io" --status-codes "200,201" --mime-types "text/html" --limit 10

# Filter by hostname instead of pattern
uv run cc-vec search --url-host-names "github.io,github.com" --limit 10

# Filter by TLD for better performance (uses indexed column)
uv run cc-vec search --url-host-tlds "edu,gov" --limit 20

# Filter by registered domain (uses indexed column)
uv run cc-vec search --url-host-registered-domains "github.com,example.com" --limit 15

# Filter by URL path (for specific site sections)
uv run cc-vec search --url-host-names "github.io" --url-paths "/blog/%,/docs/%" --limit 10

# Query across multiple Common Crawl datasets
uv run cc-vec search --url-patterns "%.edu" --crawl-ids "CC-MAIN-2024-33,CC-MAIN-2024-30" --limit 20

# List available Common Crawl datasets
uv run cc-vec list-crawls

# List all available filter columns (no API keys needed)
uv run cc-vec list-filter-columns
uv run cc-vec list-filter-columns --output json

# Vector operations (require OPENAI_API_KEY)
# Create vector store with processed content (OpenAI handles chunking with token limits)
uv run cc-vec index --url-patterns "%.github.io" --vector-store-name "ml-research" --limit 50 --chunk-size 800 --overlap 400

# Vector store name is optional - will auto-generate if not provided
uv run cc-vec index --url-patterns "%.github.io" --limit 50

# Using with alternative OpenAI-compatible endpoints (Ollama example)
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama  # Ollama doesn't require a real key
export OPENAI_EMBEDDING_MODEL=nomic-embed-text
uv run cc-vec index --url-patterns "%.github.io" --vector-store-name "local-research" --limit 50

# Using with Llama Stack
export OPENAI_BASE_URL=http://localhost:8321/v1
uv run cc-vec index --url-patterns "%.edu" --vector-store-name "education" --limit 100

# With self-signed certificates or local development (disable SSL verification)
export OPENAI_BASE_URL=https://localhost:8443/v1
export OPENAI_VERIFY_SSL=false  # ⚠️ Use only in development with trusted endpoints
export OPENAI_API_KEY=your-key
uv run cc-vec index --url-patterns "%.github.io" --vector-store-name "local-dev" --limit 50

# List cc-vec vector stores (default - only shows stores created by cc-vec)
uv run cc-vec list --output json

# List ALL vector stores (including non-cc-vec stores)
uv run cc-vec list --all

# Query vector store by ID for RAG
uv run cc-vec query "What is machine learning?" --vector-store-id "vs-123abc" --limit 5

# Query vector store by name
uv run cc-vec query "Explain deep learning" --vector-store-name "ml-research" --limit 3

```

## 1.5. 🦙 Local Llama Stack Setup (Optional)

For running cc-vec with local models via Llama Stack + Ollama, use the standalone manager script:

**Prerequisites:**
- Ollama installed and running (`ollama serve`)
- Docker (for Docker backend) or uv (for UV backend)

**First-time setup:**

```bash
# Install and start Ollama first
# macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# Run setup (pulls required models, installs dependencies)
uv run llama-stack-helper setup --backend docker

# Or for UV backend:
uv run llama-stack-helper setup --backend uv
```

**Start Llama Stack:**

```bash
# Docker backend (recommended)
uv run llama-stack-helper start --backend docker

# Or UV backend
uv run llama-stack-helper start --backend uv
```

**Check status:**

```bash
uv run llama-stack-helper status
```

**View logs:**

```bash
# Show last 20 lines
uv run llama-stack-helper logs

# Follow logs in real-time
uv run llama-stack-helper logs --follow
```

**Stop Llama Stack:**

```bash
uv run llama-stack-helper stop --backend docker
```

**Use with cc-vec:**

Once Llama Stack is running, set the environment variables:

```bash
# Set Llama Stack environment variables in your current shell
eval "$(uv run llama-stack-helper env)"

# Now use cc-vec normally with your Athena credentials
export ATHENA_OUTPUT_BUCKET=s3://your-bucket/
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

uv run cc-vec index --url-patterns "%.edu" --limit 10
```

The `env` command outputs (using your configured models):
```bash
export OPENAI_BASE_URL=http://localhost:8321/v1
export OPENAI_API_KEY=none
export OPENAI_VERIFY_SSL=false
export OPENAI_EMBEDDING_MODEL=toshk0/nomic-embed-text-v2-moe:Q6_K  # or your custom model
export OPENAI_EMBEDDING_DIMENSIONS=768  # or your custom dimensions
```

**Default models** (automatically pulled during setup):
- `llama3.2:3b` - Inference model
- `toshk0/nomic-embed-text-v2-moe:Q6_K` - Embedding model (768 dimensions)

**Custom models** (optional):

You can customize which models to use by setting environment variables before running setup:

```bash
export LLAMA_STACK_INFERENCE_MODEL=llama3.2:1b
export LLAMA_STACK_EMBEDDING_MODEL=nomic-embed-text
export LLAMA_STACK_EMBEDDING_DIMENSIONS=768

# Now run setup - it will pull your custom models
uv run llama-stack-helper setup
```

These models will be:
1. Downloaded into Ollama during setup
2. Configured in the Llama Stack run.yaml
3. Used automatically by cc-vec when you run `eval "$(uv run llama-stack-helper env)"`

## 2. 📦 Python Library

```python
import os
from cc_vec import (
    search,
    stats,
    fetch,
    index,
    list_vector_stores,
    query_vector_store,
    list_crawls,
    FilterConfig,
    VectorStoreConfig,
)

# For alternative endpoints, set environment variables before importing
# Example: Using Ollama
# os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
# os.environ["OPENAI_API_KEY"] = "ollama"
# os.environ["OPENAI_EMBEDDING_MODEL"] = "nomic-embed-text"

# Example: Using Llama Stack
# os.environ["OPENAI_BASE_URL"] = "http://localhost:8321/v1"
# os.environ["OPENAI_API_KEY"] = "your-llama-stack-key"

# Example: With self-signed certificates (disable SSL verification)
# ⚠️ Use only in development with trusted endpoints
# os.environ["OPENAI_BASE_URL"] = "https://localhost:8443/v1"
# os.environ["OPENAI_VERIFY_SSL"] = "false"
# os.environ["OPENAI_API_KEY"] = "your-key"

# Basic search and stats (no OpenAI key needed)
filter_config = FilterConfig(url_patterns=["%.github.io"])

stats_response = stats(filter_config)
print(f"Estimated records: {stats_response.estimated_records:,}")
print(f"Estimated size: {stats_response.estimated_size_mb:.2f} MB")
print(f"Athena cost: ${stats_response.estimated_cost_usd:.4f}")

results = search(filter_config, limit=10)
print(f"Found {len(results)} URLs")
for result in results[:3]:
    print(f"  {result.url} (Status: {result.status})")

# Advanced filtering - multiple criteria
filter_config = FilterConfig(
    url_patterns=["%.github.io", "%.github.com"],
    url_host_names=["github.io"],
    url_host_tlds=["io", "com"],  # Filter by TLD (uses indexed column)
    url_host_registered_domains=["github.com"],  # Filter by domain (uses indexed column)
    url_paths=["/blog/%", "/docs/%"],  # Filter by URL path
    crawl_ids=["CC-MAIN-2024-33", "CC-MAIN-2024-30"],  # Query multiple crawls
    status_codes=[200, 201],
    mime_types=["text/html"],
    charsets=["utf-8"],
    languages=["en"],
)

results = search(filter_config, limit=20)
print(f"Found {len(results)} URLs matching filters")

# Using indexed columns for better performance
filter_config = FilterConfig(
    url_host_tlds=["edu", "gov"],  # Much faster than url_patterns=["%.edu", "%.gov"]
    status_codes=[200],
)
results = search(filter_config, limit=50)
print(f"Found {len(results)} .edu and .gov sites")

# Fetch and process content (returns clean text)
filter_config = FilterConfig(url_patterns=["%.example.com"])
content_results = fetch(filter_config, limit=2)
print(f"Processed {len(content_results)} content records")
for record, processed in content_results:
    if processed:
        print(f"  {record.url}: {processed['word_count']} words")
        print(f"    Title: {processed.get('title', 'N/A')}")

# List available Common Crawl datasets
crawls = list_crawls()
print(f"Available crawls: {len(crawls)}")
print(f"Latest: {crawls[0]}")

# Index data in a vector store
filter_config = FilterConfig(url_patterns=["%.github.io"])
vector_config = VectorStoreConfig(
    name="ml-research",
    chunk_size=800,
    overlap=400,
    embedding_model="text-embedding-3-small",
    embedding_dimensions=1536,
)

result = index(filter_config, vector_config, limit=50)
print(f"Created vector store: {result['vector_store_name']}")
print(f"Vector Store ID: {result['vector_store_id']}")
print(f"Processed records: {result['total_fetched']}")

# List cc-vec vector stores (default - only shows stores created by cc-vec)
stores = list_vector_stores()
print(f"Available stores: {len(stores)}")
for store in stores[:3]:
    print(f"  {store['name']} (ID: {store['id']}, Status: {store['status']})")

# List ALL vector stores (including non-cc-vec stores)
all_stores = list_vector_stores(cc_vec_only=False)
print(f"All stores: {len(all_stores)}")

# Query vector store for RAG
query_results = query_vector_store("vs-123abc", "What is machine learning?", limit=5)
print(f"Query found {len(query_results.get('results', []))} relevant results")
for i, result in enumerate(query_results.get("results", []), 1):
    print(f"  {i}. Score: {result.get('score', 0):.3f}")
    print(f"     Content: {result.get('content', '')[:100]}...")
    print(f"     File: {result.get('file_id', 'N/A')}")
```


## 3. 🔌 MCP Server (Claude Desktop)

**Setup:**
1. Copy and edit the config: `cp claude_desktop_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json`
2. Update the directory path and API key in the config file
3. Restart Claude Desktop

The config uses stdio mode (required by Claude Desktop):
```json
{
  "mcpServers": {
    "cc-vec": {
      "command": "uv",
      "args": ["run", "--directory", "your-path-to-the-repo", "cc-vec", "mcp-serve", "--mode", "stdio"],
      "env": {
        "ATHENA_OUTPUT_BUCKET": "your-athena-output-bucket",
        "OPENAI_API_KEY": "your-openai-api-key-here"
        // "OPENAI_BASE_URL": "http://localhost:11434/v1"   // Optional: Use for Ollama, Llama Stack, or other endpoints
        // "OPENAI_VERIFY_SSL": "false"                     // Optional: Disable SSL verification for self-signed certs (dev only)
        // "OPENAI_EMBEDDING_MODEL": "nomic-embed-text"     // Optional: Specify custom embedding model
      }
    }
  }
}
```

**Available MCP tools:**

```
# Search and analysis (no OpenAI key needed)
cc_search - Search Common Crawl for URLs matching patterns with advanced filtering
cc_stats - Get statistics and cost estimates for patterns with advanced filtering
cc_fetch - Download actual content from matched URLs with advanced filtering
cc_list_crawls - List available Common Crawl dataset IDs

# Vector operations (require OPENAI_API_KEY)
cc_index - Create and populate vector stores from Common Crawl content with chunking config
cc_list_vector_stores - List OpenAI vector stores (defaults to cc-vec created only)
cc_query - Query vector stores for relevant content
```

**Example usage in Claude Desktop:**
- "Use cc_search to find GitHub Pages sites: url_pattern=%.github.io, limit=10"
- "Use cc_stats to analyze education sites: url_pattern=%.edu"
- "Use cc_search with indexed columns for better performance: url_host_tlds=['edu', 'gov'], limit=20"
- "Use cc_search with registered domains: url_host_registered_domains=['github.com'], limit=15"
- "Use cc_search for specific paths: url_host_names=['github.io'], url_paths=['/blog/%'], limit=10"
- "Use cc_search across multiple crawls: url_pattern=%.edu, crawl_ids=['CC-MAIN-2024-33', 'CC-MAIN-2024-30']"
- "Use cc_fetch to get content: url_host_names=['github.io'], limit=5"
- "Use cc_list_crawls to show available Common Crawl datasets"
- "Use cc_index to create vector store: vector_store_name=research, url_pattern=%.arxiv.org, limit=100, chunk_size=800"
- "Use cc_list_vector_stores to show cc-vec stores (default)"
- "Use cc_list_vector_stores with cc_vec_only=false to show all vector stores"
- "Use cc_query to search: vector_store_id=vs-123, query=machine learning"

**Note:** All filter options available in CLI (shown via `cc-vec list-filter-columns`) are also available in MCP tools.

## License

MIT
