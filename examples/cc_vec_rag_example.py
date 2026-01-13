"""
Example: Index Common Crawl content into a vector store and query it with OpenAI's file_search tool.

This demonstrates the full RAG workflow:
1. Index content from Common Crawl into an OpenAI vector store
2. Create an Assistant with the file_search tool
3. Query the vector store using the Assistant

Requirements:
    - OPENAI_API_KEY environment variable
    - ATHENA_OUTPUT_BUCKET environment variable
    - AWS credentials configured

Optional (for alternative endpoints):
    - OPENAI_BASE_URL: Custom OpenAI-compatible endpoint (e.g., http://localhost:11434/v1 for Ollama)
    - OPENAI_VERIFY_SSL: Set to "false" to disable SSL verification for self-signed certs (dev only)
    - OPENAI_EMBEDDING_MODEL: Custom embedding model (e.g., nomic-embed-text for Ollama)
"""

import os

from cc_vec import FilterConfig, index, VectorStoreConfig
from openai import OpenAI


def main():
    # Initialize OpenAI client
    # Note: If using alternative endpoints with self-signed certs, set:
    #   export OPENAI_BASE_URL=https://localhost:8443/v1
    #   export OPENAI_VERIFY_SSL=false
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 1: Index Common Crawl content into a vector store
    print("=" * 80)
    print("STEP 1: Indexing Common Crawl content into vector store")
    print("=" * 80)

    # Configure what to search for
    filter_config = FilterConfig(
        url_patterns=["%.github.io%"],  # GitHub Pages sites
        crawl_ids=["CC-MAIN-2024-33"],  # Latest crawl
        status_codes=[200],
        mime_types=["text/html"],
        languages=["en"],
    )

    # Configure the vector store
    vector_store_config = VectorStoreConfig(
        name="github-io-ml-content",
        chunk_size=800,  # Chunk text into 800 token pieces
        overlap=400,  # 400 token overlap between chunks
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
    )

    # Index the content (this will take a few moments)
    print("\nIndexing content from Common Crawl...")
    print(f"  - Searching for: {filter_config.url_patterns}")
    print(f"  - Vector store: {vector_store_config.name}")
    print(f"  - Chunk size: {vector_store_config.chunk_size} tokens")
    print(f"  - Overlap: {vector_store_config.overlap} tokens")
    print("  - Limit: 10 records\n")

    result = index(
        filter_config=filter_config,
        vector_store_config=vector_store_config,
        limit=10,  # Index 10 pages for demo
    )

    vector_store_id = result["vector_store_id"]
    print("\n✅ Vector store created!")
    print(f"  - ID: {vector_store_id}")
    print(f"  - Name: {result['vector_store_name']}")
    print(f"  - Records processed: {result['total_fetched']}")
    print(f"  - Successfully fetched: {result['successful_fetches']}")

    # Step 2: Query the vector store using the Responses API
    print("\n" + "=" * 80)
    print("STEP 2: Querying the vector store with Responses API")
    print("=" * 80)

    # Example queries
    queries = [
        "What machine learning topics are discussed in these GitHub Pages?",
        "Are there any tutorials or getting started guides?",
        "What programming languages or frameworks are mentioned?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 80}")
        print(f"Query {i}: {query}")
        print("─" * 80)

        # Create a response with file_search tool
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions="""You are a helpful assistant that answers questions based on content
            indexed from Common Crawl. Use the file_search tool to find relevant information
            from the indexed web pages. Always cite which URLs your information comes from.""",
            input=query,
            tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
            include=["file_search_call.results"],
        )

        # Extract and display the response
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        print(f"\n{content.text}\n")

                        # Show citations if available
                        if content.annotations:
                            print("Citations:")
                            seen_files = set()
                            for annotation in content.annotations:
                                if annotation.type == "file_citation":
                                    if annotation.file_id not in seen_files:
                                        print(
                                            f"  - File: {annotation.file_id} ({annotation.filename})"
                                        )
                                        seen_files.add(annotation.file_id)

    # Step 3: Cleanup (optional)
    print("\n" + "=" * 80)
    print("STEP 3: Cleanup")
    print("=" * 80)
    print("\nTo clean up resources, you can:")
    print(f"  - Delete vector store: client.vector_stores.delete('{vector_store_id}')")
    print(f"  - Or use cc-vec CLI: uv run cc-vec delete-vector-store {vector_store_id}")

    # Uncomment to auto-cleanup:
    # client.vector_stores.delete(vector_store_id)
    # print("\n✅ Cleanup complete!")

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Verify environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        exit(1)

    if not os.getenv("ATHENA_OUTPUT_BUCKET"):
        print("❌ Error: ATHENA_OUTPUT_BUCKET environment variable not set")
        print(
            "   Set it to your S3 bucket for Athena results (e.g., s3://your-bucket/athena/)"
        )
        exit(1)

    main()
