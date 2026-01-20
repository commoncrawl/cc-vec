"""cc-vec: Common Crawl vectorization toolkit."""

# Public API - types and configuration
from .types import (
    FilterConfig,
    VectorStoreConfig,
    AthenaSettings,
    StatsResponse,
    SearchResponse,
    CrawlRecord,
    ProcessedContent,
)

# Public API - simplified operations
from .api import (
    stats,
    search,
    fetch,
    index,
    index_from_files,
    list_vector_stores,
    query_vector_store,
    delete_vector_store,
    delete_vector_store_by_name,
    list_crawls,
)

__all__ = [
    # Types and configuration
    "FilterConfig",
    "VectorStoreConfig",
    "AthenaSettings",
    "StatsResponse",
    "SearchResponse",
    "CrawlRecord",
    "ProcessedContent",
    # Operations
    "stats",
    "search",
    "fetch",
    "index",
    "index_from_files",
    "list_vector_stores",
    "query_vector_store",
    "delete_vector_store",
    "delete_vector_store_by_name",
    "list_crawls",
]

__version__ = "0.1.0"


def main() -> None:
    """CLI entry point."""
    print("Hello from cc-vec!")
