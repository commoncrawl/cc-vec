"""Simplified API layer for cc-vec that handles client initialization."""

import logging
from typing import List, Dict, Any, Optional

import httpx
from openai import OpenAI
from .types import FilterConfig, CrawlRecord, StatsResponse, VectorStoreConfig
from .types.config import load_config
from .core import CCAthenaClient, CCS3Client
from .types import AthenaSettings
from .lib.search import search as search_lib
from .lib.stats import stats as stats_lib
from .lib.fetch import fetch as fetch_lib
from .lib.index import index as index_lib
from .lib.list_vector_stores import list_vector_stores as list_vector_stores_lib
from .lib.query import query_vector_store as query_vector_store_lib
from .lib.delete_vector_store import delete_vector_store as delete_vector_store_lib
from .lib.delete_vector_store import (
    delete_vector_store_by_name as delete_vector_store_by_name_lib,
)
from .lib.list_crawls import list_crawls as list_crawls_lib

logger = logging.getLogger(__name__)

_athena_client: Optional[CCAthenaClient] = None
_s3_client: Optional[CCS3Client] = None
_openai_client: Optional[OpenAI] = None


def _get_athena_client() -> CCAthenaClient:
    """Get cached Athena client."""
    global _athena_client
    if _athena_client is None:
        config = load_config()
        athena_settings = AthenaSettings(
            output_bucket=config.athena.output_bucket,
            region_name=config.athena.region_name,
            max_results=config.athena.max_results,
            timeout_seconds=config.athena.timeout_seconds,
        )
        _athena_client = CCAthenaClient(athena_settings)
    return _athena_client


def _get_s3_client() -> CCS3Client:
    """Get cached S3 client."""
    global _s3_client
    if _s3_client is None:
        _s3_client = CCS3Client()
    return _s3_client


def _get_openai_client() -> OpenAI:
    """Get cached OpenAI client."""
    global _openai_client
    if _openai_client is None:
        config = load_config()

        # Create custom httpx client if SSL verification is disabled
        http_client = None
        if not config.openai.verify_ssl:
            http_client = httpx.Client(verify=False)

        _openai_client = OpenAI(
            api_key=config.openai.api_key,
            base_url=config.openai.base_url,
            http_client=http_client,
        )
    return _openai_client


def search(
    filter_config: FilterConfig,
    limit: int = 10,
) -> List[CrawlRecord]:
    """Search Common Crawl for URLs matching filters.

    Args:
        filter_config: Filter configuration with search criteria
        limit: Maximum number of results to return

    Returns:
        List of CrawlRecord objects
    """
    athena_client = _get_athena_client()
    return search_lib(filter_config, athena_client, limit)


def stats(
    filter_config: FilterConfig,
) -> StatsResponse:
    """Get statistics for URLs matching filters.

    Args:
        filter_config: Filter configuration with search criteria

    Returns:
        StatsResponse with count and cost estimates
    """
    athena_client = _get_athena_client()
    return stats_lib(filter_config, athena_client)


def fetch(
    filter_config: FilterConfig,
    limit: int = 10,
) -> List[tuple]:
    """Fetch and process content for URLs matching filters.

    Args:
        filter_config: Filter configuration with search criteria
        limit: Maximum number of records to fetch

    Returns:
        List of (CrawlRecord, processed_content_dict) tuples
        Content is processed and cleaned but not chunked - use index() for chunking
    """
    athena_client = _get_athena_client()
    s3_client = _get_s3_client()
    return fetch_lib(filter_config, athena_client, s3_client, limit)


def index(
    filter_config: FilterConfig,
    vector_store_config: VectorStoreConfig,
    limit: int = 10,
) -> Dict[str, Any]:
    """Index processed Common Crawl content into a vector store for RAG.

    Args:
        filter_config: Filter configuration with search criteria
        vector_store_config: Vector store configuration including name and chunking params
        limit: Maximum number of records to index

    Returns:
        Dictionary with indexing results including vector store ID and chunk statistics
    """
    athena_client = _get_athena_client()
    s3_client = _get_s3_client()
    openai_client = _get_openai_client()
    return index_lib(
        filter_config,
        athena_client,
        vector_store_config,
        openai_client,
        s3_client,
        limit,
    )


def list_vector_stores(cc_vec_only: bool = True) -> List[Dict[str, Any]]:
    """List available OpenAI vector stores.

    Args:
        cc_vec_only: If True, only return vector stores created by cc-vec (default: True)

    Returns:
        List of vector store information dictionaries
    """
    openai_client = _get_openai_client()
    return list_vector_stores_lib(openai_client, cc_vec_only)


def query_vector_store(
    vector_store_id: str, query: str, *, limit: int = 5
) -> Dict[str, Any]:
    """Query a vector store for relevant content.

    Args:
        vector_store_id: ID of the vector store to query
        query: Query string to search for
        limit: Maximum number of results to return

    Returns:
        Dictionary with search results and metadata
    """
    openai_client = _get_openai_client()
    return query_vector_store_lib(vector_store_id, query, limit, openai_client)


def delete_vector_store(vector_store_id: str) -> Dict[str, Any]:
    """Delete a vector store by ID.

    Args:
        vector_store_id: ID of the vector store to delete

    Returns:
        Dictionary with deletion result
    """
    openai_client = _get_openai_client()
    return delete_vector_store_lib(vector_store_id, openai_client)


def delete_vector_store_by_name(vector_store_name: str) -> Dict[str, Any]:
    """Delete a vector store by name.

    Args:
        vector_store_name: Name of the vector store to delete

    Returns:
        Dictionary with deletion result
    """
    openai_client = _get_openai_client()
    return delete_vector_store_by_name_lib(vector_store_name, openai_client)


def list_crawls() -> List[str]:
    """List available Common Crawl crawls.

    Returns:
        List of crawl IDs sorted in descending order (newest first)
    """
    athena_client = _get_athena_client()
    return list_crawls_lib(athena_client)
