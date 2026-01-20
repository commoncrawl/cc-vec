"""Simplified API layer for cc-vec that handles client initialization."""

import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI
from .types import FilterConfig, CrawlRecord, StatsResponse, VectorStoreConfig
from .types.config import load_config
from .core import CCAthenaClient, CCS3Client
from .types import AthenaSettings
from .lib.search import search as search_lib
from .lib.stats import stats as stats_lib
from .lib.fetch import fetch as fetch_lib
from .lib.index import index as index_lib, VectorStoreLoader
from .lib.list_vector_stores import list_vector_stores as list_vector_stores_lib
from .lib.query import query_vector_store as query_vector_store_lib
from .lib.delete_vector_store import delete_vector_store as delete_vector_store_lib
from .lib.delete_vector_store import (
    delete_vector_store_by_name as delete_vector_store_by_name_lib,
)
from .lib.list_crawls import list_crawls as list_crawls_lib
from .lib.io import load_fetch_results

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
        _openai_client = OpenAI(
            api_key=config.openai.api_key,
            base_url=config.openai.base_url,
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
    limit: Optional[int] = 10,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Index processed Common Crawl content into a vector store for RAG.

    Args:
        filter_config: Filter configuration with search criteria
        vector_store_config: Vector store configuration including name and chunking params
        limit: Maximum number of records to index
        batch_size: Optional batch size for uploading files (None = all at once)

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
        batch_size,
    )


def index_from_files(
    input_dir: str,
    vector_store_config: VectorStoreConfig,
    limit: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Index pre-fetched content from files into a vector store.

    This function loads content from files saved by 'fetch --output-dir' and
    uploads them to a vector store. This allows for a two-step workflow:
    1. Fetch content once with 'cc-vec fetch --output-dir'
    2. Index into vector stores later with 'cc-vec index --input-dir'

    Args:
        input_dir: Directory containing saved fetch results
        vector_store_config: Vector store configuration including name and chunking params
        limit: Optional maximum number of files to process
        batch_size: Optional batch size for uploading files (None = all at once)

    Returns:
        Dictionary with indexing results including vector store ID and upload statistics
    """
    openai_client = _get_openai_client()

    # Load pre-fetched results
    logger.info(f"Loading pre-fetched content from {input_dir}")
    fetch_results, filter_config = load_fetch_results(input_dir, limit)

    if not fetch_results:
        logger.warning("No content files found in input directory")
        return {
            "vector_store_id": None,
            "status": "no_content",
            "total_fetched": 0,
            "successful_fetches": 0,
        }

    logger.info(f"Loaded {len(fetch_results)} files for indexing")

    # Create vector store loader and upload
    loader = VectorStoreLoader(openai_client, vector_store_config)
    vector_store_id = loader.create_vector_store()

    upload_result = loader.upload_to_vector_store(
        vector_store_id, fetch_results, batch_size=batch_size
    )

    # Check if upload failed completely
    file_counts = upload_result["file_counts"]
    if upload_result["status"] == "failed" and file_counts.completed == 0:
        error_msg = (
            f"All {file_counts.total} files failed to upload to vector store. "
            f"Failed: {file_counts.failed}, Cancelled: {file_counts.cancelled}. "
            f"Check the logs for detailed error messages."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Warn if some files failed but some succeeded
    if file_counts.failed > 0 and file_counts.completed > 0:
        logger.warning(
            f"Partial upload success: {file_counts.completed} succeeded, {file_counts.failed} failed"
        )

    return {
        "vector_store_id": vector_store_id,
        "vector_store_name": vector_store_config.name,
        "input_dir": input_dir,
        "total_fetched": len(fetch_results),
        "successful_fetches": len(fetch_results),
        "total_chunks": upload_result.get("total_chunks", len(fetch_results)),
        "total_pages": upload_result.get("total_pages", len(fetch_results)),
        "upload_status": upload_result["status"],
        "file_counts": upload_result["file_counts"],
        "batch_id": upload_result["batch_id"],
        "filenames": upload_result.get("filenames", []),
    }


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
