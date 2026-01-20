"""IO functionality for saving and loading fetch results."""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..types import CrawlRecord, FilterConfig

logger = logging.getLogger(__name__)

# File format version for manifest
MANIFEST_VERSION = "1.0"


def save_fetch_results(
    results: List[Tuple[CrawlRecord, Optional[Dict[str, Any]]]],
    output_dir: str,
    filter_config: Optional[FilterConfig] = None,
) -> Dict[str, Any]:
    """Save fetch results to files for later indexing.

    Args:
        results: List of (CrawlRecord, processed_content) tuples from fetch
        output_dir: Directory to save files to
        filter_config: Optional filter config to save in manifest

    Returns:
        Dictionary with save results including file count and manifest path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest_data = {
        "version": MANIFEST_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "filter_config": filter_config.model_dump() if filter_config else None,
        "files": [],
    }

    saved_count = 0
    skipped_count = 0

    for record, processed_content in results:
        if processed_content is None:
            skipped_count += 1
            continue

        # Generate unique filename
        file_id = str(uuid.uuid4()).replace("-", "")
        filename = f"file-{file_id}"
        filepath = output_path / filename

        # Create content in same format as VectorStoreLoader.prepare_files()
        metadata = processed_content.get("crawl_metadata", {})
        content_text = f"""Title: {processed_content.get("title", "N/A")}
URL: {metadata.get("url", str(record.url))}
Timestamp: {metadata.get("timestamp", record.timestamp)}
Status: {metadata.get("status", record.status)}
MIME Type: {metadata.get("mime", record.mime or "N/A")}
Word Count: {processed_content.get("word_count", 0)}
Meta Description: {processed_content.get("meta_description", "N/A")}

--- Content ---
{processed_content.get("text", "")}
"""

        # Write content file
        filepath.write_text(content_text, encoding="utf-8")

        # Add to manifest
        manifest_data["files"].append(
            {
                "filename": filename,
                "record": {
                    "url": str(record.url),
                    "urlkey": record.urlkey,
                    "timestamp": record.timestamp,
                    "status": record.status,
                    "mime": record.mime,
                    "digest": record.digest,
                    "length": record.length,
                    "offset": record.offset,
                    "filename": record.filename,
                    "languages": record.languages,
                    "charset": record.charset,
                },
                "word_count": processed_content.get("word_count", 0),
                "title": processed_content.get("title"),
            }
        )

        saved_count += 1
        logger.debug(f"Saved {filename} for {record.url}")

    # Write manifest
    manifest_path = output_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    logger.info(f"Saved {saved_count} files to {output_dir} (skipped {skipped_count})")

    return {
        "output_dir": str(output_path),
        "saved_count": saved_count,
        "skipped_count": skipped_count,
        "manifest_path": str(manifest_path),
    }


def load_fetch_results(
    input_dir: str,
    limit: Optional[int] = None,
) -> Tuple[List[Tuple[CrawlRecord, Dict[str, Any]]], Optional[FilterConfig]]:
    """Load previously saved fetch results.

    Supports two modes:
    1. With manifest.json - Uses metadata from manifest to reconstruct CrawlRecords
    2. Without manifest.json - Parses metadata from file headers

    Args:
        input_dir: Directory containing saved fetch results
        limit: Optional limit on number of files to load

    Returns:
        Tuple of (list of (CrawlRecord, processed_content) tuples, filter_config if available)
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    manifest_path = input_path / "manifest.json"

    if manifest_path.exists():
        return _load_with_manifest(input_path, manifest_path, limit)
    else:
        return _load_without_manifest(input_path, limit)


def _load_with_manifest(
    input_path: Path,
    manifest_path: Path,
    limit: Optional[int] = None,
) -> Tuple[List[Tuple[CrawlRecord, Dict[str, Any]]], Optional[FilterConfig]]:
    """Load fetch results using manifest.json."""
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    filter_config = None
    if manifest_data.get("filter_config"):
        filter_config = FilterConfig(**manifest_data["filter_config"])

    results = []
    files = manifest_data.get("files", [])

    if limit:
        files = files[:limit]

    for file_entry in files:
        filename = file_entry["filename"]
        filepath = input_path / filename

        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue

        # Read content
        content = filepath.read_text(encoding="utf-8")

        # Parse content to extract text
        text = _extract_text_from_content(content)

        # Reconstruct CrawlRecord
        record_data = file_entry["record"]
        record = CrawlRecord(**record_data)

        # Reconstruct processed content
        processed_content = {
            "text": text,
            "word_count": file_entry.get("word_count", len(text.split())),
            "title": file_entry.get("title"),
            "crawl_metadata": {
                "url": str(record.url),
                "status": record.status,
                "mime": record.mime,
                "timestamp": record.timestamp,
            },
        }

        results.append((record, processed_content))

    logger.info(f"Loaded {len(results)} files from {input_path}")
    return results, filter_config


def _load_without_manifest(
    input_path: Path,
    limit: Optional[int] = None,
) -> Tuple[List[Tuple[CrawlRecord, Dict[str, Any]]], Optional[FilterConfig]]:
    """Load fetch results by parsing file headers (no manifest.json)."""
    results = []

    # Find all files (excluding manifest.json if it somehow exists)
    files = sorted(
        [f for f in input_path.iterdir() if f.is_file() and f.name != "manifest.json"]
    )

    if limit:
        files = files[:limit]

    for filepath in files:
        try:
            content = filepath.read_text(encoding="utf-8")
            parsed = _parse_file_content(content)

            if parsed:
                record, processed_content = parsed
                results.append((record, processed_content))
            else:
                logger.warning(f"Could not parse file: {filepath}")

        except Exception as e:
            logger.warning(f"Error loading file {filepath}: {e}")
            continue

    logger.info(f"Loaded {len(results)} files from {input_path} (no manifest)")
    return results, None


def _parse_file_content(content: str) -> Optional[Tuple[CrawlRecord, Dict[str, Any]]]:
    """Parse file content to extract metadata and text.

    Expected format:
    Title: ...
    URL: ...
    Timestamp: ...
    Status: ...
    MIME Type: ...
    Word Count: ...
    Meta Description: ...

    --- Content ---
    <text>
    """
    # Split header and content
    if "--- Content ---" in content:
        parts = content.split("--- Content ---", 1)
        header = parts[0]
        text = parts[1].strip() if len(parts) > 1 else ""
    else:
        # No separator found, treat entire content as text
        return None

    # Parse header fields
    def get_field(field_name: str) -> Optional[str]:
        pattern = rf"^{field_name}:\s*(.*)$"
        match = re.search(pattern, header, re.MULTILINE)
        return match.group(1).strip() if match else None

    url = get_field("URL")
    if not url:
        return None

    timestamp = get_field("Timestamp") or "20240101"
    status_str = get_field("Status")
    status = int(status_str) if status_str and status_str.isdigit() else 200
    mime = get_field("MIME Type")
    if mime == "N/A":
        mime = None
    word_count_str = get_field("Word Count")
    word_count = (
        int(word_count_str)
        if word_count_str and word_count_str.isdigit()
        else len(text.split())
    )
    title = get_field("Title")
    if title == "N/A":
        title = None
    meta_description = get_field("Meta Description")
    if meta_description == "N/A":
        meta_description = None

    # Create CrawlRecord with minimal required fields
    # Note: urlkey is required, so we generate a placeholder
    urlkey = url.replace("https://", "").replace("http://", "").replace("/", ",")

    record = CrawlRecord(
        url=url,
        urlkey=urlkey,
        timestamp=timestamp,
        status=status,
        mime=mime,
    )

    processed_content = {
        "text": text,
        "word_count": word_count,
        "title": title,
        "meta_description": meta_description,
        "crawl_metadata": {
            "url": url,
            "status": status,
            "mime": mime,
            "timestamp": timestamp,
        },
    }

    return record, processed_content


def _extract_text_from_content(content: str) -> str:
    """Extract the text portion from a content file."""
    if "--- Content ---" in content:
        parts = content.split("--- Content ---", 1)
        return parts[1].strip() if len(parts) > 1 else ""
    return content
