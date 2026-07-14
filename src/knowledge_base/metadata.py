from __future__ import annotations

from datetime import datetime, timezone # Used to generate timestamps for metadata.
from pathlib import Path # Used to work with file paths. 
from typing import Any  # Allows metadata values to have different data types.


def create_base_metadata(file_path: str | Path) -> dict[str, Any]:
    """
    Create metadata shared by all supported document types.

    Args:
        file_path: Path of the uploaded document.

    Returns:
        A dictionary containing common document metadata.
    """

    path = Path(file_path) # Converts the file path into a Path object.

    if not path.exists(): # Checks whether the file exists.
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file(): # Ensures the provided path is a file, not a directory.
        raise ValueError(f"The provided path is not a file: {path}")

    file_stats = path.stat() # Retrieves file information such as size and modification time.
    #if file or folder


    return {
        "document_name": path.name,  # Stores the original file name.
        "file_type": path.suffix.lower().replace(".", ""), # Stores the file extension.
        "file_path": str(path.resolve()), # Stores the absolute file path.
        "file_size_bytes": file_stats.st_size,  # Stores the file size in bytes.
        "modified_at": datetime.fromtimestamp(
            file_stats.st_mtime,
            tz=timezone.utc,
        ).isoformat(), # Stores the file's last modified date and time.
    }


def add_page_metadata(  # Adds page-related metadata to the document.
    metadata: dict[str, Any],
    page_number: int | None,
    total_pages: int | None = None,
) -> dict[str, Any]:
    """
    Add page-related metadata without modifying the original dictionary.
    """

    updated_metadata = metadata.copy()
    updated_metadata["page_number"] = page_number

    if total_pages is not None: # Adds the total number of pages if available.
        updated_metadata["total_pages"] = total_pages # Stores the total number of pages.

    return updated_metadata # Returns the updated metadata.


def add_chunk_metadata( # Adds chunk-related metadata to the document.
    metadata: dict[str, Any],
    chunk_number: int,
) -> dict[str, Any]:
    """
    Add the chunk number to document metadata.
    """
 
    if chunk_number < 1: # Ensures the chunk number is valid.
        raise ValueError("chunk_number must be greater than or equal to 1.")

    updated_metadata = metadata.copy() # Creates a copy of the metadata to avoid modifying the original.
    updated_metadata["chunk_number"] = chunk_number # Stores the current page number.

    return updated_metadata