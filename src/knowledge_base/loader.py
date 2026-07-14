from __future__ import annotations

import csv  # Used to read CSV files and process tabular data.
from dataclasses import dataclass # Used to create a simple data class for storing document text and metadata.
from pathlib import Path  # Used to work with file and folder paths in a clean, cross-platform way.
from typing import Any   # Allows metadata values to have different data types. 

from docx import Document as DocxDocument  # Reads Microsoft Word (.docx) documents.
from pypdf import PdfReader  # Reads PDF files and extracts text from each page.


@dataclass  # Automatically creates constructor and utility methods.
class LoadedDocument:  # Stores the extracted text along with its metadata.
    """
    Represents one loaded part of a document.

    For PDF files, each page is returned as a separate LoadedDocument.
    For other supported files, the entire file is returned as one LoadedDocument.
    """

    text: str  # Stores the extracted document text.
    metadata: dict[str, Any] # Stores document information such as file name and page number.


class DocumentLoader:  # Main class responsible for loading supported document types.
    """Loads supported document types and extracts their text."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}  
    # Defines the file formats supported by the loader.

    def load(self, file_path: str | Path) -> list[LoadedDocument]: # Detects the file type and calls the appropriate loading function.
        """
        Load a document based on its file extension.

        Args:
            file_path: Path of the document to load.

        Returns:
            A list of LoadedDocument objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is unsupported or contains no text.
        """

        path = Path(file_path) # Converts the input path into a Path object.

        if not path.exists(): # Checks whether the file exists.
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():  # Ensures the provided path is a file, not a directory.
            raise ValueError(f"The provided path is not a file: {path}")

        extension = path.suffix.lower() # Retrieves the file extension in lowercase.

        if extension not in self.SUPPORTED_EXTENSIONS: 
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types are: {supported}"
            )

        loaders = {  # Maps each supported extension to its corresponding loader function.
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".txt": self._load_text,
            ".md": self._load_text,
            ".csv": self._load_csv,
        }

        documents = loaders[extension](path) # Calls the correct loader based on the file type.

        documents = [ # Removes empty documents or pages.
            document
            for document in documents
            if document.text and document.text.strip()
        ]

        if not documents:
            raise ValueError(f"No readable text was found in: {path.name}")

        return documents # Returns the loaded documents. 

    def _base_metadata(self, path: Path) -> dict[str, Any]: # Creates metadata shared across all document types.
        """Create common metadata for every loaded document."""

        return {
            "document_name": path.name, # Stores the original file name. 
            "file_type": path.suffix.lower().replace(".", ""), # Stores the file extension.
            "file_path": str(path.resolve()), # Stores the absolute file path.
            "file_size_bytes": path.stat().st_size,  # Stores the file size in bytes.
        }

    def _load_pdf(self, path: Path) -> list[LoadedDocument]:
        """Extract text from a PDF page by page."""

        reader = PdfReader(str(path)) # Opens the PDF document.
        documents: list[LoadedDocument] = [] 


        for page_index, page in enumerate(reader.pages, start=1): # Iterates through all PDF pages.
            text = page.extract_text() or "" # Extracts text from the current page.

            metadata = self._base_metadata(path)
            metadata["page_number"] = page_index # Saves the current page number. 
            metadata["total_pages"] = len(reader.pages) # Stores the total number of pages.

            documents.append(
                LoadedDocument(
                    text=text.strip(),
                    metadata=metadata,
                )
            )

        return documents

    def _load_docx(self, path: Path) -> list[LoadedDocument]:
        """Extract text from a DOCX file."""

        document = DocxDocument(str(path)) # Opens the DOCX document.

        paragraphs = [ # Extracts all non-empty paragraphs.
            paragraph.text.strip() # Extracts all non-empty paragraphs.
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        text = "\n".join(paragraphs) # Combines all paragraphs into one text.

        metadata = self._base_metadata(path)
        metadata["page_number"] = None

        return [
            LoadedDocument(
                text=text,
                metadata=metadata,
            )
        ]

    def _load_text(self, path: Path) -> list[LoadedDocument]:
        """Extract text from TXT and Markdown files."""

        text = self._read_text_with_fallback(path) # Reads TXT and Markdown files using multiple encodings.

        metadata = self._base_metadata(path)
        metadata["page_number"] = None

        return [
            LoadedDocument(
                text=text.strip(),
                metadata=metadata,
            )
        ]

    def _load_csv(self, path: Path) -> list[LoadedDocument]:
        """
        Convert every CSV row into readable text.

        Example:
        name: Ahmad | department: IT | salary: 500
        """

        rows_as_text: list[str] = []

        with path.open(
            mode="r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(csv_file) # Reads CSV rows using column headers.

            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no headers: {path.name}")

            for row_number, row in enumerate(reader, start=1): # Processes each row separately.
                values = [
                    f"{column}: {value}"
                    for column, value in row.items()
                    if value is not None and str(value).strip()
                ]

                if values:
                    rows_as_text.append(  # Converts each CSV row into readable text. 
                        f"Row {row_number}: " + " | ".join(values)
                    )

        metadata = self._base_metadata(path)
        metadata["page_number"] = None
        metadata["row_count"] = len(rows_as_text)

        return [
            LoadedDocument(
                text="\n".join(rows_as_text),
                metadata=metadata,
            )
        ]

    @staticmethod
    def _read_text_with_fallback(path: Path) -> str:
        """Read text using common encodings."""

        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"] # List of common text encodings to try.

        for encoding in encodings: # Attempts to read the file using each encoding. 
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError: # Tries the next encoding if decoding fails.
                continue

        raise ValueError(f"Could not decode text file: {path.name}")


def load_document(file_path: str | Path) -> list[LoadedDocument]: # Helper function that loads adocument using the DocumentLoader class.
    """
    Convenience function for loading one document.

    Example:
        documents = load_document("uploads/report.pdf")
    """

    loader = DocumentLoader()
    return loader.load(file_path)