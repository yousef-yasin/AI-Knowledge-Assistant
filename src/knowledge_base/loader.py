from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docx import Document as DocxDocument
from pypdf import PdfReader

from src.knowledge_base.metadata import add_page_metadata, create_base_metadata


@dataclass
class LoadedDocument:
    text: str
    metadata: dict[str, Any]


class DocumentLoader:
    """Load PDF, DOCX, TXT, Markdown and CSV documents, with OCR fallback."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}

    def __init__(self, enable_ocr: bool = True, ocr_language: str = "eng") -> None:
        self.enable_ocr = enable_ocr
        self.ocr_language = ocr_language

    def load(self, file_path: str | Path) -> list[LoadedDocument]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"The provided path is not a file: {path}")
        extension = path.suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}")
        loaders = {
            ".pdf": self._load_pdf, ".docx": self._load_docx,
            ".txt": self._load_text, ".md": self._load_text,
            ".csv": self._load_csv,
        }
        documents = [doc for doc in loaders[extension](path) if doc.text.strip()]
        if not documents:
            raise ValueError(f"No readable text was found in: {path.name}")
        return documents

    def _load_pdf(self, path: Path) -> list[LoadedDocument]:
        reader = PdfReader(str(path))
        documents: list[LoadedDocument] = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            extraction_method = "native_text"
            if self.enable_ocr and len(text) < 20:
                ocr_text = self._ocr_pdf_page(path, page_index - 1)
                if ocr_text.strip():
                    text = ocr_text.strip()
                    extraction_method = "ocr"
            metadata = add_page_metadata(
                create_base_metadata(path), page_index, len(reader.pages)
            )
            metadata["extraction_method"] = extraction_method
            documents.append(LoadedDocument(text=text, metadata=metadata))
        return documents

    def _ocr_pdf_page(self, path: Path, page_index: int) -> str:
        """Render one PDF page and OCR it. Returns empty text if OCR is unavailable."""
        try:
            import pypdfium2 as pdfium
            import pytesseract

            pdf = pdfium.PdfDocument(str(path))
            page = pdf[page_index]
            image = page.render(scale=2.5).to_pil()
            return pytesseract.image_to_string(image, lang=self.ocr_language)
        except Exception:
            return ""

    def _load_docx(self, path: Path) -> list[LoadedDocument]:
        document = DocxDocument(str(path))
        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        metadata = add_page_metadata(create_base_metadata(path), None)
        return [LoadedDocument("\n".join(paragraphs), metadata)]

    def _load_text(self, path: Path) -> list[LoadedDocument]:
        metadata = add_page_metadata(create_base_metadata(path), None)
        return [LoadedDocument(self._read_text_with_fallback(path).strip(), metadata)]

    def _load_csv(self, path: Path) -> list[LoadedDocument]:
        rows: list[str] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no headers: {path.name}")
            for number, row in enumerate(reader, 1):
                values = [f"{key}: {value}" for key, value in row.items()
                          if value is not None and str(value).strip()]
                if values:
                    rows.append(f"Row {number}: " + " | ".join(values))
        metadata = add_page_metadata(create_base_metadata(path), None)
        metadata["row_count"] = len(rows)
        return [LoadedDocument("\n".join(rows), metadata)]

    @staticmethod
    def _read_text_with_fallback(path: Path) -> str:
        for encoding in ("utf-8", "utf-8-sig", "utf-16", "cp1252"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Could not decode text file: {path.name}")
