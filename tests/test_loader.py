from pathlib import Path #

import pytest

from src.knowledge_base.loader import DocumentLoader


@pytest.fixture
def loader() -> DocumentLoader:
    return DocumentLoader()


def test_load_txt_file(tmp_path: Path, loader: DocumentLoader) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello from TXT file", encoding="utf-8")

    documents = loader.load(file_path)

    assert len(documents) == 1
    assert documents[0].text == "Hello from TXT file"
    assert documents[0].metadata["document_name"] == "sample.txt"
    assert documents[0].metadata["file_type"] == "txt"
    assert documents[0].metadata["page_number"] is None


def test_load_markdown_file(tmp_path: Path, loader: DocumentLoader) -> None:
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Title\nMarkdown content", encoding="utf-8")

    documents = loader.load(file_path)

    assert len(documents) == 1
    assert "# Title" in documents[0].text
    assert documents[0].metadata["file_type"] == "md"


def test_file_not_found(loader: DocumentLoader) -> None:
    with pytest.raises(FileNotFoundError):
        loader.load("missing_file.pdf")


def test_unsupported_file_type(
    tmp_path: Path,
    loader: DocumentLoader,
) -> None:
    file_path = tmp_path / "sample.exe"
    file_path.write_text("Unsupported content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load(file_path)


def test_empty_text_file(
    tmp_path: Path,
    loader: DocumentLoader,
) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No readable text"):
        loader.load(file_path)