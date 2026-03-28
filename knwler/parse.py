from pathlib import Path

import fitz  # PyMuPDF


def parse_pdf(file_path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    doc = fitz.open(str(file_path))
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def load_text(file_path: Path) -> str:
    """Extract text from a PDF or read a text/markdown file."""
    if file_path.suffix.lower() == ".pdf":
        return parse_pdf(file_path)
    return file_path.read_text(encoding="utf-8", errors="ignore")
