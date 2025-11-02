"""Utility functions for document processing."""

import re
from bs4 import BeautifulSoup
from typing import Dict, List, Optional


def extract_text_from_html(html_content: str) -> str:
    """Extract text content from HTML."""
    soup = BeautifulSoup(html_content, "html.parser")
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator="\n", strip=True)


def extract_text_from_xml(xml_content: str) -> str:
    """Extract text content from XML."""
    soup = BeautifulSoup(xml_content, "xml")
    return soup.get_text(separator="\n", strip=True)


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters but keep newlines
    text = re.sub(r"[^\w\s\n.,;:!?()\[\]{}\-]", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for processing."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

        if end >= text_length:
            break

    return chunks


def extract_metadata(filename: str) -> Dict[str, Optional[str]]:
    """Extract metadata from filename."""
    metadata = {"filename": filename, "type": None, "date": None, "ticker": None}

    # Extract ticker from filings path (e.g., "extracted_filings/AAPL/2024-11-01-10k-AAPL.html")
    if "fillings" in filename:
        parts = filename.split("/")
        if len(parts) >= 2:
            metadata["ticker"] = parts[-2] if parts[-2] != "fillings" else None

    # Extract date pattern (YYYY-MM-DD)
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if date_match:
        metadata["date"] = date_match.group(1)

    # Determine document type
    if "directive" in filename.lower() or "regulation" in filename.lower():
        metadata["type"] = "regulatory"
    elif "10k" in filename.lower():
        metadata["type"] = "filing"

    return metadata


def format_document_summary(
    metadata: Dict, preview: str = None, max_preview: int = 500
) -> str:
    """Format document metadata and preview for display."""
    lines = [f"**Filename:** {metadata['filename']}"]

    if metadata["type"]:
        lines.append(f"**Type:** {metadata['type']}")
    if metadata["date"]:
        lines.append(f"**Date:** {metadata['date']}")
    if metadata["ticker"]:
        lines.append(f"**Ticker:** {metadata['ticker']}")

    if preview:
        preview_text = (
            preview[:max_preview] + "..." if len(preview) > max_preview else preview
        )
        lines.append(f"\n**Preview:**\n{preview_text}")

    return "\n".join(lines)
