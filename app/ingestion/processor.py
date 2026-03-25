"""
Text processing: cleaning, chunking, and PDF parsing.
"""
import re
import io
import time
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

CHUNK_SIZE = 400       # tokens (approx chars/4)
CHUNK_OVERLAP = 50


def clean_text(text: str) -> str:
    """Remove noise and normalize whitespace."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # remove non-ASCII
    return text.strip()


def chunk_text(text: str, source: str, doc_type: str = "web") -> List[Dict]:
    """
    Chunk text into overlapping windows.
    Returns list of {text, source, type} dicts.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 4,   # chars (approx 400 tokens)
        chunk_overlap=CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [
        {"text": chunk.strip(), "source": source, "type": doc_type}
        for chunk in chunks if chunk.strip()
    ]


def process_scraped_pages(pages: List[Dict]) -> List[Dict]:
    """
    Process a list of scraped page dicts into chunks.
    """
    t0 = time.time()
    all_chunks = []
    for page in pages:
        source = page.get("source", "unknown")
        raw_text = clean_text(page.get("text", ""))

        # Add table text
        for table in page.get("tables", []):
            raw_text += " " + clean_text(table)

        if raw_text:
            chunks = chunk_text(raw_text, source, doc_type="web")
            all_chunks.extend(chunks)

    elapsed = time.time() - t0
    logger.info(f"[Processor] {len(pages)} pages → {len(all_chunks)} chunks in {elapsed:.2f}s")
    return all_chunks


def process_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    Parse PDF bytes and chunk the text content.
    """
    t0 = time.time()
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text = " ".join(
            page.extract_text() or ""
            for page in reader.pages
        )
        text = clean_text(text)
        chunks = chunk_text(text, source=filename, doc_type="pdf")
        elapsed = time.time() - t0
        logger.info(f"[Processor] PDF '{filename}' → {len(chunks)} chunks in {elapsed:.2f}s")
        return chunks
    except Exception as e:
        logger.error(f"[Processor] PDF parse error: {e}")
        return []


def process_txt(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    Parse plain text file and chunk.
    """
    text = clean_text(file_bytes.decode("utf-8", errors="ignore"))
    chunks = chunk_text(text, source=filename, doc_type="txt")
    logger.info(f"[Processor] TXT '{filename}' → {len(chunks)} chunks")
    return chunks
