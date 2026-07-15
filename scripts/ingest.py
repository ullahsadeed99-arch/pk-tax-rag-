"""Runs the full offline ingestion pipeline over the source PDFs and writes
the resulting chunks to data/processed/chunks.json for the Node backend's
indexing step to embed.

Usage: python scripts/ingest.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, RAW_PDF_DIR
from src.ingestion.chunker import chunk_blocks
from src.ingestion.cleaner import clean_pages
from src.ingestion.pdf_extractor import extract_pdf
from src.ingestion.structure_parser import parse_structure

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# (source filename, human-readable document name)
SOURCE_DOCS = [
    ("Income_Tax_Ordinance_2001.pdf", "Income Tax Ordinance 2001"),
    ("Income_Tax_Rules_2002.pdf", "Income Tax Rules 2002"),
    ("Sales_Tax_Act_1990.pdf", "Sales Tax Act 1990"),
    ("Sales_Tax_Rules_2006.pdf", "Sales Tax Rules 2006"),
]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for filename, doc_name in SOURCE_DOCS:
        pdf_path = RAW_PDF_DIR / filename
        if not pdf_path.exists():
            logger.warning("Skipping missing PDF: %s", pdf_path)
            continue

        logger.info("Processing %s ...", doc_name)
        pages = extract_pdf(pdf_path, doc_name)
        pages = clean_pages(pages)
        blocks = parse_structure(pages)
        chunks = chunk_blocks(blocks)
        logger.info(
            "  %d pages -> %d structured blocks -> %d chunks", len(pages), len(blocks), len(chunks)
        )
        all_chunks.extend(chunks)

    out_path = PROCESSED_DIR / "chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in all_chunks], f, ensure_ascii=False, indent=2)

    logger.info("Wrote %d total chunks to %s", len(all_chunks), out_path)


if __name__ == "__main__":
    main()
