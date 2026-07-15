"""Extracts text and tables from the source PDFs.

Uses PyMuPDF (fitz) for fast text extraction and pdfplumber for table
detection (fitz flattens tables into jumbled column-major text, which is
useless for the rate/Schedule tables in these documents).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import fitz
import pdfplumber

from src.ingestion.models import RawPage

logger = logging.getLogger(__name__)

# Matches a bare section-number cell (e.g. "18.", "23A.") or a bare page-number
# cell -- the two column shapes that make up an "Arrangement of Sections" table
# of contents, which pdfplumber extracts as a real table since it's grid-laid-out
# in the PDF but which is navigational front matter, not legal content.
_SECTION_CELL_RE = re.compile(r"^\d{1,3}[A-Z]{0,3}\.?$")
_PAGE_CELL_RE = re.compile(r"^\d{1,4}$")


def _looks_like_toc(rows: list[list[str]]) -> bool:
    """True if a table's first/last columns match the (section no. | ... |
    page no.) shape of a table of contents rather than real tabular data
    like a rate schedule.
    """
    if len(rows) < 4:
        return False
    matches = 0
    for row in rows:
        if len(row) < 2:
            continue
        first, last = row[0].strip(), row[-1].strip()
        first_ok = not first or bool(_SECTION_CELL_RE.match(first))
        last_ok = not last or bool(_PAGE_CELL_RE.match(last))
        if first_ok and last_ok:
            matches += 1
    return matches / len(rows) >= 0.6


def _table_to_markdown(table: list[list[Optional[str]]]) -> Optional[str]:
    """Convert a pdfplumber table (list of rows) into a markdown table string.
    Returns None for tables that are too small/degenerate to be useful, or
    that are actually a table-of-contents rather than real data.
    """
    if not table or len(table) < 2:
        return None
    rows = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in table]
    rows = [r for r in rows if any(c for c in r)]
    if len(rows) < 2:
        return None
    if _looks_like_toc(rows):
        return None
    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    header, body = rows[0], rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * n_cols) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_pdf(pdf_path: Path, doc_name: str) -> list[RawPage]:
    """Extract every page of a PDF as text plus any detected tables."""
    pages: list[RawPage] = []
    fitz_doc = fitz.open(pdf_path)

    with pdfplumber.open(pdf_path) as plumber_doc:
        n_pages = len(fitz_doc)
        for i in range(n_pages):
            text = fitz_doc[i].get_text("text")

            tables_md: list[str] = []
            try:
                raw_tables = plumber_doc.pages[i].extract_tables()
                for t in raw_tables:
                    md = _table_to_markdown(t)
                    if md:
                        tables_md.append(md)
            except Exception as exc:  # pdfplumber can choke on malformed pages
                logger.warning("Table extraction failed on %s page %d: %s", doc_name, i + 1, exc)

            pages.append(
                RawPage(
                    doc_name=doc_name,
                    source_file=pdf_path.name,
                    page_number=i + 1,
                    text=text,
                    tables_markdown=tables_md,
                )
            )

    fitz_doc.close()
    logger.info("Extracted %d pages from %s", len(pages), doc_name)
    return pages
