"""Cleans raw extracted page text: strips repeated running headers/footers,
normalizes whitespace, and fixes common PDF-extraction artifacts.
"""
from __future__ import annotations
import re
from collections import Counter

from src.ingestion.models import RawPage

# Lines like "Income Tax Rules, 2002", "Sales Tax Act, 1990", a lone page
# number, or "Chapter II – Charge of Tax________" recur on almost every
# page as running headers/footers. We detect them by frequency rather than
# hardcoding per-document strings, so the cleaner generalizes to new PDFs.
_PAGE_NUMBER_LINE = re.compile(r"^\(?\d{1,4}\)?$")
_UNDERSCORE_RULE = re.compile(r"_{5,}")


def _normalize_line(line: str) -> str:
    """Collapse whitespace/digits so near-identical header variants
    (differing only by page number) hash to the same key.
    """
    line = re.sub(r"\d+", "#", line.strip())
    return re.sub(r"\s+", " ", line)


def _find_boilerplate_lines(pages: list[RawPage]) -> set[str]:
    counts: Counter[str] = Counter()
    for p in pages:
        seen_this_page = set()
        for raw_line in p.text.splitlines():
            norm = _normalize_line(raw_line)
            if norm and norm not in seen_this_page:
                counts[norm] += 1
                seen_this_page.add(norm)
    threshold = max(5, int(len(pages) * 0.35))
    return {norm for norm, c in counts.items() if c >= threshold}


def clean_pages(pages: list[RawPage]) -> list[RawPage]:
    """Return new RawPage objects with boilerplate headers/footers removed
    and whitespace normalized. Table markdown is left untouched.
    """
    boilerplate = _find_boilerplate_lines(pages)
    cleaned: list[RawPage] = []

    for p in pages:
        kept_lines = []
        for raw_line in p.text.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if _UNDERSCORE_RULE.search(stripped):
                continue
            if _PAGE_NUMBER_LINE.match(stripped):
                continue
            if _normalize_line(raw_line) in boilerplate:
                continue
            kept_lines.append(stripped)

        text = "\n".join(kept_lines)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        cleaned.append(
            RawPage(
                doc_name=p.doc_name,
                source_file=p.source_file,
                page_number=p.page_number,
                text=text,
                tables_markdown=p.tables_markdown,
            )
        )
    return cleaned
