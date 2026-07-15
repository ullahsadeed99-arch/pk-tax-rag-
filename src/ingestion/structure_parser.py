"""Parses cleaned pages into StructuredBlocks by tracking the chapter/part/
section headings that organize Pakistani tax statutes.

Detection is heuristic/pattern-based rather than hardcoded per document (same
approach as cleaner.py) so it generalizes across the Ordinance, Rules, and
Acts:
- CHAPTER/PART headings are a "CHAPTER <roman>" / "PART <roman>" line, with
  the title on the next non-heading line.
- Section headings are "<number>. <title>.<dash><body>" lines, where
  <number> is digits optionally followed by amendment-suffix letters (e.g.
  "203H", "236HA") and <dash> is an em/en dash or hyphen.
- Front matter -- title pages and the "Arrangement of Sections" table of
  contents -- is filtered out after the fact rather than by page range:
  a TOC entry has no real body text before the next heading, so any section
  whose accumulated text falls under MIN_SECTION_BODY_CHARS is dropped as
  noise. Table-of-contents lines that use a dotted page-number leader
  (e.g. "Chapter-II ......... 29") are dropped outright so they can't be
  mistaken for real headings.
"""
from __future__ import annotations

import re
from typing import Optional

from src.ingestion.models import RawPage, StructuredBlock

_AMENDMENT_PREFIX_RE = re.compile(r"^\d{1,2}\[")
_CHAPTER_RE = re.compile(r"^CHAPTER[\s-]+([IVXLCDM0-9]+[A-Z]{0,2})\b", re.IGNORECASE)
_PART_RE = re.compile(r"^PART[\s-]+([IVXLCDM0-9]+[A-Z]{0,2})\b", re.IGNORECASE)
_SECTION_RE = re.compile(r"^(\d{1,3}[A-Z]{0,3})\.\s*(.*)$")
_TITLE_BODY_SPLIT_RE = re.compile(r"(.{0,150}?)\.\s*[—–-]+\s*(.*)", re.DOTALL)

# A run of dot/ellipsis leader chars (optionally space-separated -- some
# PDFs render the leader as a mix of "." runs and "…" glyphs with a space
# where the two meet) ending the line in a page number.
_TOC_LEADER_RE = re.compile(r"(?:[.…] ?){4,}\d+\s*$")

# Below this, a "section" is just a TOC entry or an omitted/repealed stub,
# not real legal text worth indexing. Calibrated against the Income Tax
# Ordinance's "Arrangement of Sections" front matter, whose entries top out
# around 106-129 chars -- comfortably under real operative section bodies.
MIN_SECTION_BODY_CHARS = 140


def _strip_amendment_prefix(line: str) -> str:
    """Strip a leading "<n>[" amendment-insertion marker (e.g. "3[99A. ...")
    so heading regexes still match text wrapped by a legislative amendment.
    """
    return _AMENDMENT_PREFIX_RE.sub("", line, count=1)


def _is_heading(line: str) -> bool:
    check = _strip_amendment_prefix(line)
    return bool(_CHAPTER_RE.match(check) or _PART_RE.match(check) or _SECTION_RE.match(check))


def _looks_like_date(number: str, rest: str) -> bool:
    """True if a "<number>. <rest>" match is actually a "DD.MM.YYYY" date
    from footnote text (e.g. "05.06.2010"), not a real section heading. Real
    section numbers never start with 0, and are never followed by another
    "<digits>." run.
    """
    return number.startswith("0") or bool(re.match(r"^\d{1,2}\.\d", rest))


def _split_title(text: str) -> tuple[Optional[str], str]:
    """Split "Title.--- body..." into (title, body). Falls back to
    (None, text) when no title/body divider is found near the start (e.g.
    a section whose body has no dash convention, or whose number and title
    were merged onto one line differently by the PDF extractor).
    """
    m = _TITLE_BODY_SPLIT_RE.match(text)
    if not m:
        return None, text
    return m.group(1).strip(), m.group(2).strip()


def parse_structure(pages: list[RawPage]) -> list[StructuredBlock]:
    """Turn cleaned pages into StructuredBlocks by tracking chapter/part/
    section headings as they're encountered, in document order.
    """
    if not pages:
        return []

    doc_name = pages[0].doc_name
    source_file = pages[0].source_file

    blocks: list[StructuredBlock] = []
    chapter: Optional[str] = None
    part: Optional[str] = None
    section: Optional[str] = None
    section_start_page: Optional[int] = None
    body_lines: list[str] = []

    def flush() -> None:
        nonlocal body_lines
        if section is not None:
            text = "\n".join(body_lines).strip()
            if len(text) >= MIN_SECTION_BODY_CHARS:
                title, remainder = _split_title(text)
                blocks.append(
                    StructuredBlock(
                        doc_name=doc_name,
                        source_file=source_file,
                        page_number=section_start_page or 0,
                        chapter=chapter,
                        part=part,
                        section=section,
                        section_title=title,
                        text=remainder,
                        is_table=False,
                    )
                )
        body_lines = []

    for page in pages:
        lines = [ln.strip() for ln in page.text.splitlines() if ln.strip()]
        lines = [ln for ln in lines if not _TOC_LEADER_RE.search(ln)]

        idx = 0
        while idx < len(lines):
            line = lines[idx]
            check = _strip_amendment_prefix(line)

            m_chapter = _CHAPTER_RE.match(check)
            if m_chapter:
                flush()
                label = m_chapter.group(1).upper()
                title = None
                if idx + 1 < len(lines) and not _is_heading(lines[idx + 1]):
                    title = lines[idx + 1]
                    idx += 1
                chapter = f"{label} - {title}" if title else label
                part = None
                section = None
                idx += 1
                continue

            m_part = _PART_RE.match(check)
            if m_part:
                flush()
                label = m_part.group(1).upper()
                title = None
                if idx + 1 < len(lines) and not _is_heading(lines[idx + 1]):
                    title = lines[idx + 1]
                    idx += 1
                part = f"{label} - {title}" if title else label
                section = None
                idx += 1
                continue

            m_section = _SECTION_RE.match(check)
            if m_section and not _looks_like_date(m_section.group(1), m_section.group(2)):
                flush()
                section = m_section.group(1)
                section_start_page = page.page_number
                rest = m_section.group(2).strip()
                body_lines = [rest] if rest else []
                idx += 1
                continue

            if section is not None:
                body_lines.append(line)
            idx += 1

        for table_md in page.tables_markdown:
            blocks.append(
                StructuredBlock(
                    doc_name=doc_name,
                    source_file=source_file,
                    page_number=page.page_number,
                    chapter=chapter,
                    part=part,
                    section=section,
                    section_title=None,
                    text=table_md,
                    is_table=True,
                )
            )

    flush()
    return blocks
