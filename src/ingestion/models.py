"""Shared data structures for the ingestion pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class RawPage:
    """One page of extracted content before structure parsing."""
    doc_name: str
    source_file: str
    page_number: int  # 1-indexed
    text: str
    tables_markdown: list[str] = field(default_factory=list)


@dataclass
class StructuredBlock:
    """A contiguous piece of legal text tagged with its position in the
    document hierarchy (chapter/part/section/clause), before token chunking.
    """
    doc_name: str
    source_file: str
    page_number: int
    chapter: Optional[str]
    part: Optional[str]
    section: Optional[str]       # e.g. "113" or "43" or "158I"
    section_title: Optional[str]  # e.g. "Minimum tax"
    text: str
    is_table: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    """Final unit stored in the vector database."""
    chunk_id: str
    doc_name: str
    source_file: str
    page_number: int
    chapter: Optional[str]
    part: Optional[str]
    section: Optional[str]
    section_title: Optional[str]
    is_table: bool
    text: str
    token_count: int

    def metadata(self) -> dict:
        """Chroma metadata values must be str/int/float/bool/None."""
        return {
            "doc_name": self.doc_name,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chapter": self.chapter or "",
            "part": self.part or "",
            "section": self.section or "",
            "section_title": self.section_title or "",
            "is_table": self.is_table,
        }
