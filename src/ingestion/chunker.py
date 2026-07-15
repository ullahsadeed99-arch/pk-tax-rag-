"""Splits StructuredBlocks into token-bounded Chunks for embedding.

Uses tiktoken so token counts line up with what the OpenAI embedding/chat
APIs actually see. Tables are never split (breaking a markdown table mid-row
makes it useless for retrieval) -- only long section text is windowed with
overlap, using a sliding window over the block's token stream.
"""
from __future__ import annotations

import tiktoken

from src.config import settings
from src.ingestion.models import Chunk, StructuredBlock

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _make_chunk_id(block: StructuredBlock, block_index: int, window_index: int) -> str:
    stem = block.source_file.rsplit(".", 1)[0]
    return f"{stem}-b{block_index}-w{window_index}"


def _chunk_block(block: StructuredBlock, block_index: int) -> list[Chunk]:
    tokens = _ENCODING.encode(block.text)

    if block.is_table or len(tokens) <= settings.chunk_size_tokens:
        return [
            Chunk(
                chunk_id=_make_chunk_id(block, block_index, 0),
                doc_name=block.doc_name,
                source_file=block.source_file,
                page_number=block.page_number,
                chapter=block.chapter,
                part=block.part,
                section=block.section,
                section_title=block.section_title,
                is_table=block.is_table,
                text=block.text,
                token_count=len(tokens),
            )
        ]

    step = max(1, settings.chunk_size_tokens - settings.chunk_overlap_tokens)
    chunks: list[Chunk] = []
    start = 0
    window_index = 0
    while start < len(tokens):
        window = tokens[start : start + settings.chunk_size_tokens]
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(block, block_index, window_index),
                doc_name=block.doc_name,
                source_file=block.source_file,
                page_number=block.page_number,
                chapter=block.chapter,
                part=block.part,
                section=block.section,
                section_title=block.section_title,
                is_table=block.is_table,
                text=_ENCODING.decode(window),
                token_count=len(window),
            )
        )
        window_index += 1
        start += step
    return chunks


def chunk_blocks(blocks: list[StructuredBlock]) -> list[Chunk]:
    """Turn StructuredBlocks into the final Chunks stored in the vector index."""
    chunks: list[Chunk] = []
    for block_index, block in enumerate(blocks):
        chunks.extend(_chunk_block(block, block_index))
    return chunks
