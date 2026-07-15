"""Generates a synthetic question-answer dataset from data/processed/chunks.json
for fine-tuning a Pakistani tax law assistant.

Generation runs against Groq's free API (OpenAI-compatible endpoint) rather
than OpenAI, since generating ~100k examples at OpenAI prices/quotas isn't
viable on a free account. Get a free key at https://console.groq.com/keys
and put it in GROQ_API_KEY in the project-root .env (see .env.example).

Each chunk gets a token-weighted quota of Q&A pairs (longer sections produce
more questions than short ones). Pairs are generated in small batches per
API call -- asking a model for dozens of pairs in one response risks
truncated/malformed JSON, so generation is capped at BATCH_SIZE per call and
looped until each chunk's quota is met.

Output is two files:
  data/processed/finetune_raw.jsonl    -- one {chunk_id, doc_name, section,
                                           question, answer} object per line;
                                           the source of truth, safe to resume from.
  data/processed/finetune_dataset.jsonl -- the same data reshaped into the
                                           standard chat {"messages": [...]}
                                           format used by OpenAI-style fine-tuning
                                           and most open-source LoRA trainers.

Usage:
  python scripts/generate_finetune_data.py --limit-chunks 5   # dry run
  python scripts/generate_finetune_data.py                    # full run (resumable)
  python scripts/generate_finetune_data.py --rebuild-only      # just re-emit
                                                                 the final file
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from tqdm import tqdm

from src.config import PROCESSED_DIR, ROOT_DIR

load_dotenv(ROOT_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise SystemExit(
        "GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys "
        "and put it in a .env file at the project root (see .env.example)."
    )

CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
RAW_PATH = PROCESSED_DIR / "finetune_raw.jsonl"
FINAL_PATH = PROCESSED_DIR / "finetune_dataset.jsonl"

TARGET_TOTAL = 100_000
MIN_PER_CHUNK = 3
MAX_PER_CHUNK = 80
BATCH_SIZE = 10          # pairs requested per API call
CONCURRENCY = 1          # Groq free tier caps at 12k tokens/minute -- concurrency just
                         # causes simultaneous 429s, it doesn't raise real throughput
MAX_ATTEMPTS_PER_BATCH = 5      # genuine (non-rate-limit) errors
MAX_RATE_LIMIT_RETRIES = 15     # rate limits get their own bounded retry budget, with
                                 # growing backoff, so a stuck request can't spin forever
MIN_REQUEST_INTERVAL = 6.5      # seconds between dispatches, paced to stay under the
                                 # ~12k TPM budget proactively instead of reacting to 429s
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_RETRY_SECONDS_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)
_last_request_at = 0.0
_pacing_lock = asyncio.Lock()


async def _pace() -> None:
    """Enforce a minimum gap between successive request dispatches so we stay
    under the free-tier TPM budget instead of firing bursts that 429."""
    global _last_request_at
    async with _pacing_lock:
        wait = _last_request_at + MIN_REQUEST_INTERVAL - asyncio.get_event_loop().time()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_request_at = asyncio.get_event_loop().time()

SYSTEM_PROMPT = (
    "You are a Pakistani tax law assistant. Answer questions using ONLY the "
    "provided excerpts from the Income Tax Ordinance 2001, Income Tax Rules "
    "2002, Sales Tax Act 1990, and Sales Tax Rules 2006. Cite the source for "
    "every claim using the matching [n] marker from the excerpts below. If "
    "the excerpts don't contain enough information to answer, say so plainly "
    "instead of guessing."
)

GEN_SYSTEM_PROMPT = (
    "You are generating synthetic training data (question-answer pairs) to "
    "fine-tune a Pakistani tax law assistant. Base every answer strictly on "
    "the provided excerpt -- never invent facts, section numbers, or figures "
    "not present in the text. If the excerpt is a markdown table, include "
    "some questions that look up specific values from it. Return only valid JSON."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def allocate_counts(chunks: list[dict], target_total: int) -> dict[str, int]:
    """Token-weighted quota per chunk_id, clipped to [MIN_PER_CHUNK, MAX_PER_CHUNK],
    with remainder distributed by largest fractional part so totals sum exactly.
    """
    weights = {c["chunk_id"]: max(c["token_count"], 1) for c in chunks}
    total_weight = sum(weights.values())

    raw = {cid: target_total * w / total_weight for cid, w in weights.items()}
    counts = {cid: min(MAX_PER_CHUNK, max(MIN_PER_CHUNK, round(v))) for cid, v in raw.items()}

    diff = target_total - sum(counts.values())
    increasing = diff > 0
    order = sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=increasing)
    idx = 0
    stalled = 0
    while diff != 0 and stalled < len(order):
        cid, _ = order[idx % len(order)]
        if increasing and counts[cid] < MAX_PER_CHUNK:
            counts[cid] += 1
            diff -= 1
            stalled = 0
        elif not increasing and counts[cid] > MIN_PER_CHUNK:
            counts[cid] -= 1
            diff += 1
            stalled = 0
        else:
            stalled += 1
        idx += 1
    return counts


def load_existing_counts(raw_path: Path) -> Counter:
    counts: Counter = Counter()
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                counts[json.loads(line)["chunk_id"]] += 1
    return counts


def _chunk_label(chunk: dict) -> str:
    parts = [chunk["doc_name"]]
    if chunk.get("section"):
        label = f"Section {chunk['section']}"
        if chunk.get("section_title"):
            label += f" - {chunk['section_title']}"
        parts.append(label)
    return ", ".join(parts)


def _parse_pairs(content: str) -> list[dict]:
    """Parse the model's JSON response, tolerating markdown code fences that
    open models sometimes add even when json_object mode is requested.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    data = json.loads(text)
    pairs = data.get("pairs", []) if isinstance(data, dict) else data
    return [
        p for p in pairs
        if isinstance(p, dict) and p.get("question") and p.get("answer")
    ]


async def _generate_batch(client: AsyncOpenAI, chunk: dict, n: int) -> list[dict]:
    user_prompt = (
        f"Source: {_chunk_label(chunk)}\n\n"
        f'Excerpt:\n"""\n{chunk["text"]}\n"""\n\n'
        f"Generate exactly {n} diverse question-answer pairs a Pakistani taxpayer, "
        "accountant, or tax lawyer might realistically ask about this excerpt. Vary "
        "the style across: direct factual lookups, definitions, procedural/how-to "
        'questions, applied scenarios (e.g. "If I earn X, ..."), and references to '
        "specific clauses. Keep answers accurate, self-contained, and grounded only "
        'in the excerpt above; do not say "the excerpt" or "the text" in the answer -- '
        "answer as a knowledgeable assistant would.\n\n"
        f'Return a JSON object: {{"pairs": [{{"question": "...", "answer": "..."}}, ...]}} '
        f"with exactly {n} items."
    )

    attempt = 0
    rate_limit_retries = 0
    while attempt < MAX_ATTEMPTS_PER_BATCH:
        try:
            await _pace()
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            pairs = _parse_pairs(resp.choices[0].message.content)
            if pairs:
                return pairs
            attempt += 1
        except RateLimitError as exc:
            # Free-tier token-per-minute throttling, not a real failure -- wait
            # out the window Groq tells us about (growing if it keeps happening,
            # in case it's a longer-lived cap) but bounded, so a persistently
            # unavailable model can't spin forever.
            rate_limit_retries += 1
            if rate_limit_retries > MAX_RATE_LIMIT_RETRIES:
                logger.warning(
                    "Giving up on %s after %d rate-limit retries", chunk["chunk_id"], rate_limit_retries
                )
                return []
            m = _RETRY_SECONDS_RE.search(str(exc))
            wait = float(m.group(1)) + 0.5 if m else min(60.0, 5.0 * rate_limit_retries)
            logger.info(
                "Rate limited on %s (retry %d/%d), waiting %.1fs",
                chunk["chunk_id"], rate_limit_retries, MAX_RATE_LIMIT_RETRIES, wait,
            )
            await asyncio.sleep(wait)
        except Exception as exc:
            attempt += 1
            logger.warning(
                "Batch failed for %s (attempt %d/%d): %s",
                chunk["chunk_id"], attempt, MAX_ATTEMPTS_PER_BATCH, exc,
            )
            await asyncio.sleep(2 * attempt)
    return []


async def _generate_for_chunk(
    client: AsyncOpenAI,
    chunk: dict,
    n_needed: int,
    sem: asyncio.Semaphore,
    lock: asyncio.Lock,
    out_f,
    pbar: tqdm,
) -> None:
    remaining = n_needed
    while remaining > 0:
        batch_n = min(BATCH_SIZE, remaining)
        async with sem:
            pairs = await _generate_batch(client, chunk, batch_n)
        if not pairs:
            logger.warning(
                "Giving up on %s after exhausting retries (%d pairs still short)",
                chunk["chunk_id"], remaining,
            )
            break
        async with lock:
            for p in pairs[:remaining]:
                out_f.write(json.dumps({
                    "chunk_id": chunk["chunk_id"],
                    "doc_name": chunk["doc_name"],
                    "section": chunk.get("section"),
                    "question": p["question"].strip(),
                    "answer": p["answer"].strip(),
                }, ensure_ascii=False) + "\n")
            out_f.flush()
            pbar.update(min(len(pairs), remaining))
        remaining -= len(pairs)


def build_finetune_file() -> int:
    """Reshape finetune_raw.jsonl into OpenAI chat fine-tuning format."""
    n = 0
    with open(RAW_PATH, encoding="utf-8") as src, open(FINAL_PATH, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dst.write(json.dumps({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ]
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


async def run(
    target_total: int,
    limit_chunks: int | None,
    concurrency: int,
    doc_filter: str | None = None,
) -> None:
    chunks = load_chunks()
    if doc_filter:
        needle = doc_filter.lower()
        chunks = [c for c in chunks if needle in c["doc_name"].lower() or needle in c["source_file"].lower()]
        if not chunks:
            raise SystemExit(f"No chunks matched --doc {doc_filter!r}. Check data/processed/chunks.json doc_name values.")
    if limit_chunks:
        chunks = chunks[:limit_chunks]

    counts = allocate_counts(chunks, target_total)
    existing = load_existing_counts(RAW_PATH)

    todo = [(c, counts[c["chunk_id"]] - existing.get(c["chunk_id"], 0)) for c in chunks]
    todo = [(c, n) for c, n in todo if n > 0]

    total_needed = sum(n for _, n in todo)
    if total_needed == 0:
        logger.info("Nothing to do -- quota already met for all %d chunks.", len(chunks))
        return

    logger.info(
        "Generating %d Q&A pairs across %d chunks (already have %d from a prior run).",
        total_needed, len(todo), sum(existing.values()),
    )

    client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(RAW_PATH, "a", encoding="utf-8") as out_f, tqdm(total=total_needed, unit="pair") as pbar:
        tasks = [
            _generate_for_chunk(client, chunk, n, sem, lock, out_f, pbar)
            for chunk, n in todo
        ]
        await asyncio.gather(*tasks)

    n_final = build_finetune_file()
    logger.info("Wrote %d examples to %s", n_final, FINAL_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET_TOTAL)
    parser.add_argument("--limit-chunks", type=int, default=None, help="Only process the first N chunks (dry run)")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--doc", type=str, default=None, help="Only generate for chunks whose doc_name/source_file contains this substring (case-insensitive), e.g. 'Sales Tax Act'")
    parser.add_argument("--rebuild-only", action="store_true", help="Skip generation, just rebuild finetune_dataset.jsonl from finetune_raw.jsonl")
    args = parser.parse_args()

    if args.rebuild_only:
        n = build_finetune_file()
        logger.info("Rebuilt %s with %d examples.", FINAL_PATH, n)
        return

    asyncio.run(run(args.target, args.limit_chunks, args.concurrency, args.doc))


if __name__ == "__main__":
    main()
