# AuraRAG v3.2.0 — Release Notes

**Date:** 2026-05-16  
**Tag:** `v3.2.0`  
**Type:** Patch release (bug fixes only — no breaking changes)

---

## Summary

8 bugs fixed across the API layer, engine, and session store. No API contract changes — v3.2.0 is a drop-in upgrade from v3.1.0.

## Critical fixes (upgrade recommended)

**`top_k` was silently ignored** — every query returned exactly 5 sources regardless of what you sent in the request body. Fixed end-to-end in all three layers.

**Streaming exceptions caused silent hangs** — if the LLM call failed mid-stream the SSE connection hung open until the client timed out. Now properly propagated as `{"error": "..."}` frames.

**Dedup hash set lost on restart** — uploading the same file twice after a restart would re-index all chunks. Fixed by persisting the hash set alongside the BM25 pickle (with atomic write).

**`TextLoader` crashed on non-UTF-8 `.txt` files** in C-locale containers. Fixed with explicit `encoding="utf-8", autodetect_encoding=True`.

## Other fixes

- `SessionMemoryStore.clear()` now correctly removes the session from both stores (was leaking a stale timestamp)
- `LOG_LEVEL=DEBUG` in `.env` now actually works
- Anthropic model IDs corrected: `claude-opus-4-6`, `claude-sonnet-4-6`
- `IngestResponse.message` type corrected from `Optional[str]` → `str`

## Upgrade from v3.1.0

```bash
git pull
pip install -r requirements.txt   # adds python-magic
# or
docker compose pull && docker compose up --build
```

No database migrations required. Existing ChromaDB and BM25 pickle files are compatible.

## Full changelog

See [CHANGELOG.md](CHANGELOG.md).
