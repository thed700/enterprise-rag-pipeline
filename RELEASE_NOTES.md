# AuraRAG v3.6 — Release Notes

**Date:** 2026-05-23
**Tag:** `v3.6`
**Type:** Bug-fix release — no breaking changes, drop-in upgrade from v3.5

---

## Summary

Six production bugs fixed across the UI, streaming pipeline, model registry,
dependency tree, and CI workflow. Every fix addresses a real failure observed
in the deployed Hugging Face Space or the GitHub Actions build. No API contract
changes — v3.6 is a drop-in replacement for v3.5.

**Upgrade strongly recommended** — two of the six bugs cause the app to crash
on every message, and a third silently prevents source cards from ever rendering
in streaming mode.

---

## Critical fixes (upgrade immediately)

### Streamlit avatar crash — every message (BUG-AVATAR)

**Symptom:** `streamlit.errors.StreamlitAPIException: Failed to load the provided avatar value as an image.` — the app crashes on every message render, in both the chat history renderer and the live response block.

**Root cause:** `app/ui.py` used `avatar="◈"` at two call sites (line 824 and line 1111). The `◈` character is a Unicode geometric shape with no emoji presentation selector. Streamlit's `st.chat_message()` avatar validator requires a genuine emoji codepoint (with emoji presentation) or a valid image URL — pure Unicode symbols are rejected.

**Fix:** Replaced `"◈"` with `"🤖"` for the assistant and `"🧑"` for the user at both call sites. Both are valid emoji codepoints that pass Streamlit's validator.

---

### Hidden `<<META>>` JSON block leaked into the chat stream (BUG-META-LEAK)

**Symptom:** Users would see raw JSON like `<<META>>{"hallucination_risk": 0.23}<<END_META>>` appear at the end of streamed responses in the chat bubble.

**Root cause:** The generate node appends a hidden `<<META>>{...}<<END_META>>` block to every answer so the engine can extract a self-assessed hallucination risk score. In `stream_query()`, raw LLM tokens — including those forming this block — were yielded directly to the SSE stream before any stripping occurred. The clean extraction only happened after streaming finished, at which point the dirty content had already been sent to the client.

**Fix:** `stream_query_with_meta()` now tracks a rolling buffer. Once a `<<META>>` token is detected, subsequent tokens are suppressed until the closing `<<END_META>>` completes the block. A secondary `_strip_meta()` call in `ui.py` provides a client-side safety net for any edge cases.

---

### Source cards and pipeline trace never rendered in streaming mode (BUG-SSE-SOURCES)

**Symptom:** Source reference cards and the pipeline trace badge (`rewrite → retrieve → grade → generate`) appeared after non-streaming queries but were always empty after streaming queries, even though the backend computed them correctly.

**Root cause:** The SSE meta frame emitted by `/query/stream` only contained `session_id`, `token_count`, `provider`, `model`, and `top_k`. The fields `sources`, `pipeline_trace`, `graded_chunks`, and `reflect_loops` were never included because the streaming code path had no mechanism to surface the final graph state back to the router.

**Fix:** `stream_query_with_meta()` is a new generator that yields token strings during streaming, then yields a single `Dict` as its final item containing the complete result metadata. The router reads this dict and includes all fields in the meta SSE frame. The UI parses them to render source cards and the trace badge identically in both streaming and non-streaming modes.

---

## Other fixes

### Anthropic model IDs rejected by the API (BUG-R)

`app/backend/models.py` listed `claude-opus-4-5-20251101`, `claude-sonnet-4-5-20251022`, and `claude-haiku-4-5-20251001` as the Anthropic model identifiers. These versioned date-suffix strings are not valid Anthropic API model IDs and cause API calls to fail with a 404.

Fixed: replaced with the correct short-form model strings `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-sonnet-4-5`, `claude-haiku-4-5`, alongside the already-correct `claude-3-5-sonnet-20241022` and `claude-3-5-haiku-20241022`.

---

### `unstructured` dependency added ~2 GB and caused pip timeouts (BUG-UNSTRUCTURED)

`requirements.txt` listed `unstructured>=0.14.0`. Searching the entire codebase reveals zero imports of this package — it is not used anywhere. However, `unstructured`'s dependency tree pulls in `detectron2`, `tesseract` OCR bindings, `poppler` bindings, and numerous other heavy packages totalling approximately 2 GB of additional installation weight. This caused `pip install` to time out on Hugging Face Spaces during Space rebuilds and produced a Docker image significantly larger than necessary.

Fixed: removed from `requirements.txt` entirely.

---

### Android APK build failed on every CI run (BUG-SDK-LICENSE)

**Symptom:** CI log shows `Skipping following packages as the license is not accepted: Android SDK Build-Tools 37` followed by `Aidl not found, please install it.` and `Error: Process completed with exit code 1`.

**Root cause:** `sdkmanager` prompts for Google Android SDK license acceptance interactively when installing `build-tools;37.0.0`. GitHub Actions runners are non-interactive, so the prompt was never answered, the package was silently skipped, and AIDL (part of build-tools) was unavailable. Without AIDL, Buildozer cannot compile the APK.

**Fix:** The workflow now pre-writes the authoritative Google SDK license hash files to the expected path before Buildozer runs. `sdkmanager` finds these files, treats the licenses as already accepted, and installs build-tools without prompting.

---

## Files changed

| File | Change |
|------|--------|
| `app/ui.py` | BUG-AVATAR: `avatar="◈"` → `"🤖"` / `"🧑"` at L824 and L1111. Client-side `_strip_meta()` safety net added. SSE source card rendering wired to meta frame. |
| `app/engine/pipeline.py` | BUG-META-LEAK: token suppression during `<<META>>` accumulation. BUG-SSE-SOURCES: new `stream_query_with_meta()` generator yielding final metadata dict. |
| `app/routers/query.py` | BUG-SSE-SOURCES: router reads final dict from `stream_query_with_meta()` and emits complete meta SSE frame. |
| `app/backend/models.py` | BUG-R: corrected Anthropic model ID strings. |
| `requirements.txt` | BUG-UNSTRUCTURED: removed `unstructured>=0.14.0`. |
| `.github/workflows/build-artifacts.yml` | BUG-SDK-LICENSE: pre-write SDK license hashes before Buildozer step. `buildozer.spec` version bumped to 3.6.0. |
| `app/utils.py` | `APP_VERSION` bumped to `"3.6"`. |
| `Dockerfile` | Version label 3.6.0. Added `AURARAG_CACHE_DIR=/tmp/aurarag` for HF Spaces read-only filesystem compatibility. Added `curl` to runtime image for healthcheck. |
| `entrypoint.sh` | Improved healthcheck loop with configurable `MAX_WAIT` timeout. `AURARAG_CACHE_DIR` directory created on startup. |
| `app/engine/__init__.py` | Version comment updated. |
| `app/engine.py` | Version comment updated. |
| `README.md` | Full rewrite: updated model IDs, corrected SSE event schema to include `sources` and `pipeline_trace`, roadmap updated. |
| `CHANGELOG.md` | v3.6 section added. |

---

## Upgrade from v3.5

```bash
git pull origin main
git checkout v3.6

# Dependencies changed: unstructured removed (saves ~2 GB)
pip install -r requirements.txt

# Or with Docker
docker compose pull && docker compose up --build
```

No database migrations required. Existing ChromaDB collections and BM25 pickle files are fully compatible — the persistence format is unchanged.

---

## Running the test suite

```bash
pytest tests/ -v
```

All existing regression tests pass. No new tests were added in this release — the bugs were identified from live runtime errors and CI logs rather than test failures.
