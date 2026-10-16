"""
tests/test_v37.py — regression tests for the AuraRAG v3.7 bug fixes.

Run: pytest tests/ -v
Uses a REAL Chroma + BM25 + LangGraph state machine; only the embedding model
and cross-encoder are replaced with tiny deterministic fakes.
"""

import asyncio
import io
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessageChunk


# ─────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────

class _FakeCrossEncoder:
    def __init__(self, *a, **k):
        pass

    def predict(self, pairs):
        return [0.0] * len(pairs)


@pytest.fixture
def real_engine(tmp_path, monkeypatch):
    """A RAGEngine on real Chroma/BM25 in a temp dir, with fake models."""
    from app.engine import pipeline

    collection = f"t{abs(hash(str(tmp_path)))}"   # same name for every build() in a test
    # Must stay patched for the whole test: ingest_documents() reads it later too.
    monkeypatch.setattr(pipeline.settings, "CHROMA_COLLECTION", collection)

    def build():
        with patch.object(pipeline, "CrossEncoder", _FakeCrossEncoder), \
             patch.object(pipeline, "HuggingFaceEmbeddings",
                          lambda **kw: DeterministicFakeEmbedding(size=32)), \
             patch.object(pipeline, "_resolve_persist_dir", return_value=tmp_path):
            return pipeline.RAGEngine()

    build.path = tmp_path
    return build


def _doc(text, source="a.txt"):
    return Document(page_content=text, metadata={"source": source})


# ─────────────────────────────────────────────
# BUG-STREAM-LEAK / BUG-META-UNCLOSED
# ─────────────────────────────────────────────

class TestMetaStripper:
    @staticmethod
    def _run(tokens):
        from app.engine.pipeline import _MetaStripper
        st = _MetaStripper()
        return "".join(st.feed(t) for t in tokens) + st.flush()

    def test_plain_text_passes_through(self):
        assert self._run(["Hello ", "world", "!"]) == "Hello world!"

    def test_whole_block_in_one_token(self):
        out = self._run(['Answer.\n<<META>>{"hallucination_risk": 0.1}<<END_META>>'])
        assert out == "Answer."

    def test_marker_split_across_tokens(self):
        toks = ["Answer.", "\n", "<<", "META", ">>", '{"hallucination_risk"', ": 0.2}", "<<END", "_META", ">>"]
        out = self._run(toks)
        assert out == "Answer."
        assert "<" not in out and "META" not in out and ">" not in out

    def test_every_possible_split_point(self):
        full = 'The answer is 42.\n<<META>>{"hallucination_risk": 0.3}<<END_META>>'
        for i in range(1, len(full)):
            assert self._run([full[:i], full[i:]]) == "The answer is 42.", i

    def test_char_by_char(self):
        full = 'ok <b>fine</b> < not a marker <<META>>{"x":1}<<END_META>>'
        assert self._run(list(full)) == "ok <b>fine</b> < not a marker"

    def test_unclosed_block_is_dropped(self):
        assert self._run(["Answer.", "<<META>>{\"hallucination_ri"]) == "Answer."

    def test_lone_angle_brackets_are_not_lost(self):
        assert self._run(["a << b", " and c < d"]) == "a << b and c < d"

    def test_extract_risk_handles_unclosed_block(self):
        from app.engine.pipeline import _extract_hallucination_risk
        clean, risk = _extract_hallucination_risk('Hi.\n<<META>>{"hallucination_risk": 0.9')
        assert clean == "Hi."
        assert risk == 0.5   # unparseable -> neutral default


# ─────────────────────────────────────────────
# BUG-CTX-TRUNC
# ─────────────────────────────────────────────

class TestGenerateContext:
    def test_llm_receives_full_chunk_not_display_snippet(self):
        from app.engine import pipeline

        marker = "TAIL_MARKER_" + "z" * 8
        long_text = "x" * 450 + marker          # marker sits beyond char 300
        state = {
            "question": "q", "provider": "OpenAI", "model": "m", "api_key": "k",
            "relevant_docs": [_doc(long_text)], "top_k": 5,
            "chat_history_text": "No prior conversation.", "system_prompts": {},
        }
        seen = {}

        class _LLM:
            async def ainvoke(self, messages):
                seen["prompt"] = messages[-1].content
                return MagicMock(content="ok <<META>>{\"hallucination_risk\": 0.1}<<END_META>>")

        with patch.object(pipeline, "build_llm", return_value=_LLM()), \
             patch.object(pipeline, "_maybe_tagged", side_effect=lambda llm, tags: llm):
            out = asyncio.run(pipeline._node_generate(state))

        assert marker in seen["prompt"], "LLM context was truncated to the 300-char display snippet"
        assert out["answer"] == "ok"


# ─────────────────────────────────────────────
# BUG-STREAM-REFLECT
# ─────────────────────────────────────────────

class _FakeStreamGraph:
    """astream() yields two generate passes (a reflect loop) then final updates."""

    def __init__(self, passes, final_answer):
        self.passes, self.final_answer = passes, final_answer

    async def astream(self, state, stream_mode=None):
        for step, tokens in self.passes:
            for t in tokens:
                yield ("messages", (AIMessageChunk(content=t),
                                    {"langgraph_node": "generate", "langgraph_step": step}))
        yield ("updates", {"generate": {"answer": self.final_answer, "question": state["question"],
                                        "pipeline_trace": ["rewrite", "generate", "reflect", "generate"],
                                        "reflect_loops": 1, "graded_count": 2,
                                        "relevant_docs": [_doc("src text")], "top_k": 5}})


class TestStreamingReflect:
    def _stream(self, engine, graph):
        async def run():
            out = []
            with patch.object(engine, "_get_or_rebuild_graph", return_value=graph):
                async for item in engine.stream_query_with_meta(
                    "q", "OpenAI", "m", "k", session_id="s", top_k=5
                ):
                    out.append(item)
            return out
        return asyncio.run(run())

    def test_reset_emitted_and_only_final_answer_persisted(self, real_engine):
        engine = real_engine()
        engine.ingest_documents([_doc("seed document text")])
        meta = '<<META>>{"hallucination_risk": 0.9}<<END_META>>'
        graph = _FakeStreamGraph(
            passes=[(3, ["First ", "draft.", "<<ME", "TA>>{\"hallucination_risk\": 0.9}", "<<END_META>>"]),
                    (6, ["Better ", "answer."])],
            final_answer="Better answer.",
        )
        items = self._stream(engine, graph)

        kinds = ["reset" if isinstance(i, dict) and i.get("event") == "reset"
                 else "meta" if isinstance(i, dict) else "tok" for i in items]
        assert kinds.count("reset") == 1 and kinds[-1] == "meta"
        reset_at = kinds.index("reset")
        after = "".join(i for i, k in zip(items, kinds) if k == "tok" and items.index(i) > reset_at)
        assert after.strip() == "Better answer."
        shown_all = "".join(i for i, k in zip(items, kinds) if k == "tok")
        assert "<<" not in shown_all and "META" not in shown_all and ">>" not in shown_all

        history = engine._session_store.format_history("s")
        assert "Assistant: Better answer." in history
        assert "First draft" not in history

        final = items[-1]
        assert final["reflect_loops"] == 1 and final["sources"]

    def test_router_forwards_reset_frame(self, real_engine):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.routers.query import router

        engine = MagicMock()

        async def fake(**kwargs):
            yield "draft"
            yield {"event": "reset"}
            yield "final"
            yield {"pipeline_trace": ["generate"], "graded_chunks": 1, "reflect_loops": 1, "sources": []}

        engine.stream_query_with_meta = fake
        app = FastAPI()
        app.state.engine = engine
        app.state.limiter = __import__("app.limiter", fromlist=["limiter"]).limiter
        app.include_router(router)
        body = TestClient(app).post("/query/stream", json={"question": "q"}).text
        assert '"reset": true' in body
        assert body.index('"reset"') < body.index('"final"')
        assert body.rstrip().endswith("[DONE]")


# ─────────────────────────────────────────────
# BUG-SYNC-PROMPTS
# ─────────────────────────────────────────────

class TestSyncQueryPrompts:
    def test_query_forwards_system_prompts(self, real_engine):
        engine = real_engine()
        engine.aquery = AsyncMock(return_value={"answer": "x"})
        engine.query("q", "OpenAI", "m", "k", system_prompts={"generate": "CUSTOM"})
        assert engine.aquery.call_args.kwargs["system_prompts"] == {"generate": "CUSTOM"}

    def test_query_inside_running_loop_is_rejected(self, real_engine):
        engine = real_engine()

        async def inner():
            with pytest.raises(RuntimeError, match="running event loop"):
                engine.query("q", "OpenAI", "m", "k")
        asyncio.run(inner())


# ─────────────────────────────────────────────
# Ingestion: BUG-HASH-EARLY / BUG-HASH-SOURCE / BUG-BM25-MISSING (real Chroma)
# ─────────────────────────────────────────────

class TestIngestionIntegration:
    def test_same_text_from_two_files_is_kept_for_both(self, real_engine):
        engine = real_engine()
        text = "Quarterly revenue grew twelve percent year over year."
        r1 = engine.ingest_documents([_doc(text, "q1.txt")])
        r2 = engine.ingest_documents([_doc(text, "q2.txt")])
        assert r1["chunks_ingested"] == 1 and r2["chunks_ingested"] == 1
        assert {d.metadata["source"] for d in engine._all_docs} == {"q1.txt", "q2.txt"}

    def test_reupload_of_same_file_is_still_deduplicated(self, real_engine):
        engine = real_engine()
        engine.ingest_documents([_doc("Alpha beta gamma.", "a.txt")])
        r = engine.ingest_documents([_doc("Alpha beta gamma.", "a.txt")])
        assert r["chunks_ingested"] == 0 and r["duplicates_skipped"] == 1

    def test_failed_vector_write_does_not_poison_hashes(self, real_engine):
        engine = real_engine()
        engine.ingest_documents([_doc("First file content.", "a.txt")])
        with patch.object(engine._chroma, "add_documents", side_effect=RuntimeError("disk full")):
            with pytest.raises(RuntimeError):
                engine.ingest_documents([_doc("Second file content.", "b.txt")])
        r = engine.ingest_documents([_doc("Second file content.", "b.txt")])
        assert r["chunks_ingested"] == 1, "failed ingest permanently blocked this content"

    def test_duplicates_within_one_batch(self, real_engine):
        engine = real_engine()
        r = engine.ingest_documents([_doc("Same line.", "a.txt"), _doc("Same line.", "a.txt")])
        assert r["chunks_ingested"] == 1 and r["duplicates_skipped"] == 1

    def test_empty_chunks_are_ignored(self, real_engine):
        engine = real_engine()
        r = engine.ingest_documents([_doc("   \n  ", "blank.pdf")])
        assert r["chunks_ingested"] == 0

    def test_bm25_rebuilt_from_chroma_when_pickle_is_lost(self, real_engine):
        e1 = real_engine()
        e1.ingest_documents([_doc("Neural networks learn representations.", "a.txt"),
                             _doc("Gradient descent minimises loss.", "a.txt")])
        pkl = pathlib.Path(real_engine.path) / "bm25.pkl"
        assert pkl.exists()
        pkl.unlink()                                  # simulate lost/ephemeral file

        e2 = real_engine()                            # fresh start on same Chroma dir
        assert e2._bm25_retriever is not None, "BM25 not rebuilt from Chroma"
        assert len(e2._all_docs) == 2
        # engine is usable: hybrid retrieval works and re-upload is deduplicated
        assert e2.retrieve("gradient descent", top_k=2)
        assert e2.ingest_documents([_doc("Gradient descent minimises loss.", "a.txt")])["chunks_ingested"] == 0

    def test_old_text_only_hash_pickles_upgrade_cleanly(self, real_engine):
        e1 = real_engine()
        e1.ingest_documents([_doc("Legacy content here.", "a.txt")])
        e2 = real_engine()      # reload from the saved pickle
        r = e2.ingest_documents([_doc("Legacy content here.", "a.txt")])
        assert r["chunks_ingested"] == 0


# ─────────────────────────────────────────────
# Grading config
# ─────────────────────────────────────────────

class TestGradeTokens:
    def test_default_budget_fits_ten_scores(self):
        from app.utils import Settings
        assert Settings().GRADE_MAX_TOKENS >= 200

    def test_full_json_score_reply_parses(self):
        from app.engine.pipeline import _safe_json_object
        import json
        reply = json.dumps({"scores": {str(i): 0.9 for i in range(1, 11)}, "reason": "all match"})
        assert len(_safe_json_object(reply)["scores"]) == 10


# ─────────────────────────────────────────────
# Ingest helpers: BUG-NAN-TEXT / BUG-CSV-ENCODING / BUG-EMPTY-PDF
# ─────────────────────────────────────────────

class TestIngestHelpers:
    def test_missing_values_do_not_become_text(self):
        from app.backend.ingest import dataframe_to_documents
        df = pd.DataFrame({
            "a": [1.0, None], "b": ["x", pd.NA],
            "c": pd.to_datetime(["2024-01-01", None]),
        })
        docs = dataframe_to_documents(df, source_name="t.csv")
        joined = " ".join(d.page_content for d in docs)
        for bad in ("nan", "NaN", "NaT", "<NA>", "None"):
            assert bad not in joined, bad

    def test_cp1251_csv_is_readable(self, tmp_path):
        from app.backend.ingest import load_documents_from_file
        f = tmp_path / "ru.csv"
        f.write_bytes("город,страна\nТашкент,Узбекистан\n".encode("cp1251"))
        docs = load_documents_from_file(str(f), original_name="ru.csv")
        assert "Ташкент" in docs[0].page_content

    def test_utf8_bom_csv_is_readable(self, tmp_path):
        from app.backend.ingest import load_documents_from_file
        f = tmp_path / "bom.csv"
        f.write_bytes("\ufeffname,city\nAkmal,Tashkent\n".encode("utf-8"))
        docs = load_documents_from_file(str(f), original_name="bom.csv")
        assert docs[0].page_content.startswith("name: Akmal")

    def test_blank_txt_yields_no_documents(self, tmp_path):
        from app.backend.ingest import load_documents_from_file
        f = tmp_path / "blank.txt"
        f.write_text("  \n\n ")
        assert load_documents_from_file(str(f), original_name="blank.txt") == []


# ─────────────────────────────────────────────
# main.py: BUG-CORS-ENV / BUG-INGEST-500 / limiter
# ─────────────────────────────────────────────

class TestMainApp:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_ORIGINS", "https://example.com")
        import importlib
        import app.utils as utils
        utils.get_settings.cache_clear()
        from app.limiter import limiter
        limiter.reset()                     # rate-limit state is shared across tests
        with patch("app.engine.pipeline.RAGEngine") as eng_cls:
            import app.main as main
            importlib.reload(main)
            eng = MagicMock()
            eng.ingest_documents.return_value = {"chunks_ingested": 1, "duplicates_skipped": 0, "status": "success"}
            eng.health.return_value = {"vector_store": "ready", "bm25_index": "ready",
                                       "docs_indexed": "1", "bm25_docs": "1", "active_sessions": "0"}
            main.app.state.engine = eng
            from fastapi.testclient import TestClient
            yield TestClient(main.app), main, eng
        utils.get_settings.cache_clear()

    def test_cors_uses_settings_not_os_environ_only(self, client):
        c, main, _ = client
        assert main._allowed_origins == ["https://example.com"]

    def test_single_shared_limiter(self, client):
        c, main, _ = client
        from app.limiter import limiter
        import app.routers.query as q
        assert main.limiter is limiter and main.app.state.limiter is limiter

    def test_corrupt_pdf_returns_422_not_500(self, client):
        c, _, _ = client
        r = c.post("/ingest", files=[("files", ("bad.pdf", io.BytesIO(b"not a pdf"), "application/pdf"))])
        assert r.status_code == 422
        assert "bad.pdf" in r.json()["detail"]

    def test_scanned_or_empty_file_returns_400(self, client):
        c, _, _ = client
        r = c.post("/ingest", files=[("files", ("e.txt", io.BytesIO(b"   "), "text/plain"))])
        assert r.status_code == 400

    def test_good_csv_ingests(self, client):
        c, _, eng = client
        r = c.post("/ingest", files=[("files", ("d.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv"))])
        assert r.status_code == 201
        assert eng.ingest_documents.called

    def test_unsupported_extension_is_415(self, client):
        c, _, _ = client
        r = c.post("/ingest", files=[("files", ("x.exe", io.BytesIO(b"MZ"), "application/octet-stream"))])
        assert r.status_code == 415

    def test_oversize_upload_is_413_and_tmp_file_removed(self, client, monkeypatch):
        c, main, _ = client
        monkeypatch.setattr(main.settings, "MAX_UPLOAD_MB", 0.0001)   # ~100 bytes
        import tempfile, os
        before = set(os.listdir(tempfile.gettempdir()))
        r = c.post("/ingest", files=[("files", ("big.txt", io.BytesIO(b"a" * 5000), "text/plain"))])
        assert r.status_code == 413
        leaked = [f for f in set(os.listdir(tempfile.gettempdir())) - before if f.endswith(".txt")]
        assert leaked == []


# ─────────────────────────────────────────────
# Version / housekeeping
# ─────────────────────────────────────────────

class TestVersion:
    def test_version_is_3_7(self):
        from app.utils import APP_VERSION
        import app
        assert APP_VERSION == "3.7" and app.__version__ == "3.7"

    def test_dead_modules_are_gone(self):
        root = pathlib.Path(__file__).resolve().parents[1] / "app"
        assert not (root / "engine_core.py").exists()
        assert not (root / "engine.py").exists()

    def test_model_registry_has_no_retired_gemini_1_5(self):
        from app.constants import PROVIDER_MODELS
        assert not [m for m in PROVIDER_MODELS["Google Gemini"] if "1.5" in m]

    def test_every_model_has_a_label_or_is_documented(self):
        from app.backend.models import PROVIDER_MODELS, MODEL_LABELS
        for provider in ("OpenAI", "Google Gemini"):
            for m in PROVIDER_MODELS[provider]:
                assert m in MODEL_LABELS, m


# ─────────────────────────────────────────────
# End-to-end through the REAL LangGraph runtime (scripted fake LLM)
# ─────────────────────────────────────────────

class TestEndToEndGraph:
    @staticmethod
    def _scripted_llm(replies):
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage
        return GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))

    GRADE = '{"scores": {"1": 0.9, "2": 0.9, "3": 0.9}, "reason": "ok"}'

    def _engine_with_docs(self, real_engine):
        engine = real_engine()
        engine.ingest_documents([_doc("Tashkent is the capital of Uzbekistan.", "geo.txt"),
                                 _doc("Samarkand is famous for the Registan.", "geo.txt")])
        return engine

    def test_non_stream_query_full_pipeline(self, real_engine):
        from app.engine import pipeline
        engine = self._engine_with_docs(real_engine)
        llm = self._scripted_llm([
            "capital of Uzbekistan",                                              # rewrite
            self.GRADE,                                                           # grade
            'Tashkent.\n<<META>>{"hallucination_risk": 0.1}<<END_META>>',        # generate
        ])
        with patch.object(pipeline, "build_llm", return_value=llm):
            r = asyncio.run(engine.aquery("What is the capital?", "OpenAI", "m", "k", session_id="e2e"))
        assert r["answer"] == "Tashkent."
        assert r["pipeline_trace"] == ["rewrite", "retrieve", "grade", "generate"]
        assert r["sources"] and "<<" not in r["answer"]

    def test_stream_with_reflect_loop_resets_and_persists_final_only(self, real_engine):
        from app.engine import pipeline
        engine = self._engine_with_docs(real_engine)
        llm = self._scripted_llm([
            "capital uzbekistan",                                                  # rewrite
            self.GRADE,                                                            # grade
            'Maybe Bukhara. <<META>>{"hallucination_risk": 0.95}<<END_META>>',   # generate #1 (risky)
            "Tashkent capital Uzbekistan",                                         # reflect
            self.GRADE,                                                            # grade again
            'Tashkent is the capital. <<META>>{"hallucination_risk": 0.05}<<END_META>>',  # generate #2
        ])

        async def run():
            items = []
            with patch.object(pipeline, "build_llm", return_value=llm):
                async for it in engine.stream_query_with_meta(
                    "What is the capital?", "OpenAI", "m", "k", session_id="e2e-stream"
                ):
                    items.append(it)
            return items

        items = asyncio.run(run())
        resets = [i for i in items if isinstance(i, dict) and i.get("event") == "reset"]
        meta = items[-1]
        shown_after_reset = "".join(
            i for i in items[items.index(resets[0]) + 1:] if isinstance(i, str)
        ) if resets else ""

        assert len(resets) == 1, f"expected exactly one reset, got {len(resets)}: {items}"
        assert shown_after_reset.strip() == "Tashkent is the capital."
        assert meta["reflect_loops"] == 1
        assert meta["pipeline_trace"].count("generate") == 2
        everything = "".join(i for i in items if isinstance(i, str))
        assert "<<" not in everything and "META" not in everything and ">>" not in everything
        history = engine._session_store.format_history("e2e-stream")
        assert "Tashkent is the capital." in history and "Bukhara" not in history
