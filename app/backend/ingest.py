"""
backend/ingest.py — AuraRAG v4.0 Document Ingestion Module
Author: Akmal Raxmatov (github: thed700)

Changes v4.0:
  FEAT-DS:  Expanded file support — CSV, JSON, XLSX, Parquet in addition to
            the existing PDF and TXT loaders.
  FEAT-DF:  Dataframe rows are serialised into rich, searchable natural-language
            strings ("Column1: Value1 | Column2: Value2") before being passed
            to RecursiveCharacterTextSplitter → ChromaDB + BM25.
  FEAT-ENV: Dual-deployment detection (Hugging Face Spaces vs local Docker).
            HF Spaces sets HF_SPACE_ID; the loader falls back gracefully when
            optional heavy dependencies (openpyxl, pyarrow, fastparquet) are
            absent from the HF image.
  FEAT-META: Each document chunk carries rich metadata: source filename,
             file_type, row_count (for dataframes), and sheet name (XLSX).

Integration:
  Replace the inline ingest logic in app/main.py's `ingest_documents()` route
  with a call to `load_uploaded_file(upload, tmp_path, suffix)`.  The function
  returns List[Document] ready for engine.ingest_documents().

  Example (inside main.py):
      from app.backend.ingest import load_uploaded_file, SUPPORTED_EXTENSIONS

      @app.post("/ingest", ...)
      async def ingest_documents(request, files):
          eng = _engine_or_503(request)
          all_docs = []
          for upload in files:
              suffix = os.path.splitext(upload.filename or "")[-1].lower()
              if suffix not in SUPPORTED_EXTENSIONS:
                  raise HTTPException(415, f"Unsupported type '{suffix}'.")
              tmp_path = await _stream_upload_to_tmp(upload, suffix)
              try:
                  docs = load_uploaded_file(upload.filename, tmp_path, suffix)
                  all_docs.extend(docs)
              finally:
                  if os.path.exists(tmp_path):
                      os.unlink(tmp_path)
          result = await asyncio.to_thread(eng.ingest_documents, all_docs)
          return IngestResponse(...)
"""

from __future__ import annotations

import logging
import os
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SUPPORTED EXTENSIONS
# ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt", ".csv", ".json", ".xlsx", ".parquet"}

# ─────────────────────────────────────────────
# RUNTIME ENVIRONMENT DETECTION
# ─────────────────────────────────────────────

_IS_HF_SPACE: bool = bool(os.environ.get("HF_SPACE_ID"))

if _IS_HF_SPACE:
    logger.info("Ingest: running on Hugging Face Spaces — optional deps may be absent.")
else:
    logger.info("Ingest: running in local Docker mode — full dependency set expected.")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _row_to_text(row: dict) -> str:
    """
    Serialise a single dataframe row into a pipe-separated string.

    Example:
        {"Name": "Alice", "Age": 30, "Dept": "Engineering"}
        → "Name: Alice | Age: 30 | Dept: Engineering"

    NaN / None values are replaced with an empty string to keep chunks clean.
    """
    parts = []
    for col, val in row.items():
        if val is None or (isinstance(val, float) and val != val):  # NaN check
            val = ""
        parts.append(f"{col}: {val}")
    return " | ".join(parts)


def _df_to_documents(df: "pandas.DataFrame", source_name: str, extra_meta: dict | None = None) -> List[Document]:  # noqa: F821
    """
    Convert every row of a pandas DataFrame into a LangChain Document.

    Each Document's page_content is the pipe-serialised row string.
    Metadata includes the source filename, file_type, and row index so
    citations in the UI can point back to the origin row.
    """
    docs: List[Document] = []
    base_meta: dict = {"source": source_name, **(extra_meta or {})}

    for idx, row in df.iterrows():
        text = _row_to_text(row.to_dict())
        if not text.strip():
            continue
        meta = {**base_meta, "row": int(idx)}  # type: ignore[arg-type]
        docs.append(Document(page_content=text, metadata=meta))

    logger.info("Serialised %d row(s) from '%s'.", len(docs), source_name)
    return docs


# ─────────────────────────────────────────────
# LOADER FUNCTIONS  (one per file type)
# ─────────────────────────────────────────────

def _load_pdf(filename: str, tmp_path: str) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(tmp_path)
    docs   = loader.load()
    for doc in docs:
        doc.metadata["source"]    = filename
        doc.metadata["file_type"] = "pdf"
    logger.info("PDF '%s': loaded %d page(s).", filename, len(docs))
    return docs


def _load_txt(filename: str, tmp_path: str) -> List[Document]:
    from langchain_community.document_loaders import TextLoader

    # BUG-W (carried): UTF-8 + autodetect_encoding for non-UTF-8 files
    loader = TextLoader(tmp_path, encoding="utf-8", autodetect_encoding=True)
    docs   = loader.load()
    for doc in docs:
        doc.metadata["source"]    = filename
        doc.metadata["file_type"] = "txt"
    logger.info("TXT '%s': loaded %d document(s).", filename, len(docs))
    return docs


def _load_csv(filename: str, tmp_path: str) -> List[Document]:
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required to ingest CSV files. "
            "Install it with: pip install pandas"
        )

    df = pd.read_csv(tmp_path, dtype=str)
    df.fillna("", inplace=True)
    logger.info("CSV '%s': read %d rows × %d cols.", filename, len(df), len(df.columns))
    return _df_to_documents(
        df, filename,
        extra_meta={"file_type": "csv", "row_count": len(df)},
    )


def _load_json(filename: str, tmp_path: str) -> List[Document]:
    """
    Supports both JSON arrays (list of objects → rows) and nested JSON objects.

    For a flat array of objects each object becomes a Document row.
    For a plain nested object, the entire JSON is stringified into a single
    Document so nothing is lost.  For a top-level key that maps to an array
    (common API response pattern), the array is unpacked as rows.
    """
    import json as _json

    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]

    with open(tmp_path, encoding="utf-8") as fh:
        raw = _json.load(fh)

    # --- Case 1: top-level list of objects → DataFrame rows
    if isinstance(raw, list) and all(isinstance(item, dict) for item in raw):
        if pd is not None:
            df = pd.DataFrame(raw).astype(str)
            df.fillna("", inplace=True)
            logger.info("JSON '%s': %d records (array).", filename, len(df))
            return _df_to_documents(
                df, filename,
                extra_meta={"file_type": "json", "row_count": len(df)},
            )
        # Fallback without pandas
        docs = []
        for idx, obj in enumerate(raw):
            text = _row_to_text(obj)
            docs.append(Document(
                page_content=text,
                metadata={"source": filename, "file_type": "json", "row": idx},
            ))
        logger.info("JSON '%s': %d records (no pandas).", filename, len(docs))
        return docs

    # --- Case 2: dict with one key pointing to an array  (e.g. {"data": [...]})
    if isinstance(raw, dict) and pd is not None:
        for key, val in raw.items():
            if isinstance(val, list) and all(isinstance(i, dict) for i in val):
                df = pd.DataFrame(val).astype(str)
                df.fillna("", inplace=True)
                logger.info("JSON '%s': key=%r, %d records.", filename, key, len(df))
                return _df_to_documents(
                    df, filename,
                    extra_meta={"file_type": "json", "json_key": key, "row_count": len(df)},
                )

    # --- Case 3: arbitrary JSON — serialise the whole thing as one document
    text = _json.dumps(raw, ensure_ascii=False, indent=2)
    logger.info("JSON '%s': serialised as single document (%d chars).", filename, len(text))
    return [Document(
        page_content=text,
        metadata={"source": filename, "file_type": "json"},
    )]


def _load_xlsx(filename: str, tmp_path: str) -> List[Document]:
    """
    Load all sheets from an Excel workbook.  Each sheet is treated as an
    independent table; sheet name is stored in metadata.

    Requires openpyxl:  pip install openpyxl
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required to ingest XLSX files. "
            "Install it with: pip install pandas openpyxl"
        )
    try:
        import openpyxl  # noqa: F401 — verifies the engine is present
    except ImportError:
        raise ImportError(
            "openpyxl is required to read .xlsx files. "
            "Install it with: pip install openpyxl"
        )

    xl    = pd.ExcelFile(tmp_path, engine="openpyxl")
    docs: List[Document] = []

    for sheet in xl.sheet_names:
        df = xl.parse(sheet, dtype=str)
        df.fillna("", inplace=True)
        sheet_docs = _df_to_documents(
            df, filename,
            extra_meta={
                "file_type":  "xlsx",
                "sheet_name": sheet,
                "row_count":  len(df),
            },
        )
        docs.extend(sheet_docs)
        logger.info(
            "XLSX '%s', sheet '%s': %d rows.", filename, sheet, len(df)
        )

    return docs


def _load_parquet(filename: str, tmp_path: str) -> List[Document]:
    """
    Load a Parquet file.  Tries pyarrow first, then fastparquet as fallback.

    Requires pyarrow:  pip install pyarrow
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required to ingest Parquet files. "
            "Install it with: pip install pandas pyarrow"
        )

    try:
        df = pd.read_parquet(tmp_path, engine="pyarrow")
    except ImportError:
        try:
            df = pd.read_parquet(tmp_path, engine="fastparquet")
        except ImportError:
            raise ImportError(
                "pyarrow or fastparquet is required to read Parquet files. "
                "Install with: pip install pyarrow"
            )

    df = df.astype(str)
    df.fillna("", inplace=True)
    logger.info("Parquet '%s': %d rows × %d cols.", filename, len(df), len(df.columns))
    return _df_to_documents(
        df, filename,
        extra_meta={"file_type": "parquet", "row_count": len(df)},
    )


# ─────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────

_LOADERS = {
    ".pdf":     _load_pdf,
    ".txt":     _load_txt,
    ".csv":     _load_csv,
    ".json":    _load_json,
    ".xlsx":    _load_xlsx,
    ".parquet": _load_parquet,
}


def load_uploaded_file(filename: str, tmp_path: str, suffix: str) -> List[Document]:
    """
    Dispatch to the correct loader based on file extension.

    Args:
        filename:  Original upload filename (used in Document metadata).
        tmp_path:  Path to the temporary file on disk.
        suffix:    Lowercase file extension including dot (e.g. ".csv").

    Returns:
        List[Document] — ready for engine.ingest_documents().

    Raises:
        ValueError:  If the extension is not in SUPPORTED_EXTENSIONS.
        ImportError: If a required optional dependency is missing.
        Exception:   Any loader-level parsing error is re-raised with context.
    """
    if suffix not in _LOADERS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    loader_fn = _LOADERS[suffix]
    try:
        docs = loader_fn(filename, tmp_path)
    except (ImportError, ValueError):
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to parse '{filename}' (type={suffix}): {exc}"
        ) from exc

    if not docs:
        logger.warning("No content extracted from '%s'.", filename)

    return docs
