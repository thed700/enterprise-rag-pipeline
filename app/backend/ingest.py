"""
ingest.py — structured document ingestion helpers for AuraRAG.

Supports PDF, TXT, CSV, JSON, XLSX/XLS, and Parquet. Structured files are
converted into row-level text documents before chunking so Chroma and BM25
receive retrievable natural-language text instead of opaque tables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

SUPPORTED_UPLOAD_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".csv",
    ".json",
    ".xlsx",
    ".xls",
    ".parquet",
}

def _safe_string(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_safe_string(item) for item in value if _safe_string(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # type: ignore[no-any-return]
        except Exception:
            pass
    text = str(value).strip()
    return text

def _row_to_text(row: pd.Series) -> str:
    parts: list[str] = []
    for col, value in row.items():
        value_text = _safe_string(value)
        if value_text == "":
            continue
        parts.append(f"{col}: {value_text}")
    return " | ".join(parts)

def dataframe_to_documents(
    frame: pd.DataFrame,
    *,
    source_name: str,
    sheet_name: str | None = None,
    file_type: str = "dataframe",
) -> List[Document]:
    """
    Convert a dataframe into searchable row-level documents.
    """
    if frame is None or frame.empty:
        return []

    docs: List[Document] = []
    normalised = frame.copy()
    normalised = normalised.reset_index(drop=True)

    # Make sure object columns are serializable and stable.
    for column in normalised.columns:
        normalised[column] = normalised[column].map(_safe_string)

    for idx, row in normalised.iterrows():
        row_text = _row_to_text(row)
        if not row_text:
            continue
        metadata = {
            "source": source_name,
            "file_type": file_type,
            "row_index": int(idx),
        }
        if sheet_name:
            metadata["sheet_name"] = sheet_name
        docs.append(Document(page_content=row_text, metadata=metadata))

    return docs

def _load_json_frame(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        if not payload:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in payload):
            return pd.json_normalize(payload, sep=".")
        return pd.DataFrame({"value": payload})

    if isinstance(payload, dict):
        # Common shapes:
        #  - {"records": [...]}
        #  - nested objects
        if len(payload) == 1:
            only_value = next(iter(payload.values()))
            if isinstance(only_value, list) and all(isinstance(item, dict) for item in only_value):
                return pd.json_normalize(only_value, sep=".")
        return pd.json_normalize(payload, sep=".")

    return pd.DataFrame({"value": [payload]})

def _load_xlsx_frames(path: str) -> list[tuple[str | None, pd.DataFrame]]:
    workbook = pd.read_excel(path, sheet_name=None)
    if isinstance(workbook, dict):
        return [(sheet, frame) for sheet, frame in workbook.items()]
    return [(None, workbook)]

def load_documents_from_file(
    path: str,
    *,
    original_name: str | None = None,
) -> List[Document]:
    """
    Load a single file from disk and convert it into LangChain Documents.
    """
    suffix = Path(original_name or path).suffix.lower()
    source_name = original_name or Path(path).name

    if suffix == ".pdf":
        docs = PyPDFLoader(path).load()
        for doc in docs:
            doc.metadata["source"] = source_name
            doc.metadata["file_type"] = "pdf"
        return docs

    if suffix == ".txt":
        docs = TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()
        for doc in docs:
            doc.metadata["source"] = source_name
            doc.metadata["file_type"] = "txt"
        return docs

    if suffix == ".csv":
        frame = pd.read_csv(path)
        return dataframe_to_documents(frame, source_name=source_name, file_type="csv")

    if suffix == ".json":
        frame = _load_json_frame(path)
        return dataframe_to_documents(frame, source_name=source_name, file_type="json")

    if suffix in {".xlsx", ".xls"}:
        docs: List[Document] = []
        for sheet_name, frame in _load_xlsx_frames(path):
            docs.extend(
                dataframe_to_documents(
                    frame,
                    source_name=source_name,
                    sheet_name=sheet_name,
                    file_type="xlsx",
                )
            )
        return docs

    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        return dataframe_to_documents(frame, source_name=source_name, file_type="parquet")

    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported: "
        + ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
    )

def load_documents_from_uploads(upload_paths: Sequence[tuple[str, str]]) -> List[Document]:
    """
    Convenience helper for ingesting many already-saved uploads.
    Each item is (path, original_name).
    """
    all_docs: List[Document] = []
    for path, original_name in upload_paths:
        all_docs.extend(load_documents_from_file(path, original_name=original_name))
    return all_docs
