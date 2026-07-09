from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

from services.session_store import (
    OPENAI_READY,
    PDF_DIR,
    SESSION_PDF_META_PATH,
    StudySession,
    get_session,
)


def read_pdf_documents(path: Path, filename: str) -> list[Document]:
    reader = PdfReader(str(path))
    docs: list[Document] = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        docs.append(
            Document(
                page_content=f"[page {page_no}]\n{text}",
                metadata={"page": page_no, "source": filename},
            )
        )

    return docs


def docs_to_text(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs).strip()


def chunk_document(doc: Document, size: int = 900, overlap: int = 150) -> list[Document]:
    text = doc.page_content
    chunks: list[Document] = []
    start = 0
    step = max(1, size - overlap)

    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(Document(page_content=piece, metadata=dict(doc.metadata)))
        start += step

    return chunks


def chunk_documents(docs: list[Document]) -> list[Document]:
    chunks: list[Document] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks


def build_index(session: StudySession) -> None:
    if not session.page_documents or not OPENAI_READY:
        session.vectorstore = None
        return

    try:
        session.vectorstore = FAISS.from_documents(
            chunk_documents(session.page_documents),
            OpenAIEmbeddings(model="text-embedding-3-small"),
        )
    except Exception:
        session.vectorstore = None


def load_session_pdf_meta() -> dict[str, dict[str, str]]:
    if not SESSION_PDF_META_PATH.exists():
        return {}

    try:
        payload = json.loads(SESSION_PDF_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def save_session_pdf_meta(payload: dict[str, dict[str, str]]) -> None:
    SESSION_PDF_META_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def persist_session_pdf(session: StudySession) -> None:
    if not session.pdf_path or not session.pdf_name:
        return

    payload = load_session_pdf_meta()
    payload[session.session_id] = {
        "pdf_name": session.pdf_name,
        "pdf_file": session.pdf_path.name,
    }
    save_session_pdf_meta(payload)


def remove_session_pdf_meta(session_id: str) -> None:
    payload = load_session_pdf_meta()
    if session_id in payload:
        del payload[session_id]
        save_session_pdf_meta(payload)


def restore_session_pdf(session: StudySession) -> bool:
    if session.pdf_text:
        return True

    meta = load_session_pdf_meta().get(session.session_id)
    if not isinstance(meta, dict):
        return False

    pdf_name = meta.get("pdf_name")
    pdf_file = meta.get("pdf_file")
    if not pdf_name or not pdf_file:
        return False

    path = PDF_DIR / Path(pdf_file).name
    if not path.exists():
        return False

    page_docs = read_pdf_documents(path, pdf_name)
    text = docs_to_text(page_docs)
    if not text:
        return False

    session.pdf_path = path
    session.pdf_name = pdf_name
    session.page_documents = page_docs
    session.pdf_text = text
    build_index(session)
    return True


def enrich_sessions_with_pdf_meta(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = load_session_pdf_meta()
    enriched: list[dict[str, Any]] = []

    for session in sessions:
        item = dict(session)
        session_id = item.get("session_id")
        meta = payload.get(session_id) if session_id else None
        if meta and not item.get("pdf_name"):
            item["pdf_name"] = meta.get("pdf_name")
        enriched.append(item)

    return enriched


def _keyword_score(text: str, words: list[str]) -> int:
    lower = text.lower()
    return sum(lower.count(word) for word in words)


def keyword_fallback_chunks(session: StudySession, query: str, k: int = 5) -> list[Document]:
    docs = chunk_documents(session.page_documents or [])
    if not docs:
        return []

    words = [
        word.strip().lower()
        for word in query.replace("/", " ").replace("_", " ").split()
        if len(word.strip()) >= 2
    ]
    scored = sorted(docs, key=lambda doc: _keyword_score(doc.page_content, words), reverse=True)
    has_score = any(_keyword_score(doc.page_content, words) > 0 for doc in scored[:k])
    return scored[:k] if has_score else docs[:k]


def retrieve_pdf_chunks(session_id: str, query: str, k: int = 5) -> list[Document]:
    session = get_session(session_id)
    if not session.pdf_text:
        return []

    if session.vectorstore is not None:
        try:
            return session.vectorstore.as_retriever(search_kwargs={"k": k}).invoke(query)
        except Exception:
            pass

    return keyword_fallback_chunks(session, query, k=k)


def retrieve_pdf_context(session_id: str, query: str, k: int = 5) -> str:
    docs = retrieve_pdf_chunks(session_id, query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)
