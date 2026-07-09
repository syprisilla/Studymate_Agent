from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
STATIC_DIR = BASE_DIR / "static"
SESSION_PDF_META_PATH = DATA_DIR / "session_pdfs.json"

for path in (DATA_DIR, PDF_DIR, STATIC_DIR):
    path.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_READY = bool(os.getenv("OPENAI_API_KEY"))
MAX_RAG_RETRIES = 2


@dataclass
class StudySession:
    session_id: str
    pdf_path: Optional[Path] = None
    pdf_name: Optional[str] = None
    pdf_text: str = ""
    page_documents: list[Document] | None = None
    vectorstore: object | None = None


SESSIONS: dict[str, StudySession] = {}


def get_session(session_id: str) -> StudySession:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = StudySession(session_id=session_id)

        try:
            from services.pdf_store import restore_session_pdf

            restore_session_pdf(SESSIONS[session_id])
        except Exception:
            pass

    return SESSIONS[session_id]
