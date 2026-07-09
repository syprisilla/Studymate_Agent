from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Optional, TypedDict

import fitz
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ToolRetryMiddleware
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from pypdf import PdfReader
from services.chat_history import (
    create_chat_session,
    delete_chat_session,
    get_chat_messages,
    list_chat_sessions,
    save_chat_message,
    update_chat_pdf_name,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
STATIC_DIR = BASE_DIR / "static"
SESSION_PDF_META_PATH = DATA_DIR / "session_pdfs.json"

for path in (DATA_DIR, PDF_DIR, STATIC_DIR):
    path.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_READY = bool(os.getenv("OPENAI_API_KEY"))
MAX_RAG_RETRIES = 2


# =========================================================================
# 0. 채팅 히스토리 호환 래퍼
# =========================================================================

def make_chat_title(text: str, max_len: int = 30) -> str:
    title = re.sub(r"\s+", " ", str(text or "")).strip()
    if not title:
        return "새 대화"
    return title[:max_len]


def normalize_message_role(role: str) -> str:
    if role in {"assistant", "ai", "bot"}:
        return "ai"
    if role in {"human", "user"}:
        return "user"
    return role or "ai"


def normalize_message_item(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        role = normalize_message_role(str(message.get("role", "ai")))
        content = message.get("content", "")
        return {
            **message,
            "role": role,
            "content": content,
        }

    role = normalize_message_role(str(getattr(message, "role", "ai")))
    content = getattr(message, "content", "")

    return {
        "role": role,
        "content": content,
    }


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []

    return [normalize_message_item(message) for message in messages]


def normalize_session_item(session: Any) -> dict[str, Any]:
    if isinstance(session, dict):
        session_id = (
            session.get("session_id")
            or session.get("id")
            or session.get("thread_id")
            or "default"
        )

        title = session.get("title") or session.get("name") or "새 대화"
        pdf_name = session.get("pdf_name")

        return {
            **session,
            "session_id": session_id,
            "title": title,
            "pdf_name": pdf_name,
        }

    session_id = getattr(session, "session_id", None) or getattr(session, "id", None) or "default"
    title = getattr(session, "title", None) or getattr(session, "name", None) or "새 대화"
    pdf_name = getattr(session, "pdf_name", None)

    return {
        "session_id": session_id,
        "title": title,
        "pdf_name": pdf_name,
    }


def normalize_session_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get("sessions") or payload.get("items") or payload.get("data") or []
    else:
        items = payload

    if not isinstance(items, list):
        return []

    return [normalize_session_item(item) for item in items]


def safe_list_chat_sessions() -> list[dict[str, Any]]:
    try:
        return normalize_session_list(list_chat_sessions())
    except Exception:
        return []


def safe_get_chat_messages(session_id: str) -> list[dict[str, Any]]:
    try:
        return normalize_messages(get_chat_messages(session_id))
    except Exception:
        return []


def safe_create_chat_session(
    session_id: Optional[str] = None,
    title: str = "새 대화",
    pdf_name: Optional[str] = None,
) -> dict[str, Any]:
    session_id = session_id or str(uuid.uuid4())

    call_attempts = [
        lambda: create_chat_session(
            session_id=session_id,
            title=title,
            pdf_name=pdf_name,
        ),
        lambda: create_chat_session(
            session_id=session_id,
            title=title,
        ),
        lambda: create_chat_session(
            session_id=session_id,
        ),
        lambda: create_chat_session(session_id),
        lambda: create_chat_session(),
    ]

    for call in call_attempts:
        try:
            result = call()
            if isinstance(result, dict):
                result_session_id = result.get("session_id") or result.get("id") or session_id
                return {
                    **result,
                    "session_id": result_session_id,
                    "title": result.get("title") or title,
                    "pdf_name": result.get("pdf_name") or pdf_name,
                }

            return {
                "ok": True,
                "session_id": session_id,
                "title": title,
                "pdf_name": pdf_name,
            }
        except TypeError:
            continue
        except Exception:
            break

    return {
        "ok": True,
        "session_id": session_id,
        "title": title,
        "pdf_name": pdf_name,
    }


def safe_save_chat_message(
    session_id: str,
    role: str,
    content: str,
    route: Optional[str] = None,
    pdf_name: Optional[str] = None,
    used_tools: Optional[list[str]] = None,
    evidence: Optional[list[str]] = None,
) -> None:
    role = normalize_message_role(role)
    save_role = "assistant" if role == "ai" else role

    call_attempts = [
        lambda: save_chat_message(
            session_id=session_id,
            role=save_role,
            content=content,
            route=route,
            pdf_name=pdf_name,
            used_tools=used_tools or [],
            evidence=evidence or [],
        ),
        lambda: save_chat_message(
            session_id=session_id,
            role=save_role,
            content=content,
            pdf_name=pdf_name,
        ),
        lambda: save_chat_message(
            session_id=session_id,
            role=save_role,
            content=content,
        ),
        lambda: save_chat_message(session_id, save_role, content),
    ]

    for call in call_attempts:
        try:
            call()
            return
        except TypeError:
            continue
        except Exception:
            return


def safe_update_chat_pdf_name(session_id: str, pdf_name: str) -> None:
    call_attempts = [
        lambda: update_chat_pdf_name(
            session_id=session_id,
            pdf_name=pdf_name,
        ),
        lambda: update_chat_pdf_name(session_id, pdf_name),
    ]

    for call in call_attempts:
        try:
            call()
            return
        except TypeError:
            continue
        except Exception:
            return


def history_to_langchain_messages(messages: list[dict[str, Any]], limit: int = 20) -> list:
    result = []

    for message in messages[-limit:]:
        role = normalize_message_role(str(message.get("role", "")))
        content = str(message.get("content", ""))

        if not content:
            continue

        if role == "user":
            result.append(HumanMessage(content=content))
        elif role in {"ai", "assistant"}:
            result.append(AIMessage(content=content))

    return result


# =========================================================================
# 1. PDF 세션 저장소
# =========================================================================

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
            restore_session_pdf(SESSIONS[session_id])
        except NameError:
            # 모듈 로딩 중 get_session이 먼저 호출되는 경우를 대비한 안전장치
            pass
        except Exception:
            # 저장된 PDF 복원 실패가 전체 서버 실행을 막지 않게 한다.
            pass

    return SESSIONS[session_id]


# =========================================================================
# 2. PDF 읽기 / 청크 / RAG 인덱싱
# =========================================================================

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
                metadata={
                    "page": page_no,
                    "source": filename,
                },
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
        end = start + size
        piece = text[start:end].strip()

        if piece:
            chunks.append(
                Document(
                    page_content=piece,
                    metadata=dict(doc.metadata),
                )
            )

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

    chunks = chunk_documents(session.page_documents)

    try:
        session.vectorstore = FAISS.from_documents(
            chunks,
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

    scored = sorted(
        docs,
        key=lambda doc: _keyword_score(doc.page_content, words),
        reverse=True,
    )

    has_score = any(_keyword_score(doc.page_content, words) > 0 for doc in scored[:k])
    return scored[:k] if has_score else docs[:k]


def retrieve_pdf_chunks(session_id: str, query: str, k: int = 5) -> list[Document]:
    session = get_session(session_id)

    if not session.pdf_text:
        return []

    if session.vectorstore is not None:
        try:
            return session.vectorstore.as_retriever(
                search_kwargs={"k": k}
            ).invoke(query)
        except Exception:
            pass

    return keyword_fallback_chunks(session, query, k=k)


def retrieve_pdf_context(session_id: str, query: str, k: int = 5) -> str:
    docs = retrieve_pdf_chunks(session_id, query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)


def evidence_from_docs(docs: list[Document]) -> list[str]:
    pages: dict[int, str] = {}

    for doc in docs:
        page = doc.metadata.get("page")
        source = doc.metadata.get("source", "PDF")

        if isinstance(page, int):
            pages[page] = f"page {page} - {source}"

    return [pages[p] for p in sorted(pages)]


def extract_evidence_from_text(text: str, default_source: str = "PDF") -> list[str]:
    found: dict[int, str] = {}

    patterns = [
        r"page\s+(\d+)\s*-\s*([^,\n]+)",
        r"page\s+(\d+)",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            page = int(match.group(1))

            if len(match.groups()) >= 2 and match.group(2):
                source = match.group(2).strip()
            else:
                source = default_source

            source = source.strip(" .)")
            found[page] = f"page {page} - {source}"

    return [found[p] for p in sorted(found)]


def extract_evidence_from_messages(messages: list, default_source: str = "PDF") -> list[str]:
    text_parts: list[str] = []

    for message in messages:
        content = getattr(message, "content", "")

        if isinstance(content, str):
            text_parts.append(content)

    return extract_evidence_from_text("\n".join(text_parts), default_source=default_source)


def extract_tool_names(messages: list) -> list[str]:
    names: list[str] = []

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            for call in tool_calls:
                name = call.get("name")

                if name and name not in names:
                    names.append(name)

        if getattr(message, "type", None) == "tool":
            name = getattr(message, "name", None)

            if name and name not in names:
                names.append(name)

    return names


def last_ai_content(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)

    return str(messages[-1].content) if messages else ""


# =========================================================================
# 3. 구조화 출력 모델
# =========================================================================

class RouteDecision(BaseModel):
    route: Literal[
        "concept",
        "summary",
        "plan",
        "quiz",
        "review",
        "followup",
        "unknown",
    ] = Field(
        description=(
            "concept=PDF 내용/CS 개념 설명 질문, "
            "summary=PDF 전체 요약/정리 요청, "
            "plan=공부 계획/일정 요청, "
            "quiz=예상문제/퀴즈 요청, "
            "review=사용자 답안 채점/오답 복습 요청, "
            "followup=직전 답변을 이어서 더 묻는 요청, "
            "unknown=PDF/공부와 무관한 요청"
        )
    )
    reason: str = Field(description="이 route로 분류한 짧은 이유")


class RelevanceCheck(BaseModel):
    is_relevant: bool
    reason: str


class StudyAnswer(BaseModel):
    title: str = Field(
        description=(
            "사용자에게 보여줄 자연스러운 한국어 제목. "
            "절대 StudyAnswer, JSON, Pydantic, schema 같은 내부 용어를 쓰지 말 것."
        )
    )
    core: str = Field(description="핵심 요약 2~3문장")
    details: list[str] = Field(description="주요 설명 목록")
    exam_points: list[str] = Field(description="시험 대비 포인트")
    examples: list[str] = Field(description="예시 또는 실습 포인트")
    check_questions: list[str] = Field(description="사용자가 스스로 점검할 확인 질문 1~3개")


class ReviewFeedback(BaseModel):
    is_correct: bool
    correct_concept: str = Field(description="PDF 근거에 따른 올바른 개념")
    mistake_reason: str = Field(description="헷갈린 부분. 정답이면 '해당 없음'")
    retry_question: str = Field(description="다시 풀어볼 문제 1개")
    encouragement: str = Field(description="짧은 격려")


def format_study_answer(answer: StudyAnswer, evidence: list[str]) -> str:
    details = "\n".join(f"- {item}" for item in answer.details)
    exam_points = "\n".join(f"- {item}" for item in answer.exam_points)
    examples = "\n".join(f"- {item}" for item in answer.examples)
    questions = "\n".join(f"- {item}" for item in answer.check_questions)
    evidence_text = "\n".join(f"- {item}" for item in evidence) if evidence else "- PDF 검색 근거 없음"

    return (
        f"## {answer.title}\n\n"
        f"### 핵심\n{answer.core}\n\n"
        f"### 설명\n{details}\n\n"
        f"### 시험 포인트\n{exam_points}\n\n"
        f"### 예시\n{examples}\n\n"
        f"### 확인 질문\n{questions}\n\n"
        f"### 참고한 PDF 위치\n{evidence_text}"
    )


def format_review_feedback(feedback: ReviewFeedback, evidence: list[str]) -> str:
    verdict = "정답에 가깝습니다 ✅" if feedback.is_correct else "다시 확인이 필요합니다 ❌"
    evidence_text = "\n".join(f"- {item}" for item in evidence) if evidence else "- PDF 검색 근거 없음"

    return (
        f"## 채점 결과: {verdict}\n\n"
        f"### 올바른 개념\n{feedback.correct_concept}\n\n"
        f"### 놓친 부분\n{feedback.mistake_reason}\n\n"
        f"### 다시 풀어볼 문제\n{feedback.retry_question}\n\n"
        f"### 참고한 PDF 위치\n{evidence_text}\n\n"
        f"{feedback.encouragement}"
    )


# =========================================================================
# 4. Tool 정의
# =========================================================================

@tool
def pdf_retriever(session_id: str, query: str) -> str:
    """업로드된 PDF에서 질문과 관련 있는 내용을 검색한다. session_id와 query가 필요하다."""
    docs = retrieve_pdf_chunks(session_id, query, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)
    evidence = evidence_from_docs(docs)

    if not context:
        return "현재 세션에 업로드된 PDF가 없습니다."

    return (
        f"[RAG 검색 결과]\n"
        f"참고 페이지: {', '.join(evidence) if evidence else '없음'}\n\n"
        f"{context[:5000]}"
    )


@tool
def pdf_summary_tool(session_id: str) -> str:
    """업로드된 PDF 전체 요약에 필요한 핵심 문단을 검색한다. session_id가 필요하다."""
    docs = retrieve_pdf_chunks(session_id, "전체 요약 핵심 개념 목차 중요한 내용", k=6)
    context = "\n\n".join(doc.page_content for doc in docs)
    evidence = evidence_from_docs(docs)

    if not context:
        return "현재 세션에 업로드된 PDF가 없습니다."

    return (
        f"[PDF 요약 근거]\n"
        f"참고 페이지: {', '.join(evidence) if evidence else '없음'}\n\n"
        f"{context[:5500]}"
    )


@tool
def make_quiz_from_pdf(session_id: str, topic: str = "전체", count: int = 5) -> str:
    """업로드된 PDF 내용을 바탕으로 예상문제를 만들기 위한 근거를 검색한다."""
    docs = retrieve_pdf_chunks(session_id, f"{topic} 예상문제 시험 중요 개념 명령어 비교", k=6)
    context = "\n\n".join(doc.page_content for doc in docs)
    evidence = evidence_from_docs(docs)

    if not context:
        return "현재 세션에 업로드된 PDF가 없습니다."

    return (
        f"[예상문제 생성 근거]\n"
        f"참고 페이지: {', '.join(evidence) if evidence else '없음'}\n\n"
        f"아래 PDF 근거를 바탕으로 예상문제 {count}개를 만드세요.\n"
        f"반드시 문제 영역과 정답/해설 영역을 분리하세요.\n"
        f"객관식/단답형/서술형을 섞고 같은 개념을 반복하지 마세요.\n\n"
        f"{context[:5000]}"
    )


@tool
def study_plan_tool(session_id: str, days: int = 3) -> str:
    """업로드된 PDF 내용을 기준으로 공부 계획을 세우기 위한 핵심 내용을 검색한다."""
    docs = retrieve_pdf_chunks(session_id, "공부 계획 학습 순서 핵심 개념 실습 시험 대비", k=6)
    context = "\n\n".join(doc.page_content for doc in docs)
    evidence = evidence_from_docs(docs)

    if not context:
        return "현재 세션에 업로드된 PDF가 없습니다."

    return (
        f"[공부 계획 생성 근거]\n"
        f"참고 페이지: {', '.join(evidence) if evidence else '없음'}\n\n"
        f"아래 PDF 근거를 바탕으로 {days}일 공부 계획을 세우세요.\n"
        f"각 날짜마다 목표, 공부할 내용, 확인 문제를 포함하세요.\n"
        f"PDF 한 챕터 기준이면 너무 긴 계획을 만들지 마세요.\n\n"
        f"{context[:5000]}"
    )


TOOLS = [
    pdf_retriever,
    pdf_summary_tool,
    make_quiz_from_pdf,
    study_plan_tool,
]


def build_tool_agent():
    return create_agent(
        model=init_chat_model(f"openai:{MODEL_NAME}", temperature=0.2),
        tools=TOOLS,
        system_prompt=(
            "너는 CS 전공 학습 도우미 StudyMate다. "
            "사용자가 버튼으로 기능을 고르는 것이 아니라 자연어로 요청하면, "
            "의도에 맞는 Tool을 스스로 선택해 실행해야 한다. "
            "반드시 업로드된 PDF 근거를 먼저 사용하고, 한국어로 답한다. "
            "원문을 길게 복사하지 말고 시험 공부에 바로 쓰기 좋게 재구성한다. "
            "답변 마지막에는 사용한 Tool과 참고한 PDF 페이지를 간단히 언급한다."
        ),
        middleware=[
            ToolRetryMiddleware(
                max_retries=2,
                backoff_factor=2.0,
                initial_delay=0.5,
                on_failure="continue",
            ),
            ToolCallLimitMiddleware(run_limit=6),
        ],
    )


# =========================================================================
# 5. Corrective RAG 서브그래프
# =========================================================================

class StudyRAGState(TypedDict, total=False):
    session_id: str
    query: str
    mode: Literal["concept", "summary"]
    retry_count: int
    documents: list[Document]
    relevant_documents: list[Document]
    answer: str
    evidence: list[str]
    trace: list[str]


def collect_pdf_chunks(state: StudyRAGState) -> dict:
    docs = retrieve_pdf_chunks(state["session_id"], state["query"], k=6)
    trace = state.get("trace", []) + [
        f"collect_pdf_chunks: PDF에서 후보 청크 {len(docs)}개 검색"
    ]

    return {
        "documents": docs,
        "trace": trace,
    }


def grade_pdf_relevance(state: StudyRAGState) -> dict:
    if not OPENAI_READY:
        trace = state.get("trace", []) + [
            "grade_pdf_relevance: OPENAI_API_KEY 없음 → 관련성 평가 생략"
        ]

        return {
            "relevant_documents": state.get("documents", []),
            "trace": trace,
        }

    llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0)
    checker = llm.with_structured_output(RelevanceCheck)
    relevant: list[Document] = []

    for doc in state.get("documents", []):
        try:
            result = checker.invoke(
                f"질문: {state['query']}\n\n"
                f"문단:\n{doc.page_content[:1000]}\n\n"
                "이 문단이 질문에 답하는 데 실제로 도움이 되는지 판단하세요."
            )

            if result.is_relevant:
                relevant.append(doc)
        except Exception:
            relevant.append(doc)

    trace = state.get("trace", []) + [
        f"grade_pdf_relevance: 관련 청크 {len(relevant)}개 선별"
    ]

    return {
        "relevant_documents": relevant,
        "trace": trace,
    }


def should_retry_pdf_search(state: StudyRAGState) -> str:
    if not OPENAI_READY:
        return "synthesize"

    if len(state.get("relevant_documents", [])) >= 2:
        return "synthesize"

    if state.get("retry_count", 0) >= MAX_RAG_RETRIES:
        return "synthesize"

    return "rewrite_query"


def rewrite_pdf_query(state: StudyRAGState) -> dict:
    if not OPENAI_READY:
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "trace": state.get("trace", []) + [
                "rewrite_pdf_query: OPENAI_API_KEY 없음 → 재작성 생략"
            ],
        }

    llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0.3)

    try:
        new_query = llm.invoke(
            f"다음 검색어로 PDF에서 충분한 관련 내용을 찾지 못했습니다: {state['query']}\n"
            "같은 의도를 유지하면서 더 구체적인 검색어 1개만 출력하세요."
        ).content.strip()
    except Exception:
        new_query = state["query"]

    trace = state.get("trace", []) + [
        f"rewrite_pdf_query: 검색어 재작성 → {new_query}"
    ]

    return {
        "query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": trace,
    }


def synthesize_study_answer(state: StudyRAGState) -> dict:
    docs = state.get("relevant_documents") or state.get("documents") or []

    if not docs:
        trace = state.get("trace", []) + ["synthesize_study_answer: 관련 문서 없음"]

        return {
            "answer": "PDF에서 관련 내용을 찾지 못했습니다. 질문을 조금 더 구체적으로 해주세요.",
            "evidence": [],
            "trace": trace,
        }

    context = "\n\n".join(doc.page_content for doc in docs)[:5500]
    evidence = evidence_from_docs(docs)

    if not OPENAI_READY:
        trace = state.get("trace", []) + [
            "synthesize_study_answer: OPENAI_API_KEY 없음 → 원문 기반 응답"
        ]

        return {
            "answer": f"PDF에서 관련 내용을 찾았습니다.\n\n{context[:1400]}",
            "evidence": evidence,
            "trace": trace,
        }

    llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0.2)
    parser = llm.with_structured_output(StudyAnswer)

    if state["mode"] == "summary":
        mode_instruction = (
            "PDF 전체를 요약하세요. "
            "핵심 개념, 주요 내용, 시험 포인트, 확인 질문을 한 번만 정리하세요."
        )
    else:
        mode_instruction = (
            "질문에 해당하는 PDF 개념을 설명하세요. "
            "반드시 1. 한 줄 정의, 2. 왜 필요한지, 3. 구조/동작 방식, "
            "4. PDF 기반 예시, 5. 비슷한 개념 비교, 6. 시험 포인트, "
            "7. 확인 질문 순서의 내용을 포함하세요. "
            "비슷한 개념이 있으면 비교해서 설명하세요."
        )

    try:
        structured = parser.invoke(
            [
                SystemMessage(
                    content=(
                        "너는 PDF 기반 CS 학습 도우미다. "
                        "반드시 주어진 PDF 근거 안에서만 답한다. "
                        "원문을 그대로 길게 복사하지 말고 시험 공부에 바로 쓸 수 있게 재구성한다. "
                        "제목에는 StudyAnswer, JSON, Pydantic, schema 같은 내부 구현 용어를 절대 쓰지 않는다."
                    )
                ),
                HumanMessage(
                    content=(
                        f"지시: {mode_instruction}\n"
                        f"질문: {state['query']}\n\n"
                        f"PDF 근거:\n{context}"
                    )
                ),
            ]
        )

        answer = format_study_answer(structured, evidence)
    except Exception:
        answer = (
            "PDF 근거를 바탕으로 정리하면 다음과 같습니다.\n\n"
            f"{context[:1800]}\n\n"
            "### 참고한 PDF 위치\n"
            + ("\n".join(f"- {item}" for item in evidence) if evidence else "- PDF 검색 근거 없음")
        )

    trace = state.get("trace", []) + [
        f"synthesize_study_answer: 구조화 출력 생성, 근거 페이지 {len(evidence)}개"
    ]

    return {
        "answer": answer,
        "evidence": evidence,
        "trace": trace,
    }


def build_study_rag_graph():
    builder = StateGraph(StudyRAGState)

    builder.add_node("collect_pdf_chunks", collect_pdf_chunks)
    builder.add_node("grade_pdf_relevance", grade_pdf_relevance)
    builder.add_node("rewrite_pdf_query", rewrite_pdf_query)
    builder.add_node("synthesize_study_answer", synthesize_study_answer)

    builder.add_edge(START, "collect_pdf_chunks")
    builder.add_edge("collect_pdf_chunks", "grade_pdf_relevance")
    builder.add_conditional_edges(
        "grade_pdf_relevance",
        should_retry_pdf_search,
        {
            "synthesize": "synthesize_study_answer",
            "rewrite_query": "rewrite_pdf_query",
        },
    )
    builder.add_edge("rewrite_pdf_query", "collect_pdf_chunks")
    builder.add_edge("synthesize_study_answer", END)

    return builder.compile()


study_rag_graph = build_study_rag_graph()


def run_study_rag(session_id: str, query: str, mode: Literal["concept", "summary"]) -> dict:
    return study_rag_graph.invoke(
        {
            "session_id": session_id,
            "query": query,
            "mode": mode,
            "retry_count": 0,
            "documents": [],
            "relevant_documents": [],
            "trace": [
                f"StudyRAG 시작: mode={mode}, query={query}"
            ],
        }
    )


# =========================================================================
# 6. Main LangGraph
# =========================================================================

ROUTER_PROMPT = """
당신은 StudyMate 라우터입니다.
StudyMate는 사용자가 버튼으로 기능을 선택하는 서비스가 아닙니다.
사용자의 자연어 요청을 보고 Agent가 스스로 route를 선택해야 합니다.

아래 7개 중 하나로 분류하세요.

- concept: PDF 내용이나 CS 개념 설명 질문
- summary: 현재 PDF 전체 요약/정리 요청
- plan: 공부 계획/일정 요청
- quiz: 예상문제, 문제 내줘, 퀴즈 요청
- review: 사용자가 자신의 답을 채점받거나 오답 복습을 요청
- followup: 직전 AI 답변에 이어서 추가 설명/추가 내용/쉽게 설명을 요청
  예: "더 자세히", "무슨 말이야", "쉽게 설명해줘", "이거 말고 더 있어?",
      "이외에 더 알아야 할 거 있어?", "방금 내용에서 추가로 중요한 거 있어?"
- unknown: PDF/공부와 무관한 질문

최근 대화:
{history}
"""


def keyword_route(message: str) -> str:
    text = message.lower()

    if any(word in text for word in [
        "더 자세히",
        "무슨 말",
        "쉽게",
        "다시 설명",
        "왜 그런",
        "이거 말고",
        "이외",
        "추가적으로",
        "추가로",
        "방금",
        "그 밖에",
    ]):
        return "followup"

    if any(word in text for word in ["채점", "오답", "복습", "틀린", "맞아?", "맞나요"]):
        return "review"

    if any(word in text for word in ["요약", "정리"]):
        return "summary"

    if any(word in text for word in ["계획", "일정", "플랜"]):
        return "plan"

    if any(word in text for word in ["문제", "퀴즈", "예상"]):
        return "quiz"

    return "concept"


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    session_id: str
    route: str
    final_response: dict
    trace: list[str]
    used_tools: list[str]
    evidence: list[str]


def router_node(state: State) -> dict:
    session = get_session(state["session_id"])
    message = state["messages"][-1].content
    trace = state.get("trace", [])

    if OPENAI_READY:
        try:
            llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0)
            history_text = "\n".join(f"{m.type}: {m.content}" for m in state["messages"][-8:])
            decision = llm.with_structured_output(RouteDecision).invoke(
                ROUTER_PROMPT.format(history=history_text)
            )
            route = decision.route
            reason = decision.reason
        except Exception as exc:
            route = keyword_route(message)
            reason = f"LLM 라우팅 실패 → 키워드 fallback: {exc}"
    else:
        route = keyword_route(message)
        reason = "OPENAI_API_KEY 없음 → 키워드 fallback 라우팅"

    trace = trace + [
        f"router_node: route={route}",
        f"router_node: reason={reason}",
    ]

    if route != "unknown" and not session.pdf_text:
        return {
            "route": "no_pdf",
            "trace": trace + ["router_node: PDF 없음 → no_pdf로 전환"],
            "final_response": {
                "type": "no_pdf",
                "summary": "현재 업로드된 PDF가 없습니다. 먼저 PDF를 업로드한 뒤 질문해주세요.",
            },
        }

    return {
        "route": route,
        "trace": trace,
    }


def route_selector(state: State) -> str:
    return state.get("route", "unknown")


def concept_node(state: State) -> dict:
    query = state["messages"][-1].content
    result = run_study_rag(state["session_id"], query, mode="concept")

    return {
        "final_response": {
            "type": "concept",
            "summary": result["answer"],
        },
        "trace": state.get("trace", []) + result.get("trace", []),
        "used_tools": ["study_rag_graph", "FAISS_retriever"],
        "evidence": result.get("evidence", []),
    }


def summary_node(state: State) -> dict:
    result = run_study_rag(
        state["session_id"],
        "전체 요약 핵심 개념 목차 중요한 내용",
        mode="summary",
    )

    return {
        "final_response": {
            "type": "summary",
            "summary": result["answer"],
        },
        "trace": state.get("trace", []) + result.get("trace", []),
        "used_tools": ["study_rag_graph", "FAISS_retriever"],
        "evidence": result.get("evidence", []),
    }


def run_tool_agent_node(state: State) -> dict:
    session = get_session(state["session_id"])
    route = state.get("route", "quiz")

    if not OPENAI_READY:
        return {
            "final_response": {
                "type": route,
                "summary": "OPENAI_API_KEY가 없어 Tool Agent를 실행할 수 없습니다.",
            },
            "trace": state.get("trace", []) + [
                "tool_agent_node: OPENAI_API_KEY 없음"
            ],
            "used_tools": [],
            "evidence": [],
        }

    agent = build_tool_agent()

    context = SystemMessage(
        content=(
            f"현재 session_id: {session.session_id}\n"
            f"현재 PDF 이름: {session.pdf_name}\n"
            "Tool 호출 시 session_id 인자에는 반드시 위 session_id를 그대로 넣어라.\n"
            f"현재 route는 {route}이다.\n"
            "사용자 요청을 보고 필요한 Tool을 스스로 선택해 실행하라."
        )
    )

    user_message = state["messages"][-1]

    try:
        result = agent.invoke({"messages": [context, user_message]})
        used_tools = extract_tool_names(result["messages"])
        answer = last_ai_content(result["messages"])
        evidence = extract_evidence_from_messages(
            result["messages"],
            default_source=session.pdf_name or "PDF",
        )

        trace = state.get("trace", []) + [
            "tool_agent_node: create_agent 실행",
            f"tool_agent_node: 사용 Tool={used_tools if used_tools else '감지 안 됨'}",
            f"tool_agent_node: 근거 페이지 {len(evidence)}개 추출",
        ]

    except Exception as exc:
        used_tools = []
        evidence = []
        answer = f"Tool Agent 실행 중 오류가 발생했습니다: {exc}"
        trace = state.get("trace", []) + [
            f"tool_agent_node: 실행 실패 → {exc}"
        ]

    return {
        "final_response": {
            "type": route,
            "summary": answer,
        },
        "trace": trace,
        "used_tools": used_tools,
        "evidence": evidence,
    }


def review_node(state: State) -> dict:
    user_answer = state["messages"][-1].content
    last_ai = next(
        (message for message in reversed(state["messages"][:-1]) if isinstance(message, AIMessage)),
        None,
    )
    prior_context = last_ai.content if last_ai else "(직전 AI 답변 없음)"

    docs = retrieve_pdf_chunks(state["session_id"], user_answer, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)[:4500]
    evidence = evidence_from_docs(docs)

    if not OPENAI_READY:
        return {
            "final_response": {
                "type": "review",
                "summary": (
                    "OPENAI_API_KEY가 없어 자동 채점은 제한됩니다.\n\n"
                    f"직전 답변:\n{prior_context[:800]}\n\n"
                    f"PDF 근거:\n{context[:1000]}"
                ),
            },
            "trace": state.get("trace", []) + ["review_node: fallback 채점"],
            "used_tools": ["pdf_retriever"],
            "evidence": evidence,
        }

    llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0.2)
    parser = llm.with_structured_output(ReviewFeedback)

    try:
        feedback = parser.invoke(
            [
                SystemMessage(
                    content=(
                        "너는 PDF 기반 CS 학습 도우미다. "
                        "사용자의 답을 직전 문제/답변과 PDF 근거에 비추어 채점한다. "
                        "단정적으로 혼내지 말고, 올바른 개념과 다시 풀 문제를 제시한다."
                    )
                ),
                HumanMessage(
                    content=(
                        f"직전 AI 답변 또는 문제:\n{prior_context}\n\n"
                        f"사용자 답변:\n{user_answer}\n\n"
                        f"PDF 근거:\n{context}"
                    )
                ),
            ]
        )

        summary = format_review_feedback(feedback, evidence)

    except Exception:
        summary = (
            "채점 중 오류가 발생해 PDF 근거를 우선 정리합니다.\n\n"
            f"직전 문제/답변:\n{prior_context[:1000]}\n\n"
            f"사용자 답변:\n{user_answer}\n\n"
            "### 참고한 PDF 위치\n"
            + ("\n".join(f"- {item}" for item in evidence) if evidence else "- PDF 검색 근거 없음")
        )

    return {
        "final_response": {
            "type": "review",
            "summary": summary,
        },
        "trace": state.get("trace", []) + [
            "review_node: 직전 AI 답변 + PDF 근거로 채점"
        ],
        "used_tools": ["pdf_retriever"],
        "evidence": evidence,
    }


def followup_node(state: State) -> dict:
    last_ai = next(
        (message for message in reversed(state["messages"][:-1]) if isinstance(message, AIMessage)),
        None,
    )

    if last_ai is None:
        return {
            "final_response": {
                "type": "followup",
                "summary": "아직 이전 답변이 없습니다. PDF 내용에 대해 다시 질문해주세요.",
            },
            "trace": state.get("trace", []) + ["followup_node: 직전 AI 답변 없음"],
            "used_tools": [],
            "evidence": [],
        }

    user_request = state["messages"][-1].content

    if not OPENAI_READY:
        answer = f"직전 답변을 다시 정리하면 다음과 같습니다.\n\n{last_ai.content}"
    else:
        try:
            llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0)
            answer = llm.invoke(
                "아래는 직전 AI 답변입니다. 새로운 검색을 하지 말고, "
                "직전 답변에 담긴 내용만 바탕으로 사용자의 후속 요청에 맞게 설명하세요.\n\n"
                f"직전 답변:\n{last_ai.content}\n\n"
                f"사용자 후속 요청:\n{user_request}"
            ).content
        except Exception:
            answer = f"직전 답변을 다시 정리하면 다음과 같습니다.\n\n{last_ai.content}"

    return {
        "final_response": {
            "type": "followup",
            "summary": answer,
        },
        "trace": state.get("trace", []) + [
            "followup_node: 이전 대화의 직전 AI 답변을 기반으로 재설명"
        ],
        "used_tools": ["memory"],
        "evidence": [],
    }


def no_pdf_node(state: State) -> dict:
    return {
        "trace": state.get("trace", []) + ["no_pdf_node: PDF 업로드 안내"]
    }


def unknown_node(state: State) -> dict:
    return {
        "final_response": {
            "type": "unknown",
            "summary": (
                "저는 업로드된 PDF를 기반으로 자연어 요청을 분석해 "
                "개념 설명, 요약, 공부 계획, 예상문제, 답안 채점, 후속 설명을 수행하는 StudyMate입니다. "
                "PDF 내용과 관련된 질문을 해주세요."
            ),
        },
        "trace": state.get("trace", []) + ["unknown_node: 지원 범위 안내"],
        "used_tools": [],
        "evidence": [],
    }


def finalize_response_node(state: State) -> dict:
    response = state.get("final_response") or {
        "type": "error",
        "summary": "응답을 생성하지 못했습니다.",
    }

    return {
        "messages": [AIMessage(content=response["summary"])]
    }


def build_graph():
    builder = StateGraph(State)

    builder.add_node("router", router_node)
    builder.add_node("concept_node", concept_node)
    builder.add_node("summary_node", summary_node)
    builder.add_node("tool_agent_node", run_tool_agent_node)
    builder.add_node("review_node", review_node)
    builder.add_node("followup_node", followup_node)
    builder.add_node("no_pdf_node", no_pdf_node)
    builder.add_node("unknown_node", unknown_node)
    builder.add_node("finalize_response", finalize_response_node)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_selector,
        {
            "concept": "concept_node",
            "summary": "summary_node",
            "plan": "tool_agent_node",
            "quiz": "tool_agent_node",
            "review": "review_node",
            "followup": "followup_node",
            "no_pdf": "no_pdf_node",
            "unknown": "unknown_node",
        },
    )

    builder.add_edge("concept_node", "finalize_response")
    builder.add_edge("summary_node", "finalize_response")
    builder.add_edge("tool_agent_node", "finalize_response")
    builder.add_edge("review_node", "finalize_response")
    builder.add_edge("followup_node", "finalize_response")
    builder.add_edge("no_pdf_node", "finalize_response")
    builder.add_edge("unknown_node", "finalize_response")
    builder.add_edge("finalize_response", END)

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()


# =========================================================================
# 7. FastAPI
# =========================================================================

app = FastAPI(title="StudyMate Agent")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str
    route: str
    session_id: str
    pdf_name: Optional[str] = None
    used_tools: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    ok: bool
    pdf_name: str
    session_id: str
    text_length: int
    has_vectorstore: bool


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "StudyMate Agent",
        "openai_ready": OPENAI_READY,
        "model": MODEL_NAME,
        "langgraph_ready": True,
        "tools": [item.name for item in TOOLS],
        "routes": [
            "concept",
            "summary",
            "plan",
            "quiz",
            "review",
            "followup",
            "unknown",
        ],
        "chat_history_ready": True,
    }


@app.get("/api/graph/mermaid", response_class=PlainTextResponse)
def graph_mermaid() -> str:
    return graph.get_graph().draw_mermaid()


# =========================================================================
# 8. 채팅 세션 API
# 기존 /api/chats 유지 + 프론트에서 쓰는 /api/chat/sessions 추가
# =========================================================================

@app.post("/api/chats")
def create_chat():
    session = safe_create_chat_session()
    get_session(session["session_id"])
    return session


@app.get("/api/chats")
def chats():
    return {
        "sessions": enrich_sessions_with_pdf_meta(safe_list_chat_sessions())
    }


@app.get("/api/chats/{session_id}")
def chat_messages(session_id: str):
    return {
        "session_id": session_id,
        "messages": safe_get_chat_messages(session_id),
    }


@app.delete("/api/chats/{session_id}")
def delete_chat(session_id: str):
    deleted = delete_chat_session(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="대화방을 찾을 수 없습니다.")

    if session_id in SESSIONS:
        del SESSIONS[session_id]

    remove_session_pdf_meta(session_id)

    return {
        "ok": True,
        "session_id": session_id,
    }


@app.get("/api/chat/sessions")
def api_chat_sessions():
    return {
        "sessions": enrich_sessions_with_pdf_meta(safe_list_chat_sessions())
    }


@app.post("/api/chat/sessions")
def api_create_chat_session():
    new_session_id = str(uuid.uuid4())

    session = safe_create_chat_session(
        session_id=new_session_id,
        title="새 대화",
        pdf_name=None,
    )

    session_id = session.get("session_id") or new_session_id
    get_session(session_id)

    return {
        "ok": True,
        "session_id": session_id,
        "title": session.get("title") or "새 대화",
        "pdf_name": session.get("pdf_name"),
    }


@app.get("/api/chat/messages")
def api_chat_messages(session_id: str = "default"):
    return {
        "session_id": session_id,
        "messages": safe_get_chat_messages(session_id),
    }


# =========================================================================
# 9. PDF API
# =========================================================================

@app.get("/api/pdf/status")
def pdf_status(session_id: str = "default"):
    session = get_session(session_id)

    return {
        "has_pdf": bool(session.pdf_text),
        "pdf_name": session.pdf_name,
        "text_length": len(session.pdf_text),
        "openai_ready": OPENAI_READY,
        "has_vectorstore": session.vectorstore is not None,
    }


@app.get("/api/pdf/page-image")
def pdf_page_image(session_id: str = "default", page: int = 1):
    session = get_session(session_id)

    if not session.pdf_path or not session.pdf_path.exists():
        raise HTTPException(status_code=404, detail="업로드된 PDF가 없습니다.")

    if page < 1:
        raise HTTPException(status_code=400, detail="page는 1 이상이어야 합니다.")

    try:
        with fitz.open(str(session.pdf_path)) as doc:
            if page > len(doc):
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF에 {page}페이지가 없습니다.",
                )

            pdf_page = doc.load_page(page - 1)
            matrix = fitz.Matrix(2.0, 2.0)
            pix = pdf_page.get_pixmap(matrix=matrix, alpha=False)
            image_bytes = pix.tobytes("png")

        return Response(content=image_bytes, media_type="image/png")

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PDF 페이지 이미지를 생성하지 못했습니다: {exc}",
        )


@app.post("/api/pdf/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile = File(...), session_id: str = Form("default")):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

    safe_create_chat_session(
        session_id=session_id,
        title="새 대화",
        pdf_name=None,
    )

    safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    path = PDF_DIR / safe_name

    with path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    page_docs = read_pdf_documents(path, file.filename)
    text = docs_to_text(page_docs)

    if not text:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

        raise HTTPException(
            status_code=400,
            detail="PDF에서 텍스트를 추출하지 못했습니다. 스캔본이면 OCR이 필요합니다.",
        )

    session = get_session(session_id)
    session.pdf_path = path
    session.pdf_name = file.filename
    session.page_documents = page_docs
    session.pdf_text = text

    build_index(session)
    persist_session_pdf(session)
    safe_update_chat_pdf_name(session_id, file.filename)

    return UploadResponse(
        ok=True,
        pdf_name=session.pdf_name,
        session_id=session_id,
        text_length=len(text),
        has_vectorstore=session.vectorstore is not None,
    )


# =========================================================================
# 10. Chat API
# =========================================================================

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session = get_session(req.session_id)

    safe_create_chat_session(
        session_id=req.session_id,
        title=make_chat_title(req.message),
        pdf_name=session.pdf_name,
    )

    previous_messages = safe_get_chat_messages(req.session_id)
    history_messages = history_to_langchain_messages(previous_messages, limit=20)
    current_message = HumanMessage(content=req.message)
    graph_messages = history_messages + [current_message]

    safe_save_chat_message(
        session_id=req.session_id,
        role="user",
        content=req.message,
        pdf_name=session.pdf_name,
    )

    config = {
        "configurable": {
            "thread_id": req.session_id,
        }
    }

    result = graph.invoke(
        {
            "messages": graph_messages,
            "session_id": req.session_id,
            "trace": [
                "chat: 사용자 자연어 요청 수신",
                f"chat: 이전 저장 메시지 {len(previous_messages)}개 참고",
            ],
            "used_tools": [],
            "evidence": [],
        },
        config=config,
    )

    final_response = result.get("final_response") or {}
    answer = final_response.get("summary") or result["messages"][-1].content
    route = result.get("route", "unknown")
    used_tools = result.get("used_tools", [])
    evidence = result.get("evidence", [])
    trace = result.get("trace", [])

    safe_save_chat_message(
        session_id=req.session_id,
        role="assistant",
        content=answer,
        route=route,
        pdf_name=session.pdf_name,
        used_tools=used_tools,
        evidence=evidence,
    )

    return ChatResponse(
        answer=answer,
        route=route,
        session_id=req.session_id,
        pdf_name=session.pdf_name,
        used_tools=used_tools,
        evidence=evidence,
        trace=trace,
    )


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)