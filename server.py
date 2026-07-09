from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import fitz
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from agent.graph import graph
from agent.schemas import (
    ChatRequest,
    ChatResponse,
    QuizAnswerRequest,
    QuizSet,
    QuizStartRequest,
    UploadResponse,
)
from agent.tools import TOOLS, evidence_from_docs
from services.chat_history import delete_chat_session
from services.chat_store import (
    history_to_langchain_messages,
    make_chat_title,
    safe_create_chat_session,
    safe_get_chat_messages,
    safe_list_chat_sessions,
    safe_save_chat_message,
    safe_update_chat_pdf_name,
)
from services.pdf_store import (
    build_index,
    docs_to_text,
    enrich_sessions_with_pdf_meta,
    persist_session_pdf,
    read_pdf_documents,
    remove_session_pdf_meta,
    retrieve_pdf_chunks,
)
from services.quiz_store import (
    get_current_question,
    get_wrong_answer,
    list_wrong_answers,
    reset_quiz,
    start_quiz,
    submit_answer,
)
from services.session_store import (
    MODEL_NAME,
    OPENAI_READY,
    PDF_DIR,
    SESSIONS,
    STATIC_DIR,
    get_session,
)


logger = logging.getLogger("studymate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="StudyMate Agent")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "request_failed method=%s path=%s elapsed_ms=%.2f",
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.info(
        "request_done method=%s path=%s status=%s elapsed_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )

    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(
        "unhandled_exception method=%s path=%s error=%s",
        request.method,
        request.url.path,
        exc,
    )

    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": "internal_server_error",
            "detail": "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        },
    )


def model_to_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()

    if hasattr(item, "dict"):
        return item.dict()

    if isinstance(item, dict):
        return item

    return {}


def find_pdf_by_original_name(pdf_name: str | None) -> Path | None:
    if not pdf_name:
        return None

    target_name = Path(pdf_name).name

    for candidate in PDF_DIR.glob(f"*_{target_name}"):
        if candidate.exists() and candidate.is_file():
            return candidate

    direct_path = PDF_DIR / target_name
    if direct_path.exists() and direct_path.is_file():
        return direct_path

    return None


def ensure_session_pdf_loaded(session_id: str, pdf_name: str | None = None):
    session = get_session(session_id)

    if session.pdf_path and session.pdf_path.exists() and session.pdf_text:
        return session

    if pdf_name:
        pdf_path = find_pdf_by_original_name(pdf_name)

        if pdf_path:
            page_docs = read_pdf_documents(pdf_path, Path(pdf_name).name)
            text = docs_to_text(page_docs)

            if text:
                session.pdf_path = pdf_path
                session.pdf_name = Path(pdf_name).name
                session.page_documents = page_docs
                session.pdf_text = text
                build_index(session)
                persist_session_pdf(session)
                safe_update_chat_pdf_name(session_id, session.pdf_name)

    return session


def normalize_quiz_questions(
    questions: list[dict[str, Any]],
    docs: list,
    pdf_name: str | None,
) -> list[dict[str, Any]]:
    fallback_page = None

    for doc in docs:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            fallback_page = page
            break

    normalized: list[dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
        choices = question.get("choices") or []

        normalized.append(
            {
                "id": question.get("id") or f"q{index}",
                "question": str(question.get("question", "")).strip(),
                "choices": [str(choice).strip() for choice in choices],
                "answer": str(question.get("answer", "")).strip(),
                "explanation": str(question.get("explanation", "")).strip(),
                "page": question.get("page") or fallback_page,
                "pdf_name": question.get("pdf_name") or pdf_name,
            }
        )

    return [item for item in normalized if item["question"] and item["answer"]]


def format_quiz_question(question: dict[str, Any] | None) -> str:
    if not question:
        return "출제할 문제가 없습니다."

    choices = question.get("choices") or []

    lines = [
        f"{question.get('number', 1)}번 문제",
        "",
        question.get("question", ""),
        "",
    ]

    for choice in choices:
        lines.append(str(choice))

    lines.append("")
    lines.append("답을 입력해주세요. 예: A 또는 B")

    return "\n".join(lines)


def format_quiz_feedback(result: dict[str, Any]) -> str:
    is_correct = result.get("is_correct")
    page = result.get("page")
    pdf_name = result.get("pdf_name")

    lines: list[str] = []

    if is_correct:
        lines.append("정답입니다.")
    else:
        lines.append("오답입니다.")
        lines.append("")
        lines.append(f"내 답: {result.get('user_answer')}")
        lines.append(f"정답: {result.get('correct_answer')}")

    lines.append("")
    lines.append("해설:")
    lines.append(result.get("explanation") or "해설이 없습니다.")

    if page:
        lines.append("")
        lines.append("PDF 근거:")
        lines.append(f"- page {page} - {pdf_name or 'PDF'}")

    if result.get("finished"):
        lines.append("")
        lines.append("퀴즈가 끝났습니다.")
        lines.append(f"현재 오답 수: {result.get('wrong_count', 0)}개")
    else:
        next_question = result.get("next_question")

        if next_question:
            lines.append("")
            lines.append("다음 문제입니다.")
            lines.append("")
            lines.append(format_quiz_question(next_question))

    return "\n".join(lines)


def format_wrong_answer_detail(item: dict[str, Any]) -> str:
    lines = [
        f"오답 복습 - {item.get('number')}번 문제",
        "",
        "문제:",
        item.get("question", ""),
        "",
    ]

    choices = item.get("choices") or []

    for choice in choices:
        lines.append(str(choice))

    lines.extend(
        [
            "",
            f"내 답: {item.get('user_answer')}",
            f"정답: {item.get('correct_answer')}",
            "",
            "해설:",
            item.get("explanation", ""),
        ]
    )

    if item.get("page"):
        lines.extend(
            [
                "",
                "PDF 근거:",
                f"- page {item.get('page')} - {item.get('pdf_name') or 'PDF'}",
            ]
        )

    return "\n".join(lines)


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
            "web_search",
            "hybrid",
            "unknown",
        ],
        "chat_history_ready": True,
        "quiz_ready": True,
    }


@app.get("/api/graph/mermaid", response_class=PlainTextResponse)
def graph_mermaid() -> str:
    return graph.get_graph().draw_mermaid()


@app.post("/api/chats")
def create_chat():
    session = safe_create_chat_session()
    get_session(session["session_id"])
    return session


@app.get("/api/chats")
def chats():
    return {"sessions": enrich_sessions_with_pdf_meta(safe_list_chat_sessions())}


@app.get("/api/chats/{session_id}")
def chat_messages(session_id: str):
    session = get_session(session_id)

    return {
        "session_id": session_id,
        "pdf_name": session.pdf_name,
        "has_pdf": bool(session.pdf_text),
        "messages": safe_get_chat_messages(session_id),
    }


@app.delete("/api/chats/{session_id}")
def delete_chat(session_id: str):
    deleted = delete_chat_session(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="대화방을 찾을 수 없습니다.")

    if session_id in SESSIONS:
        del SESSIONS[session_id]

    reset_quiz(session_id)
    remove_session_pdf_meta(session_id)

    return {"ok": True, "session_id": session_id}


@app.get("/api/chat/sessions")
def api_chat_sessions():
    return {"sessions": enrich_sessions_with_pdf_meta(safe_list_chat_sessions())}


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
    session = get_session(session_id)

    return {
        "session_id": session_id,
        "pdf_name": session.pdf_name,
        "has_pdf": bool(session.pdf_text),
        "messages": safe_get_chat_messages(session_id),
    }


@app.get("/api/pdf/status")
def pdf_status(session_id: str = "default", pdf_name: str | None = None):
    session = ensure_session_pdf_loaded(session_id, pdf_name)

    return {
        "has_pdf": bool(session.pdf_text),
        "pdf_name": session.pdf_name,
        "text_length": len(session.pdf_text),
        "openai_ready": OPENAI_READY,
        "has_vectorstore": session.vectorstore is not None,
    }


@app.get("/api/pdf/page-image")
def pdf_page_image(
    session_id: str = "default",
    page: int = 1,
    pdf_name: str | None = None,
):
    if page < 1:
        raise HTTPException(status_code=400, detail="page는 1 이상이어야 합니다.")

    session = ensure_session_pdf_loaded(session_id, pdf_name)
    pdf_path = session.pdf_path if session.pdf_path and session.pdf_path.exists() else None

    if pdf_path is None and pdf_name:
        pdf_path = find_pdf_by_original_name(pdf_name)

        if pdf_path:
            page_docs = read_pdf_documents(pdf_path, Path(pdf_name).name)
            text = docs_to_text(page_docs)

            if text:
                session.pdf_path = pdf_path
                session.pdf_name = Path(pdf_name).name
                session.page_documents = page_docs
                session.pdf_text = text
                build_index(session)
                persist_session_pdf(session)
                safe_update_chat_pdf_name(session_id, session.pdf_name)

    if pdf_path is None:
        raise HTTPException(
            status_code=404,
            detail="현재 세션에 연결된 PDF 파일을 찾지 못했습니다.",
        )

    try:
        with fitz.open(str(pdf_path)) as doc:
            if page > len(doc):
                raise HTTPException(
                    status_code=404,
                    detail=f"PDF에 {page}페이지가 없습니다.",
                )

            pdf_page = doc.load_page(page - 1)
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
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

    safe_create_chat_session(session_id=session_id, title="새 대화", pdf_name=None)

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
    reset_quiz(session_id)

    return UploadResponse(
        ok=True,
        pdf_name=session.pdf_name,
        session_id=session_id,
        text_length=len(text),
        has_vectorstore=session.vectorstore is not None,
    )


@app.post("/api/quiz/start")
def quiz_start(req: QuizStartRequest):
    session = ensure_session_pdf_loaded(req.session_id)

    if not session.pdf_text:
        raise HTTPException(
            status_code=400,
            detail="현재 업로드된 PDF가 없습니다. 먼저 PDF를 업로드해주세요.",
        )

    safe_create_chat_session(
        session_id=req.session_id,
        title=make_chat_title(req.topic),
        pdf_name=session.pdf_name,
    )

    count = max(1, min(req.count, 10))

    docs = retrieve_pdf_chunks(
        req.session_id,
        f"{req.topic} 예상문제 시험 중요 개념 정답 해설",
        k=8,
    )

    context = "\n\n".join(doc.page_content for doc in docs)
    evidence = evidence_from_docs(docs)

    if not context:
        raise HTTPException(
            status_code=400,
            detail="PDF에서 예상문제를 만들 근거를 찾지 못했습니다.",
        )

    llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0.2)
    structured_llm = llm.with_structured_output(QuizSet)

    quiz_set = structured_llm.invoke(
        [
            HumanMessage(
                content=(
                    "아래 PDF 근거만 사용해서 예상문제를 만들어라.\n"
                    "반드시 한국어로 작성해라.\n"
                    "문제는 객관식 위주로 만들어라.\n"
                    "answer에는 정답 선택지 문자만 넣어라. 예: A\n"
                    "choices에는 A, B, C, D 형식의 선택지를 넣어라.\n"
                    "explanation에는 왜 그 답인지 자세히 설명해라.\n"
                    "page에는 근거가 되는 PDF 페이지 번호를 넣어라.\n"
                    f"pdf_name에는 '{session.pdf_name}' 값을 넣어라.\n"
                    f"문제 수: {count}개\n"
                    f"주제: {req.topic}\n\n"
                    f"[PDF 근거]\n{context[:7000]}\n\n"
                    f"[참고 페이지]\n{', '.join(evidence)}"
                )
            )
        ]
    )

    raw_questions = [model_to_dict(item) for item in quiz_set.questions]
    questions = normalize_quiz_questions(raw_questions, docs, session.pdf_name)

    if not questions:
        raise HTTPException(
            status_code=500,
            detail="예상문제를 생성하지 못했습니다.",
        )

    state = start_quiz(req.session_id, questions)
    first_question = get_current_question(req.session_id)
    answer = format_quiz_question(first_question)

    safe_save_chat_message(
        session_id=req.session_id,
        role="user",
        content=req.topic,
        pdf_name=session.pdf_name,
    )

    safe_save_chat_message(
        session_id=req.session_id,
        role="assistant",
        content=answer,
        route="quiz",
        pdf_name=session.pdf_name,
        used_tools=["retrieve_pdf_chunks", "quiz_generator"],
        evidence=evidence,
    )

    return {
        "ok": True,
        "session_id": req.session_id,
        "pdf_name": session.pdf_name,
        "count": len(state.questions),
        "question": first_question,
        "answer": answer,
        "route": "quiz",
        "used_tools": ["retrieve_pdf_chunks", "quiz_generator"],
        "evidence": evidence,
        "trace": [
            "quiz_start: PDF 근거 검색",
            f"quiz_start: 예상문제 {len(state.questions)}개 생성",
            "quiz_start: 첫 번째 문제만 사용자에게 출력",
        ],
    }


@app.post("/api/quiz/answer")
def quiz_answer(req: QuizAnswerRequest):
    session = get_session(req.session_id)
    result = submit_answer(req.session_id, req.answer)

    if not result.get("ok"):
        return {
            "ok": False,
            "answer": result.get("message", "진행 중인 퀴즈가 없습니다."),
            "wrong_answers": list_wrong_answers(req.session_id),
        }

    answer = format_quiz_feedback(result)
    evidence = (
        [f"page {result.get('page')} - {result.get('pdf_name')}"]
        if result.get("page")
        else []
    )

    safe_save_chat_message(
        session_id=req.session_id,
        role="user",
        content=req.answer,
        pdf_name=session.pdf_name,
    )

    safe_save_chat_message(
        session_id=req.session_id,
        role="assistant",
        content=answer,
        route="quiz_answer",
        pdf_name=session.pdf_name,
        used_tools=["quiz_store", "answer_checker"],
        evidence=evidence,
    )

    return {
        "ok": True,
        "session_id": req.session_id,
        "is_correct": result.get("is_correct"),
        "answer": answer,
        "wrong_item": result.get("wrong_item"),
        "wrong_answers": list_wrong_answers(req.session_id),
        "finished": result.get("finished"),
        "next_question": result.get("next_question"),
        "route": "quiz_answer",
        "used_tools": ["quiz_store", "answer_checker"],
        "evidence": evidence,
        "trace": [
            "quiz_answer: 사용자 답안 수신",
            "quiz_answer: 현재 문제 정답과 비교",
            "quiz_answer: 오답이면 오답 목록에 저장",
        ],
    }


@app.get("/api/quiz/wrongs")
def quiz_wrongs(session_id: str = "default"):
    return {
        "ok": True,
        "session_id": session_id,
        "wrong_answers": list_wrong_answers(session_id),
    }


@app.get("/api/quiz/wrongs/{wrong_id}")
def quiz_wrong_detail(wrong_id: str, session_id: str = "default"):
    item = get_wrong_answer(session_id, wrong_id)

    if not item:
        raise HTTPException(status_code=404, detail="오답 기록을 찾지 못했습니다.")

    return {
        "ok": True,
        "wrong": item,
        "answer": format_wrong_answer_detail(item),
    }


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
    graph_messages = history_messages + [HumanMessage(content=req.message)]

    safe_save_chat_message(
        session_id=req.session_id,
        role="user",
        content=req.message,
        pdf_name=session.pdf_name,
    )

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
            "retry_count": 0,
            "max_retries": 1,
        },
        config={"configurable": {"thread_id": req.session_id}},
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