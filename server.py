from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import fitz
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage

from agent.graph import graph
from agent.schemas import ChatRequest, ChatResponse, UploadResponse
from agent.tools import TOOLS
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
)
from services.session_store import (
    MODEL_NAME,
    OPENAI_READY,
    PDF_DIR,
    SESSIONS,
    STATIC_DIR,
    get_session,
)


app = FastAPI(title="StudyMate Agent")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
        "routes": ["concept", "summary", "plan", "quiz", "review", "followup", "unknown"],
        "chat_history_ready": True,
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
    return {"session_id": session_id, "messages": safe_get_chat_messages(session_id)}


@app.delete("/api/chats/{session_id}")
def delete_chat(session_id: str):
    deleted = delete_chat_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="대화방을 찾을 수 없습니다.")

    if session_id in SESSIONS:
        del SESSIONS[session_id]

    remove_session_pdf_meta(session_id)
    return {"ok": True, "session_id": session_id}


@app.get("/api/chat/sessions")
def api_chat_sessions():
    return {"sessions": enrich_sessions_with_pdf_meta(safe_list_chat_sessions())}


@app.post("/api/chat/sessions")
def api_create_chat_session():
    new_session_id = str(uuid.uuid4())
    session = safe_create_chat_session(session_id=new_session_id, title="새 대화", pdf_name=None)
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
    return {"session_id": session_id, "messages": safe_get_chat_messages(session_id)}


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
                raise HTTPException(status_code=404, detail=f"PDF에 {page}페이지가 없습니다.")

            pdf_page = doc.load_page(page - 1)
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
            image_bytes = pix.tobytes("png")

        return Response(content=image_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF 페이지 이미지를 생성하지 못했습니다: {exc}")


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
        raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출하지 못했습니다. 스캔본이면 OCR이 필요합니다.")

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
