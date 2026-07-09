from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
HISTORY_PATH = DATA_DIR / "chat_history.json"


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_history_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text(
            json.dumps({"sessions": {}}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_history() -> dict[str, Any]:
    ensure_history_file()

    try:
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"sessions": {}}


def save_history(data: dict[str, Any]) -> None:
    ensure_history_file()
    HISTORY_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_title_from_message(message: str) -> str:
    text = " ".join(message.strip().split())

    if not text:
        return "새 대화"

    if len(text) > 24:
        return text[:24] + "..."

    return text


def create_chat_session(
    session_id: str | None = None,
    title: str | None = None,
    pdf_name: str | None = None,
) -> dict[str, Any]:
    data = load_history()
    session_id = session_id or uuid.uuid4().hex
    created_at = now_text()

    data["sessions"][session_id] = {
        "session_id": session_id,
        "title": title or "새 대화",
        "pdf_name": pdf_name,
        "created_at": created_at,
        "updated_at": created_at,
        "messages": [],
    }

    save_history(data)

    return data["sessions"][session_id]


def ensure_chat_session(session_id: str, first_message: str | None = None) -> dict[str, Any]:
    data = load_history()
    sessions = data.setdefault("sessions", {})

    if session_id not in sessions:
        created_at = now_text()
        sessions[session_id] = {
            "session_id": session_id,
            "title": make_title_from_message(first_message or "새 대화"),
            "pdf_name": None,
            "created_at": created_at,
            "updated_at": created_at,
            "messages": [],
        }
        save_history(data)

    return sessions[session_id]


def list_chat_sessions() -> list[dict[str, Any]]:
    data = load_history()
    sessions = list(data.get("sessions", {}).values())

    sessions.sort(
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )

    return [
        {
            "session_id": item.get("session_id"),
            "title": item.get("title") or "새 대화",
            "pdf_name": item.get("pdf_name"),
            "created_at": item.get("created_at"),
            "updated_at": item.get("updated_at"),
            "message_count": len(item.get("messages", [])),
        }
        for item in sessions
    ]


def get_chat_session(session_id: str) -> dict[str, Any]:
    data = load_history()
    session = data.get("sessions", {}).get(session_id)

    if not session:
        return {
            "session_id": session_id,
            "title": "새 대화",
            "pdf_name": None,
            "messages": [],
        }

    return session


def get_chat_messages(session_id: str) -> list[dict[str, Any]]:
    session = get_chat_session(session_id)
    return session.get("messages", [])


def save_chat_message(
    session_id: str,
    role: str,
    content: str,
    route: str | None = None,
    pdf_name: str | None = None,
    used_tools: list[str] | None = None,
    evidence: list[str] | None = None,
) -> None:
    data = load_history()
    sessions = data.setdefault("sessions", {})

    if session_id not in sessions:
        created_at = now_text()
        sessions[session_id] = {
            "session_id": session_id,
            "title": make_title_from_message(content) if role == "user" else "새 대화",
            "pdf_name": pdf_name,
            "created_at": created_at,
            "updated_at": created_at,
            "messages": [],
        }

    session = sessions[session_id]

    if role == "user" and session.get("title") in [None, "", "새 대화"]:
        session["title"] = make_title_from_message(content)

    if pdf_name:
        session["pdf_name"] = pdf_name

    session["messages"].append(
        {
            "role": role,
            "content": content,
            "route": route,
            "pdf_name": pdf_name,
            "used_tools": used_tools or [],
            "evidence": evidence or [],
            "created_at": now_text(),
        }
    )

    session["updated_at"] = now_text()
    save_history(data)


def update_chat_pdf_name(session_id: str, pdf_name: str) -> None:
    data = load_history()
    sessions = data.setdefault("sessions", {})

    if session_id not in sessions:
        created_at = now_text()
        sessions[session_id] = {
            "session_id": session_id,
            "title": pdf_name,
            "pdf_name": pdf_name,
            "created_at": created_at,
            "updated_at": created_at,
            "messages": [],
        }
    else:
        sessions[session_id]["pdf_name"] = pdf_name
        sessions[session_id]["updated_at"] = now_text()

        if sessions[session_id].get("title") in [None, "", "새 대화"]:
            sessions[session_id]["title"] = pdf_name

    save_history(data)


def delete_chat_session(session_id: str) -> bool:
    data = load_history()
    sessions = data.setdefault("sessions", {})

    if session_id not in sessions:
        return False

    del sessions[session_id]
    save_history(data)
    return True
