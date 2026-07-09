from __future__ import annotations

import re
import uuid
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage

from services.chat_history import (
    create_chat_session,
    get_chat_messages,
    list_chat_sessions,
    save_chat_message,
    update_chat_pdf_name,
)


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
        return {
            **message,
            "role": role,
            "content": message.get("content", ""),
        }

    return {
        "role": normalize_message_role(str(getattr(message, "role", "ai"))),
        "content": getattr(message, "content", ""),
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
        return {
            **session,
            "session_id": session_id,
            "title": session.get("title") or session.get("name") or "새 대화",
            "pdf_name": session.get("pdf_name"),
        }

    return {
        "session_id": getattr(session, "session_id", None) or getattr(session, "id", None) or "default",
        "title": getattr(session, "title", None) or getattr(session, "name", None) or "새 대화",
        "pdf_name": getattr(session, "pdf_name", None),
    }


def normalize_session_list(payload: Any) -> list[dict[str, Any]]:
    items = payload.get("sessions") or payload.get("items") or payload.get("data") or [] if isinstance(payload, dict) else payload
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
        lambda: create_chat_session(session_id=session_id, title=title, pdf_name=pdf_name),
        lambda: create_chat_session(session_id=session_id, title=title),
        lambda: create_chat_session(session_id=session_id),
        lambda: create_chat_session(session_id),
        lambda: create_chat_session(title=title),
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
            return {"ok": True, "session_id": session_id, "title": title, "pdf_name": pdf_name}
        except TypeError:
            continue
        except Exception:
            break

    return {"ok": True, "session_id": session_id, "title": title, "pdf_name": pdf_name}


def safe_save_chat_message(
    session_id: str,
    role: str,
    content: str,
    route: Optional[str] = None,
    pdf_name: Optional[str] = None,
    used_tools: Optional[list[str]] = None,
    evidence: Optional[list[str]] = None,
) -> None:
    save_role = "assistant" if normalize_message_role(role) == "ai" else normalize_message_role(role)
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
        lambda: save_chat_message(session_id=session_id, role=save_role, content=content, pdf_name=pdf_name),
        lambda: save_chat_message(session_id=session_id, role=save_role, content=content),
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
    for call in (
        lambda: update_chat_pdf_name(session_id=session_id, pdf_name=pdf_name),
        lambda: update_chat_pdf_name(session_id, pdf_name),
    ):
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
