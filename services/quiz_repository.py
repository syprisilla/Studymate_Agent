from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from services.database import get_connection


def _question_to_row(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": question.get("id"),
        "number": question.get("number"),
        "question": question.get("question", ""),
        "choices": json.loads(question.get("choices_json") or "[]")
        if "choices_json" in question
        else question.get("choices", []),
        "answer": question.get("correct_answer", question.get("answer", "")),
        "explanation": question.get("explanation", ""),
        "page": question.get("page"),
        "pdf_name": question.get("pdf_name"),
    }


def _wrong_answer_to_response(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "question_id": row["question_id"],
        "number": row["number"],
        "question": row["question"],
        "choices": json.loads(row["choices_json"] or "[]"),
        "user_answer": row["user_answer"],
        "correct_answer": row["correct_answer"],
        "explanation": row["explanation"],
        "page": row["page"],
        "pdf_name": row["pdf_name"],
    }


def save_quiz(
    *,
    quiz_id: str,
    session_id: str,
    pdf_name: str | None,
    active: bool,
    current_index: int,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO quizzes
            (id, session_id, pdf_name, active, current_index, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (quiz_id, session_id, pdf_name, int(active), current_index),
        )


def save_quiz_question(
    *,
    quiz_id: str,
    session_id: str,
    question: dict[str, Any],
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO quiz_questions
            (id, quiz_id, session_id, number, question, choices_json, correct_answer,
             explanation, page, pdf_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question["id"],
                quiz_id,
                session_id,
                question.get("number"),
                question.get("question", ""),
                json.dumps(question.get("choices", []), ensure_ascii=False),
                question.get("answer", ""),
                question.get("explanation", ""),
                question.get("page"),
                question.get("pdf_name"),
            ),
        )


def save_quiz_attempt(
    *,
    quiz_id: str | None,
    question_id: str,
    session_id: str,
    user_answer: str,
    correct_answer: str,
    is_correct: bool,
) -> str:
    attempt_id = f"attempt_{uuid4().hex}"
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO quiz_attempts
            (id, quiz_id, question_id, session_id, user_answer, correct_answer, is_correct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                quiz_id,
                question_id,
                session_id,
                user_answer,
                correct_answer,
                int(is_correct),
            ),
        )
    return attempt_id


def save_wrong_answer(
    *,
    quiz_id: str | None,
    session_id: str,
    item: dict[str, Any],
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO wrong_answers
            (id, quiz_id, question_id, session_id, number, question, choices_json,
             user_answer, correct_answer, explanation, page, pdf_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item["id"],
                quiz_id,
                item["question_id"],
                session_id,
                item.get("number"),
                item.get("question", ""),
                json.dumps(item.get("choices", []), ensure_ascii=False),
                item.get("user_answer"),
                item.get("correct_answer"),
                item.get("explanation", ""),
                item.get("page"),
                item.get("pdf_name"),
            ),
        )


def update_quiz_progress(
    *,
    quiz_id: str | None,
    current_index: int,
    active: bool,
) -> None:
    if not quiz_id:
        return

    with get_connection() as conn:
        conn.execute(
            """
            UPDATE quizzes
            SET current_index = ?, active = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (current_index, int(active), quiz_id),
        )


def deactivate_active_quizzes(session_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE quizzes
            SET active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE session_id = ? AND active = 1
            """,
            (session_id,),
        )


def load_active_quiz(session_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        quiz = conn.execute(
            """
            SELECT *
            FROM quizzes
            WHERE session_id = ? AND active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()

        if quiz is None:
            return None

        question_rows = conn.execute(
            """
            SELECT *
            FROM quiz_questions
            WHERE session_id = ? AND quiz_id = ?
            ORDER BY number ASC
            """,
            (session_id, quiz["id"]),
        ).fetchall()

    return {
        "quiz_id": quiz["id"],
        "session_id": quiz["session_id"],
        "active": bool(quiz["active"]),
        "current_index": quiz["current_index"],
        "questions": [_question_to_row(dict(row)) for row in question_rows],
    }


def load_wrong_answers(session_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM wrong_answers
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        ).fetchall()

    return [_wrong_answer_to_response(row) for row in rows]


def load_wrong_answer(session_id: str, wrong_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM wrong_answers
            WHERE session_id = ? AND id = ?
            LIMIT 1
            """,
            (session_id, wrong_id),
        ).fetchone()

    return _wrong_answer_to_response(row) if row else None


def delete_quiz_history(session_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM wrong_answers WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM quiz_attempts WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM quiz_questions WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM quizzes WHERE session_id = ?", (session_id,))
