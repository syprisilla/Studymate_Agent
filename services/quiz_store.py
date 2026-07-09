from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class QuizState:
    session_id: str
    active: bool = False
    current_index: int = 0
    questions: list[dict[str, Any]] = field(default_factory=list)
    wrong_answers: list[dict[str, Any]] = field(default_factory=list)


QUIZ_STATES: dict[str, QuizState] = {}


def get_quiz_state(session_id: str) -> QuizState:
    if session_id not in QUIZ_STATES:
        QUIZ_STATES[session_id] = QuizState(session_id=session_id)

    return QUIZ_STATES[session_id]


def start_quiz(session_id: str, questions: list[dict[str, Any]]) -> QuizState:
    state = get_quiz_state(session_id)
    state.active = True
    state.current_index = 0
    state.questions = []

    for index, question in enumerate(questions, start=1):
        question_id = question.get("id") or f"q{index}_{uuid4().hex[:8]}"
        state.questions.append(
            {
                "id": question_id,
                "number": index,
                "question": question.get("question", ""),
                "choices": question.get("choices", []),
                "answer": str(question.get("answer", "")).strip(),
                "explanation": question.get("explanation", ""),
                "page": question.get("page"),
                "pdf_name": question.get("pdf_name"),
            }
        )

    return state


def get_current_question(session_id: str) -> dict[str, Any] | None:
    state = get_quiz_state(session_id)

    if not state.active:
        return None

    if state.current_index >= len(state.questions):
        state.active = False
        return None

    return state.questions[state.current_index]


def submit_answer(session_id: str, user_answer: str) -> dict[str, Any]:
    state = get_quiz_state(session_id)
    question = get_current_question(session_id)

    if not question:
        return {
            "ok": False,
            "finished": True,
            "message": "진행 중인 퀴즈가 없습니다.",
        }

    normalized_user_answer = str(user_answer).strip().lower()
    normalized_correct_answer = str(question.get("answer", "")).strip().lower()

    is_correct = normalized_user_answer == normalized_correct_answer

    wrong_item = None

    if not is_correct:
        wrong_item = {
            "id": f"wrong_{uuid4().hex[:10]}",
            "question_id": question["id"],
            "number": question["number"],
            "question": question["question"],
            "choices": question.get("choices", []),
            "user_answer": user_answer,
            "correct_answer": question.get("answer", ""),
            "explanation": question.get("explanation", ""),
            "page": question.get("page"),
            "pdf_name": question.get("pdf_name"),
        }
        state.wrong_answers.append(wrong_item)

    state.current_index += 1

    finished = state.current_index >= len(state.questions)

    if finished:
        state.active = False
        next_question = None
    else:
        next_question = state.questions[state.current_index]

    return {
        "ok": True,
        "is_correct": is_correct,
        "question": question,
        "user_answer": user_answer,
        "correct_answer": question.get("answer", ""),
        "explanation": question.get("explanation", ""),
        "page": question.get("page"),
        "pdf_name": question.get("pdf_name"),
        "wrong_item": wrong_item,
        "finished": finished,
        "next_question": next_question,
        "wrong_count": len(state.wrong_answers),
    }


def list_wrong_answers(session_id: str) -> list[dict[str, Any]]:
    return get_quiz_state(session_id).wrong_answers


def get_wrong_answer(session_id: str, wrong_id: str) -> dict[str, Any] | None:
    state = get_quiz_state(session_id)

    for item in state.wrong_answers:
        if item.get("id") == wrong_id:
            return item

    return None


def reset_quiz(session_id: str) -> None:
    if session_id in QUIZ_STATES:
        del QUIZ_STATES[session_id]