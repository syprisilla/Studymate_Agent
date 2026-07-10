from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from services.quiz_repository import (
    deactivate_active_quizzes,
    delete_quiz_history as delete_quiz_history_from_db,
    load_active_quiz,
    load_wrong_answer,
    load_wrong_answers,
    save_quiz,
    save_quiz_attempt,
    save_quiz_question,
    save_wrong_answer,
    update_quiz_progress,
)


logger = logging.getLogger("studymate.quiz_store")


@dataclass
class QuizState:
    session_id: str
    quiz_id: str | None = None
    active: bool = False
    current_index: int = 0
    questions: list[dict[str, Any]] = field(default_factory=list)
    wrong_answers: list[dict[str, Any]] = field(default_factory=list)


QUIZ_STATES: dict[str, QuizState] = {}


def _load_wrong_answers_safely(session_id: str) -> list[dict[str, Any]]:
    try:
        return load_wrong_answers(session_id)
    except Exception:
        logger.exception("failed_to_load_wrong_answers session_id=%s", session_id)
        return []


def _restore_quiz_state(session_id: str) -> QuizState | None:
    try:
        active_quiz = load_active_quiz(session_id)
    except Exception:
        logger.exception("failed_to_restore_quiz_state session_id=%s", session_id)
        return None

    if not active_quiz:
        return None

    return QuizState(
        session_id=session_id,
        quiz_id=active_quiz.get("quiz_id"),
        active=bool(active_quiz.get("active")),
        current_index=int(active_quiz.get("current_index") or 0),
        questions=active_quiz.get("questions") or [],
        wrong_answers=_load_wrong_answers_safely(session_id),
    )


def get_quiz_state(session_id: str) -> QuizState:
    if session_id not in QUIZ_STATES:
        restored_state = _restore_quiz_state(session_id)
        QUIZ_STATES[session_id] = restored_state or QuizState(
            session_id=session_id,
            wrong_answers=_load_wrong_answers_safely(session_id),
        )

    return QUIZ_STATES[session_id]


def start_quiz(session_id: str, questions: list[dict[str, Any]]) -> QuizState:
    state = get_quiz_state(session_id)
    quiz_id = f"quiz_{uuid4().hex}"
    pdf_name = next((item.get("pdf_name") for item in questions if item.get("pdf_name")), None)

    state.quiz_id = quiz_id
    state.active = True
    state.current_index = 0
    state.questions = []
    state.wrong_answers = _load_wrong_answers_safely(session_id)

    try:
        deactivate_active_quizzes(session_id)
        save_quiz(
            quiz_id=quiz_id,
            session_id=session_id,
            pdf_name=pdf_name,
            active=True,
            current_index=0,
        )
    except Exception:
        logger.exception("failed_to_save_quiz session_id=%s quiz_id=%s", session_id, quiz_id)

    for index, question in enumerate(questions, start=1):
        question_id = question.get("id") or f"q{index}_{uuid4().hex[:8]}"
        question_item = {
            "id": question_id,
            "number": index,
            "question": question.get("question", ""),
            "choices": question.get("choices", []),
            "answer": str(question.get("answer", "")).strip(),
            "explanation": question.get("explanation", ""),
            "page": question.get("page"),
            "pdf_name": question.get("pdf_name"),
        }
        state.questions.append(question_item)

        try:
            save_quiz_question(
                quiz_id=quiz_id,
                session_id=session_id,
                question=question_item,
            )
        except Exception:
            logger.exception(
                "failed_to_save_quiz_question session_id=%s quiz_id=%s question_id=%s",
                session_id,
                quiz_id,
                question_id,
            )

    return state


def get_current_question(session_id: str) -> dict[str, Any] | None:
    state = get_quiz_state(session_id)

    if not state.active:
        return None

    if state.current_index >= len(state.questions):
        state.active = False
        try:
            update_quiz_progress(
                quiz_id=state.quiz_id,
                current_index=state.current_index,
                active=False,
            )
        except Exception:
            logger.exception("failed_to_finish_quiz session_id=%s", session_id)
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

    try:
        save_quiz_attempt(
            quiz_id=state.quiz_id,
            question_id=question["id"],
            session_id=session_id,
            user_answer=user_answer,
            correct_answer=question.get("answer", ""),
            is_correct=is_correct,
        )
    except Exception:
        logger.exception(
            "failed_to_save_quiz_attempt session_id=%s quiz_id=%s question_id=%s",
            session_id,
            state.quiz_id,
            question.get("id"),
        )

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

        try:
            save_wrong_answer(
                quiz_id=state.quiz_id,
                session_id=session_id,
                item=wrong_item,
            )
        except Exception:
            logger.exception(
                "failed_to_save_wrong_answer session_id=%s quiz_id=%s wrong_id=%s",
                session_id,
                state.quiz_id,
                wrong_item.get("id"),
            )

    state.current_index += 1

    finished = state.current_index >= len(state.questions)

    if finished:
        state.active = False
        next_question = None
    else:
        next_question = state.questions[state.current_index]

    try:
        update_quiz_progress(
            quiz_id=state.quiz_id,
            current_index=state.current_index,
            active=state.active,
        )
    except Exception:
        logger.exception(
            "failed_to_update_quiz_progress session_id=%s quiz_id=%s",
            session_id,
            state.quiz_id,
        )

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
    try:
        items = load_wrong_answers(session_id)
        if session_id in QUIZ_STATES:
            QUIZ_STATES[session_id].wrong_answers = items
        return items
    except Exception:
        logger.exception("failed_to_list_wrong_answers session_id=%s", session_id)
        return get_quiz_state(session_id).wrong_answers


def get_wrong_answer(session_id: str, wrong_id: str) -> dict[str, Any] | None:
    try:
        return load_wrong_answer(session_id, wrong_id)
    except Exception:
        logger.exception(
            "failed_to_get_wrong_answer session_id=%s wrong_id=%s",
            session_id,
            wrong_id,
        )

    for item in get_quiz_state(session_id).wrong_answers:
        if item.get("id") == wrong_id:
            return item

    return None


def reset_quiz(session_id: str) -> None:
    if session_id in QUIZ_STATES:
        state = QUIZ_STATES[session_id]
        try:
            update_quiz_progress(
                quiz_id=state.quiz_id,
                current_index=state.current_index,
                active=False,
            )
        except Exception:
            logger.exception("failed_to_deactivate_quiz session_id=%s", session_id)
        del QUIZ_STATES[session_id]
        return

    try:
        deactivate_active_quizzes(session_id)
    except Exception:
        logger.exception("failed_to_deactivate_active_quizzes session_id=%s", session_id)


def delete_quiz_history(session_id: str) -> None:
    if session_id in QUIZ_STATES:
        del QUIZ_STATES[session_id]

    try:
        delete_quiz_history_from_db(session_id)
    except Exception:
        logger.exception("failed_to_delete_quiz_history session_id=%s", session_id)
