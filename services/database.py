from __future__ import annotations

import sqlite3
from pathlib import Path

from services.session_store import DATA_DIR


DB_PATH = DATA_DIR / "studymate.db"


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS quizzes (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                pdf_name TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                current_index INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS quiz_questions (
                id TEXT PRIMARY KEY,
                quiz_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                number INTEGER NOT NULL,
                question TEXT NOT NULL,
                choices_json TEXT,
                correct_answer TEXT NOT NULL,
                explanation TEXT,
                page INTEGER,
                pdf_name TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id TEXT PRIMARY KEY,
                quiz_id TEXT,
                question_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                user_answer TEXT,
                correct_answer TEXT,
                is_correct INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS wrong_answers (
                id TEXT PRIMARY KEY,
                quiz_id TEXT,
                question_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                number INTEGER,
                question TEXT NOT NULL,
                choices_json TEXT,
                user_answer TEXT,
                correct_answer TEXT,
                explanation TEXT,
                page INTEGER,
                pdf_name TEXT,
                resolved INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_quizzes_session_id
            ON quizzes(session_id);

            CREATE INDEX IF NOT EXISTS idx_quiz_questions_session_id
            ON quiz_questions(session_id);

            CREATE INDEX IF NOT EXISTS idx_quiz_attempts_session_id
            ON quiz_attempts(session_id);

            CREATE INDEX IF NOT EXISTS idx_wrong_answers_session_id
            ON wrong_answers(session_id);
            """
        )
