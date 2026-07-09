from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal[
        "concept",
        "summary",
        "plan",
        "quiz",
        "review",
        "followup",
        "web_search",
        "hybrid",
        "unknown",
    ] = Field(
        description=(
            "concept=PDF 내용/CS 개념 설명 질문, "
            "summary=PDF 전체 요약/정리 요청, "
            "plan=공부 계획/일정 요청, "
            "quiz=예상문제/퀴즈 요청, "
            "review=사용자 답안 채점/오답 복습 요청, "
            "followup=직전 답변을 이어서 더 묻는 요청, "
            "web_search=PDF 밖의 최신 정보/외부 사례/웹 검색 요청, "
            "hybrid=PDF 근거와 외부 웹 검색이 모두 필요한 요청, "
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


class StudyRAGState(TypedDict, total=False):
    session_id: str
    query: str
    mode: Literal["concept", "summary"]
    retry_count: int
    documents: list
    relevant_documents: list
    answer: str
    evidence: list[str]
    trace: list[str]


class QuizQuestion(BaseModel):
    id: str | None = None
    question: str
    choices: list[str] = Field(default_factory=list)
    answer: str
    explanation: str
    page: int | None = None
    pdf_name: str | None = None


class QuizSet(BaseModel):
    questions: list[QuizQuestion]


class QuizStartRequest(BaseModel):
    session_id: str = "default"
    topic: str = "전체"
    count: int = 5


class QuizAnswerRequest(BaseModel):
    session_id: str = "default"
    answer: str


class QuizWrongDetailRequest(BaseModel):
    session_id: str = "default"
    wrong_id: str