from __future__ import annotations

import re

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ToolRetryMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from services.pdf_store import retrieve_pdf_chunks
from services.session_store import MODEL_NAME


def evidence_from_docs(docs: list) -> list[str]:
    pages: dict[int, str] = {}
    for doc in docs:
        page = doc.metadata.get("page")
        source = doc.metadata.get("source", "PDF")
        if isinstance(page, int):
            pages[page] = f"page {page} - {source}"
    return [pages[p] for p in sorted(pages)]


def extract_evidence_from_text(text: str, default_source: str = "PDF") -> list[str]:
    found: dict[int, str] = {}
    patterns = [r"page\s+(\d+)\s*-\s*([^,\n]+)", r"page\s+(\d+)"]

    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            page = int(match.group(1))
            source = match.group(2).strip() if len(match.groups()) >= 2 and match.group(2) else default_source
            found[page] = f"page {page} - {source.strip(' .)')}"

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


@tool
def pdf_retriever(session_id: str, query: str) -> str:
    """업로드된 PDF에서 질문과 관련 있는 내용을 검색한다. session_id와 query가 필요하다."""
    docs = retrieve_pdf_chunks(session_id, query, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)
    evidence = evidence_from_docs(docs)

    if not context:
        return "현재 세션에 업로드된 PDF가 없습니다."

    return (
        "[RAG 검색 결과]\n"
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
        "[PDF 요약 근거]\n"
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
        "[예상문제 생성 근거]\n"
        f"참고 페이지: {', '.join(evidence) if evidence else '없음'}\n\n"
        f"아래 PDF 근거를 바탕으로 예상문제 {count}개를 만드세요.\n"
        "반드시 문제 영역과 정답/해설 영역을 분리하세요.\n"
        "객관식/단답형/서술형을 섞고 같은 개념을 반복하지 마세요.\n\n"
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
        "[공부 계획 생성 근거]\n"
        f"참고 페이지: {', '.join(evidence) if evidence else '없음'}\n\n"
        f"아래 PDF 근거를 바탕으로 {days}일 공부 계획을 세우세요.\n"
        "각 날짜마다 목표, 공부할 내용, 확인 문제를 포함하세요.\n"
        "PDF 한 챕터 기준이면 너무 긴 계획을 만들지 마세요.\n\n"
        f"{context[:5000]}"
    )


TOOLS = [pdf_retriever, pdf_summary_tool, make_quiz_from_pdf, study_plan_tool]


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
