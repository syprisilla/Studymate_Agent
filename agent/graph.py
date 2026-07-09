from __future__ import annotations

from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.chains import run_study_rag
from agent.prompts import ROUTER_PROMPT
from agent.schemas import RouteDecision, ReviewFeedback
from agent.tools import (
    build_tool_agent,
    evidence_from_docs,
    extract_evidence_from_messages,
    extract_tool_names,
    last_ai_content,
)
from services.pdf_store import retrieve_pdf_chunks
from services.session_store import MODEL_NAME, OPENAI_READY, get_session


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    session_id: str
    route: str
    final_response: dict
    trace: list[str]
    used_tools: list[str]
    evidence: list[str]
    retry_count: int
    max_retries: int
    quality_next: str


def format_review_feedback(feedback: ReviewFeedback, evidence: list[str]) -> str:
    verdict = "정답에 가깝습니다 ✅" if feedback.is_correct else "다시 확인이 필요합니다 ❌"
    evidence_text = "\n".join(f"- {item}" for item in evidence) if evidence else "- PDF 검색 근거 없음"

    return (
        f"## 채점 결과: {verdict}\n\n"
        f"### 올바른 개념\n{feedback.correct_concept}\n\n"
        f"### 놓친 부분\n{feedback.mistake_reason}\n\n"
        f"### 다시 풀어볼 문제\n{feedback.retry_question}\n\n"
        f"### 참고한 PDF 위치\n{evidence_text}\n\n"
        f"{feedback.encouragement}"
    )


def keyword_route(message: str) -> str:
    text = message.lower()

    if any(
        word in text
        for word in [
            "더 자세히",
            "무슨 말",
            "쉽게",
            "다시 설명",
            "왜 그런",
            "이거 말고",
            "이외",
            "추가적으로",
            "추가로",
            "방금",
            "그 밖에",
        ]
    ):
        return "followup"
    if any(word in text for word in ["채점", "오답", "복습", "틀린", "맞아?", "맞나요"]):
        return "review"
    if any(word in text for word in ["요약", "정리"]):
        return "summary"
    if any(word in text for word in ["계획", "일정", "플랜"]):
        return "plan"
    if any(word in text for word in ["문제", "퀴즈", "예상"]):
        return "quiz"
    return "concept"


def router_node(state: State) -> dict:
    session = get_session(state["session_id"])
    message = state["messages"][-1].content
    trace = state.get("trace", [])

    if OPENAI_READY:
        try:
            llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0)
            output_parser = PydanticOutputParser(pydantic_object=RouteDecision)
            history_text = "\n".join(f"{m.type}: {m.content}" for m in state["messages"][-8:])

            prompt = PromptTemplate.from_template(
                ROUTER_PROMPT
                + "\n\n"
                + "반드시 아래 형식 지침을 따르는 JSON으로만 답해라.\n"
                + "{format_instructions}"
            )

            router_chain = prompt | llm | output_parser
            decision = router_chain.invoke(
                {
                    "history": history_text,
                    "format_instructions": output_parser.get_format_instructions(),
                }
            )

            route = decision.route
            reason = decision.reason
            trace = trace + ["router_node: PromptTemplate | LLM | PydanticOutputParser Chain 사용"]
        except Exception as exc:
            route = keyword_route(message)
            reason = f"LLM 라우팅 실패 → 키워드 fallback: {exc}"
    else:
        route = keyword_route(message)
        reason = "OPENAI_API_KEY 없음 → 키워드 fallback 라우팅"

    trace = trace + [f"router_node: route={route}", f"router_node: reason={reason}"]

    if route != "unknown" and not session.pdf_text:
        return {
            "route": "no_pdf",
            "trace": trace + ["router_node: PDF 없음 → no_pdf로 전환"],
            "final_response": {
                "type": "no_pdf",
                "summary": "현재 업로드된 PDF가 없습니다. 먼저 PDF를 업로드한 뒤 질문해주세요.",
            },
            "quality_next": "finish",
        }

    return {
        "route": route,
        "trace": trace,
        "quality_next": "finish",
    }


def route_selector(state: State) -> str:
    return state.get("route", "unknown")


def quality_selector(state: State) -> str:
    return state.get("quality_next", "finish")


def concept_node(state: State) -> dict:
    query = state["messages"][-1].content
    result = run_study_rag(state["session_id"], query, mode="concept")

    return {
        "final_response": {"type": "concept", "summary": result["answer"]},
        "trace": state.get("trace", []) + result.get("trace", []),
        "used_tools": ["study_rag_graph", "FAISS_retriever"],
        "evidence": result.get("evidence", []),
    }


def summary_node(state: State) -> dict:
    result = run_study_rag(
        state["session_id"],
        "전체 요약 핵심 개념 목차 중요한 내용",
        mode="summary",
    )

    return {
        "final_response": {"type": "summary", "summary": result["answer"]},
        "trace": state.get("trace", []) + result.get("trace", []),
        "used_tools": ["study_rag_graph", "FAISS_retriever"],
        "evidence": result.get("evidence", []),
    }


def run_tool_agent_node(state: State) -> dict:
    session = get_session(state["session_id"])
    route = state.get("route", "quiz")

    if not OPENAI_READY:
        return {
            "final_response": {
                "type": route,
                "summary": "OPENAI_API_KEY가 없어 Tool Agent를 실행할 수 없습니다.",
            },
            "trace": state.get("trace", []) + ["tool_agent_node: OPENAI_API_KEY 없음"],
            "used_tools": [],
            "evidence": [],
        }

    agent = build_tool_agent()
    context = SystemMessage(
        content=(
            f"현재 session_id: {session.session_id}\n"
            f"현재 PDF 이름: {session.pdf_name}\n"
            "Tool 호출 시 session_id 인자에는 반드시 위 session_id를 그대로 넣어라.\n"
            f"현재 route는 {route}이다.\n"
            "사용자 요청을 보고 필요한 Tool을 스스로 선택해 실행하라."
        )
    )

    try:
        result = agent.invoke({"messages": [context, state["messages"][-1]]})
        used_tools = extract_tool_names(result["messages"])
        answer = last_ai_content(result["messages"])
        evidence = extract_evidence_from_messages(
            result["messages"],
            default_source=session.pdf_name or "PDF",
        )
        trace = state.get("trace", []) + [
            "tool_agent_node: create_agent 실행",
            f"tool_agent_node: 사용 Tool={used_tools if used_tools else '감지 안 됨'}",
            f"tool_agent_node: 근거 페이지 {len(evidence)}개 추출",
        ]
    except Exception as exc:
        used_tools = []
        evidence = []
        answer = f"Tool Agent 실행 중 오류가 발생했습니다: {exc}"
        trace = state.get("trace", []) + [f"tool_agent_node: 실행 실패 → {exc}"]

    return {
        "final_response": {"type": route, "summary": answer},
        "trace": trace,
        "used_tools": used_tools,
        "evidence": evidence,
    }


def review_node(state: State) -> dict:
    user_answer = state["messages"][-1].content
    last_ai = next(
        (message for message in reversed(state["messages"][:-1]) if isinstance(message, AIMessage)),
        None,
    )
    prior_context = last_ai.content if last_ai else "(직전 AI 답변 없음)"

    docs = retrieve_pdf_chunks(state["session_id"], user_answer, k=5)
    context = "\n\n".join(doc.page_content for doc in docs)[:4500]
    evidence = evidence_from_docs(docs)

    if not OPENAI_READY:
        return {
            "final_response": {
                "type": "review",
                "summary": (
                    "OPENAI_API_KEY가 없어 자동 채점은 제한됩니다.\n\n"
                    f"직전 답변:\n{prior_context[:800]}\n\n"
                    f"PDF 근거:\n{context[:1000]}"
                ),
            },
            "trace": state.get("trace", []) + ["review_node: fallback 채점"],
            "used_tools": ["pdf_retriever"],
            "evidence": evidence,
        }

    llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0.2)
    output_parser = PydanticOutputParser(pydantic_object=ReviewFeedback)

    prompt = PromptTemplate.from_template(
        """
너는 PDF 기반 CS 학습 도우미다.
사용자의 답을 직전 문제/답변과 PDF 근거에 비추어 채점한다.
단정적으로 혼내지 말고, 올바른 개념과 다시 풀 문제를 제시한다.

직전 AI 답변 또는 문제:
{prior_context}

사용자 답변:
{user_answer}

PDF 근거:
{context}

반드시 아래 형식 지침을 따르는 JSON으로만 답해라.

{format_instructions}
"""
    )

    review_chain = prompt | llm | output_parser

    try:
        feedback = review_chain.invoke(
            {
                "prior_context": prior_context,
                "user_answer": user_answer,
                "context": context,
                "format_instructions": output_parser.get_format_instructions(),
            }
        )
        summary = format_review_feedback(feedback, evidence)
        trace = state.get("trace", []) + [
            "review_node: PromptTemplate | LLM | PydanticOutputParser Chain으로 채점"
        ]
    except Exception:
        summary = (
            "채점 중 오류가 발생해 PDF 근거를 우선 정리합니다.\n\n"
            f"직전 문제/답변:\n{prior_context[:1000]}\n\n"
            f"사용자 답변:\n{user_answer}\n\n"
            "### 참고한 PDF 위치\n"
            + ("\n".join(f"- {item}" for item in evidence) if evidence else "- PDF 검색 근거 없음")
        )
        trace = state.get("trace", []) + ["review_node: 채점 Chain 실패 → PDF 근거 fallback"]

    return {
        "final_response": {"type": "review", "summary": summary},
        "trace": trace,
        "used_tools": ["pdf_retriever"],
        "evidence": evidence,
    }


def followup_node(state: State) -> dict:
    last_ai = next(
        (message for message in reversed(state["messages"][:-1]) if isinstance(message, AIMessage)),
        None,
    )

    if last_ai is None:
        return {
            "final_response": {
                "type": "followup",
                "summary": "아직 이전 답변이 없습니다. PDF 내용에 대해 다시 질문해주세요.",
            },
            "trace": state.get("trace", []) + ["followup_node: 직전 AI 답변 없음"],
            "used_tools": [],
            "evidence": [],
        }

    user_request = state["messages"][-1].content

    if not OPENAI_READY:
        answer = f"직전 답변을 다시 정리하면 다음과 같습니다.\n\n{last_ai.content}"
        trace = state.get("trace", []) + ["followup_node: OPENAI_API_KEY 없음 → 직전 답변 재출력"]
    else:
        try:
            llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0)
            prompt = PromptTemplate.from_template(
                """
아래는 직전 AI 답변이다.
새로운 검색을 하지 말고, 직전 답변에 담긴 내용만 바탕으로 사용자의 후속 요청에 맞게 설명해라.

직전 답변:
{last_answer}

사용자 후속 요청:
{user_request}
"""
            )

            followup_chain = prompt | llm | StrOutputParser()
            answer = followup_chain.invoke(
                {
                    "last_answer": last_ai.content,
                    "user_request": user_request,
                }
            )
            trace = state.get("trace", []) + [
                "followup_node: PromptTemplate | LLM | StrOutputParser Chain으로 후속 설명"
            ]
        except Exception:
            answer = f"직전 답변을 다시 정리하면 다음과 같습니다.\n\n{last_ai.content}"
            trace = state.get("trace", []) + ["followup_node: 후속 설명 Chain 실패 → 직전 답변 fallback"]

    return {
        "final_response": {"type": "followup", "summary": answer},
        "trace": trace,
        "used_tools": ["memory"],
        "evidence": [],
    }


def response_looks_failed(summary: str) -> bool:
    return (
        not summary
        or "응답을 생성하지 못했습니다" in summary
        or "오류가 발생했습니다" in summary
        or "실행 중 오류" in summary
    )


def quality_check_node(state: State) -> dict:
    response = state.get("final_response") or {}
    summary = str(response.get("summary", "")).strip()
    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", 1))

    if response_looks_failed(summary) and retry_count < max_retries:
        return {
            "retry_count": retry_count + 1,
            "quality_next": "retry",
            "trace": state.get("trace", []) + [
                f"quality_check_node: 응답 품질 부족 → router로 재시도 {retry_count + 1}/{max_retries}"
            ],
        }

    return {
        "quality_next": "finish",
        "trace": state.get("trace", []) + ["quality_check_node: 응답 품질 확인 완료 → finalize_response"],
    }


def no_pdf_node(state: State) -> dict:
    return {"trace": state.get("trace", []) + ["no_pdf_node: PDF 업로드 안내"]}


def unknown_node(state: State) -> dict:
    return {
        "final_response": {
            "type": "unknown",
            "summary": (
                "저는 업로드된 PDF를 기반으로 자연어 요청을 분석해 "
                "개념 설명, 요약, 공부 계획, 예상문제, 답안 채점, 후속 설명을 수행하는 StudyMate입니다. "
                "PDF 내용과 관련된 질문을 해주세요."
            ),
        },
        "trace": state.get("trace", []) + ["unknown_node: 지원 범위 안내"],
        "used_tools": [],
        "evidence": [],
    }


def finalize_response_node(state: State) -> dict:
    response = state.get("final_response") or {
        "type": "error",
        "summary": "응답을 생성하지 못했습니다.",
    }

    return {"messages": [AIMessage(content=response["summary"])]}


def build_graph():
    builder = StateGraph(State)

    builder.add_node("router", router_node)
    builder.add_node("concept_node", concept_node)
    builder.add_node("summary_node", summary_node)
    builder.add_node("tool_agent_node", run_tool_agent_node)
    builder.add_node("review_node", review_node)
    builder.add_node("followup_node", followup_node)
    builder.add_node("quality_check_node", quality_check_node)
    builder.add_node("no_pdf_node", no_pdf_node)
    builder.add_node("unknown_node", unknown_node)
    builder.add_node("finalize_response", finalize_response_node)

    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_selector,
        {
            "concept": "concept_node",
            "summary": "summary_node",
            "plan": "tool_agent_node",
            "quiz": "tool_agent_node",
            "review": "review_node",
            "followup": "followup_node",
            "no_pdf": "no_pdf_node",
            "unknown": "unknown_node",
        },
    )

    for node_name in (
        "concept_node",
        "summary_node",
        "tool_agent_node",
        "review_node",
        "followup_node",
    ):
        builder.add_edge(node_name, "quality_check_node")

    builder.add_conditional_edges(
        "quality_check_node",
        quality_selector,
        {
            "retry": "router",
            "finish": "finalize_response",
        },
    )

    builder.add_edge("no_pdf_node", "finalize_response")
    builder.add_edge("unknown_node", "finalize_response")
    builder.add_edge("finalize_response", END)

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()