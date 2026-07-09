from __future__ import annotations

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph

from agent.schemas import RelevanceCheck, StudyAnswer, StudyRAGState
from agent.tools import evidence_from_docs
from services.pdf_store import retrieve_pdf_chunks
from services.session_store import MAX_RAG_RETRIES, MODEL_NAME, OPENAI_READY


def format_study_answer(answer: StudyAnswer, evidence: list[str]) -> str:
    details = "\n".join(f"- {item}" for item in answer.details)
    exam_points = "\n".join(f"- {item}" for item in answer.exam_points)
    examples = "\n".join(f"- {item}" for item in answer.examples)
    questions = "\n".join(f"- {item}" for item in answer.check_questions)
    evidence_text = "\n".join(f"- {item}" for item in evidence) if evidence else "- PDF 검색 근거 없음"

    return (
        f"## {answer.title}\n\n"
        f"### 핵심\n{answer.core}\n\n"
        f"### 설명\n{details}\n\n"
        f"### 시험 포인트\n{exam_points}\n\n"
        f"### 예시\n{examples}\n\n"
        f"### 확인 질문\n{questions}\n\n"
        f"### 참고한 PDF 위치\n{evidence_text}"
    )


def collect_pdf_chunks(state: StudyRAGState) -> dict:
    docs = retrieve_pdf_chunks(state["session_id"], state["query"], k=6)
    return {
        "documents": docs,
        "trace": state.get("trace", []) + [f"collect_pdf_chunks: PDF에서 후보 청크 {len(docs)}개 검색"],
    }


def grade_pdf_relevance(state: StudyRAGState) -> dict:
    if not OPENAI_READY:
        return {
            "relevant_documents": state.get("documents", []),
            "trace": state.get("trace", []) + ["grade_pdf_relevance: OPENAI_API_KEY 없음 → 관련성 평가 생략"],
        }

    llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0)
    output_parser = PydanticOutputParser(pydantic_object=RelevanceCheck)

    prompt = PromptTemplate.from_template(
        """
너는 PDF 검색 결과의 관련성을 평가하는 도우미다.

질문:
{query}

PDF 문단:
{context}

이 문단이 질문에 답하는 데 실제로 도움이 되는지 판단해라.
반드시 아래 형식 지침을 따르는 JSON으로만 답해라.

{format_instructions}
"""
    )

    relevance_chain = prompt | llm | output_parser
    relevant: list = []

    for doc in state.get("documents", []):
        try:
            result = relevance_chain.invoke(
                {
                    "query": state["query"],
                    "context": doc.page_content[:1000],
                    "format_instructions": output_parser.get_format_instructions(),
                }
            )
            if result.is_relevant:
                relevant.append(doc)
        except Exception:
            relevant.append(doc)

    return {
        "relevant_documents": relevant,
        "trace": state.get("trace", []) + [
            f"grade_pdf_relevance: PromptTemplate | LLM | PydanticOutputParser Chain으로 관련 청크 {len(relevant)}개 선별"
        ],
    }


def should_retry_pdf_search(state: StudyRAGState) -> str:
    if not OPENAI_READY:
        return "synthesize"
    if len(state.get("relevant_documents", [])) >= 2:
        return "synthesize"
    if state.get("retry_count", 0) >= MAX_RAG_RETRIES:
        return "synthesize"
    return "rewrite_query"


def rewrite_pdf_query(state: StudyRAGState) -> dict:
    if not OPENAI_READY:
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "trace": state.get("trace", []) + ["rewrite_pdf_query: OPENAI_API_KEY 없음 → 재작성 생략"],
        }

    llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0.3)

    prompt = PromptTemplate.from_template(
        """
다음 검색어로 PDF에서 충분한 관련 내용을 찾지 못했다.

기존 검색어:
{query}

같은 의도를 유지하면서 PDF 검색에 더 잘 맞는 구체적인 검색어 1개만 출력해라.
설명은 쓰지 말고 검색어만 출력해라.
"""
    )

    rewrite_chain = prompt | llm | StrOutputParser()

    try:
        new_query = rewrite_chain.invoke({"query": state["query"]}).strip()
    except Exception:
        new_query = state["query"]

    return {
        "query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "trace": state.get("trace", []) + [
            f"rewrite_pdf_query: PromptTemplate | LLM | StrOutputParser Chain으로 검색어 재작성 → {new_query}"
        ],
    }


def synthesize_study_answer(state: StudyRAGState) -> dict:
    docs = state.get("relevant_documents") or state.get("documents") or []
    if not docs:
        return {
            "answer": "PDF에서 관련 내용을 찾지 못했습니다. 질문을 조금 더 구체적으로 해주세요.",
            "evidence": [],
            "trace": state.get("trace", []) + ["synthesize_study_answer: 관련 문서 없음"],
        }

    context = "\n\n".join(doc.page_content for doc in docs)[:5500]
    evidence = evidence_from_docs(docs)

    if not OPENAI_READY:
        return {
            "answer": f"PDF에서 관련 내용을 찾았습니다.\n\n{context[:1400]}",
            "evidence": evidence,
            "trace": state.get("trace", []) + ["synthesize_study_answer: OPENAI_API_KEY 없음 → 원문 기반 응답"],
        }

    llm = init_chat_model(f"openai:{MODEL_NAME}", temperature=0.2)
    output_parser = PydanticOutputParser(pydantic_object=StudyAnswer)

    mode_instruction = (
        "PDF 전체를 요약하세요. 핵심 개념, 주요 내용, 시험 포인트, 확인 질문을 한 번만 정리하세요."
        if state["mode"] == "summary"
        else (
            "질문에 해당하는 PDF 개념을 설명하세요. 반드시 1. 한 줄 정의, 2. 왜 필요한지, "
            "3. 구조/동작 방식, 4. PDF 기반 예시, 5. 비슷한 개념 비교, "
            "6. 시험 포인트, 7. 확인 질문 순서의 내용을 포함하세요. "
            "비슷한 개념이 있으면 비교해서 설명하세요."
        )
    )

    prompt = PromptTemplate.from_template(
        """
너는 PDF 기반 CS 학습 도우미다.
반드시 주어진 PDF 근거 안에서만 답한다.
원문을 그대로 길게 복사하지 말고 시험 공부에 바로 쓸 수 있게 재구성한다.
제목에는 StudyAnswer, JSON, Pydantic, schema 같은 내부 구현 용어를 절대 쓰지 않는다.

지시:
{mode_instruction}

질문:
{query}

PDF 근거:
{context}

반드시 아래 형식 지침을 따르는 JSON으로만 답해라.

{format_instructions}
"""
    )

    answer_chain = prompt | llm | output_parser

    try:
        structured = answer_chain.invoke(
            {
                "mode_instruction": mode_instruction,
                "query": state["query"],
                "context": context,
                "format_instructions": output_parser.get_format_instructions(),
            }
        )
        answer = format_study_answer(structured, evidence)
    except Exception:
        answer = (
            "PDF 근거를 바탕으로 정리하면 다음과 같습니다.\n\n"
            f"{context[:1800]}\n\n"
            "### 참고한 PDF 위치\n"
            + ("\n".join(f"- {item}" for item in evidence) if evidence else "- PDF 검색 근거 없음")
        )

    return {
        "answer": answer,
        "evidence": evidence,
        "trace": state.get("trace", []) + [
            f"synthesize_study_answer: PromptTemplate | LLM | PydanticOutputParser Chain으로 구조화 출력 생성, 근거 페이지 {len(evidence)}개"
        ],
    }


def build_study_rag_graph():
    builder = StateGraph(StudyRAGState)
    builder.add_node("collect_pdf_chunks", collect_pdf_chunks)
    builder.add_node("grade_pdf_relevance", grade_pdf_relevance)
    builder.add_node("rewrite_pdf_query", rewrite_pdf_query)
    builder.add_node("synthesize_study_answer", synthesize_study_answer)

    builder.add_edge(START, "collect_pdf_chunks")
    builder.add_edge("collect_pdf_chunks", "grade_pdf_relevance")
    builder.add_conditional_edges(
        "grade_pdf_relevance",
        should_retry_pdf_search,
        {
            "synthesize": "synthesize_study_answer",
            "rewrite_query": "rewrite_pdf_query",
        },
    )
    builder.add_edge("rewrite_pdf_query", "collect_pdf_chunks")
    builder.add_edge("synthesize_study_answer", END)

    return builder.compile()


study_rag_graph = build_study_rag_graph()


def run_study_rag(session_id: str, query: str, mode: Literal["concept", "summary"]) -> dict:
    return study_rag_graph.invoke(
        {
            "session_id": session_id,
            "query": query,
            "mode": mode,
            "retry_count": 0,
            "documents": [],
            "relevant_documents": [],
            "trace": [f"StudyRAG 시작: mode={mode}, query={query}"],
        }
    )