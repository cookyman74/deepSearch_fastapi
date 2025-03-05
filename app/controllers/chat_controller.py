# app/controllers/chat_controller.py

import os
import time
import logging
import openai
from app.models.chat_models import ChatRequest, ChatResponse, ResearchSettings
from app.services.research_service import deep_research
from app.services.reporting_service import write_final_report

logger = logging.getLogger(__name__)


def handle_chat(request: ChatRequest) -> ChatResponse:
    """
    챗봇(리서치) 요청을 처리하는 컨트롤러 함수.

    사용자의 초기 질문과 피드백 항목을 결합하여 최종 연구 보고서를 생성합니다.
    deep_research를 통해 연구를 수행하고, write_final_report를 통해 보고서를 작성합니다.

    Args:
        request (ChatRequest): 사용자가 보낸 챗봇 요청 데이터

    Returns:
        ChatResponse: 최종 연구 보고서, 사용된 검색 쿼리, 참조 URL, 실행 통계를 담은 응답 객체
    """
    start_time = time.time()

    # 환경 변수에서 모델명 가져오기 (없으면 기본값 사용)
    research_model = os.getenv("RESEARCH_MODEL", "gpt-4o")
    reporting_model = os.getenv("REPORTING_MODEL", "gpt-4o")

    # 설정 확인 (없으면 기본값 사용)
    if request.settings is None:
        request.settings = ResearchSettings()

    # 초기 질문과 피드백 항목 결합
    combined_query = f"초기 질문: {request.query}\n"
    if request.feedback_items:
        for item in request.feedback_items:
            if item.answer:
                combined_query += f"피드백 - 질문: {item.question}, 답변: {item.answer}\n"
            else:
                combined_query += f"피드백 - 질문: {item.question}\n"

    logger.info(
        f"Starting research for query: '{request.query}' with breadth={request.settings.breadth}, depth={request.settings.depth}")

    try:
        client = openai

        # 연구 수행 (동기식 인터페이스 사용)
        research_results = deep_research(
            query=combined_query,
            breadth=request.settings.breadth,
            depth=request.settings.depth,
            client=client,
            model=research_model,
            max_total_queries=os.getenv("MAX_QUERIES", 20)
        )

        logger.info(
            f"Research completed. Found {len(research_results.get('learnings', []))} learnings and {len(research_results.get('visited_urls', []))} references")

        # 최종 보고서 작성
        final_report = write_final_report(
            prompt=combined_query,
            learnings=research_results.get("learnings", []),
            visited_urls=research_results.get("visited_urls", []),
            search_queries=research_results.get("search_queries_used", []),
            execution_stats=research_results.get("execution_stats", {}),
            client=client,
            model=reporting_model
        )

        # 실행 시간 추가
        execution_stats = research_results.get("execution_stats", {})
        execution_stats["total_execution_time"] = time.time() - start_time

        logger.info(f"Final report generated. Length: {len(final_report)} characters")

        return ChatResponse(
            final_report=final_report,
            search_queries_used=research_results.get("search_queries_used", []),
            sources=research_results.get("visited_urls", []),
            execution_stats=execution_stats
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)

        # 오류 발생 시 기본 보고서 생성
        error_report = (
            f"# 연구 중 오류 발생\n\n"
            f"죄송합니다. 연구 과정에서 오류가 발생했습니다: {str(e)}\n\n"
            f"다시 시도하거나, 다른 연구 주제를 입력해 주세요."
        )

        return ChatResponse(
            final_report=error_report,
            search_queries_used=[combined_query],
            sources=[],
            execution_stats={"error": str(e), "total_execution_time": time.time() - start_time}
        )