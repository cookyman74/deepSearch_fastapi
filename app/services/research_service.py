# app/services/research_service.py

import os
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional, Set
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field
from app.utils.utils import llm_call, JSON_llm, system_prompt
from firecrawl import FirecrawlApp  # Firecrawl API 클라이언트 (설치 및 환경변수 설정 필요)

logger = logging.getLogger(__name__)


# 결과 처리를 위한 Pydantic 모델 정의
class SerpResultResponse(BaseModel):
    learnings: List[str] = Field(..., description="검색 결과에서 추출한 주요 학습 내용")
    followUpQuestions: List[str] = Field(..., description="검색 결과를 바탕으로 생성한 후속 질문")


class ResearchStats(BaseModel):
    execution_time_seconds: float = Field(0.0, description="실행 시간(초)")
    queries_executed: int = Field(0, description="실행된 검색 쿼리 수")
    total_tokens_used: int = Field(0, description="사용된 총 토큰 수")


class ResearchResults(BaseModel):
    learnings: List[str] = Field([], description="수집된 모든 학습 내용")
    visited_urls: List[str] = Field([], description="방문한 모든 URL")
    search_queries_used: List[str] = Field([], description="사용된 모든 검색 쿼리")
    execution_stats: ResearchStats = Field(default_factory=ResearchStats, description="실행 통계")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def firecrawl_search(query: str, timeout: int = 15000, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Firecrawl 검색 API를 호출하여 결과를 반환하는 함수.

    Args:
        query (str): 검색할 쿼리
        timeout (int): 요청 제한 시간 (밀리초)
        limit (int): 검색 결과 최대 개수

    Returns:
        List[Dict[str, Any]]: 검색 결과 목록 (각 항목은 딕셔너리)
    """
    try:
        api_key = os.getenv("FIRECRAWL_API_KEY", "")
        if not api_key:
            logger.error("FIRECRAWL_API_KEY environment variable not set")
            return []

        app_instance = FirecrawlApp(api_key=api_key)
        response = app_instance.search(
            query=query,
            params={"timeout": timeout, "limit": limit, "scrapeOptions": {"formats": ["markdown"]}}
        )
        data = response.get("data", [])
        logger.info(f"Firecrawl search for '{query}' returned {len(data)} results")
        return data
    except Exception as e:
        logger.error(f"Firecrawl search error for query '{query}': {e}")
        return []


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=5)
)
def generate_serp_queries(
        query: str,
        client,
        model: str,
        num_queries: int = 3,
        learnings: Optional[List[str]] = None
) -> List[str]:
    """
    사용자의 쿼리와 이전 연구 결과를 바탕으로 SERP 검색 쿼리 목록을 생성합니다.

    Args:
        query (str): 초기 연구 주제
        client: OpenAI API 클라이언트
        model (str): 사용할 LLM 모델명
        num_queries (int): 생성할 검색 쿼리 수
        learnings (Optional[List[str]]): 이전 학습 내용 (옵션)

    Returns:
        List[str]: 생성된 검색 쿼리 목록
    """
    # 입력 유효성 검증 및 제한
    num_queries = min(max(1, num_queries), 5)  # 1-5 사이로 제한
    query = query[:500] if len(query) > 500 else query  # 쿼리 길이 제한

    # 이전 학습 내용이 있으면 최대 3개만 포함
    learnings_text = ""
    if learnings and len(learnings) > 0:
        selected_learnings = learnings[:3] if len(learnings) > 3 else learnings
        learnings_text = "\n이전 연구 결과:\n- " + "\n- ".join(selected_learnings)

    prompt = (
        f"다음 사용자 입력을 기반으로 주제 '{query}'에 대해 연구를 진행하기 위한 {num_queries}개의 검색 쿼리를 생성하세요. "
        "각 쿼리는 구체적이어야 하며, 검색 엔진에서 효과적으로 작동할 수 있도록 작성되어야 합니다."
        f"{learnings_text}"
    )

    try:
        logger.info(f"Generating search queries for '{query}'")
        response_text = llm_call(
            prompt=prompt,
            model=model,
            client=client,
            custom_system_prompt=system_prompt(),
            temperature=0.3
        )

        # 줄바꿈으로 구분된 쿼리 추출
        queries = [q.strip() for q in response_text.split("\n") if q.strip()]

        # 숫자나 불릿 포인트 등 제거
        import re
        cleaned_queries = []
        for q in queries:
            # 줄 시작 부분의 숫자, 마침표, 대시 등 제거
            cleaned = re.sub(r'^[\d\.\-\*]+\s*', '', q)
            if cleaned:
                cleaned_queries.append(cleaned)

        result = cleaned_queries[:num_queries]
        logger.info(f"Generated {len(result)} search queries")
        return result
    except Exception as e:
        logger.error(f"Error generating search queries: {e}")
        # 실패 시 기본 쿼리 생성
        default_queries = [f"{query} research", f"{query} analysis", f"{query} examples"]
        return default_queries[:num_queries]


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=5)
)
def process_serp_results(
        query: str,
        search_results: List[Dict[str, Any]],
        client,
        model: str,
        num_learnings: int = 5,
        num_follow_up: int = 3
) -> Dict[str, List[str]]:
    """
    검색 결과를 처리하여 주요 학습 내용과 후속 질문을 추출합니다.

    Args:
        query (str): 원래 검색 쿼리
        search_results (List[Dict[str, Any]]): Firecrawl API 검색 결과 목록
        client: OpenAI API 클라이언트
        model (str): 사용할 LLM 모델명
        num_learnings (int): 추출할 최대 학습 내용 수
        num_follow_up (int): 추출할 최대 후속 질문 수

    Returns:
        Dict[str, List[str]]: {"learnings": [...], "followUpQuestions": [...]}
    """
    if not search_results:
        logger.warning(f"No search results to process for query: {query}")
        return {"learnings": [], "followUpQuestions": []}

    # 각 검색 결과의 'markdown' 필드를 취합 (토큰 제한 고려)
    contents = []
    total_content_length = 0
    max_content_length = 10000  # 최대 컨텐츠 길이 제한

    for item in search_results:
        markdown = item.get("markdown", "").strip()
        if markdown:
            # 내용 길이 제한
            content_to_add = markdown[:2500]
            if total_content_length + len(content_to_add) <= max_content_length:
                contents.append(content_to_add)
                total_content_length += len(content_to_add)
            else:
                # 최대 길이 도달 시 중단
                break

    if not contents:
        logger.warning(f"No valid content found in search results for query: {query}")
        return {"learnings": [], "followUpQuestions": []}

    contents_str = "\n\n".join(contents)

    prompt = (
        f"다음은 '{query}'에 대한 검색 결과 일부 내용입니다:\n{contents_str}\n\n"
        f"이 내용을 바탕으로 다음 JSON 형식으로 응답하세요:\n"
        "{\n"
        f'  "learnings": [검색 결과에서 추출한 주요 학습 내용 {num_learnings}개],\n'
        f'  "followUpQuestions": [검색 결과를 바탕으로 생성한 후속 질문 {num_follow_up}개]\n'
        "}\n"
        "학습 내용은 명확하고 사실적이어야 하며, 후속 질문은 연구를 더 깊이 진행하는 데 도움이 되어야 합니다."
    )

    try:
        logger.info(f"Processing search results for query: '{query}'")
        response_obj = JSON_llm(
            user_prompt=prompt,
            schema=SerpResultResponse,
            client=client,
            custom_system_prompt=system_prompt(),
            model=model,
            temperature=0.3
        )

        if response_obj is None:
            logger.warning(f"Failed to process search results for query: {query}")
            return {"learnings": [], "followUpQuestions": []}

        # 결과 제한
        follow_ups = response_obj.followUpQuestions[:num_follow_up]
        learnings_limited = response_obj.learnings[:num_learnings]

        logger.info(f"Extracted {len(learnings_limited)} learnings and {len(follow_ups)} follow-up questions")
        return {"learnings": learnings_limited, "followUpQuestions": follow_ups}

    except Exception as e:
        logger.error(f"Error processing search results: {e}")
        return {"learnings": [], "followUpQuestions": []}


async def async_deep_research(
        query: str,
        breadth: int,
        depth: int,
        client,
        model: str,
        max_total_queries: int = 20,
        unique_learnings: Optional[Set[str]] = None,
        unique_urls: Optional[Set[str]] = None,
        unique_queries: Optional[Set[str]] = None,
        current_depth: int = 1,
        total_tokens: int = 0
) -> ResearchResults:
    """
    주제를 비동기적으로 탐색하여 SERP 쿼리를 생성하고, 검색 결과를 처리합니다.

    Args:
        query (str): 시작 연구 주제
        breadth (int): 각 단계에서 생성할 검색 쿼리 수
        depth (int): 최대 연구 진행 단계 수
        client: OpenAI API 클라이언트
        model (str): 사용할 LLM 모델명
        max_total_queries (int): 최대 총 쿼리 수 제한
        unique_learnings (Optional[Set[str]]): 수집된 학습 내용 (중복 방지용 Set)
        unique_urls (Optional[Set[str]]): 수집된 URL (중복 방지용 Set)
        unique_queries (Optional[Set[str]]): 사용된 쿼리 (중복 방지용 Set)
        current_depth (int): 현재 연구 깊이 (재귀 호출용)
        total_tokens (int): 누적 토큰 사용량

    Returns:
        ResearchResults: 연구 결과 객체
    """
    # 초기화
    start_time = time.time()

    if unique_learnings is None:
        unique_learnings = set()
    if unique_urls is None:
        unique_urls = set()
    if unique_queries is None:
        unique_queries = set()

    # 현재 쿼리 추가
    unique_queries.add(query)

    # 최대 쿼리 수 도달 체크
    if len(unique_queries) >= max_total_queries:
        logger.warning(f"Maximum total queries limit reached ({max_total_queries})")
        return ResearchResults(
            learnings=list(unique_learnings),
            visited_urls=list(unique_urls),
            search_queries_used=list(unique_queries),
            execution_stats=ResearchStats(
                execution_time_seconds=time.time() - start_time,
                queries_executed=len(unique_queries),
                total_tokens_used=total_tokens
            )
        )

    # 현재 단계의 SERP 쿼리 생성
    current_breadth = max(1, min(breadth, max_total_queries - len(unique_queries)))
    previous_learnings = list(unique_learnings)[:5]  # 최대 5개 이전 학습 사용

    queries = generate_serp_queries(
        query=query,
        client=client,
        model=model,
        num_queries=current_breadth,
        learnings=previous_learnings
    )

    # 새 쿼리만 추가 (중복 방지)
    new_queries = [q for q in queries if q not in unique_queries]
    for q in new_queries:
        unique_queries.add(q)

    logger.info(f"Depth {current_depth}/{depth}: Processing {len(new_queries)} unique queries")

    # 각 쿼리에 대한 태스크 생성 (비동기 처리)
    tasks = []
    for q in new_queries:
        # 검색 결과 가져오기
        tasks.append(asyncio.create_task(_process_query(
            query=q,
            client=client,
            model=model,
            unique_urls=unique_urls
        )))

    # 모든 태스크 완료 대기
    results = await asyncio.gather(*tasks)

    # 결과 처리
    all_follow_ups = []
    for result in results:
        # 학습 내용 추가
        for learning in result.get("learnings", []):
            unique_learnings.add(learning)

        # 후속 질문 수집
        all_follow_ups.extend(result.get("followUpQuestions", []))

    # 더 깊은 탐색이 필요하고 후속 질문이 있는 경우 재귀 호출
    if current_depth < depth and all_follow_ups:
        # 후속 질문에서 최대 2개 선택
        selected_follow_ups = all_follow_ups[:2]
        new_query = " ".join(selected_follow_ups)

        # 재귀적으로 더 깊은 탐색 실행
        await async_deep_research(
            query=new_query,
            breadth=max(1, breadth // 2),  # 깊이가 깊어질수록 breadth 감소
            depth=depth,
            client=client,
            model=model,
            max_total_queries=max_total_queries,
            unique_learnings=unique_learnings,
            unique_urls=unique_urls,
            unique_queries=unique_queries,
            current_depth=current_depth + 1,
            total_tokens=total_tokens
        )

    # 실행 시간 계산
    execution_time = time.time() - start_time

    # 결과 반환
    return ResearchResults(
        learnings=list(unique_learnings),
        visited_urls=list(unique_urls),
        search_queries_used=list(unique_queries),
        execution_stats=ResearchStats(
            execution_time_seconds=execution_time,
            queries_executed=len(unique_queries),
            total_tokens_used=total_tokens
        )
    )


async def _process_query(
        query: str,
        client,
        model: str,
        unique_urls: Set[str]
) -> Dict[str, Any]:
    """
    단일 쿼리를 처리하는 비동기 헬퍼 함수
    """
    # Firecrawl API를 호출하여 검색 결과 가져오기
    search_results = firecrawl_search(query)

    # URL 추출 및 저장
    for item in search_results:
        url = item.get("url")
        if url:
            unique_urls.add(url)

    # 검색 결과 처리
    return process_serp_results(query, search_results, client, model)


def deep_research(
        query: str,
        breadth: int,
        depth: int,
        client,
        model: str,
        max_total_queries: int = 20
) -> Dict[str, Any]:
    """
    동기식 인터페이스로 비동기 연구 함수를 실행합니다.

    Args:
        query (str): 시작 연구 주제
        breadth (int): 각 단계에서 생성할 검색 쿼리 수
        depth (int): 연구 진행 단계 수
        client: OpenAI API 클라이언트
        model (str): 사용할 LLM 모델명
        max_total_queries (int): 최대 총 쿼리 수 제한

    Returns:
        Dict[str, Any]: 연구 결과
    """
    # 입력 검증
    breadth = min(max(1, breadth), 5)  # 1-5 사이로 제한
    depth = min(max(1, depth), 3)  # 1-3 사이로 제한

    logger.info(f"Starting deep research for query: '{query}' with breadth={breadth}, depth={depth}")

    try:
        # 비동기 함수 실행을 위한 이벤트 루프 설정
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 비동기 함수 실행
        result = loop.run_until_complete(
            async_deep_research(
                query=query,
                breadth=breadth,
                depth=depth,
                client=client,
                model=model,
                max_total_queries=max_total_queries
            )
        )

        # 이벤트 루프 종료
        loop.close()

        # 결과 반환
        return {
            "learnings": result.learnings,
            "visited_urls": result.visited_urls,
            "search_queries_used": result.search_queries_used,
            "execution_stats": result.execution_stats.dict()
        }

    except Exception as e:
        logger.error(f"Error in deep research: {e}", exc_info=True)
        # 오류 시 빈 결과 반환
        return {
            "learnings": [],
            "visited_urls": [],
            "search_queries_used": [query],
            "execution_stats": {
                "execution_time_seconds": 0,
                "queries_executed": 0,
                "total_tokens_used": 0,
                "error": str(e)
            }
        }
