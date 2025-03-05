# app/services/reporting_service.py

import logging
import tiktoken  # OpenAI의 토큰 계산 라이브러리
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.utils.utils import llm_call, system_prompt

logger = logging.getLogger(__name__)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """텍스트의 토큰 수를 계산합니다."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # 대략적인 토큰 수 추정: 단어 수 * 1.3
        return int(len(text.split()) * 1.3)


def categorize_learnings(
        learnings: List[str],
        client,
        model: str
) -> Dict[str, List[str]]:
    """
    학습 내용을 주제별로 분류합니다.

    Args:
        learnings: 수집된 학습 내용 목록
        client: OpenAI API 클라이언트
        model: 사용할 LLM 모델명

    Returns:
        주제별로 분류된 학습 내용 딕셔너리
    """
    if not learnings:
        return {}

    # 학습 내용이 너무 많을 경우 일부만 선택
    max_learnings = 50
    selected_learnings = learnings[:max_learnings] if len(learnings) > max_learnings else learnings

    learnings_str = "\n".join([f"- {l}" for l in selected_learnings])

    prompt = (
        "다음은 연구 과정에서 수집된 학습 내용입니다. 이 내용을 3-5개의 주요 주제로 분류하세요:\n\n"
        f"{learnings_str}\n\n"
        "각 주제별로 관련 학습 내용을 그룹화하고, 주제명과 해당 내용 번호를 JSON 형식으로 반환하세요. 예시:\n"
        "{\n"
        '  "주제 1": [0, 2, 5, ...],\n'
        '  "주제 2": [1, 3, 4, ...],\n'
        '  ...\n'
        "}"
    )

    try:
        response = llm_call(
            prompt=prompt,
            model=model,
            client=client,
            temperature=0.2
        )

        import json
        categories = json.loads(response)

        result = {}
        for category, indices in categories.items():
            # 인덱스가 유효한지 확인
            valid_indices = [i for i in indices if 0 <= i < len(selected_learnings)]
            if valid_indices:
                result[category] = [selected_learnings[i] for i in valid_indices]

        return result

    except Exception as e:
        logger.error(f"Error categorizing learnings: {e}")
        # 분류 실패 시 하나의 일반 카테고리로 반환
        return {"일반 정보": selected_learnings}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)
def generate_report_section(
        section_name: str,
        section_content: str,
        prompt: str,
        client,
        model: str
) -> str:
    """
    보고서의 한 섹션을 생성합니다.

    Args:
        section_name: 섹션 이름
        section_content: 섹션 관련 내용
        prompt: 원래 사용자 프롬프트
        client: OpenAI API 클라이언트
        model: 사용할 LLM 모델명

    Returns:
        생성된 섹션 텍스트
    """
    user_prompt = (
        f"사용자가 제시한 프롬프트: '{prompt}'\n\n"
        f"다음은 '{section_name}' 섹션과 관련된 학습 내용입니다:\n\n{section_content}\n\n"
        f"위 내용을 바탕으로 '{section_name}' 섹션을 작성하세요. "
        "논리적이고 체계적으로 정보를 구성하며, 가능한 한 사실에 근거해야 합니다. "
        "마크다운 형식으로 제목, 소제목, 목록 등을 적절히 사용하세요."
    )

    return llm_call(
        prompt=user_prompt,
        model=model,
        client=client,
        temperature=0.4
    )


def write_final_report(
        prompt: str,
        learnings: List[str],
        visited_urls: List[str],
        search_queries: List[str],
        execution_stats: Dict[str, Any],
        client,
        model: str = "gpt-4o"
) -> str:
    """
    모든 연구 결과를 바탕으로 최종 보고서를 생성합니다.

    Args:
        prompt (str): 사용자가 제시한 최종 프롬프트(연구 주제 및 초기 질문)
        learnings (List[str]): 연구 과정에서 수집된 학습 내용 목록
        visited_urls (List[str]): 연구 과정에서 참조된 URL 목록
        search_queries (List[str]): 사용된 검색 쿼리 목록
        execution_stats (Dict[str, Any]): 실행 통계 정보
        client: OpenAI API 클라이언트 (예: openai 모듈)
        model (str): 사용할 LLM 모델명 (기본값: "gpt-4o")

    Returns:
        str: 마크다운 형식의 최종 연구 보고서
    """
    logger.info("Starting final report generation...")

    # 학습 내용이 없으면 기본 메시지 반환
    if not learnings:
        return (
            "# 연구 보고서\n\n"
            "## 개요\n\n"
            "연구 과정에서 충분한 정보를 수집하지 못했습니다. "
            "다른 검색어나 더 구체적인 연구 주제를 시도해보세요."
        )

    # 1. 학습 내용 분류
    categorized_learnings = categorize_learnings(learnings, client, model)

    # 2. 각 섹션 생성 및 결합
    final_report = []

    # 2.1 제목 및 개요
    overview_content = (
        f"연구 주제: {prompt}\n"
        f"총 {len(learnings)} 개의 학습 내용과 {len(visited_urls)} 개의 참조 자료를 수집했습니다.\n"
        f"사용된 검색 쿼리: {', '.join(search_queries[:5])}..."
    )

    overview = generate_report_section(
        section_name="개요",
        section_content=overview_content,
        prompt=prompt,
        client=client,
        model=model
    )

    final_report.append(f"# {prompt} 연구 보고서\n\n## 개요\n\n{overview}")

    # 2.2 각 카테고리별 섹션 생성
    for category, items in categorized_learnings.items():
        category_content = "\n".join([f"- {item}" for item in items])

        section = generate_report_section(
            section_name=category,
            section_content=category_content,
            prompt=prompt,
            client=client,
            model=model
        )

        final_report.append(f"## {category}\n\n{section}")

    # 2.3 결론 및 추가 연구 방향
    conclusion_prompt = (
        f"연구 주제 '{prompt}'에 대한 결론과 추가 연구 방향을 제시하세요. "
        "지금까지 수집한 정보의 한계점과 후속 연구에서 다룰 수 있는 질문들을 포함하세요."
    )

    conclusion = llm_call(
        prompt=conclusion_prompt,
        model=model,
        client=client,
        temperature=0.5
    )

    final_report.append(f"## 결론 및 추가 연구 방향\n\n{conclusion}")

    # 2.4 연구 방법론 및 통계
    methodology = (
        "## 연구 방법론\n\n"
        "이 보고서는 자동화된 딥 리서치 시스템을 통해 작성되었습니다. "
        f"총 {len(search_queries)} 개의 검색 쿼리를 이용하여 {len(visited_urls)} 개의 자료를 분석했으며, "
        f"{len(learnings)} 개의 관련 정보를 추출했습니다.\n\n"
        "### 실행 통계\n\n"
        f"- 실행 시간: {execution_stats.get('execution_time_seconds', 0):.1f} 초\n"
        f"- 실행된 쿼리 수: {execution_stats.get('queries_executed', 0)}\n"
        f"- 사용된 토큰 수: {execution_stats.get('total_tokens_used', 0)}\n"
    )

    final_report.append(methodology)

    # 2.5 출처 목록
    references = (
            "## 출처\n\n" +
            "\n".join([f"- {url}" for url in visited_urls[:30]])  # 최대 30개 URL 표시
    )

    if len(visited_urls) > 30:
        references += f"\n\n*및 {len(visited_urls) - 30}개 추가 출처*"

    final_report.append(references)

    # 최종 보고서 결합
    full_report = "\n\n".join(final_report)
    logger.info("Final report generation completed.")

    return full_report