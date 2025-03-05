# app/utils/utils.py

from datetime import datetime
from typing import Optional, Any, Dict, List, Union
from pydantic import BaseModel
import openai
import json
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def system_prompt() -> str:
    """시스템 프롬프트 생성"""
    now = datetime.now().isoformat()
    return (
        f"당신은 전문 연구원입니다. 오늘 날짜는 {now}입니다. "
        "정확하고 체계적인 답변을 제공하세요."
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def llm_call(
        prompt: str,
        model: str = "gpt-4o-mini",
        client=None,
        custom_system_prompt: Optional[str] = None,
        temperature: float = 0.7
) -> str:
    """
    LLM API 호출 함수 (재시도 로직 포함)

    Args:
        prompt: 사용자 프롬프트
        model: 사용할 모델
        client: OpenAI 클라이언트 (None이면 기본 openai 모듈 사용)
        custom_system_prompt: 커스텀 시스템 프롬프트 (None이면 기본값 사용)
        temperature: 응답의 창의성 조절 (0~1)

    Returns:
        LLM 응답 텍스트
    """
    if client is None:
        client = openai

    # 시스템 프롬프트 설정
    sys_prompt = custom_system_prompt if custom_system_prompt else system_prompt()

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        logger.info(f"Calling LLM with model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def JSON_llm(
        user_prompt: str,
        schema: BaseModel,
        client=None,
        custom_system_prompt: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2
) -> Optional[BaseModel]:
    """
    JSON 구조화 응답을 반환하는 LLM API 호출

    Args:
        user_prompt: 사용자 프롬프트
        schema: Pydantic 모델 (응답 구조 정의)
        client: OpenAI 클라이언트
        custom_system_prompt: 커스텀 시스템 프롬프트
        model: LLM 모델명
        temperature: 응답의 창의성 조절 (0~1)

    Returns:
        schema에 맞는 구조화된 객체 또는 None
    """
    if client is None:
        client = openai

    # 시스템 프롬프트 설정 (JSON 응답 요청 추가)
    sys_prompt = custom_system_prompt if custom_system_prompt else system_prompt()
    sys_prompt += "\n다음 형식에 맞춰 JSON 형태로만 응답하세요. 추가 설명은 하지 마세요."

    # JSON 스키마 정보 추가
    schema_json = schema.schema_json()
    user_prompt_with_schema = f"{user_prompt}\n\n응답 JSON 스키마: {schema_json}"

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt_with_schema}
    ]

    try:
        logger.info(f"Calling LLM for JSON response with model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}  # JSON 응답 형식 지정
        )

        content = response.choices[0].message.content

        try:
            # JSON 파싱 및 스키마 검증
            data = json.loads(content)

            # Pydantic 버전에 따라 다른 메서드 사용
            try:
                # Pydantic v2
                return schema.model_validate(data)
            except AttributeError:
                return schema.model_validate(data)

        except Exception as e:
            logger.error(f"JSON parsing/validation error: {e}")
            logger.debug(f"Raw content: {content}")
            return None

    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise


# DeepSearch 관련 유틸리티 함수

def generate_search_queries(topic: str, client=None, model: str = "gpt-4o") -> List[str]:
    """
    주요 주제에서 검색 쿼리 목록을 생성합니다.

    Args:
        topic: 검색할 주제
        client: OpenAI 클라이언트
        model: 사용할 모델

    Returns:
        생성된 검색 쿼리 목록
    """
    prompt = f"""
    다음 주제에 대한 포괄적인 연구를 위해 5개의 구체적인 검색 쿼리를 생성하세요:
    '{topic}'

    다양한 측면을 다루는 쿼리를 만들되, 검색 엔진에서 효과적으로 작동할 수 있도록 
    구체적이고 명확하게 작성하세요. 쿼리만 줄바꿈으로 구분하여 반환하세요.
    """

    response = llm_call(
        prompt=prompt,
        model=model,
        client=client,
        temperature=0.7  # 다양한 쿼리를 위해 약간 높은 temperature
    )

    # 응답에서 쿼리 추출 (줄바꿈으로 구분된 목록)
    queries = [q.strip() for q in response.split('\n') if q.strip()]
    return queries


def summarize_search_results(query: str, results: List[Dict[str, Any]], client=None, model: str = "gpt-4o") -> str:
    """
    검색 결과를 요약합니다.

    Args:
        query: 원래 검색 쿼리
        results: 검색 결과 목록 (각 항목은 title, snippet, url 등을 포함하는 딕셔너리)
        client: OpenAI 클라이언트
        model: 사용할 모델

    Returns:
        검색 결과 요약
    """
    # 검색 결과 텍스트 구성
    results_text = ""
    for i, result in enumerate(results, 1):
        results_text += f"[{i}] {result.get('title', 'No Title')}\n"
        results_text += f"URL: {result.get('url', 'No URL')}\n"
        results_text += f"Snippet: {result.get('snippet', 'No Snippet')}\n\n"

    prompt = f"""
    다음은 '{query}'에 대한 검색 결과입니다:

    {results_text}

    이 결과를 바탕으로 다음을 수행하세요:
    1. 주요 정보 요약 (가장 관련성 높은 정보 중심)
    2. 검색 결과에서 발견된 주요 관점이나 의견
    3. 추가 조사가 필요한 부분

    한 페이지 분량으로 요약하세요.
    """

    return llm_call(
        prompt=prompt,
        model=model,
        client=client,
        temperature=0.3  # 정확한 요약을 위해 낮은 temperature
    )


def combine_research_findings(
        topic: str,
        summaries: List[str],
        client=None,
        model: str = "gpt-4o"
) -> str:
    """
    여러 검색 결과 요약을 종합하여 최종 연구 보고서를 생성합니다.

    Args:
        topic: 원래 연구 주제
        summaries: 각 검색 쿼리에 대한 결과 요약 목록
        client: OpenAI 클라이언트
        model: 사용할 모델

    Returns:
        최종 연구 보고서
    """
    combined_summaries = "\n\n".join([f"요약 {i + 1}:\n{summary}" for i, summary in enumerate(summaries)])

    prompt = f"""
    다음은 '{topic}'에 대한 여러 검색 결과 요약입니다:

    {combined_summaries}

    이 정보를 바탕으로 종합적인 연구 보고서를 작성하세요. 보고서는 다음 구조를 따라야 합니다:

    1. 개요: 주제 소개 및 주요 발견사항
    2. 주요 측면: 주제의 중요한 측면들을 분석 (하위 섹션으로 구분)
    3. 다양한 관점: 주제에 대한 서로 다른 시각이나 의견
    4. 결론: 주요 발견사항 요약 및 종합적 평가
    5. 추가 연구 방향: 더 탐구할 가치가 있는 영역

    객관적이고 균형 잡힌 시각으로 작성하되, 특히 신뢰할 수 있는 정보와 출처를 강조하세요.
    """

    return llm_call(
        prompt=prompt,
        model=model,
        client=client,
        temperature=0.4  # 창의적이면서도 일관된 보고서를 위한 중간 temperature
    )