# app/services/feedback_service.py

import logging
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.models.feedback_models import FeedbackResponse
from app.models.chat_models import FeedbackItem  # FeedbackQuestion 대신 FeedbackItem 사용
from app.utils.utils import JSON_llm, system_prompt

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def generate_feedback(
        query: str,
        client,
        model: str = "gpt-4o-mini",
        max_feedbacks: int = 3
) -> FeedbackResponse:
    """
    사용자의 연구 주제(또는 질문)를 바탕으로 후속 질문(피드백)을 생성하는 서비스 함수입니다.

    Args:
        query (str): 사용자가 제시한 연구 주제 또는 질문.
        client: OpenAI API 클라이언트.
        model (str): 사용할 LLM 모델명. 기본값은 "gpt-4o-mini".
        max_feedbacks (int): 생성할 최대 후속 질문 수.

    Returns:
        FeedbackResponse: 생성된 후속 질문 목록을 포함하는 응답 객체.
    """
    # 매개변수 유효성 검증: 1~5 사이로 제한
    max_feedbacks = min(max(1, max_feedbacks), 5)

    # 토큰 제한을 고려하여 매우 긴 쿼리 잘라내기
    if len(query) > 1000:
        logger.warning(f"Query too long ({len(query)} chars). Truncating to 1000 chars.")
        query = query[:1000] + "..."

    # LLM에게 전달할 프롬프트 구성
    prompt = (
        f"사용자가 제시한 연구 주제: '{query}'에 대해 최대 {max_feedbacks}개의 후속 질문을 생성하세요. "
        "각 질문은 주제를 더 깊이 이해하거나 연구 방향을 명확히 하는 데 도움이 되어야 합니다. "
        "예를 들어, 주제의 특정 측면에 대한 자세한 정보, 사용자의 관점이나 경험, 특정 예시 등을 요청할 수 있습니다.\n\n"
        "응답은 다음 JSON 형식으로 작성하세요:\n"
        "{\n"
        '  "feedback_questions": [\n'
        "    {\n"
        '      "question": "질문 내용",\n'
        '      "answer": null\n'
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "질문은 명확하고 구체적이어야 하며, 직접적인 yes/no 대신 더 상세한 응답을 유도해야 합니다."
    )

    # 시스템 프롬프트에 전문성 강조 추가
    sys_prompt = system_prompt() + " 사용자의 연구를 돕기 위한 전문적인 후속 질문을 생성해야 합니다."

    try:
        logger.info(f"Generating feedback questions for query: '{query}'")

        # JSON_llm 함수를 이용하여 LLM API 호출
        response = JSON_llm(
            user_prompt=prompt,
            schema=FeedbackResponse,
            client=client,
            custom_system_prompt=sys_prompt,
            model=model,
            temperature=0.2  # 안정적인 JSON 응답을 위해 낮은 temperature 사용
        )

        # 응답 유효성 검증
        if response is None or not response.feedback_questions:
            logger.warning("Invalid or empty response from LLM. Using fallback.")
            return _create_fallback_response(query, max_feedbacks)

        # 최대 피드백 수로 제한
        if len(response.feedback_questions) > max_feedbacks:
            logger.info(f"Limiting feedback questions from {len(response.feedback_questions)} to {max_feedbacks}")
            response.feedback_questions = response.feedback_questions[:max_feedbacks]

        logger.info(f"Successfully generated {len(response.feedback_questions)} feedback questions")
        return response

    except Exception as e:
        logger.error(f"Error generating feedback: {str(e)}", exc_info=True)
        return _create_fallback_response(query, max_feedbacks)


def _create_fallback_response(query: str, max_feedbacks: int) -> FeedbackResponse:
    """
    LLM 호출 실패 시 사용할 기본 더미 피드백 질문을 생성합니다.
    """
    dummy_questions = []
    default_questions = [
        f"'{query}'에 대해 더 자세히 알려주실 수 있나요?",
        f"이 주제에 관련된 구체적인 예시나 사례가 있나요?",
        f"이 주제를 연구하게 된 특별한 이유나 배경이 있나요?",
        f"이 주제에서 특히 집중하고 싶은 특정 측면이 있나요?",
        f"이 주제와 관련하여 이미 알고 있는 정보나 자료가 있나요?"
    ]

    for i in range(min(max_feedbacks, len(default_questions))):
        dummy_questions.append(FeedbackItem(
            question=default_questions[i],
            answer=None
        ))

    return FeedbackResponse(feedback_questions=dummy_questions)
