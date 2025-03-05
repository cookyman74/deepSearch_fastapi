# app/controllers/feedback_controller.py

import os
import logging
import openai
from app.models.feedback_models import FeedbackRequest, FeedbackResponse
from app.services.feedback_service import generate_feedback

logger = logging.getLogger(__name__)


def handle_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    피드백 요청을 처리하는 컨트롤러 함수.

    사용자가 제시한 연구 주제에 대해 후속 질문(피드백)을 생성하기 위해
    feedback_service의 generate_feedback 함수를 호출합니다.

    Args:
        request (FeedbackRequest): 사용자가 보낸 피드백 요청 데이터

    Returns:
        FeedbackResponse: 생성된 후속 질문 목록을 담은 응답 객체
    """
    # 환경 변수에서 모델명 가져오기 (없으면 기본값 사용)
    model = os.getenv("FEEDBACK_MODEL", "gpt-4o-mini")

    logger.info(f"Processing feedback request for query: '{request.query}'")

    try:
        # OpenAI API 클라이언트로 openai 모듈을 사용합니다.
        client = openai
        response = generate_feedback(
            query=request.query,
            client=client,
            model=model,
            max_feedbacks=request.max_feedbacks
        )

        logger.info(f"Successfully generated {len(response.feedback_questions)} feedback questions")
        return response

    except Exception as e:
        logger.error(f"Error handling feedback request: {e}", exc_info=True)
        # 오류 발생 시 기본 응답 생성
        return FeedbackResponse(feedback_questions=[
            f"'{request.query}'에 대해 더 자세히 알려주실 수 있나요?"
        ])