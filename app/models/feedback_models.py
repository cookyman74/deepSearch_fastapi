from typing import List
from pydantic import BaseModel, Field

class FeedbackRequest(BaseModel):
    """
    피드백 요청 모델

    사용자가 연구 주제나 질문을 입력할 때, 후속 질문(피드백)을 요청하기 위한 데이터 구조입니다.
    """
    query: str = Field(..., description="사용자가 제시한 연구 주제 또는 질문", min_length=3)
    max_feedbacks: int = Field(3, description="생성할 최대 후속 질문 개수 (기본값: 3)", ge=1, le=5)

    class Config:
        schema_extra = {
            "example": {
                "query": "인공지능의 윤리적 영향에 대해 알고 싶습니다",
                "max_feedbacks": 3
            }
        }


class FeedbackQuestion(BaseModel):
    question: str = Field(..., description="피드백 질문")
    type: str = Field(..., description="질문 유형 (예: factual, conceptual, opinion)")
    rationale: str = Field(..., description="이 질문을 하는 이유")


class FeedbackResponse(BaseModel):
    """
    피드백 응답 모델

    LLM을 통해 생성된 후속 질문들을 담는 응답 데이터 구조입니다.
    """
    feedback_questions: List[FeedbackQuestion] = Field(..., description="생성된 후속 질문 목록")