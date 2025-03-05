# app/models/chat_models.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ResearchSettings(BaseModel):
    breadth: int = Field(2, description="각 연구 단계에서 생성할 검색 쿼리의 수 (연구 범위)", ge=1, le=5)
    depth: int = Field(2, description="연구 진행 단계 수 (연구 깊이)", ge=1, le=3)


class FeedbackItem(BaseModel):
    question: str = Field(..., description="피드백 질문")
    answer: Optional[str] = Field(None, description="사용자의 답변")


class ChatRequest(BaseModel):
    """
    챗봇(리서치) 요청 모델

    사용자가 챗봇 서비스에 연구 주제와 피드백 답변, 연구 범위 및 깊이를 입력할 때 사용됩니다.
    """
    query: str = Field(..., description="사용자가 제시한 초기 연구 질문 또는 주제", min_length=3)
    feedback_items: List[FeedbackItem] = Field(
        default_factory=list,
        description="피드백 질문과 답변 항목 목록"
    )
    settings: Optional[ResearchSettings] = Field(
        default_factory=ResearchSettings,
        description="연구 설정"
    )

    class Config:
        schema_extra = {
            "example": {
                "query": "인공지능의 윤리적 영향",
                "feedback_items": [
                    {"question": "당신의 주제에 대한 최근 연구를 확인하셨나요?", "answer": "최근 논문을 참조했습니다"},
                    {"question": "관련 기사나 뉴스를 찾아보셨나요?", "answer": "관련 기사를 확인했습니다"}
                ],
                "settings": {
                    "breadth": 3,
                    "depth": 2
                }
            }
        }


class ChatResponse(BaseModel):
    """
    챗봇(리서치) 응답 모델

    딥 리서치 및 보고서 작성 결과로 생성된 최종 연구 보고서를 담습니다.
    """
    final_report: str = Field(..., description="생성된 최종 연구 보고서 (Markdown 형식)")
    search_queries_used: List[str] = Field(..., description="연구에 사용된 검색 쿼리 목록")
    sources: List[str] = Field(..., description="참조된 소스 URL 목록")
    execution_stats: Dict[str, Any] = Field(..., description="실행 통계 (토큰 수, 실행 시간 등)")

    class Config:
        schema_extra = {
            "example": {
                "final_report": "# 연구 보고서\n\n## 개요\n... (보고서 내용)",
                "search_queries_used": ["인공지능 윤리적 영향 최근 연구", "AI 윤리 가이드라인"],
                "sources": ["https://example.org/ai-ethics", "https://example.com/research-paper"],
                "execution_stats": {
                    "total_tokens": 8523,
                    "execution_time_seconds": 12.5,
                    "queries_executed": 6
                }
            }
        }