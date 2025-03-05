from fastapi import APIRouter
from app.models.feedback_models import FeedbackRequest, FeedbackResponse
from app.controllers.feedback_controller import handle_feedback

router = APIRouter()

@router.post("/", response_model=FeedbackResponse, tags=["Feedback"])
def feedback_route(request: FeedbackRequest):
    """
    피드백 요청 엔드포인트

    사용자가 연구 주제나 질문을 입력하면 후속 질문(피드백)을 생성하여 반환합니다.
    """
    return handle_feedback(request)
