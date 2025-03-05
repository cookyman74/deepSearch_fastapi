from fastapi import APIRouter
from app.models.chat_models import ChatRequest, ChatResponse
from app.controllers.chat_controller import handle_chat

router = APIRouter()

@router.post("/", response_model=ChatResponse, tags=["Chat"])
def chat_route(request: ChatRequest):
    """
    챗봇(리서치) 요청 엔드포인트

    사용자의 초기 질문과 피드백 항목을 기반으로 연구를 수행하고,
    최종 연구 보고서를 생성하여 반환합니다.
    """
    return handle_chat(request)
