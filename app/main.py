import os

from fastapi import FastAPI
from dotenv import load_dotenv

from app.routes import feedback_routes, chat_routes
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

app = FastAPI(
    title=os.getenv("PROJECT_NAME", "my app"),
    description=os.getenv("PROJECT_DESCRIPTION", "my project"),
    version=os.getenv("PROJECT_VERSION", "0.0.1"),
)

# 피드백 관련 엔드포인트를 /feedback 경로로 등록.
app.include_router(feedback_routes.router, prefix="/feedback", tags=["Feedback"])
# 챗봇 관련 엔드포인트
app.include_router(chat_routes.router, prefix="/chat", tags=["Chat"])

app.add_middleware(
    CORSMiddleware, # type: ignore
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
def read_root():
    """
    서비스 상태를 체크
    :return: {}
    """
    return {"message": "FastAPI 챗봇 서비스가 정상 작동 중입니다."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True)