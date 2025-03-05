이 프로젝트는 사용자 질문에 대해 심층적인 리서치를 수행하고 종합적인 보고서를 생성하는 FastAPI 기반 챗봇 서비스입니다. 피드백 수집, 다단계 검색 및 분석, 최종 보고서 생성 기능을 제공합니다.

## 주요 기능
- **피드백 질문 생성**: 사용자의 초기 질문에 대해 추가 맥락을 얻기 위한 후속 질문 생성
- **딥 리서치**: 여러 단계(깊이와 넓이)로 검색을 수행하여 포괄적인 정보 수집
- **자동 보고서 생성**: 수집된 정보를 바탕으로 구조화된 연구 보고서 작성

## 프로젝트 구조
```bash
fastapi_chatbot/
├── .env                      # API 키 등 환경변수 파일
├── requirements.txt          # 프로젝트 의존성 목록
├── README.md                 # 프로젝트 소개 및 실행 방법
└── app/
    ├── main.py               # FastAPI 애플리케이션 엔트리 포인트
    ├── controllers/          # 컨트롤러 계층: 요청 처리 및 응답 조립
    │   ├── feedback_controller.py   # 피드백 관련 요청 처리
    │   └── chat_controller.py       # 챗봇(리서치) 관련 요청 처리
    ├── models/               # Pydantic 모델 및 스키마 정의
    │   ├── feedback_models.py       # 피드백 요청/응답 모델
    │   ├── chat_models.py           # 챗봇(리서치) 요청/응답 모델
    │   └── research_models.py       # 연구 결과 관련 모델
    ├── routes/               # 라우터: 엔드포인트 정의
    │   ├── feedback_routes.py       # 피드백 엔드포인트
    │   └── chat_routes.py           # 챗봇 엔드포인트
    ├── services/             # 서비스 계층: 비즈니스 로직 구현
    │   ├── feedback_service.py      # 피드백 질문 생성 로직
    │   ├── research_service.py      # 딥 리서치 및 검색 로직
    │   └── reporting_service.py     # 최종 보고서 생성 로직
    └── utils/                # 유틸리티 함수 및 공통 모듈
        └── utils.py          # LLM 호출, 시스템 프롬프트 등
```


## 주요 구성 요소
### 모델 (Models)
- **FeedbackRequest/Response**: 피드백 질문 생성 요청 및 응답 모델
- **ChatRequest/Response**: 챗봇(리서치) 요청 및 응답 모델
- **ResearchSettings**: 연구 깊이와 넓이 설정 모델
- **FeedbackItem**: 피드백 질문과 답변을 포함하는 모델

### 서비스 (Services)
- **feedback_service**: 사용자 질문에 대한 후속 질문 생성
- **research_service**: 다단계 검색을 통한 정보 수집 및 분석
    - FireCrawl API를 활용한 웹 검색
    - 검색 결과에서 주요 학습 내용 추출
    - 재귀적 검색을 통한 심층 연구
- **reporting_service**: 최종 연구 보고서 생성

### 컨트롤러 (Controllers)
- **feedback_controller**: 피드백 요청 처리 및 서비스 호출
- **chat_controller**: 챗봇(리서치) 요청 처리 및 서비스 호출

### 유틸리티 (Utilities)
- OpenAI API 호출 래퍼 함수
- JSON 응답 처리 함수
- 시스템 프롬프트 생성 함수

### 환경변수
```env
OPENAI_API_KEY=your_openai_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
FEEDBACK_MODEL=gpt-4o-mini
RESEARCH_MODEL=gpt-4o
REPORTING_MODEL=gpt-4o
MAX_QUERIES=20
```

### 서버 실행
```bash
uvicorn app.main:app --reload
```


## 사용방법
### 1. 피드백 생성 (프롬프트 추가 개선)
사용자의 초기 질문에 대한 후속 질문(피드백)을 생성합니다.
```bash
POST /feedback
```

- 사용자 요청
```json
{
  "query": "인공지능의 윤리적 영향에 대해 알고 싶습니다",
  "max_feedbacks": 3
}
```

- 서버 답변 
```json
{
  "feedback_questions": [
    {"question": "특정 산업 분야에서의 AI 윤리 문제에 관심이 있으신가요?", "answer": null},
    {"question": "AI의 윤리적 영향 중 특별히 우려하는 측면이 있으신가요?", "answer": null},
    {"question": "현재 AI 윤리 규제에 대한 배경 지식이 있으신가요?", "answer": null}
  ]
}
```


### 2. 챗봇(리서치 & 결과보고)
사용자의 질문 및 피드백 답변을 바탕으로 심층 연구를 수행하고 최종 보고서를 생성합니다.
```bash
POST /chat
```

- 사용자 요청
```json
{
  "query": "인공지능의 윤리적 영향",
  "feedback_items": [
    {"question": "특정 산업 분야에서의 AI 윤리 문제에 관심이 있으신가요?", "answer": "의료 분야에 관심이 있습니다"},
    {"question": "AI의 윤리적 영향 중 특별히 우려하는 측면이 있으신가요?", "answer": "편향성과 결정 투명성에 관심이 있습니다"}
  ],
  "settings": {
    "breadth": 3,
    "depth": 2
  }
}
```

- 서버 응답 
```json
{
  "final_report": "# 인공지능의 윤리적 영향 연구 보고서\n\n## 개요\n...(보고서 내용)...주절주절 답변 내용들...",
  "search_queries_used": ["의료 AI 윤리적 영향", "인공지능 결정 투명성", "..."],
  "sources": ["https://example.org/ai-ethics", "...", "검색된 링크 자료들.."],
  "execution_stats": {
    "execution_time_seconds": 45.2,
    "queries_executed": 8,
    "total_tokens_used": 12500
  }
}
```