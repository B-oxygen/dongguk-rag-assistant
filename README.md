# 누비AI RAG 실습 — Streamlit AI 비서

강의 슬라이드(`index.html`)에서 다룬 내용을 **실제로 동작하는 코드**로 구현한 실습 프로젝트입니다.
Google Gemini · LangChain · FAISS · Streamlit으로 문서 기반 질의응답(RAG) AI 비서를 만들고,
Google Cloud Run으로 배포합니다.

## 프로젝트 구조

| 파일 | 역할 | 대응 슬라이드 |
|---|---|---|
| `data/startup_faq.md` | AI 비서가 학습할 지식베이스(가상 스타트업 FAQ) | 07 RAG |
| `rag.py` | RAG 파이프라인: 로드 → 분할 → 임베딩 → FAISS → 검색 → 생성 | 06·08·09·10 |
| `app.py` | Streamlit 웹 UI(질문 입력 → 답변 + 출처) | 10·11 |
| `Dockerfile` | Cloud Run 배포용 컨테이너 | 12 |

## 1. 사전 준비

이미 `.venv`와 의존성이 설치되어 있습니다. 새 환경이라면:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. API 키 설정

Google AI Studio에서 무료로 Gemini API 키를 발급받습니다: <https://aistudio.google.com/apikey>

```bash
cp .env.example .env
# .env 파일을 열어 GOOGLE_API_KEY 값을 채웁니다.
```

## 3. 실행

```bash
.venv/bin/python -m streamlit run app.py
```

> ℹ️ 이 `.venv`는 다른 경로에서 만들어진 뒤 이동되어 `streamlit` 실행 스크립트의 경로가
> 깨져 있습니다. 그래서 `streamlit run` 대신 위처럼 **`python -m streamlit`**로 실행합니다.
> 완전히 고치려면 venv를 새로 만드세요:
> `python -m venv .venv && .venv/bin/pip install -r requirements.txt`

브라우저에서 `http://localhost:8501` 이 열립니다. 예시 질문:

- "Pro 요금제 가격은?"
- "환불 정책 알려줘"
- "고객센터 운영시간은?"

> 첫 실행 시 문서를 임베딩해 `faiss_index/` 폴더에 저장합니다(수 초 소요).
> 이후 실행은 저장된 인덱스를 재사용합니다. 문서를 수정했다면 사이드바의
> **"🔄 지식베이스 재색인"** 버튼을 누르세요.

## 4. 나만의 문서로 바꾸기

`data/` 폴더에 `.md` 파일을 추가하거나 교체한 뒤 재색인하면 됩니다.
(PDF를 쓰려면 `rag.py`의 로더를 `PyPDFLoader` / `unstructured`로 바꾸세요 — 관련 패키지는 이미 설치되어 있습니다.)

## 5. Vertex AI로 전환 (선택)

`langchain-google-genai` 4.x는 동일한 클래스로 Vertex AI 백엔드를 지원합니다.
`rag.py`의 모델 생성 부분을 아래처럼 바꾸면 됩니다(GCP 프로젝트 + 인증 필요):

```python
ChatGoogleGenerativeAI(model="gemini-2.5-flash", project="your-gcp-project", vertexai=True)
```

## 6. Cloud Run 배포 (슬라이드 12)

```bash
gcloud run deploy nubi-assistant \
  --source . \
  --region asia-northeast3 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_API_KEY=발급받은_키
```

`--source .` 옵션이 `Dockerfile`을 빌드해 Artifact Registry에 올리고 Cloud Run에 배포합니다.
배포가 끝나면 출력된 URL로 전 세계에서 접속할 수 있습니다.
