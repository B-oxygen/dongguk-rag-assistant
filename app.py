"""Streamlit AI 비서 웹앱 (강의 슬라이드 10~11).

실행: `streamlit run app.py`
"""

from __future__ import annotations

import os

# macOS FAISS/OpenMP 충돌(OMP: Error #15) 회피 — 다른 라이브러리 임포트 전에 설정.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import streamlit as st

from rag import AIAssistant, create_assistant

st.set_page_config(page_title="나의 AI 비서", page_icon="🤖", layout="centered")


@st.cache_resource(show_spinner="지식베이스를 색인하는 중입니다…")
def load_assistant() -> AIAssistant:
    """비서를 1회만 생성하고 캐시합니다(FAISS 인덱스 재사용)."""
    return create_assistant()


def has_api_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


st.title("🤖 나의 AI 비서")
st.caption("Vertex AI · LangChain · FAISS · Streamlit 기반 RAG 데모")

# ── 사이드바: 안내 및 인덱스 재생성 ──────────────────────────────────
with st.sidebar:
    st.subheader("ℹ️ 정보")
    st.write("'누비AI' 고객지원 문서를 학습한 RAG 비서입니다.")
    st.write(f"모델: `{os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}`")
    if st.button("🔄 지식베이스 재색인"):
        st.cache_resource.clear()
        st.rerun()
    st.divider()
    st.caption(
        "**예시 질문**\n\n"
        "- Pro 요금제 가격은?\n"
        "- 환불 정책 알려줘\n"
        "- 고객센터 운영시간은?\n"
        "- 무료 체험 기간은 며칠이야?"
    )

# ── API 키 확인 (없으면 설정 안내 후 중단) ──────────────────────────
if not has_api_key():
    st.error(
        "**GOOGLE_API_KEY가 설정되지 않았습니다.**\n\n"
        "1. https://aistudio.google.com/apikey 에서 API 키를 발급받으세요.\n"
        "2. 프로젝트 루트에 `.env` 파일을 만들고 아래 한 줄을 추가하세요:\n\n"
        "   ```\n   GOOGLE_API_KEY=발급받은_키\n   ```\n\n"
        "3. 앱을 다시 실행하세요."
    )
    st.stop()

# ── 슬라이드 11의 입력 → 버튼 → 답변 UI ─────────────────────────────
question = st.text_input("질문을 입력하세요:", placeholder="예) 무료 체험 기간은 며칠인가요?")

if st.button("답변 생성", type="primary") and question:
    try:
        assistant = load_assistant()
        with st.spinner("생각 중…"):
            result = assistant.ask(question)
    except Exception as exc:  # 인증/네트워크/모델 오류를 사용자에게 안내 (UI 경계)
        st.error("답변 생성 중 오류가 발생했습니다. API 키와 네트워크를 확인하세요.")
        st.exception(exc)
    else:
        st.markdown("### 💬 답변")
        st.write(result.text)
        with st.expander(f"📚 참고한 문서 {len(result.sources)}건 보기"):
            for i, doc in enumerate(result.sources, start=1):
                source = doc.metadata.get("source", "알 수 없음")
                st.markdown(f"**{i}. `{source}`**")
                st.caption(doc.page_content[:300] + "…")
