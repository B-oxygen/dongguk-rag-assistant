"""RAG 파이프라인 — 문서 로드 → 분할 → 임베딩 → FAISS 저장 → 검색 → LLM 생성.

강의 슬라이드 06(RAG 개념), 08~09(LangChain 워크플로우), 10(FAISS 구조)을
실제 코드로 구현한 모듈입니다. 각 단계 옆에 대응하는 슬라이드 번호를 주석으로 표시했습니다.
"""

from __future__ import annotations

import os

# macOS에서 faiss-cpu와 다른 라이브러리가 각각 OpenMP(libomp)를 로드하면
# "OMP: Error #15"로 크래시가 납니다. faiss 임포트 전에 중복 로드를 허용합니다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # .env 파일의 GOOGLE_API_KEY 등을 환경변수로 로드

# ── 기본 설정 (환경변수로 오버라이드 가능) ─────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "faiss_index"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")

# RAG 답변 프롬프트 — 슬라이드 05(환각 방지)의 원칙을 그대로 적용했습니다.
PROMPT = ChatPromptTemplate.from_template(
    """당신은 '누비AI'의 고객지원 AI 비서입니다.
아래 <문서> 내용에만 근거하여 한국어로 친절하고 정확하게 답변하세요.
문서에 없는 내용은 추측하지 말고 "제공된 문서에서 관련 내용을 찾을 수 없습니다."라고 답하세요.

<문서>
{context}
</문서>

질문: {question}
답변:"""
)


@dataclass(frozen=True)
class Answer:
    """LLM의 최종 답변과, 그 근거가 된 원본 문서 조각."""

    text: str
    sources: list[Document]


def get_embeddings() -> Embeddings:
    """텍스트를 벡터로 변환하는 임베딩 모델 (슬라이드 09-③ Embedding)."""
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_llm() -> BaseChatModel:
    """답변을 생성하는 Gemini 모델 (슬라이드 07 RAG의 LLM)."""
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.2)


def load_and_split(data_dir: Path) -> list[Document]:
    """① Document Loader → ② Text Splitter (슬라이드 09-①,②).

    data/ 폴더의 모든 .md 문서를 읽어 검색에 적합한 크기의 청크로 나눕니다.
    """
    loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_documents(documents)


def _format_docs(docs: list[Document]) -> str:
    """검색된 문서 조각들을 하나의 컨텍스트 문자열로 결합."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class AIAssistant:
    """RAG 기반 질의응답 비서. Retriever와 LLM 체인을 감쌉니다."""

    def __init__(self, vectorstore: FAISS, llm: BaseChatModel, *, k: int = 4) -> None:
        # ⑤ Retriever — 질문과 유사도가 높은 상위 k개 조각을 검색 (슬라이드 09-⑤)
        self._retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        # LCEL 체인: 프롬프트 → LLM → 문자열 파싱
        self._chain = PROMPT | llm | StrOutputParser()

    def ask(self, question: str) -> Answer:
        """질문을 받아 검색 → 생성 순으로 답변을 만듭니다."""
        docs = self._retriever.invoke(question)  # ⑤ 검색
        context = _format_docs(docs)  # 컨텍스트 구성
        text = self._chain.invoke(  # ⑥ LLM 생성 (슬라이드 09-⑥)
            {"context": context, "question": question}
        )
        return Answer(text=text, sources=docs)


def create_assistant(
    *,
    embeddings: Embeddings | None = None,
    llm: BaseChatModel | None = None,
    data_dir: Path = DATA_DIR,
    index_dir: Path = INDEX_DIR,
    force_rebuild: bool = False,
) -> AIAssistant:
    """비서 인스턴스를 생성합니다.

    모델을 주입하지 않으면 기본 Gemini 모델을 사용합니다(테스트 시 가짜 모델 주입 가능).
    디스크에 저장된 FAISS 인덱스가 있으면 재사용하고, 없으면 새로 만들어 저장합니다.
    """
    embeddings = embeddings or get_embeddings()
    llm = llm or get_llm()

    if index_dir.exists() and not force_rebuild:
        # ④ 저장된 Vector Store 재사용 (슬라이드 10 — 재색인 비용 절약)
        vectorstore = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )
    else:
        # ③ 임베딩 → ④ FAISS Vector Store 신규 생성 후 저장 (슬라이드 09-③,④ / 10)
        chunks = load_and_split(data_dir)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(index_dir))

    return AIAssistant(vectorstore, llm)
