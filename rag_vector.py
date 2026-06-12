import chromadb
import os
import json
import uuid
import requests
from dotenv import load_dotenv

try:
    from google import genai as google_genai
except ImportError:
    google_genai = None
try:
    import openai
except Exception:
    openai = None

# ── RAG / ChromaDB 연동 모듈 ────────────────────────────────────────────
#
#  외부에서 사용하는 함수:
#   • embed_and_store(session_id, stock_name, created_at, news_list)
#       → 분석 완료 후 orchestrator/app.py 에서 호출
#   • query_similar_news(query_text, n_results=3, exclude_session_id=None)
#       → app.py 뉴스 목록 하단 "유사 뉴스 Top 3" 표시에 활용
#
#  ChromaDB 최초 실행 전 설치:
#   pip install chromadb
#  Ollama 임베딩 사용 시 로컬 Ollama 11434 포트가 켜져 있어야 합니다.

load_dotenv()

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "stock_news"
GEMINI_EMBED_MODEL = "gemini-embedding-001"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_EMBED_API_URL = os.environ.get("OLLAMA_EMBED_API_URL", f"{OLLAMA_BASE_URL}/api/embed")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"))
print(f"[RAG] Ollama embed endpoint={OLLAMA_EMBED_API_URL}, model={OLLAMA_EMBED_MODEL}")

# ── 클라이언트 초기화 ─────────────────────────────────────────
def _get_gemini_client():
    if google_genai is None:
        raise RuntimeError("google-genai 패키지가 설치되어 있지 않습니다. pip install google-genai 를 실행하세요.")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(".env 파일에 GEMINI_API_KEY 가 설정되어 있지 않습니다.")
    return google_genai.Client(api_key=api_key)


def _get_collection():
    client_chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return client_chroma.get_or_create_collection(name=COLLECTION_NAME)


# ── 임베딩 함수 ───────────────────────────────────────────────
def _get_ollama_embedding(text: str) -> list:
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "input": text,
    }
    try:
        response = requests.post(OLLAMA_EMBED_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings") or data.get("data")
        if isinstance(embeddings, list) and embeddings:
            first = embeddings[0]
            if isinstance(first, dict):
                return first.get("embedding") or first.get("vector")
            return first
        raise RuntimeError(f"Ollama 임베딩 응답 형식 오류: {data}")
    except Exception as e:
        raise RuntimeError(f"Ollama 임베딩 호출 실패: {e}")


def _get_gemini_embedding(text: str) -> list:
    if google_genai is None:
        raise RuntimeError("google-genai 패키지가 설치되어 있지 않습니다. pip install google-genai 를 실행하세요.")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(".env 파일에 GEMINI_API_KEY 가 설정되어 있지 않습니다.")
    gemini = _get_gemini_client()
    result = gemini.models.embed_content(
        model=GEMINI_EMBED_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def _get_openai_embedding(text: str) -> list:
    if openai is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. pip install openai 를 실행하세요.")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(".env 파일에 OPENAI_API_KEY 가 설정되어 있지 않습니다.")
    # Support new openai-python (1.0+) interface and fall back if older
    try:
        # New client interface
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=api_key) if hasattr(openai.OpenAI, '__call__') else openai.OpenAI()
            resp = client.embeddings.create(model="text-embedding-3-small", input=text)
            return resp.data[0].embedding
        # Fallback to older interface
        if hasattr(openai, "Embedding"):
            openai.api_key = api_key
            resp = openai.Embedding.create(model="text-embedding-3-small", input=text)
            return resp["data"][0]["embedding"]
        raise RuntimeError("지원되지 않는 openai 라이브러리 인터페이스입니다.")
    except Exception as e:
        raise RuntimeError(f"OpenAI 임베딩 호출 실패: {e}")


def get_embedding(text: str) -> list:
    """
    텍스트를 벡터로 변환합니다.
    1) OPENAI_API_KEY가 설정되어 있으면 OpenAI 임베딩을 우선 사용합니다.
    2) OpenAI가 없거나 실패하면 Ollama를 시도합니다.
    3) 그 다음 Gemini로 폴백합니다.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_api_key:
        try:
            return _get_openai_embedding(text)
        except Exception as e_openai:
            print(f"[RAG] OpenAI 임베딩 실패: {e_openai}")
            print("[RAG] Ollama 임베딩 시도")
            try:
                return _get_ollama_embedding(text)
            except Exception as e_ollama:
                print(f"[RAG] Ollama 임베딩 실패: {e_ollama}")
                print(f"[RAG] Gemini로 폴백 시도")
                return _get_gemini_embedding(text)
    else:
        try:
            return _get_ollama_embedding(text)
        except Exception as e_ollama:
            print(f"[RAG] Ollama 임베딩 실패: {e_ollama}")
            print(f"[RAG] OpenAI API 키가 없어서 Gemini로 폴백 시도")
            return _get_gemini_embedding(text)


# ── ChromaDB 저장 ─────────────────────────────────────────────
def embed_and_store(session_id: str, stock_name: str, created_at: str, news_list: list) -> int:
    """
    뉴스 분석 결과(10개)를 임베딩하여 ChromaDB에 저장합니다.
    이미 저장된 session_id는 건너뜁니다.

    반환: 새로 저장된 뉴스 개수
    """
    if not news_list:
        return 0

    collection = _get_collection()

    # 이미 저장된 session_id 확인
    existing = collection.get(include=["metadatas"])
    existing_sessions = {
        meta.get("session_id")
        for meta in existing["metadatas"]
        if meta.get("session_id")
    }
    if session_id in existing_sessions:
        return 0  # 이미 저장됨 → 건너뜀

    saved = 0
    for news in news_list:
        document = f"""제목: {news.get('title', '')}
요약: {news.get('description', '')}
분석근거: {news.get('reason', '')}"""
        try:
            embedding = get_embedding(document)
            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[document],
                embeddings=[embedding],
                metadatas=[{
                    "session_id": session_id,
                    "stock_name": stock_name,
                    "created_at": created_at,
                    "title": news.get("title", ""),
                    "sentiment": news.get("sentiment", "중립"),
                    "reason": news.get("reason", ""),
                    "link": news.get("link", ""),
                }],
            )
            saved += 1
        except Exception as e:
            print(f"[RAG] 임베딩 저장 실패: {e}")

    return saved


# ── ChromaDB 유사 뉴스 검색 ───────────────────────────────────
def query_similar_news(query_text: str, n_results: int = 3, exclude_session_id: str = None) -> list:
    """
    쿼리 텍스트와 가장 유사한 뉴스를 ChromaDB에서 검색합니다.

    반환: [{"title", "sentiment", "reason", "stock_name", "created_at", "link", "distance"}, ...]
    """
    collection = _get_collection()

    # 저장된 문서가 없으면 빈 리스트 반환
    total = collection.count()
    if total == 0:
        return []

    # 현재 세션 뉴스는 제외 (자기 자신 제외)
    where_filter = None
    if exclude_session_id:
        where_filter = {"session_id": {"$ne": exclude_session_id}}

    try:
        embedding = get_embedding(query_text)
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": min(n_results, total),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = collection.query(**kwargs)
    except Exception as e:
        print(f"[RAG] 유사 뉴스 검색 실패: {e}")
        return []

    output = []
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for meta, dist in zip(metadatas, distances):
        output.append({
            "title": meta.get("title", ""),
            "sentiment": meta.get("sentiment", "중립"),
            "reason": meta.get("reason", ""),
            "stock_name": meta.get("stock_name", ""),
            "created_at": meta.get("created_at", ""),
            "link": meta.get("link", ""),
            "distance": round(float(dist), 4),
        })

    return output


# ── 독립 실행 (전체 chat_history.json 일괄 임베딩) ───────────
if __name__ == "__main__":
    HISTORY_FILE = "chat_history.json"
    if not os.path.exists(HISTORY_FILE):
        print(f"{HISTORY_FILE} 파일이 없습니다.")
        exit()

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_saved = 0
    for session_id, info in data.items():
        news_list = info.get("news_context_raw", [])
        if not news_list:
            continue
        saved = embed_and_store(
            session_id=session_id,
            stock_name=info.get("stock_name", ""),
            created_at=info.get("created_at", ""),
            news_list=news_list,
        )
        if saved > 0:
            print(f"[{info['stock_name']}] {saved}건 저장 완료")
        else:
            print(f"[{info['stock_name']}] 이미 저장됨 → 건너뜀")
        total_saved += saved

    print(f"\nChromaDB 저장 완료 — 총 {total_saved}건 신규 저장")
