import os
import json
import re
import html
from datetime import datetime

#  기본적인 공통 보조 기능과 로컬 JSON 파일 제어를 담당

#   •  load_all_history() :  chat_history.json  파일로부터 이전 기록을 안전하게 읽어오는 함수
#   •  save_current_history() : 현재 대화 및 분석 상태를 누적 백업하는 함수
#   •  clean_html() : 뉴스 제목/요약문 안의 HTML 태그 정제 함수
#   •  get_jaccard_similarity() : 문자열 간 유사도 파악 함수

# 이전 대화 및 분석 결과를 저장할 로컬 JSON 파일 경로
HISTORY_FILE = "chat_history.json"

def load_all_history() -> dict:
    """
    로컬 JSON 파일(chat_history.json)로부터 기존의 모든 대화/분석 히스토리 데이터를 불러옵니다.
    파일이 없거나 읽기 에러 발생 시 빈 딕셔너리를 반환합니다.
    """
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_current_history(session_id: str, stock_name: str, news_context: str, chat_history: list, news_context_raw: list = None, risks: list = None):
    """
    현재 진행 중인 주식 분석 세션의 데이터를 로컬 JSON 파일에 누적하여 저장합니다.
    """
    if not session_id:
        return
    all_hist = load_all_history()
    # 해당 세션 ID의 키 아래에 데이터 저장
    all_hist[session_id] = {
        "stock_name": stock_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "news_context": news_context,
        "news_context_raw": news_context_raw if news_context_raw is not None else [],
        "risks": risks if risks is not None else [],
        "chat_history": chat_history
    }
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(all_hist, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print("히스토리 저장 오류:", e)

def clean_html(text: str) -> str:
    """
    네이버 뉴스 API 결과(제목, 요약문)에 섞여 있는 HTML 태그(예: <b>, &quot; 등)를 제거하고 디코딩합니다.
    """
    clean_re = re.compile('<.*?>')
    cleaned_text = re.sub(clean_re, '', text)
    return html.unescape(cleaned_text)


