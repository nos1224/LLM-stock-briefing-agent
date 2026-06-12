import os
import re
import requests
from datetime import datetime
from email.utils import parsedate_to_datetime
from difflib import SequenceMatcher
from urllib.parse import urlparse
from utils import clean_html

# 네이버 검색 API 통신 및 데이터 필터링만 전문으로 담당

#  • collect_news(query) : 네이버 뉴스 API로부터 정확도순 15건, 최신순 15건 수집
#  • filter_news(news_list) : 저품질 기사 필터링, 14일 경과 기사 제외 및 고유 키 기반 중복 필터링 적용 (최종 10건 유지)

def _normalize_title_for_dedupe(title: str) -> str:
    normalized = re.sub(r"\[[^\]]+\]|\([^)]+\)", " ", title.lower())
    normalized = re.sub(r"(종합|단독|속보|영상|포토|그래픽|기획|인터뷰|특징주)", " ", normalized)
    normalized = re.sub(r"[^0-9a-z가-힣]", "", normalized)
    return normalized


def _is_similar_title(title: str, seen_titles: list, threshold: float = 0.72) -> bool:
    normalized_title = _normalize_title_for_dedupe(title)
    if not normalized_title:
        return False

    for seen_title in seen_titles:
        if normalized_title in seen_title or seen_title in normalized_title:
            return True
        if SequenceMatcher(None, normalized_title, seen_title).ratio() >= threshold:
            return True
    return False


def collect_news(query: str) -> list:
    """
    네이버 뉴스 API를 통해 정확도순(sim) 15건, 최신순(date) 15건의 기사를 수집합니다.
    """
    client_id = os.environ.get("NAVER_CLIENT_ID", "").strip()
    client_secret = os.environ.get("NAVER_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return []
    
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    news_dict = {}
    
    # 정확도순, 최신순 교차 수집
    for sort_type in ["sim", "date"]:
        params = {
            "query": query,
            "display": 15,
            "sort": sort_type
        }
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                for item in items:
                    link = item.get("link", "")
                    if link not in news_dict:
                        news_dict[link] = {
                            "title": clean_html(item.get("title", "")),
                            "description": clean_html(item.get("description", "")),
                            "link": link,
                            "pub_date": item.get("pubDate", "")
                        }
        except Exception as e:
            print(f"뉴스 수집 오류 ({sort_type}):", e)
    return list(news_dict.values())

def filter_news(news_list: list) -> list:
    """
    수집된 뉴스 데이터에 대해 저품질 뉴스 제외, 오래된 뉴스 제외, 유사 제목 중복 제거를 진행합니다.
    """
    filtered = []
    latest_dt = None
    parsed_news_list = []
    
    # 1단계: 가장 최신 기사의 발행일 기준점 파악
    for news in news_list:
        pub_date_str = news.get("pub_date", "").strip()
        pub_dt = None
        if pub_date_str:
            try:
                pub_dt = parsedate_to_datetime(pub_date_str)
                if latest_dt is None or pub_dt > latest_dt:
                    latest_dt = pub_dt
            except Exception:
                pass
        parsed_news_list.append((news, pub_dt))
        
    if latest_dt is None:
        latest_dt = datetime.now()
        
    seen_keys = set()
    seen_titles = []
    # 2단계: 필터링 조건 적용
    for news, pub_dt in parsed_news_list:
        title = news.get("title", "").strip()
        description = news.get("description", "").strip()
        link = news.get("link", "").strip()
        
        # 저품질 필터링 (글자수 제한)
        if len(title) < 8 or len(description) < 20:
            continue
            
        # 14일 초과 여부 필터링
        if pub_dt:
            delta = latest_dt - pub_dt
            if delta.days > 14:
                continue
                
        # 중복 기사 필터링 (Dedupe Key)
        try:
            parsed_url = urlparse(link)
            normalized_title = "".join(title.lower().split())
            dedupe_key = parsed_url.netloc + parsed_url.path + normalized_title[:40]
        except Exception:
            dedupe_key = link + title
            
        if dedupe_key in seen_keys or _is_similar_title(title, seen_titles):
            continue
        seen_keys.add(dedupe_key)
        seen_titles.append(_normalize_title_for_dedupe(title))
        filtered.append(news)
        
    return filtered[:10]
