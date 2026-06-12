import json
import os
import streamlit as st
import requests
from concurrent.futures import ThreadPoolExecutor

try:
    import openai
except ImportError:
    openai = None

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "512"))
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.3"))
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "90"))

OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", os.environ.get("OLLAMA_TIMEOUT", "90")))
GEMINI_MODEL = OLLAMA_MODEL  # 기존 app.py 호환을 위한 매핑

def _get_openai_client():
    if openai is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. pip install openai 를 실행하세요.")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(".env 파일에 OPENAI_API_KEY 가 설정되어 있지 않습니다.")
    if hasattr(openai, "OpenAI"):
        return openai.OpenAI(api_key=api_key)
    openai.api_key = api_key
    return openai


def call_openai(messages: list, format_json: bool = False, max_tokens: int = OPENAI_MAX_TOKENS, temperature: float = OPENAI_TEMPERATURE) -> str:
    if not messages:
        return ""
    client = _get_openai_client()
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if format_json:
        payload["temperature"] = temperature
    try:
        if hasattr(client, "chat"):
            response = client.chat.completions.create(**payload)
            return response.choices[0].message.content
        return client.ChatCompletion.create(**payload).choices[0].message["content"]
    except Exception as e:
        raise RuntimeError(f"OpenAI 호출 오류가 발생했습니다. (상세 오류: {e})")


def _analyze_single_news(news: dict) -> dict:
    prompt = """당신은 전문 금융 감성 분석가입니다. 아래 제공되는 기사의 제목과 요약을 분석하여, 이 뉴스가 해당 기업의 주가에 미칠 영향을 "호재", "악재", "중립" 중 하나로 평가하고 그 구체적인 근거를 한국어로 설명해 주세요.

[기사 정보]
제목: {title}
요약: {description}

반드시 아래 형식의 JSON 객체로만 응답해 주세요. 추가적인 설명이나 텍스트는 출력하지 마세요:
{
  "sentiment": "호재" | "악재" | "중립",
  "reason": "평가한 구체적인 근거(한글 문장)"
}"""
    prompt = prompt.replace("{title}", news.get('title', '')).replace("{description}", news.get('description', ''))
    prompt += "\n\n추가 지침: 모든 응답 값은 반드시 한국어로만 작성하세요. 중국어, 일본어, 영어 문장, markdown code fence는 절대 출력하지 마세요."
    news_copy = news.copy()
    try:
        response_text = call_openai([{"role": "user", "content": prompt}], format_json=True)
        data = json.loads(response_text.strip())
        news_copy["sentiment"] = data.get("sentiment", "중립")
        news_copy["reason"] = data.get("reason", "분석 불가")
    except Exception as e:
        news_copy["sentiment"] = "중립"
        news_copy["reason"] = f"감성 분석 오류: {str(e)}"
    return news_copy


def analyze_sentiments(news_list: list, active_google_key: str = None) -> list:
    if not news_list:
        return []
    max_workers = min(4, len(news_list))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_analyze_single_news, news) for news in news_list]
        analyzed_news = [future.result() for future in futures]
    return analyzed_news


def analyze_risks(news_list: list, active_google_key: str = None) -> list:
    news_summary = []
    for i, news in enumerate(news_list, 1):
        news_summary.append(f"[{i}] {news['title']}\n- 감성: {news['sentiment']}\n- 근거: {news['reason']}")
    news_summary_str = "\n\n".join(news_summary)

    prompt = """당신은 리스크 관리 전문가입니다. 다음 수집된 뉴스 및 개별 감성 분석 결과들을 분석하여, 투자자가 해당 종목에 투자할 때 직면할 수 있는 투자 리스크 요인을 최소 3가지에서 최대 5가지 도출해 주세요. 거시경제(Macro), 기업 실적, 경쟁 구도, 주가 변동성 등의 관점을 반영하여 구체적으로 작성해야 합니다.

[뉴스 및 분석 데이터]
{news_summary_str}

반드시 아래 형식의 JSON 객체로만 응답해 주세요. 추가적인 설명이나 텍스트는 출력하지 마세요:
{
  "risks": [
    "리스크 요인 1 (구체적인 영향 및 이유 포함)",
    "리스크 요인 2 ...",
    ...
  ]
}"""
    prompt = prompt.replace("{news_summary_str}", news_summary_str)
    prompt += "\n\n추가 지침: 모든 응답 값은 반드시 한국어로만 작성하세요. 중국어, 일본어, 영어 문장, markdown code fence는 절대 출력하지 마세요."
    try:
        response_text = call_openai([{"role": "user", "content": prompt}], format_json=True)
        data = json.loads(response_text.strip())
        return data.get("risks", ["추출된 리스크 요인이 없습니다."])
    except Exception as e:
        return [f"리스크 분석 중 오류 발생: {str(e)}"]


def _sanitize_recommendations(text: str) -> str:
    replacements = {
        "강력 매수": "매우 긍정적 전망",
        "매수 추천": "관심 종목 분석",
        "매수 권유": "관심 요인 제시",
        "매수를 권장": "긍정적으로 평가",
        "매수 요인": "호재 요인",
        "매도 요인": "리스크 요인",
        "매수를 유도": "판단을 보조",
        "매도 추천": "보수적인 접근",
        "투자 추천": "투자 검토",
        "매수할 것": "진입을 신중히 검토할 것",
        "매도할 것": "비중 조절을 신중히 검토할 것"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def generate_briefing(news_list: list, risks: list, stock_name: str, active_google_key: str = None) -> str:
    news_summary = []
    for i, news in enumerate(news_list, 1):
        news_summary.append(f"[{i}] {news['title']}\n- 감성: {news['sentiment']}\n- 분석 근거: {news['reason']}")
    news_summary_str = "\n\n".join(news_summary)
    risks_str = "\n".join([f"- {risk}" for risk in risks])

    prompt = """당신은 전문 금융 분석가입니다. 수집된 뉴스 및 감성 분석 결과와 투자 리스크 요인을 종합하여 해당 종목에 대한 투자 요약 브리핑 리포트를 작성해 주세요.

[종목명]
{stock_name}

[뉴스 및 감성 분석 요약]
{news_summary_str}

[도출된 투자 리스크]
{risks_str}

[작성 지침]
- 반드시 한국어로 구조화하여 마크다운 양식으로 작성해 주세요.
- 다음 항목을 포함해야 합니다:
  1. **뉴스 분석 요약**: 최근 수집된 뉴스들의 핵심 트렌드를 종합 설명합니다.
  2. **호재와 악재**: 주요 호재 요인과 악재 요인을 일목요연하게 정리합니다.
  3. **투자 리스크 전망**: 분석된 리스크 요인이 향후 주가나 기업에 미칠 잠재적 영향을 종합적으로 분석합니다.
  4. **최종 브리핑 요약**: 투자자 관점의 핵심 메시지를 3줄 요약하여 서술합니다.
- (주의: "매수", "매도", "추천", "강력 추천" 등 투자 권유를 나타내는 직접적인 표현은 사용하지 마십시오.)

최종 보고서 작성 시작:"""
    prompt = prompt.replace("{stock_name}", stock_name)
    prompt = prompt.replace("{news_summary_str}", news_summary_str)
    prompt = prompt.replace("{risks_str}", risks_str)
    prompt += "\n\n추가 지침: 반드시 한국어 문장만 작성하세요. 중국어, 일본어, 영어 문장은 금지합니다. 필요한 영문 약어(HBM, AI 등)만 짧게 사용할 수 있습니다. ```markdown 또는 ``` 같은 코드펜스는 절대 출력하지 마세요."
    try:
        response_text = call_openai([{"role": "user", "content": prompt}])
        briefing = response_text.strip()
        briefing = _sanitize_recommendations(briefing)
        return briefing
    except Exception as e:
        return f"브리핑 생성 중 오류 발생: {str(e)}"


def run_direct_gemini_chat(
    user_question: str,
    active_google_key: str = None,
    model_name: str = None,
    tech_analysis: dict = None,
    history_analysis: dict = None,
    bull_result: dict = None,
    bear_result: dict = None,
) -> str:
    """
    분석 완료 후 대화방에서 유저의 후속 질문에 대답합니다.
    tech_analysis, history_analysis, bull_result, bear_result 를 받아 풍부한 컨텍스트로 답변합니다.
    """
    tech_analysis = tech_analysis or {}
    history_analysis = history_analysis or {}
    bull_result = bull_result or {}
    bear_result = bear_result or {}

    # 기술적 분석 컨텍스트
    tech_str = ""
    if tech_analysis:
        details_str = "\n".join(f"  - {d}" for d in tech_analysis.get("details", []))
        tech_str = f"""
[기술적 분석]
신호: {tech_analysis.get('signal', '-')}
요약: {tech_analysis.get('summary', '-')}
{details_str}"""

    # 과거 패턴 컨텍스트
    history_str = ""
    if history_analysis:
        history_str = f"""
[과거 패턴 분석]
요약: {history_analysis.get('summary', '-')}
계절성: {history_analysis.get('seasonality_insight', '-')}
리스크 수준: {history_analysis.get('risk_level', '-')}"""

    # Bull 컨텍스트
    bull_str = ""
    if bull_result.get("arguments"):
        args = "\n".join(f"  - {a}" for a in bull_result["arguments"])
        bull_str = f"""
[강세 논거 (Bull) - 신뢰도: {bull_result.get('confidence', '-')}]
{args}"""

    # Bear 컨텍스트
    bear_str = ""
    if bear_result.get("arguments"):
        args = "\n".join(f"  - {a}" for a in bear_result["arguments"])
        bear_str = f"""
[약세 논거 (Bear) - 신뢰도: {bear_result.get('confidence', '-')}]
{args}"""

    prompt = f"""당신은 전문 주식 투자 분석 에이전트입니다.
사용자가 수집된 최근 뉴스 정보 및 기존 대화 맥락을 기반으로 후속 질문을 하고 있습니다. 이에 대해 친절하고 전문적으로 한국어로 답변해 주세요.

[수집된 뉴스 정보]
{st.session_state.get("news_context", "")}
{tech_str}
{history_str}
{bull_str}
{bear_str}

[이전 대화 기록]
"""
    for msg in st.session_state["chat_history"]:
        prompt += f"{msg['role'].upper()}: {msg['content']}\n"

    prompt += f"""
USER: {user_question}
ASSISTANT:"""
    prompt += "\n\n추가 지침: 반드시 한국어 문장만 작성하세요. 중국어, 일본어, 영어 문장은 금지합니다. 필요한 영문 약어만 짧게 사용할 수 있습니다."

    try:
        response_text = call_openai([{"role": "user", "content": prompt}])
        return response_text
    except Exception as e:
        return f"답변 중 오류 발생: {str(e)}"