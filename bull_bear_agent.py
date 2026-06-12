import os
import json
import requests

try:
    import openai
except ImportError:
    openai = None

# Bull agent : 매수 관점 논거 생성
# Bear agent : 매도 관점 논거 생성
# 두 에이전트는 같은 데이터를 받아 서로 반대 입장에서 분석합니다.

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "512"))
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.4"))
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "90"))


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
    try:
        if hasattr(client, "chat"):
            response = client.chat.completions.create(**payload)
            return response.choices[0].message.content
        return client.ChatCompletion.create(**payload).choices[0].message["content"]
    except Exception as e:
        raise RuntimeError(f"OpenAI 호출 오류가 발생했습니다. (상세 오류: {e})")


def _build_context(
    stock_name: str,
    news_analysis: list,
    tech_analysis: dict,
    history_analysis: dict,
) -> str:
    """Bull/Bear 공통 컨텍스트 문자열 생성."""
    news_lines = []
    for i, n in enumerate(news_analysis[:5], 1):
        news_lines.append(f"  [{i}] {n.get('title','')} → {n.get('sentiment','')} ({n.get('reason','')})")
    news_str = "\n".join(news_lines) if news_lines else "  (없음)"

    return f"""[종목명] {stock_name}

[뉴스 감성 요약]
{news_str}

[기술적 분석]
  신호: {tech_analysis.get('signal', '-')}
  요약: {tech_analysis.get('summary', '-')}

[과거 패턴]
  요약: {history_analysis.get('summary', '-')}
  계절성: {history_analysis.get('seasonality_insight', '-')}
  리스크 수준: {history_analysis.get('risk_level', '-')}"""


def run_bull_agent(
    stock_name: str,
    news_analysis: list,
    tech_analysis: dict,
    history_analysis: dict,
) -> dict:
    """
    긍정적(매수) 관점에서 투자 논거 3~5가지를 생성합니다.
    반환: {arguments: list[str], confidence: "높음"|"보통"|"낮음"}
    """
    context = _build_context(stock_name, news_analysis, tech_analysis, history_analysis)

    prompt = f"""당신은 강세론자(Bull) 투자 분석가입니다. 아래 데이터를 바탕으로 "{stock_name}" 종목에 대한 긍정적 투자 논거를 3~5가지 도출하세요. 호재 요인, 기술적 강세 신호, 유리한 과거 패턴 등을 근거로 활용하세요.

{context}

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "arguments": [
    "긍정적 논거 1 (구체적 근거 포함)",
    "긍정적 논거 2 ...",
    ...
  ],
  "confidence": "높음" | "보통" | "낮음"
}}

주의: "매수 추천", "매수 권유" 등 직접적 투자 권유 표현은 사용하지 마세요.
추가 지침: 반드시 한국어로만 작성하세요. markdown code fence 금지."""

    try:
        result = call_openai([{"role": "user", "content": prompt}], format_json=True)
        data = json.loads(result.strip())
        return {
            "arguments": data.get("arguments", []),
            "confidence": data.get("confidence", "보통"),
        }
    except Exception as e:
        return {
            "arguments": [f"강세 논거 생성 오류: {str(e)}"],
            "confidence": "낮음",
        }


def run_bear_agent(
    stock_name: str,
    news_analysis: list,
    tech_analysis: dict,
    history_analysis: dict,
) -> dict:
    """
    부정적(매도) 관점에서 투자 논거 3~5가지를 생성합니다.
    반환: {arguments: list[str], confidence: "높음"|"보통"|"낮음"}
    """
    context = _build_context(stock_name, news_analysis, tech_analysis, history_analysis)

    prompt = f"""당신은 약세론자(Bear) 투자 분석가입니다. 아래 데이터를 바탕으로 "{stock_name}" 종목에 대한 부정적 투자 논거를 3~5가지 도출하세요. 악재 요인, 기술적 약세 신호, 불리한 과거 패턴, 리스크 요인 등을 근거로 활용하세요.

{context}

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "arguments": [
    "부정적 논거 1 (구체적 근거 포함)",
    "부정적 논거 2 ...",
    ...
  ],
  "confidence": "높음" | "보통" | "낮음"
}}

주의: "매도 추천", "매도 권유" 등 직접적 투자 권유 표현은 사용하지 마세요.
추가 지침: 반드시 한국어로만 작성하세요. markdown code fence 금지."""

    try:
        result = call_openai([{"role": "user", "content": prompt}], format_json=True)
        data = json.loads(result.strip())
        return {
            "arguments": data.get("arguments", []),
            "confidence": data.get("confidence", "보통"),
        }
    except Exception as e:
        return {
            "arguments": [f"약세 논거 생성 오류: {str(e)}"],
            "confidence": "낮음",
        }