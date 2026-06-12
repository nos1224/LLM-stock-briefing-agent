import os
import json
import requests

try:
    import openai
except ImportError:
    openai = None

# Summary agent
# 모든 에이전트 결과를 종합하여 최종 투자 의견과 브리핑을 생성합니다.

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "512"))
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.3"))
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "90"))

OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "90"))


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


def _call_ollama(messages: list) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3},
    }
    response = requests.post(OLLAMA_API_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    response.raise_for_status()
    return response.json()["message"]["content"]


def _sanitize(text: str) -> str:
    replacements = {
        "강력 매수": "매우 긍정적 전망",
        "매수 추천": "관심 종목 분석",
        "매수 권유": "관심 요인 제시",
        "매수를 권장": "긍정적으로 평가",
        "매도 추천": "보수적인 접근",
        "투자 추천": "투자 검토",
        "매수할 것": "진입을 신중히 검토할 것",
        "매도할 것": "비중 조절을 신중히 검토할 것",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def run_summary_agent(
    stock_name: str,
    news_analysis: list,
    risks: list,
    tech_analysis: dict,
    history_analysis: dict,
    bull_result: dict,
    bear_result: dict,
) -> str:
    """
    모든 에이전트 결과를 종합하여 최종 마크다운 브리핑 리포트를 생성합니다.
    """
    # 뉴스 요약
    news_lines = []
    for i, n in enumerate(news_analysis[:5], 1):
        news_lines.append(f"[{i}] {n.get('title','')} → {n.get('sentiment','')} / {n.get('reason','')}")
    news_str = "\n".join(news_lines)

    # 리스크
    risks_str = "\n".join(f"- {r}" for r in risks)

    # Bull 논거
    bull_str = "\n".join(f"- {a}" for a in bull_result.get("arguments", []))

    # Bear 논거
    bear_str = "\n".join(f"- {a}" for a in bear_result.get("arguments", []))

    prompt = f"""당신은 수석 금융 분석가입니다. 아래 모든 에이전트의 분석 결과를 종합하여 "{stock_name}"에 대한 최종 투자 브리핑 리포트를 마크다운 형식으로 작성하세요.
각 항목은 단순 나열이 아니라 분석가의 시각으로 충분히 서술해 주세요. 각 섹션 최소 3~5문장 이상 작성하세요.

[뉴스 감성 분석 요약]
{news_str}

[기술적 분석]
신호: {tech_analysis.get('signal', '-')}
{tech_analysis.get('summary', '')}
세부 내용: {', '.join(tech_analysis.get('details', []))}

[과거 패턴 분석]
{history_analysis.get('summary', '')}
계절성: {history_analysis.get('seasonality_insight', '')}
리스크 수준: {history_analysis.get('risk_level', '-')}

[강세 논거 (Bull)]
{bull_str}
강세 신뢰도: {bull_result.get('confidence', '-')}

[약세 논거 (Bear)]
{bear_str}
약세 신뢰도: {bear_result.get('confidence', '-')}

[주요 리스크 요인]
{risks_str}

[작성 지침]
반드시 한국어 마크다운으로 작성하고 아래 항목을 빠짐없이 포함하세요. 각 항목은 충분한 분량으로 서술하세요:

1. **종합 시황 요약**
   - 최근 뉴스 흐름, 기술적 흐름, 과거 패턴을 종합한 현재 종목 상황을 5문장 이상 서술하세요.

2. **호재 및 악재 분석**
   - 주요 호재 요인과 악재 요인을 각각 구체적으로 서술하세요. 뉴스 감성 분석 결과를 근거로 활용하세요.

3. **기술적 분석 해석**
   - 이동평균, RSI, MACD, 거래량 등 지표를 바탕으로 현재 주가 흐름의 의미를 투자자 관점에서 해석하세요.

4. **계절성 및 과거 패턴 인사이트**
   - 과거 데이터에서 도출된 계절성과 패턴이 현재 시점에 주는 시사점을 서술하세요.

5. **강세 vs 약세 논거 비교**
   - Bull/Bear 논거를 아래 표 형식으로 정리하세요:

| 구분 | 내용 |
|------|------|
| 🟢 강세 논거 | (Bull 논거 요약) |
| 🔴 약세 논거 | (Bear 논거 요약) |

6. **리스크 요인**
   - 핵심 리스크 요인이 실제로 투자에 미칠 영향을 구체적으로 서술하세요.

7. **최종 투자 관점 요약**
   - 투자자가 이 종목을 바라볼 때 핵심적으로 고려해야 할 사항을 3~5줄로 요약하세요. (투자 권유 표현 금지)

markdown code fence(```) 절대 출력하지 마세요.
중국어·일본어·영어 문장 금지. 영문 약어(RSI, MACD, AI 등)만 허용."""

    try:
        result = call_openai([{"role": "user", "content": prompt}], max_tokens=2500)
        briefing = _sanitize(result.strip())
        return briefing
    except Exception as e:
        return f"최종 브리핑 생성 오류: {str(e)}"