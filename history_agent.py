import os
import json
import requests

try:
    import openai
except ImportError:
    openai = None

try:
    import yfinance as yf
except ImportError:
    yf = None

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "512"))
OPENAI_TEMPERATURE = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))
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


def fetch_history_data(ticker: str) -> dict:
    """
    2년치 일봉 데이터를 수집하고 월별 수익률, 변동성, 최대낙폭을 계산합니다.
    """
    if yf is None:
        return {"error": "yfinance가 설치되어 있지 않습니다."}
    try:
        hist = yf.Ticker(ticker).history(period="2y", interval="1d")
        if hist.empty or len(hist) < 60:
            return {"error": f"'{ticker}' 과거 데이터가 부족합니다."}

        close = hist["Close"].dropna()

        # 월별 수익률 계산
        monthly = close.resample("ME").last()
        monthly_returns = monthly.pct_change().dropna()
        monthly_avg = {}
        for date, ret in monthly_returns.items():
            month_key = date.strftime("%m월")
            if month_key not in monthly_avg:
                monthly_avg[month_key] = []
            monthly_avg[month_key].append(float(ret) * 100)

        seasonality = {
            month: round(sum(vals) / len(vals), 2)
            for month, vals in monthly_avg.items()
        }

        # 연간 수익률 (최근 1년)
        if len(close) >= 252:
            yearly_return = (float(close.iloc[-1]) / float(close.iloc[-252]) - 1) * 100
        else:
            yearly_return = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100

        # 최대 낙폭 (MDD)
        peak = close.cummax()
        drawdown = (close - peak) / peak
        mdd = float(drawdown.min()) * 100

        # 변동성 (연환산)
        daily_returns = close.pct_change().dropna()
        volatility = float(daily_returns.std() * (252 ** 0.5) * 100)

        # 최근 3개월 추세
        if len(close) >= 60:
            trend_3m = (float(close.iloc[-1]) / float(close.iloc[-60]) - 1) * 100
        else:
            trend_3m = 0.0

        return {
            "ticker": ticker,
            "yearly_return": round(yearly_return, 2),
            "mdd": round(mdd, 2),
            "volatility": round(volatility, 2),
            "trend_3m": round(trend_3m, 2),
            "seasonality": seasonality,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_history(stock_name: str, history_data: dict) -> dict:
    """
    과거 데이터 지표를 Ollama로 해석하여 패턴 인사이트를 반환합니다.
    반환: {summary: str, seasonality_insight: str, risk_level: "높음"|"보통"|"낮음"}
    """
    if history_data.get("error"):
        return {
            "summary": f"과거 데이터 수집 실패: {history_data['error']}",
            "seasonality_insight": "",
            "risk_level": "보통",
        }

    seasonality_str = ", ".join(
        f"{k}: {v:+.1f}%" for k, v in history_data["seasonality"].items()
    )

    prompt = f"""당신은 주식 퀀트 분석가입니다. 아래 과거 데이터를 분석하여 패턴과 계절성 인사이트를 도출하세요.

[종목명] {stock_name}
[1년 수익률] {history_data['yearly_return']}%
[최대 낙폭(MDD)] {history_data['mdd']}%
[연환산 변동성] {history_data['volatility']}%
[최근 3개월 추세] {history_data['trend_3m']}%
[월별 평균 수익률] {seasonality_str}

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "summary": "과거 성과와 패턴에 대한 2~3줄 요약 (한국어)",
  "seasonality_insight": "계절성 분석 결과 (어느 달이 강세/약세인지 한국어로)",
  "risk_level": "높음" | "보통" | "낮음"
}}

추가 지침: 반드시 한국어로만 작성하세요. markdown code fence 금지."""

    try:
        result = call_openai([{"role": "user", "content": prompt}], format_json=True)
        data = json.loads(result.strip())
        return {
            "summary": data.get("summary", ""),
            "seasonality_insight": data.get("seasonality_insight", ""),
            "risk_level": data.get("risk_level", "보통"),
        }
    except Exception as e:
        return {
            "summary": f"과거 패턴 분석 오류: {str(e)}",
            "seasonality_insight": "",
            "risk_level": "보통",
        }
