import os
import requests

try:
    import openai
except ImportError:
    openai = None

try:
    import yfinance as yf
except ImportError:
    yf = None

# yfinance 기반 기술적 분석 에이전트
# - get_ticker_symbol() : 종목명 → 티커 변환 (OpenAI 활용)
# - fetch_technical_data() : 주가 데이터 수집
# - analyze_technical() : 이동평균, RSI, MACD 등 지표 계산 및 OpenAI 해석

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


def get_ticker_symbol(stock_name: str) -> str:
    """
    한국어 종목명을 yfinance 티커로 변환합니다.
    예: '삼성전자' → '005930.KS', '테슬라' → 'TSLA'
    """
    prompt = f"""다음 주식 종목명에 해당하는 yfinance 티커 심볼을 반환하세요.
한국 주식이면 .KS 또는 .KQ를 붙이고, 미국 주식이면 그냥 심볼만 반환하세요.

종목명: {stock_name}

반드시 아래 JSON 형식으로만 응답하세요:
{{"ticker": "티커심볼"}}"""
    try:
        import json
        result = call_openai([{"role": "user", "content": prompt}], format_json=True)
        data = json.loads(result.strip())
        return data.get("ticker", "").strip()
    except Exception:
        return ""


def fetch_technical_data(ticker: str) -> dict:
    """
    yfinance로 주가 데이터를 수집하고 기술적 지표를 계산합니다.
    반환: {prices, ma5, ma20, ma60, rsi, macd, macd_signal, volume_avg, current_price, 52w_high, 52w_low}
    """
    if yf is None:
        return {"error": "yfinance가 설치되어 있지 않습니다."}
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="6mo", interval="1d")
        if hist.empty or len(hist) < 20:
            return {"error": f"'{ticker}' 데이터를 불러올 수 없습니다. 티커를 확인하세요."}

        close = hist["Close"].dropna()
        volume = hist["Volume"].dropna()

        # 이동평균
        ma5  = float(close.rolling(5).mean().iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(min(60, len(close))).mean().iloc[-1])

        # RSI (14일)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        current_price = float(close.iloc[-1])
        high_52w = float(close.tail(252).max())
        low_52w  = float(close.tail(252).min())

        # 최근 20일 가격 리스트 (스파크라인용)
        recent_prices = [round(float(p), 2) for p in close.tail(20).tolist()]

        return {
            "ticker": ticker,
            "current_price": current_price,
            "ma5": round(ma5, 2),
            "ma20": round(ma20, 2),
            "ma60": round(ma60, 2),
            "rsi": round(rsi, 2),
            "macd": round(float(macd_line.iloc[-1]), 4),
            "macd_signal": round(float(signal_line.iloc[-1]), 4),
            "volume_avg_20": int(volume.tail(20).mean()),
            "volume_latest": int(volume.iloc[-1]),
            "high_52w": round(high_52w, 2),
            "low_52w": round(low_52w, 2),
            "recent_prices": recent_prices,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_technical(stock_name: str, tech_data: dict) -> dict:
    """
    수집된 기술적 지표를 Ollama로 해석하여 요약과 신호를 반환합니다.
    반환: {signal: "강세"|"약세"|"중립", summary: str, details: list}
    """
    import json

    if tech_data.get("error"):
        return {
            "signal": "중립",
            "summary": f"기술적 데이터 수집 실패: {tech_data['error']}",
            "details": [],
        }

    prompt = f"""당신은 전문 주식 기술적 분석가입니다. 아래 기술적 지표를 분석하여 현재 주가 흐름을 평가하세요.

[종목명] {stock_name}
[현재가] {tech_data['current_price']}
[이동평균] MA5={tech_data['ma5']}, MA20={tech_data['ma20']}, MA60={tech_data['ma60']}
[RSI(14)] {tech_data['rsi']} (30 이하=과매도, 70 이상=과매수)
[MACD] {tech_data['macd']}, Signal={tech_data['macd_signal']}
[52주 고가] {tech_data['high_52w']} / [52주 저가] {tech_data['low_52w']}
[거래량 20일평균] {tech_data['volume_avg_20']} / [최근 거래량] {tech_data['volume_latest']}

반드시 아래 JSON 형식으로만 응답하세요. 추가 텍스트 없이:
{{
  "signal": "강세" | "약세" | "중립",
  "summary": "전체 기술적 흐름 2~3줄 요약 (한국어)",
  "details": [
    "이동평균 분석 내용",
    "RSI 분석 내용",
    "MACD 분석 내용",
    "거래량 분석 내용"
  ]
}}

추가 지침: 반드시 한국어로만 작성하세요. markdown code fence 금지."""

    try:
        result = call_openai([{"role": "user", "content": prompt}], format_json=True)
        data = json.loads(result.strip())
        return {
            "signal": data.get("signal", "중립"),
            "summary": data.get("summary", ""),
            "details": data.get("details", []),
        }
    except Exception as e:
        return {
            "signal": "중립",
            "summary": f"기술적 분석 해석 오류: {str(e)}",
            "details": [],
        }