from concurrent.futures import ThreadPoolExecutor, as_completed

from news_agent import collect_news, filter_news
from ai_agent import analyze_sentiments, analyze_risks
from technical_agent import get_ticker_symbol, fetch_technical_data, analyze_technical
from history_agent import fetch_history_data, analyze_history
from bull_bear_agent import run_bull_agent, run_bear_agent
from summary_agent import run_summary_agent

# Orchestrator : 전체 멀티에이전트 파이프라인을 관리합니다.
#
# 실행 순서:
#   1단계 [병렬] News agent + Technical agent + History agent
#   2단계 [순차] Bull agent + Bear agent  (1단계 결과 필요)
#   3단계 [순차] Risk agent              (뉴스 분석 결과 필요)
#   4단계 [순차] Summary agent           (전체 결과 종합)


def run_pipeline(stock_name: str, status_callback=None) -> dict:
    """
    전체 분석 파이프라인을 실행하고 결과 딕셔너리를 반환합니다.

    status_callback(step: str) : UI 진행 상황 업데이트 함수 (선택)

    반환 딕셔너리 키:
      news_raw, news_analyzed, ticker,
      tech_data, tech_analysis,
      history_data, history_analysis,
      bull, bear, risks, briefing
    """

    def _notify(msg: str):
        if status_callback:
            status_callback(msg)

    # ── 0단계: 티커 조회 ──────────────────────────────────────
    _notify("0단계: 종목 티커 조회 중...")
    ticker = get_ticker_symbol(stock_name)
    _notify(f"0단계 완료 ✅ 티커: {ticker or '조회 실패'}")

    # ── 1단계: 병렬 실행 ─────────────────────────────────────
    _notify("1단계: 뉴스 수집 · 기술적 분석 · 과거 패턴 분석 병렬 실행 중...")

    news_raw = []
    news_filtered = []
    tech_data = {}
    history_data = {}

    def _run_news():
        raw = collect_news(stock_name)
        filtered = filter_news(raw)
        return raw, filtered

    def _run_technical():
        if not ticker:
            return {"error": "티커 조회 실패"}
        td = fetch_technical_data(ticker)
        return td

    def _run_history():
        if not ticker:
            return {"error": "티커 조회 실패"}
        hd = fetch_history_data(ticker)
        return hd

    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_news = executor.submit(_run_news)
        fut_tech = executor.submit(_run_technical)
        fut_hist = executor.submit(_run_history)

        news_raw, news_filtered = fut_news.result()
        tech_data = fut_tech.result()
        history_data = fut_hist.result()

    if not news_filtered:
        raise ValueError("유효한 뉴스를 수집하지 못했습니다. 네이버 API 설정을 확인하세요.")

    _notify(f"1단계 완료 ✅ 뉴스 {len(news_filtered)}건 / 기술적 데이터 {'수집완료' if not tech_data.get('error') else '실패'} / 과거 데이터 {'수집완료' if not history_data.get('error') else '실패'}")

    # ── 2단계: 뉴스 감성 분석 ────────────────────────────────
    _notify("2단계: 뉴스 감성 분석 중 (Ollama)...")
    news_analyzed = analyze_sentiments(news_filtered)
    _notify(f"2단계 완료 ✅ {len(news_analyzed)}건 감성 분석")

    # ── 3단계: 기술적·과거 해석 + Bull/Bear 병렬 실행 ────────
    _notify("3단계: 기술적 해석 · 과거 패턴 해석 · 강세/약세 논거 생성 병렬 실행 중...")

    tech_analysis = {}
    history_analysis = {}
    bull_result = {}
    bear_result = {}

    def _run_tech_analysis():
        return analyze_technical(stock_name, tech_data)

    def _run_hist_analysis():
        return analyze_history(stock_name, history_data)

    # 먼저 기술적·과거 해석을 병렬로 돌리고
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_ta = executor.submit(_run_tech_analysis)
        fut_ha = executor.submit(_run_hist_analysis)
        tech_analysis = fut_ta.result()
        history_analysis = fut_ha.result()

    # Bull/Bear 는 위 결과가 필요하므로 이후 병렬 실행
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_bull = executor.submit(run_bull_agent, stock_name, news_analyzed, tech_analysis, history_analysis)
        fut_bear = executor.submit(run_bear_agent, stock_name, news_analyzed, tech_analysis, history_analysis)
        bull_result = fut_bull.result()
        bear_result = fut_bear.result()

    _notify("3단계 완료 ✅ 강세/약세 논거 생성")

    # ── 4단계: 리스크 분석 ───────────────────────────────────
    _notify("4단계: 투자 리스크 요인 분석 중...")
    risks = analyze_risks(news_analyzed)
    _notify(f"4단계 완료 ✅ {len(risks)}개 리스크 감지")

    # ── 5단계: 최종 브리핑 ───────────────────────────────────
    _notify("5단계: 최종 종합 브리핑 생성 중...")
    briefing = run_summary_agent(
        stock_name=stock_name,
        news_analysis=news_analyzed,
        risks=risks,
        tech_analysis=tech_analysis,
        history_analysis=history_analysis,
        bull_result=bull_result,
        bear_result=bear_result,
    )
    _notify("5단계 완료 ✅ 브리핑 생성")

    return {
        "news_raw": news_raw,
        "news_analyzed": news_analyzed,
        "ticker": ticker,
        "tech_data": tech_data,
        "tech_analysis": tech_analysis,
        "history_data": history_data,
        "history_analysis": history_analysis,
        "bull": bull_result,
        "bear": bear_result,
        "risks": risks,
        "briefing": briefing,
    }
