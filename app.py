import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv

try:
    import yfinance as yf
except ImportError:
    yf = None

load_dotenv()

from utils import load_all_history, save_current_history
from ai_agent import run_direct_gemini_chat, GEMINI_MODEL
from orchestrator import run_pipeline

try:
    from rag_vector import embed_and_store, query_similar_news
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

# ── 세션 상태 초기화 ──────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "news_context": "",
    "news_context_raw": [],
    "risks": [],
    "current_stock": "",
    "session_id": "",
    "tech_analysis": {},
    "history_analysis": {},
    "bull": {},
    "bear": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.set_page_config(page_title="Stock-Agent: AI 투자 뉴스 비서", page_icon="📈", layout="centered")


# ── 시장 대시보드 ──────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_market_data():
    from concurrent.futures import ThreadPoolExecutor
    markets = {"KOSPI": "^KS11", "NASDAQ": "^IXIC", "S&P 500": "^GSPC", "USD/KRW": "KRW=X"}

    def _fetch_one(name, ticker):
        if yf is None:
            return name, {"ticker": ticker, "current": None, "change": None, "change_percent": None, "history": None, "error": "데이터를 불러올 수 없습니다"}
        try:
            history = yf.Ticker(ticker).history(period="1mo", interval="1d")
            close_prices = history["Close"].dropna()
            if len(close_prices) < 2:
                raise ValueError("Not enough data")
            cur = float(close_prices.iloc[-1])
            prev = float(close_prices.iloc[-2])
            chg = cur - prev
            chg_pct = (chg / prev) * 100 if prev else 0
            return name, {"ticker": ticker, "current": cur, "change": chg, "change_percent": chg_pct, "history": close_prices.tail(20), "error": None}
        except Exception:
            return name, {"ticker": ticker, "current": None, "change": None, "change_percent": None, "history": None, "error": "데이터를 불러올 수 없습니다"}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_fetch_one, n, t) for n, t in markets.items()]
        return dict(f.result() for f in futures)


def build_market_sparkline(history, line_color):
    if history is None or len(history) < 2:
        return ""
    values = [float(v) for v in history.values]
    mn, mx = min(values), max(values)
    rng = mx - mn
    w, h = 100, 36
    pts = []
    for i, v in enumerate(values):
        x = (i / (len(values) - 1)) * w
        y = h - 5 - ((v - mn) / rng) * (h - 10) if rng else h / 2
        pts.append(f"{x:.2f},{y:.2f}")
    return (f'<svg class="market-sparkline" viewBox="0 0 {w} {h}" preserveAspectRatio="none" aria-hidden="true">'
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="{line_color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" /></svg>')


def render_market_card(name, data):
    is_down = (data.get("change") or 0) < 0
    cls = "market-down" if is_down else "market-up"
    sign = "-" if is_down else "+"
    if data.get("error"):
        return f'<div class="market-card"><div class="market-name">{name}</div><div class="market-error">데이터를 불러올 수 없습니다</div></div>'
    lc = "#EF4444" if is_down else "#22C55E"
    spark = build_market_sparkline(data["history"], lc)
    val = f'{data["current"]:,.2f}'
    return (f'<div class="market-card"><div class="market-card-top"><span class="market-name">{name}</span>'
            f'<span class="market-ticker">{data["ticker"]}</span></div>'
            f'<div class="market-value">{val}</div>'
            f'<div class="{cls}">{sign}{abs(data["change"]):,.2f} ({sign}{abs(data["change_percent"]):.2f}%)</div>'
            f'{spark}</div>')


def render_market_dashboard():
    market_data = load_market_data()
    cards = "".join(render_market_card(n, d) for n, d in market_data.items())
    st.markdown(
        f'<section class="market-dashboard"><div class="market-section-title">Market Overview</div>'
        f'<div class="market-grid">{cards}</div></section>',
        unsafe_allow_html=True,
    )


# ── CSS ───────────────────────────────────────────────────────
def load_css(css_path: str = "style.css"):
    with open(css_path, encoding="utf-8") as f:
        css_content = "\n".join(line for line in f.readlines() if not line.startswith("#"))
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_css()

# ── 타이틀 ────────────────────────────────────────────────────
st.markdown('<div class="main-title">💡 AI 주식 분석 비서</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">뉴스 · 기술적 분석 · 과거 패턴 · Bull/Bear 논거를 종합한 에이전트 브리핑</div>', unsafe_allow_html=True)

render_market_dashboard()

active_google_key = os.environ.get("GOOGLE_API_KEY", "").strip()

# ── 사이드바 ──────────────────────────────────────────────────
st.sidebar.title("📁 대화 히스토리")
if st.sidebar.button("➕ 새 주식 분석 시작", use_container_width=True):
    for key in ["chat_history", "news_context", "news_context_raw", "risks",
                "current_stock", "session_id", "tech_analysis", "history_analysis", "bull", "bear"]:
        st.session_state[key] = [] if key in ("chat_history", "news_context_raw", "risks") else {}  if key in ("tech_analysis", "history_analysis", "bull", "bear") else ""
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📜 이전 분석 목록")
all_hist = load_all_history()
if not all_hist:
    st.sidebar.write("이전 분석 내역이 없습니다.")
else:
    for sess_id, details in sorted(all_hist.items(), key=lambda x: x[1]["created_at"], reverse=True):
        t = datetime.strptime(details["created_at"], "%Y-%m-%d %H:%M:%S").strftime("%m-%d %H:%M")
        if st.sidebar.button(f"📈 {details['stock_name']} ({t})", key=sess_id, use_container_width=True):
            st.session_state["session_id"] = sess_id
            st.session_state["current_stock"] = details["stock_name"]
            st.session_state["news_context"] = details.get("news_context", "")
            st.session_state["news_context_raw"] = details.get("news_context_raw", [])
            st.session_state["risks"] = details.get("risks", [])
            st.session_state["tech_analysis"] = details.get("tech_analysis", {})
            st.session_state["history_analysis"] = details.get("history_analysis", {})
            st.session_state["bull"] = details.get("bull", {})
            st.session_state["bear"] = details.get("bear", {})
            st.session_state["chat_history"] = details["chat_history"]
            st.rerun()

# ── 메인 영역 ─────────────────────────────────────────────────
if not st.session_state["current_stock"]:
    st.subheader("🔍 주식 종목 분석 시작")
    stock_input = st.text_input("분석할 주식 종목명을 입력하세요", placeholder="예: 삼성전자, 테슬라, SK하이닉스")

    st.markdown("<div style='margin-top:1rem;margin-bottom:0.5rem;font-weight:700;color:#1E1E23;font-size:0.95rem;'>🔥 실시간 인기 종목 빠른 검색</div>", unsafe_allow_html=True)
    clicked_stock = None
    for row_stocks, key_prefix in [
        (["삼성전자", "SK하이닉스", "테슬라"], "pop1"),
        (["엔비디아", "애플", "에코프로"], "pop2"),
    ]:
        cols = st.columns(3)
        for i, stock in enumerate(row_stocks):
            if cols[i].button(stock, key=f"{key_prefix}_{stock}", use_container_width=True):
                clicked_stock = stock

    st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)
    start_btn = st.button("에이전트 분석 시작", use_container_width=True)

    target_stock = ""
    should_start = False
    if start_btn:
        if not stock_input.strip():
            st.warning("분석할 종목명을 입력해 주세요.")
        else:
            target_stock = stock_input.strip()
            should_start = True
    elif clicked_stock:
        target_stock = clicked_stock
        should_start = True

    if should_start:
        st.session_state["current_stock"] = target_stock
        st.session_state["session_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            with st.status("" \
            "에이전트 분석 진행 중...", expanded=True) as status:
                def _notify(msg):
                    status.write(msg)

                result = run_pipeline(target_stock, status_callback=_notify)
                status.update(label="분석 완료! 🎉", state="complete", expanded=False)

            # 세션 저장
            analyzed = result["news_analyzed"]
            news_summary_text = [
                f"[{i}] {n['title']}\n- 감성: {n['sentiment']}\n- 근거: {n['reason']}\n- 링크: {n['link']}"
                for i, n in enumerate(analyzed, 1)
            ]
            st.session_state["news_context"] = "\n\n".join(news_summary_text)
            st.session_state["news_context_raw"] = analyzed
            st.session_state["risks"] = result["risks"]
            st.session_state["tech_analysis"] = result["tech_analysis"]
            st.session_state["history_analysis"] = result["history_analysis"]
            st.session_state["bull"] = result["bull"]
            st.session_state["bear"] = result["bear"]

            st.session_state["chat_history"].append({"role": "user", "content": f"**{target_stock}** 에이전트 분석 시작"})
            st.session_state["chat_history"].append({"role": "assistant", "content": result["briefing"]})

            save_current_history(
                st.session_state["session_id"],
                st.session_state["current_stock"],
                st.session_state["news_context"],
                st.session_state["chat_history"],
                news_context_raw=analyzed,
                risks=result["risks"],
            )

            # ── RAG: 뉴스 임베딩 → ChromaDB 저장 ──
            if RAG_AVAILABLE:
                try:
                    embed_and_store(
                        session_id=st.session_state["session_id"],
                        stock_name=st.session_state["current_stock"],
                        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        news_list=analyzed,
                    )
                except Exception:
                    pass  # RAG 실패해도 메인 흐름 중단 안 함

            st.rerun()
        except Exception as e:
            st.session_state["current_stock"] = ""
            st.session_state["session_id"] = ""
            st.error(f"분석 중 오류 발생: {str(e)}")

else:
    st.markdown(f"### 💬 **{st.session_state['current_stock']}** 에이전트 분석")

    # ── 감성 통계 배너 ──
    raw_news = st.session_state.get("news_context_raw", [])
    if raw_news:
        good = sum(1 for n in raw_news if n.get("sentiment") == "호재")
        bad  = sum(1 for n in raw_news if n.get("sentiment") == "악재")
        neu  = sum(1 for n in raw_news if n.get("sentiment") == "중립")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div style="background:#EBFBEE;border-left:5px solid #03C75A;padding:10px;border-radius:5px;text-align:center;"><span style="color:#03C75A;font-weight:bold;font-size:0.95rem;">🟢 호재 뉴스</span><br><span style="font-size:1.3rem;font-weight:800;color:#1E1E23;">{good}건</span></div>', unsafe_allow_html=True)
        col2.markdown(f'<div style="background:#FCE8E6;border-left:5px solid #D93025;padding:10px;border-radius:5px;text-align:center;"><span style="color:#D93025;font-weight:bold;font-size:0.95rem;">🔴 악재 뉴스</span><br><span style="font-size:1.3rem;font-weight:800;color:#1E1E23;">{bad}건</span></div>', unsafe_allow_html=True)
        col3.markdown(f'<div style="background:#F1F3F4;border-left:5px solid #5F6368;padding:10px;border-radius:5px;text-align:center;"><span style="color:#5F6368;font-weight:bold;font-size:0.95rem;">🟡 중립 뉴스</span><br><span style="font-size:1.3rem;font-weight:800;color:#1E1E23;">{neu}건</span></div>', unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

    # ── 기술적 분석 요약 카드 ──
    tech = st.session_state.get("tech_analysis", {})
    hist_a = st.session_state.get("history_analysis", {})
    if tech.get("signal") or hist_a.get("summary"):
        with st.expander("📊 기술적 분석 · 과거 패턴 요약", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                signal_color = {"강세": "#03C75A", "약세": "#D93025"}.get(tech.get("signal", ""), "#5F6368")
                st.markdown(f'<b>기술적 신호:</b> <span style="color:{signal_color};font-weight:800;">{tech.get("signal","—")}</span>', unsafe_allow_html=True)
                st.markdown(tech.get("summary", ""))
                for d in tech.get("details", []):
                    st.markdown(f"- {d}")
            with c2:
                risk_color = {"높음": "#D93025", "보통": "#FF9800", "낮음": "#03C75A"}.get(hist_a.get("risk_level", ""), "#5F6368")
                st.markdown(f'<b>과거 리스크:</b> <span style="color:{risk_color};font-weight:800;">{hist_a.get("risk_level","—")}</span>', unsafe_allow_html=True)
                st.markdown(hist_a.get("summary", ""))
                st.markdown(hist_a.get("seasonality_insight", ""))

    # ── Bull / Bear 논거 ──
    bull = st.session_state.get("bull", {})
    bear = st.session_state.get("bear", {})
    if bull.get("arguments") or bear.get("arguments"):
        with st.expander("📄 강세 vs 약세 논거", expanded=False):
            b1, b2 = st.columns(2)
            with b1:
                st.markdown(f'<b style="color:#03C75A;">강세 논거</b> <span style="font-size:0.8rem;color:#888;">(신뢰도: {bull.get("confidence","—")})</span>', unsafe_allow_html=True)
                for arg in bull.get("arguments", []):
                    st.markdown(f"- {arg}")
            with b2:
                st.markdown(f'<b style="color:#D93025;">약세 논거</b> <span style="font-size:0.8rem;color:#888;">(신뢰도: {bear.get("confidence","—")})</span>', unsafe_allow_html=True)
                for arg in bear.get("arguments", []):
                    st.markdown(f"- {arg}")

    # ── 대화 히스토리 ──
    for idx, message in enumerate(st.session_state["chat_history"]):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f'<div class="user-chat-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown(f'<div class="brief-container">{message["content"]}</div>', unsafe_allow_html=True)
                if idx == 1:
                    risks = st.session_state.get("risks", [])
                    if risks:
                        st.markdown("#### ⚠️ 투자자 유의 리스크 요인")
                        risk_html = "<div style='background:#FFF8E1;border-left:6px solid #FFB300;padding:1.2rem;border-radius:8px;color:#1E1E23;font-size:0.95rem;line-height:1.6;margin-bottom:1.5rem;'><ul style='margin:0;padding-left:20px;'>"
                        for r in risks:
                            risk_html += f"<li style='margin-bottom:6px;'><b>{r}</b></li>"
                        risk_html += "</ul></div>"
                        st.markdown(risk_html, unsafe_allow_html=True)

    # ── 뉴스 카드 ──
    if raw_news:
        st.markdown("---")
        st.markdown("### 📰 분석 대상 뉴스 목록")
        for news in raw_news:
            sentiment = news.get("sentiment", "중립")
            badge = {
                "호재": '<span style="background:#EBFBEE;color:#03C75A;font-weight:bold;padding:3px 8px;border-radius:4px;font-size:0.78rem;border:1px solid #03C75A;margin-right:8px;">🟢 호재</span>',
                "악재": '<span style="background:#FCE8E6;color:#D93025;font-weight:bold;padding:3px 8px;border-radius:4px;font-size:0.78rem;border:1px solid #D93025;margin-right:8px;">🔴 악재</span>',
            }.get(sentiment, '<span style="background:#F1F3F4;color:#5F6368;font-weight:bold;padding:3px 8px;border-radius:4px;font-size:0.78rem;border:1px solid #5F6368;margin-right:8px;">🟡 중립</span>')
            with st.container(border=True):
                st.markdown(f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;"><div>{badge}<span style="font-weight:800;font-size:1.05rem;color:#1E1E23;">{news["title"]}</span></div><div style="font-size:0.78rem;color:#888;">발행일: {news["pub_date"]}</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="margin-top:8px;font-size:0.92rem;line-height:1.5;color:#555;"><b>기사 요약:</b> {news["description"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div style="margin-top:8px;padding:8px 12px;background:#F8F9FA;border-radius:6px;font-size:0.9rem;color:#333;border-left:3px solid #D2D6DA;">💡 <b>AI 판단 근거:</b> {news["reason"]}</div>', unsafe_allow_html=True)
                if news.get("link"):
                    st.markdown(f'<div style="margin-top:10px;"><a href="{news["link"]}" target="_blank" style="text-decoration:none;"><button style="background:#FFF;color:#03C75A;border:1px solid #03C75A;border-radius:4px;padding:4px 10px;font-size:0.82rem;font-weight:bold;cursor:pointer;">🔗 원문 기사 읽기</button></a></div>', unsafe_allow_html=True)

        # ── RAG: 유사 뉴스 Top 3 ──
        if RAG_AVAILABLE:
            st.markdown("---")
            st.markdown("### 🔍 과거 유사 뉴스 Top 3 (RAG)")
            st.caption("현재 분석된 뉴스 내용과 벡터 유사도가 높은 과거 뉴스를 검색한 결과입니다.")
            try:
                # 쿼리: 현재 분석된 뉴스 제목 + 감성 + 근거를 합친 텍스트
                query_lines = []
                for n in raw_news[:5]:
                    query_lines.append(f"{n.get('title','')} / {n.get('sentiment','')} / {n.get('reason','')}")
                query_text = "\n".join(query_lines)

                similar = query_similar_news(
                    query_text=query_text,
                    n_results=3,
                    exclude_session_id=st.session_state.get("session_id", ""),
                )

                if not similar:
                    st.info("아직 비교할 과거 뉴스 데이터가 없습니다. 분석을 더 진행하면 유사 뉴스가 표시됩니다.")
                else:
                    sentiment_badge_map = {
                        "호재": "🟢 호재",
                        "악재": "🔴 악재",
                        "중립": "🟡 중립",
                    }
                    rows_html = ""
                    for rank, item in enumerate(similar, 1):
                        badge_text = sentiment_badge_map.get(item["sentiment"], "🟡 중립")
                        link_html = (
                            f'<a href="{item["link"]}" target="_blank" style="color:#03C75A;font-weight:700;text-decoration:none;">🔗 원문</a>'
                            if item.get("link") else "—"
                        )
                        similarity_pct = max(0, round((1 - item["distance"]) * 100, 1))
                        rows_html += f"""
<tr>
  <td style="text-align:center;font-weight:800;color:#03C75A;">{rank}</td>
  <td style="font-weight:700;">{item['title']}</td>
  <td style="text-align:center;">{badge_text}</td>
  <td style="font-size:0.85rem;color:#555;">{item['stock_name']}</td>
  <td style="font-size:0.82rem;color:#555;">{item['created_at'][:10] if item['created_at'] else '—'}</td>
  <td style="font-size:0.85rem;color:#03C75A;font-weight:700;text-align:center;">{similarity_pct}%</td>
  <td style="text-align:center;">{link_html}</td>
</tr>"""

                    table_html = f"""
<div style="overflow-x:auto;margin-top:0.5rem;">
<table style="width:100%;border-collapse:collapse;font-size:0.9rem;border-radius:10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.07);">
  <thead>
    <tr style="background:#03C75A;color:#fff;font-weight:700;">
      <th style="padding:10px 8px;text-align:center;width:40px;">#</th>
      <th style="padding:10px 12px;text-align:left;">뉴스 제목</th>
      <th style="padding:10px 8px;text-align:center;width:70px;">감성</th>
      <th style="padding:10px 8px;text-align:left;width:90px;">종목</th>
      <th style="padding:10px 8px;text-align:left;width:90px;">분석일</th>
      <th style="padding:10px 8px;text-align:center;width:70px;">유사도</th>
      <th style="padding:10px 8px;text-align:center;width:50px;">링크</th>
    </tr>
  </thead>
  <tbody style="background:#fff;">
    {rows_html}
  </tbody>
</table>
</div>"""
                    st.markdown(table_html, unsafe_allow_html=True)

            except Exception as e:
                st.warning(f"유사 뉴스 검색 중 오류: {e}")

    # ── 후속 질문 ──
    user_input = st.chat_input("궁금한 점을 질문해 보세요 (예: 악재 뉴스 기사명이 뭐야? 기술적 신호가 뭘 의미해?)")
    if user_input:
        with st.chat_message("user"):
            st.markdown(f'<div class="user-chat-bubble">{user_input}</div>', unsafe_allow_html=True)
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.spinner("생각 중..."):
            try:
                response = run_direct_gemini_chat(
                    user_question=user_input,
                    active_google_key=active_google_key,
                    model_name=GEMINI_MODEL,
                )
                st.session_state["chat_history"].append({"role": "assistant", "content": response})
                save_current_history(
                    st.session_state["session_id"],
                    st.session_state["current_stock"],
                    st.session_state["news_context"],
                    st.session_state["chat_history"],
                    news_context_raw=st.session_state["news_context_raw"],
                    risks=st.session_state["risks"],
                )
                st.rerun()
            except Exception as e:
                st.error(f"답변 중 오류 발생: {str(e)}")