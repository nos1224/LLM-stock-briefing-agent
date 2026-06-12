import argparse
import html
import logging
import os
import re
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from dotenv import load_dotenv

import ai_agent
from ai_agent import analyze_risks, analyze_sentiments, generate_briefing
from news_agent import collect_news, filter_news


DEFAULT_STOCK_NAME = "삼성전자"
OLLAMA_TIMEOUT = 900
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "daily_report.log"


def setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("daily_stock_report")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def count_sentiments(news_list: list) -> dict:
    return {
        "good": sum(1 for item in news_list if item.get("sentiment") == "호재"),
        "bad": sum(1 for item in news_list if item.get("sentiment") == "악재"),
        "neutral": sum(1 for item in news_list if item.get("sentiment") == "중립"),
    }


def strip_markdown_code_fences(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"```(?:markdown|md)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    return text.strip()


def remove_non_korean_sentences(text: str) -> str:
    if not text:
        return ""

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", stripped):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def normalize_ai_text(text: str) -> str:
    text = strip_markdown_code_fences(text)
    text = text.replace("DISCLAIMER", "면책조항")
    text = remove_non_korean_sentences(text)
    return text.strip()


def clean_inline_markdown(text: str) -> str:
    text = strip_markdown_code_fences(text)
    text = re.sub(r"^#{1,6}\s*", "", text.strip())
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text.strip()


def clean_analysis_report(report: dict) -> dict:
    report["briefing"] = normalize_ai_text(report.get("briefing", ""))
    report["risks"] = [normalize_ai_text(str(risk)) for risk in report.get("risks", []) if normalize_ai_text(str(risk))]

    cleaned_news = []
    for news in report.get("analyzed_news", []):
        news_copy = news.copy()
        news_copy["reason"] = normalize_ai_text(news_copy.get("reason", ""))
        cleaned_news.append(news_copy)
    report["analyzed_news"] = cleaned_news
    report["sentiment_counts"] = count_sentiments(cleaned_news)
    return report


def run_stock_analysis(stock_name: str, logger: logging.Logger) -> dict:
    ai_agent.OLLAMA_TIMEOUT = OLLAMA_TIMEOUT
    logger.info("analysis_start | stock=%s | ollama_timeout_seconds=%s", stock_name, OLLAMA_TIMEOUT)

    logger.info("news_collect_start | stock=%s", stock_name)
    collected_news = collect_news(stock_name)
    logger.info("news_collect_complete | stock=%s | collected=%s", stock_name, len(collected_news))
    if not collected_news:
        logger.warning("news_collect_empty | stock=%s", stock_name)
        return {
            "success": False,
            "stock_name": stock_name,
            "error": "뉴스가 없거나 네이버 API 설정/응답에 문제가 있습니다.",
            "analyzed_news": [],
            "risks": [],
            "briefing": "",
            "sentiment_counts": {"good": 0, "bad": 0, "neutral": 0},
        }

    logger.info("news_filter_start | stock=%s | collected=%s", stock_name, len(collected_news))
    filtered_news = filter_news(collected_news)
    logger.info(
        "news_filter_complete | stock=%s | collected=%s | filtered=%s",
        stock_name,
        len(collected_news),
        len(filtered_news),
    )
    if not filtered_news:
        logger.warning("news_filter_empty | stock=%s | collected=%s", stock_name, len(collected_news))
        return {
            "success": False,
            "stock_name": stock_name,
            "error": "필터링 후 분석 가능한 뉴스가 없습니다.",
            "analyzed_news": [],
            "risks": [],
            "briefing": "",
            "sentiment_counts": {"good": 0, "bad": 0, "neutral": 0},
        }

    try:
        logger.info(
            "sentiment_start | stock=%s | news=%s | ollama_timeout_seconds=%s",
            stock_name,
            len(filtered_news),
            OLLAMA_TIMEOUT,
        )
        analyzed_news = analyze_sentiments(filtered_news, os.environ.get("GOOGLE_API_KEY", "").strip())
        logger.info("sentiment_complete | stock=%s | news=%s", stock_name, len(analyzed_news))
    except Exception as exc:
        logger.exception("sentiment_failed | stock=%s | error=%s", stock_name, exc)
        analyzed_news = []
        for item in filtered_news:
            fallback_item = item.copy()
            fallback_item["sentiment"] = "중립"
            fallback_item["reason"] = f"감성 분석 오류: {exc}"
            analyzed_news.append(fallback_item)

    try:
        logger.info(
            "risk_start | stock=%s | news=%s | ollama_timeout_seconds=%s",
            stock_name,
            len(analyzed_news),
            OLLAMA_TIMEOUT,
        )
        risks = analyze_risks(analyzed_news, os.environ.get("GOOGLE_API_KEY", "").strip())
        logger.info("risk_complete | stock=%s | risks=%s", stock_name, len(risks))
    except Exception as exc:
        logger.exception("risk_failed | stock=%s | error=%s", stock_name, exc)
        risks = [f"리스크 분석 오류: {exc}"]

    try:
        logger.info(
            "briefing_start | stock=%s | news=%s | risks=%s | ollama_timeout_seconds=%s",
            stock_name,
            len(analyzed_news),
            len(risks),
            OLLAMA_TIMEOUT,
        )
        briefing = generate_briefing(
            analyzed_news,
            risks,
            stock_name,
            os.environ.get("GOOGLE_API_KEY", "").strip(),
        )
        logger.info("briefing_complete | stock=%s | briefing_chars=%s", stock_name, len(briefing))
    except Exception as exc:
        logger.exception("briefing_failed | stock=%s | error=%s", stock_name, exc)
        briefing = f"브리핑 생성 오류: {exc}"

    return clean_analysis_report({
        "success": True,
        "stock_name": stock_name,
        "error": "",
        "analyzed_news": analyzed_news,
        "risks": risks,
        "briefing": briefing,
        "sentiment_counts": count_sentiments(analyzed_news),
    })


def sentiment_badge(sentiment: str) -> str:
    palette = {
        "호재": ("#DCFCE7", "#15803D", "#22C55E"),
        "악재": ("#FEE2E2", "#B91C1C", "#EF4444"),
        "중립": ("#FEF3C7", "#92400E", "#F59E0B"),
    }
    background, color, border = palette.get(sentiment, palette["중립"])
    return (
        f'<span style="display:inline-block;padding:3px 9px;border-radius:999px;'
        f'background:{background};color:{color};border:1px solid {border};'
        f'font-size:12px;font-weight:700;">{html.escape(sentiment)}</span>'
    )


def build_investment_perspective_text(report: dict) -> str:
    counts = report["sentiment_counts"]
    total = counts["good"] + counts["bad"] + counts["neutral"]

    if total == 0:
        perspective = "분석 가능한 뉴스가 부족하므로 추가 정보 확인이 필요합니다."
    elif counts["bad"] > counts["good"]:
        perspective = "부정적 뉴스 비중이 상대적으로 높아 단기 변동성과 리스크 점검이 우선입니다."
    elif counts["good"] > counts["bad"] and counts["bad"] == 0:
        perspective = "긍정적 뉴스 흐름이 우세하지만, 실제 투자 판단은 실적과 시장 상황을 함께 확인해야 합니다."
    else:
        perspective = "긍정 요인과 리스크가 함께 존재하므로 추세 확인과 분할 관찰이 적절합니다."

    first_risk = report.get("risks", [""])[0] if report.get("risks") else ""
    risk_sentence = f"핵심 점검 요인: {clean_inline_markdown(first_risk)}" if first_risk else "핵심 점검 요인: 추가 리스크 확인이 필요합니다."
    return f"{perspective}\n{risk_sentence}"


def build_investment_perspective_html(report: dict) -> str:
    perspective_text = html.escape(build_investment_perspective_text(report)).replace("\n", "<br>")

    return f"""
    <div style="background:#FFFFFF;border-radius:12px;padding:20px;margin-top:18px;border:1px solid #E2E8F0;">
        <h2 style="font-size:18px;margin:0 0 12px;">종합 투자 관점</h2>
        <div style="font-size:14px;line-height:1.7;color:#334155;">
            {perspective_text}
        </div>
    </div>
    """


def build_briefing_cards_html(briefing: str) -> str:
    briefing = normalize_ai_text(briefing)
    if not briefing:
        return '<div style="font-size:14px;line-height:1.75;color:#334155;">브리핑 내용이 없습니다.</div>'

    sections = []
    current_title = "핵심 브리핑"
    current_lines = []

    for raw_line in briefing.splitlines():
        line = raw_line.strip()
        if not line or line == "---":
            continue

        is_heading = line.startswith("#")
        cleaned = clean_inline_markdown(line)
        if not cleaned:
            continue

        if is_heading:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = cleaned
            current_lines = []
        else:
            current_lines.append(cleaned)

    if current_lines:
        sections.append((current_title, current_lines))
    if not sections:
        sections.append(("핵심 브리핑", [clean_inline_markdown(briefing)]))

    cards = []
    for title, lines in sections:
        paragraphs = []
        bullets = []
        for line in lines:
            if line.startswith("- "):
                bullets.append(f"<li>{html.escape(line[2:].strip())}</li>")
            else:
                paragraphs.append(f"<p style='margin:0 0 10px;'>{html.escape(line)}</p>")

        body = "".join(paragraphs)
        if bullets:
            body += "<ul style='margin:0;padding-left:20px;line-height:1.75;'>" + "".join(bullets) + "</ul>"

        cards.append(
            f"""
            <div style="border:1px solid #E2E8F0;border-radius:10px;padding:16px;margin-top:12px;background:#F8FAFC;">
                <div style="font-size:16px;font-weight:900;color:#0F172A;margin-bottom:10px;">{html.escape(title)}</div>
                <div style="font-size:14px;line-height:1.75;color:#334155;">{body}</div>
            </div>
            """
        )

    return "".join(cards)


def build_email_html(report: dict, executed_at: datetime) -> str:
    stock_name = html.escape(report["stock_name"])
    counts = report["sentiment_counts"]
    briefing_cards = build_briefing_cards_html(report.get("briefing") or "")
    investment_perspective = build_investment_perspective_html(report)
    error_message = html.escape(report.get("error", ""))

    risks_html = "".join(
        f"<li>{html.escape(str(risk))}</li>" for risk in report.get("risks", [])
    ) or "<li>확인된 리스크 요인이 없습니다.</li>"

    news_cards = []
    for index, news in enumerate(report.get("analyzed_news", []), 1):
        title = html.escape(news.get("title", "제목 없음"))
        reason = html.escape(news.get("reason", "분석 근거 없음")).replace("\n", "<br>")
        link = html.escape(news.get("link", ""))
        sentiment = news.get("sentiment", "중립")
        link_html = (
            f'<a href="{link}" style="color:#2563EB;text-decoration:none;font-weight:700;">원문 링크</a>'
            if link
            else '<span style="color:#64748B;">원문 링크 없음</span>'
        )
        news_cards.append(
            f"""
            <div style="border:1px solid #E2E8F0;border-radius:10px;padding:16px;margin-top:12px;background:#FFFFFF;">
                <div style="font-size:13px;color:#64748B;margin-bottom:8px;">NEWS {index}</div>
                <div style="font-size:16px;font-weight:800;color:#0F172A;margin-bottom:10px;">{title}</div>
                <div style="margin-bottom:10px;">{sentiment_badge(sentiment)}</div>
                <div style="font-size:14px;line-height:1.6;color:#334155;margin-bottom:10px;">
                    <b>분석 근거:</b> {reason}
                </div>
                {link_html}
            </div>
            """
        )

    if not news_cards:
        news_cards.append(
            """
            <div style="border:1px solid #E2E8F0;border-radius:10px;padding:16px;margin-top:12px;background:#FFFFFF;color:#64748B;">
                분석 가능한 뉴스가 없습니다.
            </div>
            """
        )

    status_block = ""
    if not report.get("success"):
        status_block = f"""
        <div style="background:#FEF2F2;border-left:5px solid #EF4444;border-radius:8px;padding:14px;margin:18px 0;color:#991B1B;">
            <b>분석 알림:</b> {error_message}
        </div>
        """

    return f"""
    <!doctype html>
    <html>
    <body style="margin:0;padding:0;background:#F1F5F9;font-family:Arial,'Malgun Gothic',sans-serif;color:#0F172A;">
        <div style="max-width:760px;margin:0 auto;padding:28px 18px;">
            <div style="background:#07111F;color:#FFFFFF;border-radius:14px;padding:26px 28px;box-shadow:0 14px 34px rgba(15,23,42,0.18);">
                <div style="font-size:13px;color:#94A3B8;font-weight:700;">Stock-Agent Daily Briefing</div>
                <h1 style="margin:10px 0 8px;font-size:26px;line-height:1.3;">{stock_name} 데일리 AI 브리핑</h1>
                <div style="font-size:14px;color:#CBD5E1;">분석 실행 시간: {executed_at.strftime("%Y-%m-%d %H:%M:%S")}</div>
            </div>

            {status_block}

            <div style="display:block;background:#FFFFFF;border-radius:12px;padding:18px;margin-top:18px;border:1px solid #E2E8F0;">
                <h2 style="font-size:18px;margin:0 0 14px;">뉴스 감성 요약</h2>
                <div style="display:flex;gap:10px;flex-wrap:wrap;">
                    <div style="flex:1;min-width:150px;background:#F0FDF4;border-left:5px solid #22C55E;border-radius:8px;padding:14px;">
                        <div style="font-size:13px;color:#15803D;font-weight:800;">호재</div>
                        <div style="font-size:28px;font-weight:900;color:#14532D;">{counts["good"]}</div>
                    </div>
                    <div style="flex:1;min-width:150px;background:#FEF2F2;border-left:5px solid #EF4444;border-radius:8px;padding:14px;">
                        <div style="font-size:13px;color:#B91C1C;font-weight:800;">악재</div>
                        <div style="font-size:28px;font-weight:900;color:#7F1D1D;">{counts["bad"]}</div>
                    </div>
                    <div style="flex:1;min-width:150px;background:#FFFBEB;border-left:5px solid #F59E0B;border-radius:8px;padding:14px;">
                        <div style="font-size:13px;color:#92400E;font-weight:800;">중립</div>
                        <div style="font-size:28px;font-weight:900;color:#78350F;">{counts["neutral"]}</div>
                    </div>
                </div>
            </div>

            {investment_perspective}

            <div style="background:#FFFFFF;border-radius:12px;padding:20px;margin-top:18px;border:1px solid #E2E8F0;">
                <h2 style="font-size:18px;margin:0 0 12px;">최종 AI 브리핑</h2>
                {briefing_cards}
            </div>

            <div style="background:#FFFFFF;border-radius:12px;padding:20px;margin-top:18px;border:1px solid #E2E8F0;">
                <h2 style="font-size:18px;margin:0 0 12px;">주요 리스크 요인</h2>
                <ul style="margin:0;padding-left:20px;font-size:14px;line-height:1.7;color:#334155;">{risks_html}</ul>
            </div>

            <div style="margin-top:18px;">
                <h2 style="font-size:18px;margin:0 0 12px;">주요 뉴스 목록</h2>
                {''.join(news_cards)}
            </div>

            <div style="font-size:12px;line-height:1.6;color:#64748B;margin-top:22px;padding:14px;border-top:1px solid #CBD5E1;">
                본 메일은 수집된 뉴스와 AI 분석을 기반으로 한 참고 정보이며, 특정 종목에 대한 투자 권유 또는 추천이 아닙니다.
                모든 투자 판단과 그 결과에 대한 책임은 투자자 본인에게 있습니다.
            </div>
        </div>
    </body>
    </html>
    """


def build_plain_text(report: dict, executed_at: datetime) -> str:
    lines = [
        f"[Stock-Agent] {report['stock_name']} 데일리 AI 브리핑",
        f"분석 실행 시간: {executed_at.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if report.get("error"):
        lines.extend(["분석 알림:", report["error"], ""])

    counts = report["sentiment_counts"]
    lines.extend(
        [
            f"감성 요약: 호재 {counts['good']}건 / 악재 {counts['bad']}건 / 중립 {counts['neutral']}건",
            "",
            "종합 투자 관점:",
            build_investment_perspective_text(report),
            "",
            "최종 AI 브리핑:",
            normalize_ai_text(report.get("briefing") or "브리핑 내용이 없습니다."),
            "",
            "주요 리스크 요인:",
        ]
    )
    lines.extend(f"- {risk}" for risk in report.get("risks", []))
    lines.extend(["", "주요 뉴스 목록:"])

    for index, news in enumerate(report.get("analyzed_news", []), 1):
        lines.extend(
            [
                f"{index}. {news.get('title', '제목 없음')}",
                f"   감성: {news.get('sentiment', '중립')}",
                f"   근거: {news.get('reason', '분석 근거 없음')}",
                f"   링크: {news.get('link', '')}",
            ]
        )

    lines.extend(
        [
            "",
            "면책 문구: 본 메일은 참고 정보이며 투자 권유 또는 추천이 아닙니다. 모든 투자 판단과 결과의 책임은 투자자 본인에게 있습니다.",
        ]
    )
    return "\n".join(lines)


def load_email_config() -> dict:
    return {
        "host": os.environ.get("EMAIL_HOST", "smtp.gmail.com").strip(),
        "port": int(os.environ.get("EMAIL_PORT", "587").strip()),
        "user": os.environ.get("EMAIL_USER", "").strip(),
        "password": os.environ.get("EMAIL_PASSWORD", "").strip(),
        "to": os.environ.get("EMAIL_TO", "").strip(),
    }


def send_email_report(report: dict, logger: logging.Logger) -> bool:
    config = load_email_config()
    missing = [key for key in ["user", "password", "to"] if not config[key]]
    if missing:
        logger.error("email_config_missing | fields=%s", ",".join(missing))
        return False

    executed_at = datetime.now()
    subject = f"[Stock-Agent] {report['stock_name']} 데일리 AI 브리핑"

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config["user"]
    message["To"] = config["to"]
    message.set_content(build_plain_text(report, executed_at))
    message.add_alternative(build_email_html(report, executed_at), subtype="html")

    try:
        with smtplib.SMTP(config["host"], config["port"], timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(config["user"], config["password"])
            smtp.send_message(message)
        logger.info("email_sent | stock=%s | to=%s", report["stock_name"], config["to"])
        return True
    except Exception as exc:
        logger.exception("email_send_failed | stock=%s | error=%s", report["stock_name"], exc)
        return False


def run_daily_report(stock_name: str = DEFAULT_STOCK_NAME) -> bool:
    load_dotenv()
    logger = setup_logger()
    started_at = datetime.now()
    logger.info("daily_report_start | stock=%s | started_at=%s", stock_name, started_at.isoformat())

    try:
        report = run_stock_analysis(stock_name, logger)
    except Exception as exc:
        logger.exception("daily_report_unhandled_analysis_error | stock=%s | error=%s", stock_name, exc)
        report = {
            "success": False,
            "stock_name": stock_name,
            "error": f"분석 중 예기치 못한 오류가 발생했습니다: {exc}",
            "analyzed_news": [],
            "risks": [],
            "briefing": "",
            "sentiment_counts": {"good": 0, "bad": 0, "neutral": 0},
        }

    email_sent = send_email_report(report, logger)
    elapsed = (datetime.now() - started_at).total_seconds()
    logger.info(
        "daily_report_finish | stock=%s | analysis_success=%s | email_sent=%s | elapsed_seconds=%.2f",
        stock_name,
        report.get("success"),
        email_sent,
        elapsed,
    )
    return bool(report.get("success") and email_sent)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a daily Stock-Agent AI briefing by email.")
    parser.add_argument(
        "--stock",
        default=DEFAULT_STOCK_NAME,
        help=f"분석할 종목명입니다. 기본값: {DEFAULT_STOCK_NAME}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_daily_report(args.stock)
