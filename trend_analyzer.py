# trend_analyzer.py — 行业趋势分析模块

import json
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import GEMINI_MODEL, DOMAIN, CREDIBILITY_THRESHOLD, TREND_TOP_N, MIN_PAPERS_FOR_TREND_ANALYSIS


# ── Gemini 趋势分析 Prompt ────────────────────────────────────────────────────

TREND_SYSTEM_PROMPT = """You are a research trend analyst. Given a curated set of high-credibility academic papers
on a specific domain, identify macro and micro trends.

Output ONLY valid JSON with this exact schema:
{
  "domain": "string",
  "analysis_date": "YYYY-MM-DD",
  "macro_trends": [
    {
      "trend_name": "string",
      "description": "string (2-3 sentences)",
      "evidence_papers": ["paper title 1", "paper title 2"],
      "trajectory": "emerging | growing | peaking | declining",
      "confidence": "high | medium | low"
    }
  ],
  "micro_trends": [
    {
      "trend_name": "string",
      "description": "string (1-2 sentences)",
      "evidence_papers": ["paper title"],
      "trajectory": "emerging | growing | peaking | declining"
    }
  ],
  "declining_approaches": [
    {
      "approach": "string",
      "reason": "string",
      "replaced_by": "string or null"
    }
  ],
  "future_directions": [
    {
      "direction": "string",
      "rationale": "string (based only on evidence in provided papers)",
      "confidence": "high | medium | low"
    }
  ],
  "consensus_gaps": ["string"]
}

Rules:
- Base ALL claims on the provided paper summaries only.
- Do not fabricate paper titles or findings.
- If insufficient data for a section, return an empty list for that field.
- Ensure output is valid JSON with no markdown fences or extra text."""

TREND_USER_TEMPLATE = """Domain: {domain}
Analysis Date: {date}
Number of papers analyzed: {count}

Paper Summaries (sorted by credibility score, descending):
{paper_summaries}

Analyze the trends now."""


def analyze_trends(
    papers: list[dict],
    llm: ChatGoogleGenerativeAI = None,
    domain_name: str = DOMAIN,
    top_n: int = TREND_TOP_N,
    threshold: int = CREDIBILITY_THRESHOLD,
) -> dict:
    """
    对已评分的论文进行趋势分析。

    参数:
        papers: 包含 credibility_score 的论文列表
        llm: LangChain LLM 实例（为空则自动创建）
        top_n: 取前 N 篇高分论文
        threshold: 最低可信度分数阈值

    返回:
        趋势分析 JSON dict
    """
    if llm is None:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.2)

    # 过滤 + 排序
    scored = [p for p in papers if p.get("credibility_score", 0) >= threshold]
    scored.sort(key=lambda p: p.get("credibility_score", 0), reverse=True)
    top_papers = scored[:top_n]

    if not top_papers or len(top_papers) < MIN_PAPERS_FOR_TREND_ANALYSIS:
        available_titles = [p.get("title", "") for p in top_papers]
        return {
            "insufficient_data": True,
            "paper_count": len(top_papers),
            "minimum_required": MIN_PAPERS_FOR_TREND_ANALYSIS,
            "available_papers": available_titles,
            "suggestion": (
                "Extend EXTENDED_DAYS_BACK or broaden ARXIV_QUERY in config.py "
                "to gather more high-credibility papers."
            ),
            # keep these keys so callers that don't check insufficient_data still work
            "domain": domain_name,
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "macro_trends": [],
            "micro_trends": [],
            "declining_approaches": [],
            "future_directions": [],
            "consensus_gaps": [],
        }

    # 精简论文数据，节省 token
    slim = []
    for p in top_papers:
        slim.append({
            "title": p.get("title"),
            "published_date": p.get("published_date"),
            "sub_domain": p.get("sub_domain"),
            "problem_statement": p.get("problem_statement"),
            "method_summary": p.get("method_summary"),
            "contributions": p.get("contributions"),
            "method_keywords": p.get("method_keywords"),
            "credibility_score": p.get("credibility_score"),
            "venue_detected": p.get("venue_detected"),
        })

    paper_summaries_str = json.dumps(slim, ensure_ascii=False, indent=2)
    today = datetime.now().strftime("%Y-%m-%d")

    user_prompt = TREND_USER_TEMPLATE.format(
        domain=domain_name,
        date=today,
        count=len(top_papers),
        paper_summaries=paper_summaries_str,
    )

    print(f"[Trend] 正在分析 {len(top_papers)} 篇高可信度论文的趋势...")

    response = llm.invoke([
        SystemMessage(content=TREND_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    raw = response.content.strip()
    # 防御性清洗
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        trend_data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[Trend] JSON 解析失败，返回原始文本")
        trend_data = {
            "domain": DOMAIN,
            "analysis_date": today,
            "macro_trends": [],
            "micro_trends": [],
            "declining_approaches": [],
            "future_directions": [],
            "consensus_gaps": [f"Analysis failed to parse: {raw[:200]}"],
        }

    return trend_data


# ── Markdown 渲染 ─────────────────────────────────────────────────────────────

_TRAJECTORY_EMOJI = {
    "emerging":  "🌱",
    "growing":   "📈",
    "peaking":   "🔝",
    "declining": "📉",
}


def render_trends_markdown(trend_data: dict) -> str:
    """
    将趋势分析 JSON 渲染为可读的 Markdown 章节。
    """
    lines = []
    lines.append("## 0. Industry Trend Analysis")
    lines.append(f"*Domain: {trend_data.get('domain', 'N/A')} | "
                 f"Analysis Date: {trend_data.get('analysis_date', 'N/A')}*")
    lines.append("")

    # ── Macro Trends ──
    macro = trend_data.get("macro_trends", [])
    if macro:
        lines.append("### Macro Trends")
        lines.append("")
        for t in macro:
            emoji = _TRAJECTORY_EMOJI.get(t.get("trajectory", ""), "❓")
            confidence = t.get("confidence", "unknown")
            lines.append(f"#### {emoji} {t.get('trend_name', 'Unnamed Trend')} "
                         f"({t.get('trajectory', 'unknown')}, confidence: {confidence})")
            lines.append("")
            lines.append(t.get("description", ""))
            lines.append("")
            evidence = t.get("evidence_papers", [])
            if evidence:
                lines.append("**Evidence papers:**")
                for paper in evidence:
                    lines.append(f"- {paper}")
                lines.append("")

    # ── Micro Trends ──
    micro = trend_data.get("micro_trends", [])
    if micro:
        lines.append("### Micro Trends")
        lines.append("")
        for t in micro:
            emoji = _TRAJECTORY_EMOJI.get(t.get("trajectory", ""), "❓")
            lines.append(f"- {emoji} **{t.get('trend_name', '')}** "
                         f"({t.get('trajectory', '')}): {t.get('description', '')}")
        lines.append("")

    # ── Declining Approaches ──
    declining = trend_data.get("declining_approaches", [])
    if declining:
        lines.append("### Declining Approaches")
        lines.append("")
        for d in declining:
            replaced = d.get("replaced_by")
            replaced_str = f" → Replaced by: **{replaced}**" if replaced else ""
            lines.append(f"- 📉 **{d.get('approach', '')}**: "
                         f"{d.get('reason', '')}{replaced_str}")
        lines.append("")

    # ── Future Directions ──
    futures = trend_data.get("future_directions", [])
    if futures:
        lines.append("### Future Directions")
        lines.append("")
        for f_item in futures:
            confidence = f_item.get("confidence", "unknown")
            lines.append(f"- **{f_item.get('direction', '')}** "
                         f"(confidence: {confidence})")
            lines.append(f"  - {f_item.get('rationale', '')}")
        lines.append("")

    # ── Consensus Gaps ──
    gaps = trend_data.get("consensus_gaps", [])
    if gaps:
        lines.append("### Open Consensus Gaps")
        lines.append("")
        for gap in gaps:
            lines.append(f"- {gap}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # 快速测试：用 mock 数据
    mock_trend = {
        "domain": "Sparse Representation",
        "analysis_date": "2026-02-21",
        "macro_trends": [
            {
                "trend_name": "Multimodal Sparse Learning",
                "description": "Growing interest in unified sparse representations across vision and language.",
                "evidence_papers": ["LUCID-SAE"],
                "trajectory": "emerging",
                "confidence": "medium",
            }
        ],
        "micro_trends": [],
        "declining_approaches": [],
        "future_directions": [
            {
                "direction": "Cross-modal dictionary learning",
                "rationale": "Several papers explore shared sparse codes across modalities.",
                "confidence": "medium",
            }
        ],
        "consensus_gaps": ["Scalability of sparse autoencoders to large datasets"],
    }
    print(render_trends_markdown(mock_trend))
