# paper_fetcher.py — 从 arXiv 拉取论文，支持自适应时间窗口

import arxiv
import datetime
from config import ARXIV_QUERY, DAYS_BACK, MAX_PAPERS


# ── 意图配置：控制不同搜索意图的时间窗口和最小论文数 ──────────────────────────

INTENT_CONFIG = {
    "latest":    {"min_papers": 5,  "start_days": 14,  "max_days": 60,  "step": 14},
    "recent":    {"min_papers": 10, "start_days": 30,  "max_days": 120, "step": 30},
    "landscape": {"min_papers": 20, "start_days": 90,  "max_days": 365, "step": 90},
}


# ── 核心抓取函数 ──────────────────────────────────────────────────────────────

def _fetch_from_arxiv(query: str, days: int, max_results: int = 100) -> list[dict]:
    """
    从 arXiv 拉取指定时间窗口内的论文。
    """
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers = []
    for result in client.results(search):
        if result.published < cutoff:
            continue
        papers.append({
            "arxiv_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "authors": [str(a) for a in result.authors],
            "published_date": result.published.strftime("%Y-%m-%d"),
            "abstract": result.summary.replace("\n", " "),
            "url": result.entry_id,
        })

    return papers


# ── 自适应抓取 ────────────────────────────────────────────────────────────────

def fetch_papers_adaptive(
    arxiv_query: str,
    target_intent: str = "recent",
) -> dict:
    """
    根据搜索意图自适应扩展时间窗口，直到满足最小论文数或达到最大窗口。

    Args:
        arxiv_query:    arXiv 搜索查询字符串
        target_intent:  "latest" | "recent" | "landscape"

    Returns:
        {
          "papers":          list[dict],
          "days_used":       int,
          "target_met":      bool,
          "paper_count":     int,
          "window_expanded": bool,
        }
    """
    config = INTENT_CONFIG.get(target_intent, INTENT_CONFIG["recent"])
    min_papers = config["min_papers"]
    start_days = config["start_days"]
    max_days = config["max_days"]
    step = config["step"]

    days = start_days
    papers = []
    expanded = False

    while days <= max_days:
        papers = _fetch_from_arxiv(arxiv_query, days)
        print(f"[Fetcher] Window {days}d → {len(papers)} papers found")

        if len(papers) >= min_papers:
            break

        days += step
        expanded = True

    # Final check — may have exited the loop without meeting target
    target_met = len(papers) >= min_papers

    if not target_met:
        print(
            f"[Fetcher] ⚠️ Only {len(papers)} paper(s) found after "
            f"expanding to {days}d (target: {min_papers})"
        )

    return {
        "papers": papers,
        "days_used": min(days, max_days),
        "target_met": target_met,
        "paper_count": len(papers),
        "window_expanded": expanded,
    }


# ── 旧版兼容接口 ─────────────────────────────────────────────────────────────

def fetch_recent_papers() -> list[dict]:
    """
    向后兼容的旧接口。使用 config.py 中的 ARXIV_QUERY 和 DAYS_BACK。
    """
    papers = _fetch_from_arxiv(ARXIV_QUERY, DAYS_BACK, max_results=MAX_PAPERS)
    print(f"[Fetcher] 找到 {len(papers)} 篇论文（最近 {DAYS_BACK} 天）")
    return papers


if __name__ == "__main__":
    # 测试自适应抓取
    print("=== Adaptive fetch (recent intent) ===")
    result = fetch_papers_adaptive(ARXIV_QUERY, "recent")
    print(f"  days_used={result['days_used']}, "
          f"count={result['paper_count']}, "
          f"target_met={result['target_met']}, "
          f"expanded={result['window_expanded']}")
    for p in result["papers"][:5]:
        print(f"  - [{p['published_date']}] {p['title']}")

    print(f"\n=== Legacy fetch ===")
    papers = fetch_recent_papers()
    for p in papers[:5]:
        print(f"  - [{p['published_date']}] {p['title']}")
