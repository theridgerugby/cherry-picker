# paper_fetcher.py - fetch papers from arXiv with adaptive time windows

import datetime

import arxiv

from config import ARXIV_QUERY, DAYS_BACK, MAX_PAPERS


# Intent-based adaptive window config.
INTENT_CONFIG = {
    "latest": {"min_papers": 5, "start_days": 14, "max_days": 60, "step": 14},
    "recent": {"min_papers": 10, "start_days": 30, "max_days": 120, "step": 30},
    "landscape": {"min_papers": 20, "start_days": 90, "max_days": 365, "step": 90},
}


def _fetch_from_arxiv(query: str, days: int, max_results: int = 100) -> list[dict]:
    """
    Fetch arXiv papers submitted within the last `days` days.
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
        papers.append(
            {
                "arxiv_id": result.entry_id.split("/")[-1],
                "title": result.title,
                "authors": [str(a) for a in result.authors],
                "published_date": result.published.strftime("%Y-%m-%d"),
                "abstract": result.summary.replace("\n", " "),
                "url": result.entry_id,
            }
        )

    return papers


def _strip_category_filters(query: str) -> str:
    """
    Remove all cat: filter clauses from an arXiv query string.
    Used as fallback when category-filtered query returns 0 results.

    Example:
    Input:  '(ti:"anti-ice") AND (cat:cs.LG OR cat:eess.SP)'
    Output: '(ti:"anti-ice")'
    """
    import re

    # Remove AND (...cat:...) blocks
    query = re.sub(r"\s+AND\s+\([^)]*cat:[^)]+\)", "", query)
    # Remove OR (...cat:...) blocks
    query = re.sub(r"\s+OR\s+\([^)]*cat:[^)]+\)", "", query)
    # Remove standalone cat: terms
    query = re.sub(r"\s+AND\s+cat:\S+", "", query)
    query = re.sub(r"cat:\S+\s+AND\s+", "", query)
    # Clean up extra whitespace and hanging AND/OR
    query = re.sub(r"\s+(AND|OR)\s*$", "", query.strip())
    query = re.sub(r"^\s*(AND|OR)\s+", "", query.strip())
    return query.strip()


def fetch_papers_adaptive(
    arxiv_query: str,
    target_intent: str = "recent",
) -> dict:
    """
    Expand the search window adaptively until enough papers are found.

    Returns:
        {
          "papers":          list[dict],
          "days_used":       int,
          "target_met":      bool,
          "paper_count":     int,
          "window_expanded": bool,
          "query_used":      str,
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
        print(f"[Fetcher] Window {days}d -> {len(papers)} papers found")

        if len(papers) >= min_papers:
            break

        # If we've expanded to halfway and still 0 papers,
        # try stripping category filters once.
        if len(papers) == 0 and days >= start_days * 2:
            fallback_query = _strip_category_filters(arxiv_query)
            if fallback_query != arxiv_query:
                print("[Fetcher] Zero results with category filter. Trying without categories...")
                print(f"[Fetcher] Fallback query: {fallback_query}")
                fallback_papers = _fetch_from_arxiv(fallback_query, days)
                print(f"[Fetcher] Fallback -> {len(fallback_papers)} papers")
                if len(fallback_papers) > 0:
                    # Use fallback query for all subsequent expansions.
                    arxiv_query = fallback_query
                    papers = fallback_papers
                    if len(papers) >= min_papers:
                        break

        days += step
        expanded = True

    # Final check - may have exited loop without meeting target.
    target_met = len(papers) >= min_papers

    if not target_met:
        print(
            f"[Fetcher] WARNING: Only {len(papers)} paper(s) found after "
            f"expanding to {days}d (target: {min_papers})"
        )

    return {
        "papers": papers,
        "days_used": min(days, max_days),
        "target_met": target_met,
        "paper_count": len(papers),
        "window_expanded": expanded,
        "query_used": arxiv_query,
    }


def fetch_recent_papers() -> list[dict]:
    """
    Backward-compatible legacy interface using static config values.
    """
    papers = _fetch_from_arxiv(ARXIV_QUERY, DAYS_BACK, max_results=MAX_PAPERS)
    print(f"[Fetcher] Found {len(papers)} papers (last {DAYS_BACK} days)")
    return papers


if __name__ == "__main__":
    print("=== Adaptive fetch (recent intent) ===")
    result = fetch_papers_adaptive(ARXIV_QUERY, "recent")
    print(
        f"  days_used={result['days_used']}, "
        f"count={result['paper_count']}, "
        f"target_met={result['target_met']}, "
        f"expanded={result['window_expanded']}"
    )
    for p in result["papers"][:5]:
        print(f"  - [{p['published_date']}] {p['title']}")

    print("\n=== Legacy fetch ===")
    papers = fetch_recent_papers()
    for p in papers[:5]:
        print(f"  - [{p['published_date']}] {p['title']}")
