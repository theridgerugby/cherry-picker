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
    cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)

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
                "pdf_url": getattr(result, "pdf_url", "") or "",
                "comment": getattr(result, "comment", "") or "",
                "primary_category": getattr(result, "primary_category", "") or "",
                "categories": list(getattr(result, "categories", []) or []),
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


def _split_top_level_and(query: str) -> list[str]:
    """
    Split query by top-level AND (outside parentheses and quotes).
    """
    parts = []
    buf = []
    depth = 0
    in_quote = False
    i = 0
    while i < len(query):
        ch = query[i]
        if ch == '"':
            in_quote = not in_quote
            buf.append(ch)
            i += 1
            continue

        if not in_quote:
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1

            if depth == 0 and query[i : i + 5] == " AND ":
                part = "".join(buf).strip()
                if part:
                    parts.append(part)
                buf = []
                i += 5
                continue

        buf.append(ch)
        i += 1

    last = "".join(buf).strip()
    if last:
        parts.append(last)
    return parts


def _build_recall_fallback_queries(query: str) -> list[str]:
    """
    Build progressively broader fallback queries when zero papers are returned.
    Order: minimally relaxed -> broadest.
    """
    candidates = []

    stripped = _strip_category_filters(query)
    if stripped and stripped != query:
        candidates.append(stripped)

    parts = _split_top_level_and(stripped or query)
    non_cat_parts = [p for p in parts if "cat:" not in p.lower()]

    # Keep first two semantic clauses if available.
    if len(non_cat_parts) >= 2:
        first_two = " AND ".join(non_cat_parts[:2]).strip()
        if first_two and first_two not in candidates:
            candidates.append(first_two)

    # Broadest fallback: keep only the core first clause.
    if non_cat_parts:
        core = non_cat_parts[0].strip()
        if core and core not in candidates:
            candidates.append(core)

    return candidates


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
    fallback_attempted = False

    while days <= max_days:
        papers = _fetch_from_arxiv(arxiv_query, days)
        print(f"[Fetcher] Window {days}d -> {len(papers)} papers found")

        if len(papers) >= min_papers:
            break

        # If we've expanded to halfway and still 0 papers,
        # run a one-time progressive fallback sequence.
        if len(papers) == 0 and days >= start_days * 2 and not fallback_attempted:
            fallback_attempted = True
            fallback_candidates = _build_recall_fallback_queries(arxiv_query)
            for idx, fallback_query in enumerate(fallback_candidates, start=1):
                if fallback_query == arxiv_query:
                    continue
                print(f"[Fetcher] Fallback #{idx}: {fallback_query}")
                fallback_papers = _fetch_from_arxiv(fallback_query, days)
                print(f"[Fetcher] Fallback #{idx} -> {len(fallback_papers)} papers")
                if len(fallback_papers) > 0:
                    # Use successful fallback query for subsequent expansions.
                    arxiv_query = fallback_query
                    papers = fallback_papers
                    if len(papers) >= min_papers:
                        break
            if len(papers) >= min_papers:
                break

        days += step
        expanded = True

    # Final check - may have exited loop without meeting target.
    target_met = len(papers) >= min_papers

    # If still below target (even with non-zero hits), try one recall-oriented
    # fallback pass at the final window. This fixes narrow category filters that
    # under-recall domains like biology/ocean acoustics.
    if not target_met:
        final_days = min(days, max_days)
        best_query = arxiv_query
        best_papers = papers
        fallback_candidates = _build_recall_fallback_queries(arxiv_query)
        for idx, fallback_query in enumerate(fallback_candidates, start=1):
            if fallback_query == arxiv_query:
                continue
            print(f"[Fetcher] Final fallback #{idx}: {fallback_query}")
            fallback_papers = _fetch_from_arxiv(fallback_query, final_days)
            print(f"[Fetcher] Final fallback #{idx} -> {len(fallback_papers)} papers")
            if len(fallback_papers) > len(best_papers):
                best_query = fallback_query
                best_papers = fallback_papers
            if len(fallback_papers) >= min_papers:
                break

        if best_query != arxiv_query:
            arxiv_query = best_query
            papers = best_papers
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
