"""Cached parallel orchestration for report/gaps/matrix agents."""

from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor

from agent_prompts import get_gaps_prompt, get_matrix_prompt, get_report_prompt
from cache_manager import build_shared_cache, delete_cache, make_model_from_cache
from config import DAYS_BACK, DOMAIN, GEMINI_MODEL_FAST
from report_generator import (
    GAPS_USER_TEMPLATE,
    REPORT_USER_TEMPLATE,
    _clean_json_response,
    _make_llm,
    generate_extrapolated_gaps,
    generate_report,
    render_methodology_matrix,
)

_SLIM_COPY_FIELDS = (
    "title",
    "published_date",
    "sub_domain",
    "problem_statement",
    "method_summary",
    "contributions",
    "limitations",
    "authors",
    "affiliations",
    "method_keywords",
    "methodology_matrix",
    "is_core_domain",
    "industrial_readiness_score",
    "theoretical_depth",
    "domain_specificity",
    "credibility_score",
    "url",
)
_MATRIX_USER_MSG = "Build the methodology comparison matrix from the paper data in context."


def _normalize_days(days: int | None) -> int:
    try:
        parsed = int(days) if days is not None else DAYS_BACK
    except (TypeError, ValueError):
        parsed = DAYS_BACK
    return parsed if parsed > 0 else DAYS_BACK


def _normalize_domain(domain: str) -> str:
    domain_name = str(domain).strip() if domain else ""
    return domain_name or DOMAIN


def _slim_papers_for_report_impl(papers: list[dict]) -> list[dict]:
    """Extract only the fields needed by report/gaps/matrix agents."""
    slim: list[dict] = []
    for paper in papers:
        slim_paper = {field: paper.get(field) for field in _SLIM_COPY_FIELDS}
        methodology = paper.get("methodology_matrix") or {}
        slim_paper["methodology_matrix"] = methodology
        slim_paper["authors"] = paper.get("authors") or []
        slim_paper["affiliations"] = paper.get("affiliations") or ""
        slim_paper["is_core_domain"] = paper.get("is_core_domain", True)
        slim_paper["method_type"] = methodology.get("approach_type")
        slim_paper["open_source"] = methodology.get("open_source")
        slim_paper["credibility_breakdown"] = paper.get("credibility_breakdown", {})
        slim_paper["venue_detected"] = paper.get("venue_detected")
        slim_paper["github_url"] = (
            paper.get("github_url") if paper.get("github_url_validated") else None
        )
        slim.append(slim_paper)
    return slim


def _parse_cached_gaps_response(raw_text: str) -> dict | None:
    if not raw_text:
        return None
    try:
        parsed = json.loads(_clean_json_response(raw_text))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    gaps = parsed.get("extrapolated_gaps", [])
    if isinstance(gaps, list) and gaps:
        return {"extrapolated_gaps": gaps}
    return None


def _is_valid_matrix_section(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    return "## 3." in normalized and "| Paper |" in normalized


def _ensure_matrix_section(papers: list[dict], matrix_section: str | None) -> str | None:
    if _is_valid_matrix_section(str(matrix_section or "")):
        return matrix_section
    print("[Cache] Matrix section missing or invalid after parallel run, forcing fallback.")
    llm_fast = _make_llm(deep=False)
    rebuilt = render_methodology_matrix(papers, llm_fast)
    return rebuilt if _is_valid_matrix_section(rebuilt) else rebuilt


def _run_report_agent(
    papers: list[dict],
    cache,
    domain: str,
    days: int,
    today: str,
) -> str:
    user_msg = REPORT_USER_TEMPLATE.format(
        domain=domain,
        days=days,
        paper_summaries="[Provided in context cache - refer to the paper data above]",
    )
    if cache is not None:
        model = make_model_from_cache(cache, get_report_prompt(domain, days, today))
        response = model.generate_content(user_msg)
        text = str(getattr(response, "text", "") or "").strip()
        if text:
            return text
    return generate_report(papers, domain, days)


def _run_gaps_agent(papers: list[dict], cache, domain: str) -> dict:
    user_msg = GAPS_USER_TEMPLATE.format(
        paper_summaries="[Provided in context cache - refer to the paper data above]",
    )
    if cache is not None:
        try:
            model = make_model_from_cache(cache, get_gaps_prompt(domain))
            response = model.generate_content(user_msg)
            parsed = _parse_cached_gaps_response(str(getattr(response, "text", "") or ""))
            if parsed is not None:
                return parsed
        except Exception as exc:
            print(f"[Cache] Gaps agent failed, falling back: {exc}")

    llm_deep = _make_llm(deep=True)
    return generate_extrapolated_gaps(papers, llm_deep, domain)


def _run_matrix_agent(papers: list[dict], cache) -> str | None:
    if cache is not None:
        try:
            model = make_model_from_cache(cache, get_matrix_prompt())
            response = model.generate_content(_MATRIX_USER_MSG)
            text = str(getattr(response, "text", "") or "").strip()
            if _is_valid_matrix_section(text):
                return text
            print("[Cache] Matrix output missing Section 3 format, falling back.")
        except Exception as exc:
            print(f"[Cache] Matrix agent failed, falling back: {exc}")

    llm_fast = _make_llm(deep=False)
    return render_methodology_matrix(papers, llm_fast)


def run_cached_parallel_agents_impl(
    papers: list[dict],
    domain: str,
    days: int,
) -> tuple[str, dict, str | None]:
    """Run report, gaps, and matrix agents in parallel using a shared Context Cache."""
    today = dt.datetime.now().strftime("%Y-%m-%d")
    report_days = _normalize_days(days)
    domain_name = _normalize_domain(domain)
    slim_papers = _slim_papers_for_report_impl(papers)

    cache = build_shared_cache(
        papers=slim_papers,
        domain=domain_name,
        model_name=GEMINI_MODEL_FAST,
    )

    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_report = executor.submit(
                _run_report_agent, papers, cache, domain_name, report_days, today
            )
            fut_gaps = executor.submit(_run_gaps_agent, papers, cache, domain_name)
            fut_matrix = executor.submit(_run_matrix_agent, papers, cache)
            report = fut_report.result()
            gaps_data = fut_gaps.result()
            matrix_section = fut_matrix.result()
            matrix_section = _ensure_matrix_section(papers, matrix_section)
    finally:
        delete_cache(cache)

    return report, gaps_data, matrix_section
