# agent.py - ReAct Agent entrypoint wiring all tools.

import json

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from config import DAYS_BACK, DOMAIN, GEMINI_MODEL, GEMINI_MODEL_FAST, MIN_PAPERS_TO_PROCESS
from credibility_scorer import score_paper_credibility
from paper_extractor import (
    extract_paper_info,
    load_db,
    prefilter_papers,
    store_papers_to_db,
)
from paper_fetcher import fetch_recent_papers
from trend_analyzer import analyze_trends

load_dotenv()

# Global state shared during one agent run.
_papers_raw = []
_papers_extracted = []
_llm = None
_llm_fast = None
_low_confidence_mode = False
_current_display_name: str = ""

FILTER_CONFIG = {
    "min_relevance_score": 6,
    "max_papers_after_filter": 10,
    "require_core_domain_ratio": 0.7,  # at least 70% must be core domain
}


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    return _llm


def get_llm_fast():
    global _llm_fast
    if _llm_fast is None:
        _llm_fast = ChatGoogleGenerativeAI(model=GEMINI_MODEL_FAST, temperature=0)
    return _llm_fast


def set_current_topic(display_name: str):
    """Called by app.py before running the pipeline to set search context."""
    global _current_display_name
    _current_display_name = display_name.strip()


def _apply_domain_filter(papers: list[dict], domain: str) -> list[dict]:
    scored_papers = prefilter_papers(papers, domain, get_llm_fast())
    if not scored_papers:
        print("[Filter] 0 papers available for domain filtering")
        return []

    core_count = sum(1 for p in scored_papers if p.get("is_core_domain"))
    core_ratio = core_count / len(scored_papers)
    if core_ratio < FILTER_CONFIG["require_core_domain_ratio"]:
        print(
            f"[Filter] Core-domain ratio too low: {core_ratio:.2f} "
            f"(required: {FILTER_CONFIG['require_core_domain_ratio']:.2f})"
        )

    filtered = [
        p
        for p in scored_papers
        if p.get("relevance_score", 0) >= FILTER_CONFIG["min_relevance_score"]
        and p.get("is_core_domain")
    ]

    filtered = sorted(
        filtered,
        key=lambda x: x.get("relevance_score", 0),
        reverse=True,
    )
    filtered = filtered[: FILTER_CONFIG["max_papers_after_filter"]]

    print(f"[Filter] {len(papers)} → {len(filtered)} papers after domain filter")
    return filtered


@tool
def search_arxiv(query_override: str = "") -> str:
    """
    Fetch recent papers from arXiv.
    Input: optional query override (currently ignored in this implementation).
    Output: summary list as JSON string.
    """
    global _papers_raw, _low_confidence_mode

    fetched = fetch_recent_papers()
    if not fetched:
        return "No papers found. Check network connectivity or broaden the query."

    current_domain = _current_display_name.strip() or DOMAIN
    _papers_raw = _apply_domain_filter(fetched, current_domain)
    if not _papers_raw:
        return "No papers passed the domain relevance filter. Broaden the scope or time window."

    _low_confidence_mode = len(_papers_raw) < 3
    if _low_confidence_mode:
        print(f"[Filter] Low-confidence mode: only {len(_papers_raw)} paper(s) after filtering")

    summary = [
        {
            "index": i,
            "arxiv_id": p["arxiv_id"],
            "title": p["title"],
            "date": p["published_date"],
            "relevance_score": p.get("relevance_score"),
            "detected_actual_domain": p.get("detected_actual_domain"),
        }
        for i, p in enumerate(_papers_raw)
    ]
    return json.dumps(summary, ensure_ascii=False)


@tool
def extract_and_store_paper(paper_index: str) -> str:
    """
    Extract structured info for one paper and keep it in memory until DB save.
    Input: paper index in current list.
    Output: extraction result summary.
    """
    try:
        idx = int(str(paper_index).split(":")[-1].strip())
    except ValueError:
        return f"Error: unable to parse paper_index value: {paper_index}"

    global _papers_raw, _papers_extracted
    if not _papers_raw:
        return "Error: call search_arxiv first to fetch papers."
    if idx < 0 or idx >= len(_papers_raw):
        return f"Error: index {idx} out of range (total {len(_papers_raw)})."

    paper = _papers_raw[idx]
    llm = get_llm()
    current_domain = _current_display_name.strip() or DOMAIN
    result = extract_paper_info(paper, llm, domain=current_domain)

    if result is None:
        return f"Extraction failed: {paper['title'][:60]}"

    _papers_extracted.append(result)
    return (
        f"Extracted: {result['title'][:60]}\n"
        f"  Sub-domain: {result.get('sub_domain')}\n"
        f"  Industrial readiness: {result.get('industrial_readiness_score')}/5\n"
        f"  Theoretical depth: {result.get('theoretical_depth')}/5\n"
        f"  Domain specificity: {result.get('domain_specificity')}/5\n"
        f"  Contributions: {result.get('contributions', [])}"
    )


@tool
def save_all_to_database(dummy: str = "") -> str:
    """
    Save all extracted papers to the vector DB in one batch.
    """
    global _papers_extracted
    if not _papers_extracted:
        return "No extracted papers to save."
    store_papers_to_db(_papers_extracted)
    return f"Saved {len(_papers_extracted)} papers to vector database."


@tool
def query_database(question: str) -> str:
    """
    Retrieve top relevant paper snippets from vector DB for a natural language question.
    """
    try:
        db = load_db()
        results = db.similarity_search(question, k=3)
        output = []
        for doc in results:
            meta = doc.metadata
            full = json.loads(meta.get("full_json", "{}"))
            output.append(
                {
                    "title": full.get("title"),
                    "sub_domain": full.get("sub_domain"),
                    "contributions": full.get("contributions"),
                    "method_summary": full.get("method_summary"),
                }
            )
        return json.dumps(output, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Database query failed: {e}"


@tool
def analyze_industry_trends(dummy: str = "") -> str:
    """
    Score papers and run trend analysis on papers currently in DB.
    """
    try:
        db = load_db()
        search_query = _current_display_name.strip() or DOMAIN
        results = db.similarity_search(search_query, k=50)
        papers = []
        seen_ids = set()
        for doc in results:
            meta = doc.metadata
            arxiv_id = meta.get("arxiv_id", "")
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            full = json.loads(meta.get("full_json", "{}"))
            if full:
                papers.append(full)

        if not papers:
            return "Error: vector DB is empty. Extract papers first."

        scored_papers = [score_paper_credibility(p, search_query) for p in papers]
        llm = get_llm()
        trend_data = analyze_trends(scored_papers, llm, search_query)
        return json.dumps(trend_data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Trend analysis failed: {e}"


def _build_agent_system_prompt(domain: str, days: int, min_papers: int) -> str:
    return f"""You are a research intelligence agent specialized in academic literature analysis.
Your goal is to fetch, analyze, and prepare a set of recent papers on "{domain}" for report generation.

Rules:
- After fetching papers with search_arxiv, extract up to {min_papers} papers using extract_and_store_paper.
- If fewer than {min_papers} papers remain after domain filtering, process all available papers and continue.
- If fewer than 3 papers remain after domain filtering, continue in low-confidence mode.
- Prioritize papers with higher domain specificity to {domain} (domain_specificity 4-5).
- After extracting all target papers, call save_all_to_database.
- After saving, call analyze_industry_trends to generate trend insights.
- If a tool returns an error, try a different approach and continue.
- When all steps are complete, reply with a brief summary: how many papers found, how many processed, key themes.

Today's task:
Domain: {domain}
Time range: last {days} days
Minimum papers to process: {min_papers}"""


def run_agent():
    """Run the full fetch -> extract -> save flow via tool-calling agent."""
    llm = get_llm()
    tools = [
        search_arxiv,
        extract_and_store_paper,
        save_all_to_database,
        query_database,
        analyze_industry_trends,
    ]

    current_domain = _current_display_name.strip() or DOMAIN
    system_prompt = _build_agent_system_prompt(
        domain=current_domain,
        days=DAYS_BACK,
        min_papers=MIN_PAPERS_TO_PROCESS,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        debug=False,
    )

    print("\n" + "=" * 60)
    print("arXiv Agent started")
    print(f"  Domain: {current_domain}")
    print(f"  Goal: process papers from last {DAYS_BACK} days")
    print("=" * 60 + "\n")

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Start: fetch and process papers on {current_domain} "
                        f"from the last {DAYS_BACK} days."
                    ),
                }
            ]
        }
    )

    final_output = ""
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            final_output = getattr(last_msg, "content", str(last_msg))

    print("\n" + "=" * 60)
    print("Agent completed")
    print(f"Final Answer: {final_output}")
    print("=" * 60)
    print("\nNext step: run python report_generator.py to generate report")

    return _papers_extracted


if __name__ == "__main__":
    run_agent()
