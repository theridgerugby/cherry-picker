# app.py — Streamlit 主入口：arXiv Research Intelligence Pipeline

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud injects secrets via st.secrets; local runs use .env
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ── Page config (must be first st call) ───────────────────────────────────────

st.set_page_config(
    page_title="arXiv Research Intelligence",
    page_icon="🔬",
    layout="wide",
)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 arXiv Research Intelligence")
    st.caption("10-minute literature review")

    st.divider()

    with st.expander("ℹ️ About"):
        st.markdown(
            "- **Fetch & analyze** recent arXiv papers on any topic\n"
            "- **Auto-generate** structured comparison reports with "
            "trend analysis, skill maps, and reading orders\n"
            "- **Adaptive search** expands the time window until enough "
            "papers are found"
        )


# ── Intent mapping ────────────────────────────────────────────────────────────

_INTENT_MAP = {
    "Latest (past 2 weeks)": "latest",
    "Recent (past month)": "recent",
    "Landscape (past 3 months)": "landscape",
}


# ── Step 1: Topic Input ──────────────────────────────────────────────────────

st.header("What research area are you exploring?")

topic = st.text_input(
    "Research topic",
    placeholder=(
        "e.g. sparse representation, multimodal LLM efficiency, "
        "graph neural networks for drug discovery"
    ),
    label_visibility="collapsed",
)

time_window = st.radio(
    "Time window",
    options=list(_INTENT_MAP.keys()),
    horizontal=True,
    index=1,
)
intent = _INTENT_MAP[time_window]

analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)


# ── Pipeline execution ────────────────────────────────────────────────────────

if analyze_clicked and topic.strip():

    # Late imports — only load heavy modules when user clicks Analyze
    from langchain_google_genai import ChatGoogleGenerativeAI

    from config import (
        GEMINI_MODEL, GEMINI_MODEL_FAST, THINKING_BUDGET, DOMAIN,
        MIN_PAPERS_FOR_COMPARISON, MAX_PAPERS,
    )
    from input_validator import validate_user_input, format_rejection_for_ui
    from query_generator import generate_arxiv_query
    from paper_fetcher import fetch_papers_adaptive
    from paper_extractor import extract_paper_info, store_papers_to_db
    from credibility_scorer import score_paper_credibility
    from report_generator import (
        _make_llm, generate_report, generate_extrapolated_gaps,
        render_extrapolated_gaps_markdown, inject_section_four,
        render_methodology_matrix, save_report,
    )

    # ── Step 2: Validation ────────────────────────────────────────────────

    llm_fast = _make_llm(deep=False)
    llm_deep = _make_llm(deep=True)  # required for extraction (complex Pydantic schema)

    validation = validate_user_input(topic, llm_fast)

    if not validation["is_valid"]:
        st.error(format_rejection_for_ui(validation))
        if validation.get("suggestion"):
            st.info(f"💡 **Try:** {validation['suggestion']}")
        st.stop()

    # ── Step 3: Query generation ──────────────────────────────────────────

    keywords = validation.get("extractable_keywords", [])
    query_result = generate_arxiv_query(topic, keywords, llm_fast)

    display_name = query_result["display_name"]
    arxiv_query = query_result["arxiv_query"]

    st.success(f"✅ Searching arXiv for: **{display_name}**")
    st.caption(f"Query: `{arxiv_query}`")

    if query_result.get("low_confidence_warning"):
        st.warning(query_result["low_confidence_warning"])

    # ── Step 4: Pipeline with progress ────────────────────────────────────

    try:
        with st.status("Running analysis...", expanded=True) as status:

            # 4a — Fetch papers
            st.write("📡 Fetching papers from arXiv...")
            fetch_result = fetch_papers_adaptive(arxiv_query, intent)
            papers = fetch_result["papers"]
            count = fetch_result["paper_count"]
            st.write(f"✅ Found **{count}** papers (window: {fetch_result['days_used']}d)")

            if fetch_result["window_expanded"]:
                st.write(
                    f"⚠️ Date range was expanded to {fetch_result['days_used']}d "
                    "to find enough papers."
                )

            if not papers:
                status.update(label="No papers found", state="error")
                st.stop()

            # 4b — Extract structured data
            # Cap to MAX_PAPERS — gemini-3-flash handles complex schema but is slow for 68 papers
            papers_to_extract = papers[:MAX_PAPERS]
            st.write(
                f"🤖 Extracting structured data "
                f"({len(papers_to_extract)} of {count} papers)..."
            )
            # Must use llm_deep — gemini-2.0-flash fails on the strict
            # Pydantic schema that paper_extractor.py enforces
            extracted = []
            failed = 0
            progress = st.progress(0, text="Extracting...")
            for i, paper in enumerate(papers_to_extract):
                result = extract_paper_info(paper, llm_deep)
                if result is not None:
                    extracted.append(result)
                else:
                    failed += 1
                progress.progress(
                    (i + 1) / len(papers_to_extract),
                    text=f"Extracted {len(extracted)}/{i+1} ({failed} failed)",
                )
            progress.empty()
            fail_note = f" ({failed} failed)" if failed else ""
            st.write(f"✅ Extracted **{len(extracted)}** / {len(papers_to_extract)} papers{fail_note}")

            if not extracted:
                status.update(label="Extraction failed", state="error")
                st.stop()

            # 4c — Store to DB
            st.write("💾 Storing to vector database...")
            store_papers_to_db(extracted)

            # 4d — Credibility scoring
            st.write("📊 Scoring credibility...")
            scored = [score_paper_credibility(p, display_name) for p in extracted]

            # 4e — Generate report
            st.write("📝 Generating report...")

            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=3) as executor:
                fut_report = executor.submit(generate_report, extracted)
                fut_gaps = executor.submit(
                    generate_extrapolated_gaps, extracted, llm_deep
                )
                fut_matrix = None
                if len(extracted) >= MIN_PAPERS_FOR_COMPARISON:
                    fut_matrix = executor.submit(
                        render_methodology_matrix, extracted, llm_fast
                    )

                report = fut_report.result()
                gaps_data = fut_gaps.result()
                matrix_section = fut_matrix.result() if fut_matrix else None

            # Assemble report
            if matrix_section:
                marker = "## 4."
                if marker in report:
                    report = report.replace(
                        marker, matrix_section + "\n\n" + marker
                    )
                else:
                    report += "\n\n" + matrix_section

            gaps_md = render_extrapolated_gaps_markdown(gaps_data)
            report = inject_section_four(report, gaps_md)

            save_report(report, extracted)
            st.write("✅ Report generated!")

            status.update(label="Analysis complete!", state="complete")

        # ── Store in session state ────────────────────────────────────────
        st.session_state["report"] = report
        st.session_state["papers"] = extracted
        st.session_state["last_query"] = {
            "topic": topic,
            "intent": intent,
            "display_name": display_name,
            "arxiv_query": arxiv_query,
        }

    except Exception as e:
        st.error(f"❌ Pipeline error: {e}")
        st.info("Try a different topic or extend the time window.")


# ── Step 5: Report display (persists via session_state) ───────────────────────

if "report" in st.session_state and st.session_state["report"]:
    st.divider()
    st.subheader(
        f"📄 Report: {st.session_state.get('last_query', {}).get('display_name', 'Research Report')}"
    )
    st.markdown(st.session_state["report"])

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Markdown",
            data=st.session_state["report"],
            file_name="report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇️ Download as Text",
            data=st.session_state["report"],
            file_name="report.txt",
            mime="text/plain",
            use_container_width=True,
        )

elif not analyze_clicked:
    st.info("👆 Enter a research topic and click **Analyze** to get started.")
