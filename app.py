# app.py — arXiv Research Intelligence | Apple-inspired UI

import os
import random
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cherry Picker · Research Intelligence",
    page_icon="🍒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Movie quotes ──────────────────────────────────────────────────────────────

MOVIE_QUOTES = [
    ("The greatest teacher, failure is.", "Yoda — Star Wars: The Last Jedi"),
    ("It's not who I am underneath, but what I do that defines me.", "Batman Begins"),
    ("You can't handle the truth!", "A Few Good Men"),
    ("Why so serious?", "The Dark Knight"),
    ("To infinity and beyond.", "Buzz Lightyear — Toy Story"),
    ("Just keep swimming.", "Dory — Finding Nemo"),
    ("After all this time? Always.", "Snape — Harry Potter"),
    ("I am inevitable.", "Thanos — Avengers: Endgame"),
    ("The stuff that dreams are made of.", "The Maltese Falcon"),
    ("You is kind, you is smart, you is important.", "The Help"),
    ("Life is like a box of chocolates.", "Forrest Gump"),
    ("Get busy living, or get busy dying.", "The Shawshank Redemption"),
    ("With great power comes great responsibility.", "Spider-Man"),
    ("I feel the need — the need for speed.", "Top Gun"),
    ("There is no spoon.", "The Matrix"),
    ("Elementary, my dear Watson.", "The Adventures of Sherlock Holmes"),
    ("To boldly go where no man has gone before.", "Star Trek"),
    ("I'll be back.", "The Terminator"),
    ("Roads? Where we're going, we don't need roads.", "Back to the Future"),
    ("We're gonna need a bigger boat.", "Jaws"),
    ("You had me at hello.", "Jerry Maguire"),
    ("Every passing minute is another chance to turn it all around.", "Vanilla Sky"),
    ("A dream is a wish your heart makes.", "Cinderella"),
    ("Hakuna Matata — it means no worries.", "The Lion King"),
    ("Do, or do not. There is no try.", "Yoda — The Empire Strikes Back"),
]

# ── Apple-inspired CSS with acrylic effect ────────────────────────────────────

st.markdown("""
<style>
/* ── System font stack ── */
* {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Segoe UI", Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, .stDeployButton { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Force light mode ── */
.stApp {
    background: #f7f8fc !important;
    color: #1a1a1a !important;
    min-height: 100vh;
}

/* Force all Streamlit text to be dark */
.stApp, .stApp p, .stApp span, .stApp label, .stApp div,
.stMarkdown, .stMarkdown p, .stMarkdown span,
[data-testid="stText"], [data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p {
    color: #1a1a1a !important;
}

/* ── Acrylic card ── */
.acrylic {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
    padding: 16px 40px 40px;
    margin: 0 auto;
}

/* Collapse hidden label space inside acrylic */
.acrylic .stTextInput label,
.acrylic .stRadio > label {
    display: none !important;
}

/* ── Centered container ── */
.main-container {
    max-width: 780px;
    margin: 0 auto;
    padding: 12px 24px 48px;
}

/* ── Remove Streamlit's default top padding ── */
.block-container {
    padding-top: 1rem !important;
}

/* ── Hero title ── */
.hero-title {
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -1.5px;
    color: #111827 !important;
    text-align: center;
    margin-bottom: 8px;
    line-height: 1.15;
}

.hero-sub {
    font-size: 17px;
    color: #6b7280 !important;
    text-align: center;
    margin-bottom: 40px;
    font-weight: 400;
    font-style: italic;
    letter-spacing: -0.2px;
}

/* ── Feature chips ── */
.feature-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
    margin-bottom: 28px;
}

.feature-chip {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1.5px solid #cbd5e1;
    border-radius: 14px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
}

.feature-chip .icon {
    font-size: 32px;
    line-height: 1;
    font-weight: 700;
    color: #1d4ed8 !important;
    display: block;
    margin-bottom: 10px;
}
.feature-chip .label {
    font-size: 24px;
    font-weight: 700;
    color: #111827 !important;
    display: block;
    margin-bottom: 8px;
    letter-spacing: -0.2px;
}
.feature-chip .desc {
    font-size: 13px;
    color: #4b5563 !important;
    line-height: 1.45;
    font-weight: 500;
}

/* ── Input field ── */
.stTextInput > div > div > input {
    border: 1.5px solid #d1d5db !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    font-size: 16px !important;
    background: #ffffff !important;
    color: #111827 !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input::placeholder {
    color: #9ca3af !important;
}
.stTextInput > div > div > input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    background: #ffffff !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #2563EB !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
}
.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span {
    color: #ffffff !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Secondary buttons ── */
.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    color: #374151 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 20px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #eff6ff !important;
    border-color: #2563EB !important;
    color: #2563EB !important;
}

/* ── Radio buttons ── */
.stRadio > div {
    display: flex;
    gap: 8px;
    justify-content: center;
}
.stRadio label {
    color: #374151 !important;
}
.stRadio label span {
    color: #374151 !important;
}

/* ── Stats row ── */
.stats-row {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 12px;
    margin: 24px 0;
}
.stat-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
}
.stat-num {
    font-size: 26px;
    font-weight: 700;
    color: #111827 !important;
    letter-spacing: -1px;
}
.stat-label {
    font-size: 11px;
    color: #6b7280 !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 2px;
}

/* ── Quote box ── */
.quote-box {
    background: #eff6ff;
    border-left: 3px solid #2563EB;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin: 24px 0;
}
.quote-text {
    font-size: 18px;
    font-style: italic;
    color: #1f2937 !important;
    font-weight: 500;
    margin-bottom: 6px;
}
.quote-source {
    font-size: 13px;
    color: #6b7280 !important;
    font-weight: 500;
}

/* ── Section divider ── */
.section-divider {
    height: 1px;
    background: #e5e7eb;
    margin: 32px 0;
}

/* ── Report container ── */
.report-wrapper {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.04);
    max-width: none;
    overflow-x: auto;
}
.report-wrapper p, .report-wrapper li, .report-wrapper td,
.report-wrapper th, .report-wrapper h1, .report-wrapper h2,
.report-wrapper h3, .report-wrapper h4 {
    color: #1a1a1a !important;
}

/* ── Status widget text ── */
[data-testid="stStatusWidget"] p,
[data-testid="stStatusWidget"] span {
    color: #374151 !important;
}

/* ── Alerts ── */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    border: 1.5px solid #d1d5db !important;
    background: #ffffff !important;
    color: #374151 !important;
    transition: all 0.15s ease !important;
}
.stDownloadButton > button:hover {
    background: #ffffff !important;
    border-color: #2563EB !important;
    color: #2563EB !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: #2563EB !important;
    border-radius: 4px !important;
}

/* ── Caption text ── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #6b7280 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Intent mapping ────────────────────────────────────────────────────────────

_INTENT_MAP = {
    "Latest (past 2 weeks)": "latest",
    "Recent (past month)":   "recent",
    "Landscape (past 3 months)": "landscape",
}

_EXAMPLE_TOPICS = [
    "sparse representation",
    "multimodal LLMs",
    "drug discovery GNNs",
    "RLHF alignment",
    "vision transformers",
]

# ── Layout: centered symmetric container ─────────────────────────────────────

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero-title">Cherry Picker</div>
<div class="hero-sub">hope other researchers will like this too</div>
""", unsafe_allow_html=True)

# Feature chips (3-column symmetric)
st.markdown("""
<div class="feature-row">
  <div class="feature-chip">
    <span class="icon">1</span>
    <span class="label">Real-time arXiv</span>
    <span class="desc">Papers published this week, not from a stale training set</span>
  </div>
  <div class="feature-chip">
    <span class="icon">2</span>
    <span class="label">Structured Extraction</span>
    <span class="desc">Every claim traced to a specific paper and date</span>
  </div>
  <div class="feature-chip">
    <span class="icon">3</span>
    <span class="label">Cross-paper Insights</span>
    <span class="desc">Trend analysis that ChatGPT cannot replicate</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Topic input ─────────────────────────────────────────────────────────────

# Topic input
topic_value = st.session_state.get("topic_prefill", "")
topic = st.text_input(
    "Research topic",
    value=topic_value,
    placeholder="e.g. sparse representation, RLHF, vision transformers...",
    label_visibility="collapsed",
)

# Example topic chips (symmetric row)
st.markdown("<div style='margin: 12px 0 4px; font-size:12px; color:#9ca3af; font-weight:500;'>Try an example</div>", unsafe_allow_html=True)
chip_cols = st.columns(len(_EXAMPLE_TOPICS))
for i, example in enumerate(_EXAMPLE_TOPICS):
    with chip_cols[i]:
        if st.button(example, key=f"chip_{i}"):
            st.session_state["topic_prefill"] = example
            st.rerun()

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# Time window (centered radio)
time_window = st.radio(
    "Time window",
    options=list(_INTENT_MAP.keys()),
    horizontal=True,
    index=1,
    label_visibility="collapsed",
)
intent = _INTENT_MAP[time_window]

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# Analyze button (full width, centered)
_, btn_col, _ = st.columns([1, 3, 1])
with btn_col:
    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        use_container_width=True,
    )

# ── Pipeline execution ────────────────────────────────────────────────────────

if analyze_clicked and topic.strip():

    from langchain_google_genai import ChatGoogleGenerativeAI
    from config import GEMINI_MODEL, GEMINI_MODEL_FAST, MIN_PAPERS_FOR_COMPARISON
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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Validation
    llm_fast = _make_llm(deep=False)
    validation = validate_user_input(topic, llm_fast)

    if not validation["is_valid"]:
        st.error(format_rejection_for_ui(validation))
        if validation.get("suggestion"):
            st.info(f"💡 **Try:** {validation['suggestion']}")
        st.stop()

    # Query generation
    keywords = validation.get("extractable_keywords", [])
    query_result = generate_arxiv_query(topic, keywords, llm_fast)
    display_name = query_result["display_name"]
    arxiv_query  = query_result["arxiv_query"]

    st.success(f"Searching arXiv for: **{display_name}**")
    st.caption(f"`{arxiv_query}`")

    # Random movie quote while loading
    quote, source = random.choice(MOVIE_QUOTES)
    st.markdown(f"""
    <div class="quote-box">
        <div class="quote-text">"{quote}"</div>
        <div class="quote-source">— {source}</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        status_box = st.empty()
        status_box.info("Running analysis...")
        if True:

            st.write("Fetching papers from arXiv...")
            fetch_result = fetch_papers_adaptive(arxiv_query, intent)
            papers = fetch_result["papers"]
            count  = fetch_result["paper_count"]
            st.write(f"Found **{count}** papers (window: {fetch_result['days_used']}d)")

            if fetch_result["window_expanded"]:
                st.write(f"Expanded to {fetch_result['days_used']}d to find enough papers.")

            if not papers:
                status_box.error("No papers found")
                st.stop()
                
            # --- Enforce a limit of 30 diverse papers ---
            max_limit = 30
            if len(papers) > max_limit:
                st.write(f"Filtering to {max_limit} most diverse papers (from {len(papers)})...")
                import re
                
                stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "about", "as", "of", "is", "are", "was", "were", "be", "been", "that", "which", "this", "these", "those", "from", "can", "has", "have", "had", "not"}
                def get_tokens(p):
                    text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
                    return set(re.findall(r'\b[a-z]{3,}\b', text)) - stop_words
                
                ptokens = [get_tokens(p) for p in papers]
                selected_idxs = [0]
                while len(selected_idxs) < max_limit:
                    best_i = -1
                    max_min_dist = -1.0
                    for i in range(len(papers)):
                        if i in selected_idxs:
                            continue
                        min_dist_to_sel = 1.0
                        for j in selected_idxs:
                            inter = len(ptokens[i] & ptokens[j])
                            union = len(ptokens[i] | ptokens[j])
                            dist = 1.0 - (inter / union if union > 0 else 0)
                            if dist < min_dist_to_sel:
                                min_dist_to_sel = dist
                        if min_dist_to_sel > max_min_dist:
                            max_min_dist = min_dist_to_sel
                            best_i = i
                    if best_i == -1: break
                    selected_idxs.append(best_i)
                
                papers = [papers[i] for i in selected_idxs]
            # --------------------------------------------

            # Extraction (parallel)
            st.write("Extracting structured data...")
            extracted = []
            progress  = st.progress(0, text="Extracting...")

            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=10) as ex:
                futures = {ex.submit(extract_paper_info, p, llm_fast): p for p in papers}
                done = 0
                for future in as_completed(futures):
                    done += 1
                    try:
                        r = future.result()
                        if r is not None:
                            extracted.append(r)
                    except Exception as e:
                        print(f"Extraction error: {e}")
                    progress.progress(done / len(papers), text=f"Extracted {done}/{len(papers)}")

            progress.empty()
            st.write(f"Extracted **{len(extracted)}** / {len(papers)} papers")

            if not extracted:
                status_box.error("Extraction failed")
                st.stop()

            st.write("Storing to vector database...")
            store_papers_to_db(extracted)

            st.write("Scoring credibility...")
            scored = [score_paper_credibility(p, display_name) for p in extracted]
            avg_credibility = round(
                sum(p.get("credibility_score", 0) for p in scored) / len(scored)
            ) if scored else 0

            st.write("Generating report...")
            llm_deep = _make_llm(deep=True)
            with ThreadPoolExecutor(max_workers=3) as ex:
                fut_report = ex.submit(generate_report, extracted)
                fut_gaps   = ex.submit(generate_extrapolated_gaps, extracted, llm_deep)
                fut_matrix = (
                    ex.submit(render_methodology_matrix, extracted, llm_fast)
                    if len(extracted) >= MIN_PAPERS_FOR_COMPARISON else None
                )
                report        = fut_report.result()
                gaps_data     = fut_gaps.result()
                matrix_section = fut_matrix.result() if fut_matrix else None

            if matrix_section:
                marker = "## 4."
                report = (
                    report.replace(marker, matrix_section + "\n\n" + marker)
                    if marker in report
                    else report + "\n\n" + matrix_section
                )

            gaps_md = render_extrapolated_gaps_markdown(gaps_data)
            report  = inject_section_four(report, gaps_md)
            save_report(report, extracted)
            st.write("Report generated.")
            status_box.success("Analysis complete!")

        # Store results
        st.session_state["report"]        = report
        st.session_state["papers"]        = extracted
        st.session_state["avg_cred"]      = avg_credibility
        st.session_state["days_used"]     = fetch_result["days_used"]
        st.session_state["last_query"]    = {
            "topic": topic, "intent": intent,
            "display_name": display_name, "arxiv_query": arxiv_query,
        }

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.info("Try a different topic or extend the time window.")

# ── Report display ────────────────────────────────────────────────────────────

if "report" in st.session_state and st.session_state["report"]:

    # Close the narrow main-container so report gets full width
    st.markdown('</div>', unsafe_allow_html=True)

    q = st.session_state.get("last_query", {})
    papers  = st.session_state.get("papers", [])
    domains = len(set(p.get("sub_domain", "") for p in papers if p.get("sub_domain")))

    # Stats row (4-column symmetric)
    st.markdown(f"""
    <div class="stats-row" style="max-width:1100px; margin:24px auto;">
      <div class="stat-card">
        <div class="stat-num">{len(papers)}</div>
        <div class="stat-label">Papers analyzed</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{st.session_state.get('days_used', '—')}</div>
        <div class="stat-label">Days searched</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{st.session_state.get('avg_cred', '—')}</div>
        <div class="stat-label">Avg credibility</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{domains}</div>
        <div class="stat-label">Sub-domains</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Report content
    st.markdown(f"### {q.get('display_name', 'Research Report')}")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(st.session_state["report"])

    # Download buttons
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download Markdown",
            data=st.session_state["report"],
            file_name=f"{q.get('display_name', 'report').replace(' ','_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download as Text",
            data=st.session_state["report"],
            file_name=f"{q.get('display_name', 'report').replace(' ','_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

elif not analyze_clicked:
    # Empty state
    st.markdown("""
    <div style="text-align:center; color:#9ca3af; font-size:14px; font-weight:500; margin-top:24px;">
        Enter a topic above and click <strong style="color:#2563EB">Analyze</strong> 
        to generate your report
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)  # end main-container
else:
    st.markdown('</div>', unsafe_allow_html=True)  # end main-container
