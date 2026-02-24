# app.py â€” arXiv Research Intelligence | Apple-inspired UI

import html
import inspect
import os
import random
import re
from urllib.parse import quote as url_quote
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# â”€â”€ Page config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.set_page_config(
    page_title="Cherry Picker \u00b7 Research Intelligence",
    page_icon="\U0001F352",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# â”€â”€ Movie quotes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

MOVIE_QUOTES = [
    ("The greatest teacher, failure is.", "Yoda \u2014 Star Wars: The Last Jedi"),
    ("It's not who I am underneath, but what I do that defines me.", "Batman Begins"),
    ("You can't handle the truth!", "A Few Good Men"),
    ("Why so serious?", "The Dark Knight"),
    ("To infinity and beyond.", "Buzz Lightyear \u2014 Toy Story"),
    ("Just keep swimming.", "Dory \u2014 Finding Nemo"),
    ("After all this time? Always.", "Snape \u2014 Harry Potter"),
    ("The stuff that dreams are made of.", "The Maltese Falcon"),
    ("You is kind, you is smart, you is important.", "The Help"),
    ("Life is like a box of chocolates.", "Forrest Gump"),
    ("Get busy living, or get busy dying.", "The Shawshank Redemption"),
    ("With great power comes great responsibility.", "Spider-Man"),
    ("There is no spoon.", "The Matrix"),
    ("Elementary, my dear Watson.", "The Adventures of Sherlock Holmes"),
    ("To boldly go where no man has gone before.", "Star Trek"),
    ("I'll be back.", "The Terminator"),
    ("Roads? Where we're going, we don't need roads.", "Back to the Future"),
    ("We're gonna need a bigger boat.", "Jaws"),
    ("You had me at hello.", "Jerry Maguire"),
    ("Every passing minute is another chance to turn it all around.", "Vanilla Sky"),
    ("A dream is a wish your heart makes.", "Cinderella"),
    ("Hakuna Matata \u2014 it means no worries.", "The Lion King"),
    ("Do, or do not. There is no try.", "Yoda \u2014 The Empire Strikes Back"),
    ("Cooked, or being cooked, that's a good question.", "JZ"),
    ("We cannot walk alone. And as we walk, we must make the pledge that we shall always march ahead. We cannot turn back.", "Dr. Martin Luther King Jr."),
]

# â”€â”€ UI styles â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.markdown("""
<style>
:root {
    --color-primary: #2563EB;
    --color-text: #111827;
    --color-text-secondary: #6B7280;
    --color-border: rgba(0,0,0,0.08);
    --color-surface: rgba(255,255,255,0.72);
}

* {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                 "Segoe UI", Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

#MainMenu, footer, header, .stDeployButton { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

.stApp {
    background: #ffffff !important;
    color: var(--color-text-secondary) !important;
    min-height: 100vh;
}

.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

.stApp p,
.stApp li,
.stApp td,
.stApp th,
.stApp label,
.stCaption,
[data-testid="stCaptionContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th {
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: var(--color-text-secondary) !important;
}

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: var(--color-text) !important;
}

a {
    color: var(--color-primary) !important;
}

.top-navbar {
    position: fixed;
    top: 0;
    right: 0;
    left: 0;
    z-index: 999;
    height: 52px;
    padding: 0 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255,255,255,0.82);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--color-border);
}

.nav-brand {
    font-size: 16px;
    font-weight: 600;
    color: var(--color-text);
}

.nav-actions {
    display: flex;
    align-items: center;
    gap: 16px;
}

.nav-link {
    font-size: 14px;
    line-height: 1.6;
    color: var(--color-text-secondary) !important;
    text-decoration: none;
    transition: all 0.15s ease;
}

.nav-link:hover {
    color: var(--color-primary) !important;
}

.main-container {
    max-width: 780px;
    margin: 0 auto;
    padding: 72px 24px 64px;
}

.hero-title {
    font-size: 40px;
    font-weight: 700;
    letter-spacing: -1.5px;
    color: var(--color-text) !important;
    text-align: center;
    margin-bottom: 8px;
    line-height: 1.15;
}

.hero-sub {
    font-size: 14px;
    line-height: 1.6;
    color: var(--color-text-secondary) !important;
    text-align: center;
    margin-bottom: 32px;
    font-weight: 400;
    letter-spacing: 0;
}

.feature-chip {
    background: transparent;
    border: 1px solid var(--color-border);
    border-radius: 999px;
    padding: 14px;
    text-align: center;
}

.feature-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}

.feature-chip .icon {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    line-height: 1;
    color: var(--color-text-secondary) !important;
    display: block;
    margin-bottom: 8px;
}

.feature-chip .label {
    font-size: 14px;
    line-height: 1.6;
    font-weight: 600;
    color: var(--color-text) !important;
    display: block;
    margin-bottom: 8px;
}

.feature-chip .desc {
    font-size: 14px;
    line-height: 1.6;
    color: var(--color-text-secondary) !important;
}

.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--color-text-secondary) !important;
}

.explore-label {
    margin-bottom: 8px;
}

.acrylic {
    background: var(--color-surface);
    backdrop-filter: blur(24px) saturate(180%);
    -webkit-backdrop-filter: blur(24px) saturate(180%);
    border: 1px solid var(--color-border);
    border-radius: 20px;
    padding: 40px;
    margin: 0 auto;
}

.acrylic .stTextInput label,
.acrylic .stRadio > label {
    display: none !important;
}

.stTextInput > div > div > input {
    border: 1px solid var(--color-border) !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    background: #ffffff !important;
    color: var(--color-text) !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}

.stTextInput > div > div > input::placeholder {
    color: var(--color-text-secondary) !important;
}

.stTextInput > div > div > input:hover {
    border-color: var(--color-primary) !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--color-primary) !important;
    box-shadow: none !important;
    outline: none !important;
}

.stButton > button[kind="primary"] {
    background: var(--color-primary) !important;
    color: #ffffff !important;
    border: 1px solid var(--color-primary) !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    font-weight: 600 !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}

.stButton > button[kind="primary"] *,
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
.stButton > button[kind="primary"] div {
    color: #F9FAFB !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--color-primary) !important;
    filter: brightness(0.95);
}

.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: var(--color-text-secondary) !important;
    border: 1px solid var(--color-border) !important;
    border-radius: 999px !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: all 0.15s ease !important;
}

.stButton > button[kind="secondary"]:hover {
    border-color: var(--color-primary) !important;
    color: var(--color-primary) !important;
    background: transparent !important;
}

.stButton > button:focus,
.stButton > button:active,
.stDownloadButton > button:focus,
.stDownloadButton > button:active {
    box-shadow: none !important;
    outline: none !important;
}

.stRadio > div {
    display: flex;
    gap: 8px;
    justify-content: flex-start;
}

.stRadio label {
    color: var(--color-text-secondary) !important;
}

.stRadio label span {
    color: var(--color-text-secondary) !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
}

.stRadio label:hover span {
    color: var(--color-primary) !important;
}

.spacer-24 {
    height: 24px;
}

.section-divider {
    height: 1px;
    background: var(--color-border);
    margin: 32px 0;
}

.stats-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    margin: 24px 0;
}

.stat-card {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    transition: all 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}

.stat-num {
    font-size: 14px;
    font-weight: 700;
    line-height: 1.6;
    color: var(--color-text) !important;
}

.stat-label {
    font-size: 11px;
    color: var(--color-text-secondary) !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 8px;
}

.quote-box {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 24px 0;
    animation: fadeSlideIn 0.4s ease forwards;
}

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.quote-text {
    font-size: 14px;
    line-height: 1.6;
    font-style: italic;
    color: var(--color-text) !important;
    font-weight: 500;
    margin-bottom: 8px;
}

.quote-source {
    font-size: 14px;
    line-height: 1.6;
    color: var(--color-text-secondary) !important;
    font-weight: 500;
}

.report-subheader {
    position: sticky;
    top: 52px;
    z-index: 998;
    max-width: 780px;
    margin: 0 auto 24px;
    height: 44px;
    padding: 0 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    border-bottom: 1px solid var(--color-border);
    background: #ffffff;
}

.report-subheader-title {
    min-width: 0;
    font-size: 14px;
    line-height: 1.6;
    font-weight: 600;
    color: var(--color-text) !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.report-subheader-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-shrink: 0;
}

.report-download-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--color-border);
    border-radius: 999px;
    padding: 8px 12px;
    color: var(--color-text-secondary) !important;
    text-decoration: none;
    font-size: 14px;
    line-height: 1.6;
    transition: all 0.15s ease;
}

.report-download-link:hover {
    border-color: var(--color-primary);
    color: var(--color-primary) !important;
}

.report-main-container {
    padding-top: 24px;
}

.report-wrapper {
    background: transparent;
    border: 1px solid var(--color-border);
    border-radius: 16px;
    padding: 24px;
    box-shadow: none;
    max-width: 720px;
    margin: 0 auto;
    overflow-x: auto;
}

.report-wrapper p, .report-wrapper li, .report-wrapper td,
.report-wrapper th, .report-wrapper h1, .report-wrapper h2,
.report-wrapper h3, .report-wrapper h4 {
    color: var(--color-text-secondary) !important;
}

.report-wrapper h1,
.report-wrapper h2,
.report-wrapper h3,
.report-wrapper h4 {
    color: var(--color-text) !important;
}

.report-wrapper table {
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 16px 0 !important;
}

[data-testid="stMarkdownContainer"] .report-wrapper table th,
[data-testid="stMarkdownContainer"] .report-wrapper table td,
.report-wrapper table th,
.report-wrapper table td {
    color: var(--color-text-secondary) !important;
    border: 1px solid var(--color-border) !important;
    padding: 8px !important;
    background: transparent !important;
    vertical-align: top;
}

[data-testid="stMarkdownContainer"] .report-wrapper table th,
.report-wrapper table th {
    color: var(--color-text) !important;
    font-weight: 600 !important;
}

[data-testid="stMarkdownContainer"] table {
    width: 100% !important;
    table-layout: auto !important;
}

[data-testid="stMarkdownContainer"] table th,
[data-testid="stMarkdownContainer"] table td {
    white-space: normal !important;
    word-break: break-word !important;
    overflow-wrap: anywhere !important;
}
[data-testid="stMarkdownContainer"] table th:first-child,
[data-testid="stMarkdownContainer"] table td:first-child {
    min-width: 280px;
}

[data-testid="stStatusWidget"] p,
[data-testid="stStatusWidget"] span {
    color: var(--color-text-secondary) !important;
}

.stAlert {
    border-radius: 12px !important;
    border: none !important;
}

.stProgress > div > div {
    background: var(--color-primary) !important;
    border-radius: 4px !important;
}

.loading-hint {
    font-size: 14px;
    line-height: 1.6;
    color: var(--color-text-secondary) !important;
    margin: 16px 0;
}

.extract-progress {
    position: relative;
    height: 28px;
    border-radius: 6px;
    overflow: hidden;
    background: var(--color-border);
    margin-bottom: 8px;
}

.extract-progress-fill {
    height: 100%;
    width: 0%;
    background: var(--color-primary);
    transition: width 0.15s ease;
}

.extract-progress-text {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    line-height: 1.6;
    font-weight: 600;
    color: #F9FAFB !important;
    pointer-events: none;
}

[data-testid="stProgressBar"] {
    position: relative !important;
    min-height: 28px !important;
}

[data-testid="stProgressBar"] > div:first-child {
    min-height: 28px !important;
}

[data-testid="stProgressBar"] > div:nth-child(2),
[data-testid="stProgressBar"] > p {
    position: absolute !important;
    inset: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    pointer-events: none !important;
    text-align: center !important;
    margin: 0 !important;
}

.stProgress p,
.stProgress span,
[data-testid="stProgressBar"] p,
[data-testid="stProgressBar"] span {
    color: #F9FAFB !important;
    margin: 0 !important;
}

.empty-state {
    text-align: center;
    color: var(--color-text-secondary) !important;
    font-size: 14px;
    line-height: 1.6;
    margin-top: 24px;
}

.empty-state strong {
    color: var(--color-primary) !important;
}

.footer {
    height: 64px;
    margin-top: 48px;
    border-top: 1px solid var(--color-border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: var(--color-text-secondary) !important;
}

.footer span {
    font-size: 12px;
    color: var(--color-text-secondary) !important;
}

@media (prefers-color-scheme: dark) {
    :root {
        --color-primary: #2563EB;
        --color-text: #F9FAFB;
        --color-text-secondary: #9CA3AF;
        --color-border: rgba(255,255,255,0.1);
        --color-surface: rgba(30,30,30,0.72);
    }

    .stApp {
        background: #0f0f0f !important;
    }

    .top-navbar {
        background: rgba(15,15,15,0.85);
    }

    .report-subheader {
        background: #0f0f0f;
    }

    .stTextInput > div > div > input {
        background: rgba(15,15,15,0.85) !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-navbar">
  <div class="nav-brand">&#x1F352; Cherry Picker</div>
  <div class="nav-actions">
    <a class="nav-link" href="#" target="_blank" rel="noopener noreferrer">About</a>
    <a class="nav-link" href="https://github.com/theridgerugby/cherry-picker" target="_blank" rel="noopener noreferrer">GitHub</a>
  </div>
</div>
""", unsafe_allow_html=True)

# â”€â”€ Intent mapping â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_INTENT_MAP = {
    "Latest (past 2 weeks)": "latest",
    "Recent (past month)":   "recent",
    "Landscape (past 3 months)": "landscape",
}

_EXAMPLE_TOPICS = [
    "astrophysics signals",
    "biochemistry applications",
    "quantum error correction",
    "climate model inference",
    "robotics motion planning",
]

_FOOTER_HTML = """
<div class="footer">
  <span>&copy; 2026 Cherry Picker &middot; Built with LangChain & Gemini</span>
  <span>Made for DBW Lab</span>
</div>
"""


def _sanitize_table_cell(value: object) -> str:
    text = str(value if value is not None else "-")
    text = text.replace("\n", " ").replace("\r", " ").replace("|", "\\|").strip()
    return text or "-"


def _build_summary_table_markdown(papers: list[dict]) -> str:
    lines = [
        "## 6. Summary Table",
        "",
        "| Title (short) | Sub-domain | Method Type | Industrial Readiness | Theoretical Depth | Domain Specificity | Key Contribution |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    if not papers:
        lines.append(
            "| No papers available | - | - | - | - | - | Report generated without extracted paper rows. |"
        )
        return "\n".join(lines)

    for paper in papers:
        title = paper.get("title") or "Untitled"
        title_short = title if len(title) <= 60 else f"{title[:57]}..."
        matrix = paper.get("methodology_matrix") or {}

        sub_domain = paper.get("sub_domain") or "-"
        method_type = matrix.get("approach_type") or paper.get("method_type") or "-"
        industrial = paper.get("industrial_readiness_score")
        if industrial in (None, ""):
            industrial = "-"
        theory = paper.get("theoretical_depth")
        if theory in (None, ""):
            theory = "-"
        specificity = paper.get(
            "domain_specificity",
            paper.get("relevance_to_sparse_representation"),
        )
        if specificity in (None, ""):
            specificity = "-"

        contributions = paper.get("contributions")
        if isinstance(contributions, list) and contributions:
            key_contribution = contributions[0]
        else:
            key_contribution = paper.get("key_contribution") or "-"

        lines.append(
            f"| {_sanitize_table_cell(title_short)} | {_sanitize_table_cell(sub_domain)} "
            f"| {_sanitize_table_cell(method_type)} | {_sanitize_table_cell(industrial)} "
            f"| {_sanitize_table_cell(theory)} | {_sanitize_table_cell(specificity)} "
            f"| {_sanitize_table_cell(key_contribution)} |"
        )

    return "\n".join(lines)


def _summary_section_has_data_rows(summary_section_text: str) -> bool:
    table_lines = [line.strip() for line in summary_section_text.splitlines() if line.strip().startswith("|")]
    if len(table_lines) <= 2:
        return False

    for row in table_lines[2:]:
        cell_payload = row.strip().strip("|").replace("|", "").replace("-", "").strip()
        if cell_payload:
            return True
    return False


def _ensure_summary_table_section(report_text: str, papers: list[dict]) -> str:
    section_pattern = re.compile(r"## 6\. Summary Table.*?(?=\n## |\Z)", re.DOTALL)
    match = section_pattern.search(report_text)
    fallback_section = _build_summary_table_markdown(papers)

    if match:
        if _summary_section_has_data_rows(match.group(0)):
            return report_text
        return section_pattern.sub(fallback_section, report_text, count=1)

    return report_text.rstrip() + "\n\n" + fallback_section + "\n"


def _sync_report_heading_days(report_text: str, days_used: int | None) -> str:
    if not isinstance(days_used, int) or days_used <= 0:
        return report_text

    heading_pattern = re.compile(
        r"^#\s*Research Landscape:\s*(.+?)\s*[—-]\s*Last\s+\d+\s+Days\s*$",
        re.MULTILINE,
    )

    def _replace_heading(match: re.Match[str]) -> str:
        return f"# Research Landscape: {match.group(1)} — Last {days_used} Days"

    return heading_pattern.sub(_replace_heading, report_text, count=1)

# â”€â”€ Layout: centered symmetric container â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Hero
st.markdown("""
<div class="hero-title">Cherry Picker</div>
<div class="hero-sub">hope other lazy researchers would like this too</div>
""", unsafe_allow_html=True)

# Feature chips (3-column)
st.markdown("""
<div class="feature-row">
  <div class="feature-chip">
    <span class="icon">Why us 1</span>
    <span class="label">Real-time arXiv</span>
    <span class="desc">Papers published this week, not from a stale training set</span>
  </div>
  <div class="feature-chip">
    <span class="icon">Why us 2</span>
    <span class="label">Structured Extraction</span>
    <span class="desc">Every claim traced to a specific paper and date</span>
  </div>
  <div class="feature-chip">
    <span class="icon">Why us 3</span>
    <span class="label">Cross-paper Insights</span>
    <span class="desc">Trend analysis that ChatGPT cannot replicate</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Example topic chips (single row)
st.markdown('<div class="section-label explore-label">EXPLORE AN EXAMPLE</div>', unsafe_allow_html=True)
chip_cols = st.columns(len(_EXAMPLE_TOPICS))
for i, example in enumerate(_EXAMPLE_TOPICS):
    with chip_cols[i]:
        if st.button(example, key=f"chip_{i}"):
            st.session_state["topic_prefill"] = example
            st.rerun()

st.markdown('<div class="spacer-24"></div>', unsafe_allow_html=True)

# Topic input
st.markdown('<div class="section-label">Research Topic</div>', unsafe_allow_html=True)
topic_value = st.session_state.get("topic_prefill", "")
topic = st.text_input(
    "Research topic",
    value=topic_value,
    placeholder="e.g. sparse representation, RLHF, vision transformers...",
    label_visibility="collapsed",
)

st.markdown('<div class="spacer-24"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Time Window</div>', unsafe_allow_html=True)

# Time window
time_window = st.radio(
    "Time window",
    options=list(_INTENT_MAP.keys()),
    horizontal=True,
    index=1,
    label_visibility="collapsed",
)
intent = _INTENT_MAP[time_window]

st.markdown('<div class="spacer-24"></div>', unsafe_allow_html=True)

# Analyze button (full width)
analyze_clicked = st.button(
    "Analyze",
    type="primary",
    use_container_width=True,
)

if analyze_clicked and topic.strip():

    from config import MIN_PAPERS_FOR_COMPARISON
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
    loading_hint = st.empty()
    loading_hint.markdown('<div class="loading-hint">Loading...</div>', unsafe_allow_html=True)

    # Validation
    llm_fast = _make_llm(deep=False)
    validation = validate_user_input(topic, llm_fast)

    if not validation["is_valid"]:
        loading_hint.empty()
        st.error(format_rejection_for_ui(validation))
        if validation.get("suggestion"):
            st.info(f"\U0001F4A1 **Try:** {validation['suggestion']}")
        st.stop()

    # Query generation
    keywords = validation.get("extractable_keywords", [])
    query_result = generate_arxiv_query(topic, keywords, llm_fast)
    display_name = query_result["display_name"]
    arxiv_query  = query_result["arxiv_query"]

    st.success(f"Searching arXiv for: **{display_name}**")
    st.caption(f"`{arxiv_query}`")

    # Random movie quote while loading
    movie_quote, source = random.choice(MOVIE_QUOTES)
    st.markdown(f"""
    <div class="quote-box">
        <div class="quote-text">"{movie_quote}"</div>
        <div class="quote-source">\u2014 {source}</div>
    </div>
    """, unsafe_allow_html=True)

    try:
        status_box = st.empty()
        status_box.info("Running analysis...")
        loading_hint.empty()
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
        progress_placeholder = st.empty()

        def render_extract_progress(done_count: int, total_count: int) -> None:
            total = max(total_count, 1)
            percent = int((done_count / total) * 100)
            label = html.escape(f"Extracted {done_count}/{total_count}")
            progress_placeholder.markdown(
                f"""
                <div class="extract-progress">
                  <div class="extract-progress-fill" style="width:{percent}%;"></div>
                  <div class="extract-progress-text">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        render_extract_progress(0, len(papers))

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
                render_extract_progress(done, len(papers))

        progress_placeholder.empty()
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
            report_days = fetch_result.get("days_used")
            try:
                report_sig_params = inspect.signature(generate_report).parameters
            except (TypeError, ValueError):
                report_sig_params = {}
            if "days" in report_sig_params:
                fut_report = ex.submit(generate_report, extracted, display_name, report_days)
            elif "domain" in report_sig_params:
                fut_report = ex.submit(generate_report, extracted, display_name)
            else:
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
        loading_hint.empty()
        st.error(f"Pipeline error: {e}")
        st.info("Try a different topic or extend the time window.")

# â”€â”€ Report display â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if "report" in st.session_state and st.session_state["report"]:

    st.markdown('</div>', unsafe_allow_html=True)

    q = st.session_state.get("last_query", {})
    papers = st.session_state.get("papers", [])
    domains = len(set(p.get("sub_domain", "") for p in papers if p.get("sub_domain")))

    report_title = q.get("display_name", "Research Report")
    file_stem = q.get("display_name", "report").replace(" ", "_")
    safe_title = html.escape(report_title)
    safe_file_stem = html.escape(file_stem)
    report_text = st.session_state.get("report", "")
    if isinstance(report_text, bytes):
        report_text = report_text.decode("utf-8", errors="replace")
    elif not isinstance(report_text, str):
        report_text = str(report_text)
    report_text = _ensure_summary_table_section(report_text, papers)
    report_text = _sync_report_heading_days(
        report_text,
        st.session_state.get("days_used"),
    )
    encoded_report = url_quote(report_text, safe="")

    st.markdown(f"""
    <div class="report-subheader">
      <div class="report-subheader-title">{safe_title}</div>
      <div class="report-subheader-actions">
        <a class="report-download-link" href="data:text/markdown;charset=utf-8,{encoded_report}" download="{safe_file_stem}.md">&#11015; .md</a>
        <a class="report-download-link" href="data:text/plain;charset=utf-8,{encoded_report}" download="{safe_file_stem}.txt">&#11015; .txt</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row (4-column symmetric)
    st.markdown(f"""
    <div class="stats-row" style="max-width:1100px; margin:24px auto;">
      <div class="stat-card">
        <div class="stat-num">{len(papers)}</div>
        <div class="stat-label">Papers analyzed</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{st.session_state.get('days_used', '-')}</div>
        <div class="stat-label">Days searched</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{st.session_state.get('avg_cred', '-')}</div>
        <div class="stat-label">Avg credibility</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{domains}</div>
        <div class="stat-label">Sub-domains</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-container report-main-container">', unsafe_allow_html=True)
    st.markdown('<div class="report-wrapper">', unsafe_allow_html=True)
    st.markdown(report_text)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(_FOOTER_HTML, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

elif not analyze_clicked:
    st.markdown("""
    <div class="empty-state">
        Enter a topic above and click <strong>Analyze</strong> to generate your report
    </div>
    """, unsafe_allow_html=True)
    st.markdown(_FOOTER_HTML, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown(_FOOTER_HTML, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
