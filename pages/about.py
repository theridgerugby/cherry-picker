# about.py — Cherry Picker About page

import streamlit as st

st.set_page_config(
    page_title="About · Cherry Picker",
    page_icon="\U0001f352",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --color-primary: #2563EB;
    --color-text: #111827;
    --color-text-secondary: #6B7280;
    --color-border: rgba(0,0,0,0.08);
    --color-surface: rgba(255,255,255,0.72);
}

* {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Helvetica, Arial, sans-serif !important;
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
    font-size: 16px;
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
    color: var(--color-text) !important;
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

.about-container {
    max-width: 620px;
    margin: 80px auto 64px;
    padding: 0 24px;
}

.about-back {
    font-size: 13px;
    color: var(--color-text-secondary) !important;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 40px;
    transition: color 0.15s ease;
}

.about-back:hover {
    color: var(--color-primary) !important;
}

.about-title {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.8px;
    color: var(--color-text) !important;
    margin-bottom: 48px;
    line-height: 1.2;
}

.about-section {
    margin-bottom: 40px;
}

.about-section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--color-text-secondary) !important;
    margin-bottom: 12px;
}

.about-section-body {
    font-size: 15px;
    line-height: 1.75;
    color: var(--color-text-secondary) !important;
}

.about-section-body strong {
    color: var(--color-text) !important;
    font-weight: 600;
}

.about-lesson {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
    align-items: flex-start;
}

.about-lesson-num {
    font-size: 11px;
    font-weight: 700;
    color: var(--color-primary) !important;
    min-width: 20px;
    padding-top: 3px;
    letter-spacing: 0.5px;
}

.about-lesson-body {
    font-size: 15px;
    line-height: 1.75;
    color: var(--color-text-secondary) !important;
}

.about-lesson-body strong {
    color: var(--color-text) !important;
}

.about-divider {
    height: 1px;
    background: var(--color-border);
    margin: 36px 0;
}

.about-ack {
    font-size: 14px;
    line-height: 1.75;
    color: var(--color-text-secondary) !important;
    font-style: italic;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="top-navbar">
  <div class="nav-brand">&#x1F352; Cherry Picker</div>
  <div class="nav-actions">
    <a class="nav-link" href="/" target="_self">Home</a>
    <a class="nav-link" href="/about" target="_self" style="color: var(--color-text) !important; font-weight: 600;">About</a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="about-container">
  <a class="about-back" href="/" target="_self">&larr; Cherry Picker</a>
  <div class="about-title">About this project</div>

  <div class="about-section">
    <div class="about-section-label">Why I built this</div>
    <div class="about-section-body">
      I built this because I'm a freshman who wants to look at as many fields as possible to figure out what to do with life. The best way to get a feel for a field is to read its research papers, so I came up with the idea to build an AI agent that retrieves recent papers to quickly map out industry trends.
    </div>
  </div>

  <div class="about-divider"></div>

  <div class="about-section">
    <div class="about-section-label">How it was built</div>
    <div class="about-section-body">
      The whole thing went from a random idea — mostly brainstormed in the car on a ski trip — to a working prototype in about 20 hours. I figured out the architecture with Sonnet and GPT; for most of the code implementation, my prompts are rephrased by Sonnet and then executed by either Claude Code or Codex. My job was basically just planning the agent's workflow logic — deciding what features are needed or not, listing out infrastructure, annotating when AI does something stupid — and writing prompts.
    </div>
  </div>

  <div class="about-divider"></div>

  <div class="about-section">
    <div class="about-section-label">Lessons learned</div>
    <div class="about-lesson">
      <div class="about-lesson-num">01</div>
      <div class="about-lesson-body"><strong>Context is everything.</strong> When working with multiple LLMs, you have to keep their context updated. If you brainstorm a feature with Claude, it doesn't automatically know what Codex just wrote — which can eventually lead to code conflicts. Don't forget to give them the most up-to-date files.</div>
    </div>
    <div class="about-lesson">
      <div class="about-lesson-num">02</div>
      <div class="about-lesson-body"><strong>Force a plan before you build.</strong> LLMs are generally good at writing code but rarely delete old code, which leaves you with a massive chunk of dead code stacked in your folder. Before letting the AI implement a feature, force it to generate a <code>plan.md</code> that explicitly lists what will be added and what will be removed.</div>
    </div>
    <div class="about-lesson">
      <div class="about-lesson-num">03</div>
      <div class="about-lesson-body"><strong>Don't use an LLM if a library can do the job.</strong> LLMs are inherently unstable and you can't always trust them to get it right. I wasted time trying to get Gemini to extract GitHub links until I realized I could just write a deterministic file that pulls from Papers With Code, PDF regex, and arXiv metadata. Keep it simple.</div>
    </div>
  </div>

  <div class="about-divider"></div>

  <div class="about-section">
    <div class="about-section-label">Thanks</div>
    <div class="about-ack">
      Special thanks to Google for the Gemini API credits — literally couldn't have shipped this without them. Thanks to my research partner Yikai Tang for sparking the idea during a Zoom call, and to my friends who drove to St. Louis Mountain while I zoned out and brainstormed in the passenger seat. Boring times on the road always give me interesting ideas.
    </div>
  </div>
</div>

<div class="footer" style="max-width:620px; margin:0 auto; padding:0 24px;">
  <span>&copy; 2026 Cherry Picker &middot; Built with LangChain & Gemini</span>
  <span>Made for DBW Lab</span>
</div>
""",
    unsafe_allow_html=True,
)
