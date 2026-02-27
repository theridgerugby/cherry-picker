# pages/about.py -- Cherry Picker About page

import streamlit as st

st.set_page_config(
    page_title="About · Cherry Picker",
    page_icon="🍒",
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
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    font-size: 14px !important;
    line-height: 1.6 !important;
    color: var(--color-text-secondary) !important;
}

a { color: var(--color-primary) !important; }

.top-navbar {
    position: fixed;
    top: 0; right: 0; left: 0;
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
    color: var(--color-text-secondary) !important;
    text-decoration: none;
    transition: color 0.15s ease;
}

.nav-link:hover { color: var(--color-primary) !important; }

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
    .stApp { background: #0f0f0f !important; }
    .top-navbar { background: rgba(15,15,15,0.85); }
}

/* -- About-specific styles -- */

.about-container {
    max-width: 920px;
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

.about-back:hover { color: var(--color-primary) !important; }

.about-title {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.8px;
    color: var(--color-text) !important;
    margin-bottom: 48px;
    line-height: 1.2;
}

.about-section { margin-bottom: 40px; }

.about-section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--color-text-secondary) !important;
    margin-bottom: 12px;
}

.about-body {
    font-size: 15px;
    line-height: 1.75;
    color: var(--color-text-secondary) !important;
}

.about-body strong { color: var(--color-text) !important; font-weight: 600; }

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

.about-lesson-body strong { color: var(--color-text) !important; }

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
  <div class="nav-brand">🍒 Cherry Picker</div>
  <div class="nav-actions">
    <a class="nav-link" href="/" target="_self">Home</a>
    <a class="nav-link" href="/?page=about" target="_self"
       style="color: var(--color-text) !important; font-weight: 600;">About</a>
    <a class="nav-link" href="https://github.com/theridgerugby/cherry-picker"
       target="_blank" rel="noopener noreferrer">GitHub</a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="about-container">

  <a class="about-back" href="/" target="_self">← Cherry Picker</a>

  <div class="about-title">About this project</div>

  <!-- Section 1 -->
  <div class="about-section">
    <div class="about-section-label">Why I built this</div>
    <div class="about-body">
      I built this because I'm a university freshman who wants to look at as many fields
      as possible to figure out what to do with life. The best way to get a
      feel for a field is to read its research papers, so I came up with the
      idea to build an AI agent that retrieves recent papers to quickly map out
      industry trends.
    </div>
  </div>

  <div class="about-divider"></div>

  <!-- Section 2 -->
  <div class="about-section">
    <div class="about-section-label">How it was built</div>
    <div class="about-body">
      The whole thing went from a random idea — mostly brainstormed in the car
      on a ski trip — to a working prototype in about 20 hours. I figured out
      the architecture with Sonnet and GPT; for most of the code implementation,
      my prompts are rephrased by Sonnet and then executed by either Claude Code
      or Codex. My job was basically just planning the agent's workflow logic
      (e.g. deciding what features are needed or not, listing out infrastructure,
      annotating when AI does something stupid) and writing prompts.
    </div>
  </div>

  <div class="about-divider"></div>

  <!-- Section 3 -->
  <div class="about-section">
    <div class="about-section-label">Lessons learned</div>
    <div class="about-body">
      <p><strong>01. Context is all they need.</strong> When working with multiple LLMs, you have to keep their context updated. If you brainstorm a feature with Claude, it doesn't automatically know what Codex just wrote, which can eventually lead to code conflicts. Don't forget to give them the most up-to-date files.</p>
      <p><strong>02. Force a plan before you build.</strong> LLMs are generally good at writing code but rarely delete old code, which leaves you with a massive chunk of dead code stacked in your folder. Before letting the AI implement a feature, force it to generate a <code>plan.md</code> that explicitly lists what will be added and what will be removed.</p>
      <p><strong>03. DO NOT rely on an LLM if an API can do the job.</strong> LLMs are inherently unstable and you can't always trust them to get it right. I wasted time trying to get Gemini to extract GitHub links until I realized I could just write a deterministic file that pulls from Papers With Code, PDF regex, and arXiv metadata. Keep the job simple.</p>
    </div>
  </div>

  <div class="about-divider"></div>

  <!-- Section 4 -->
  <div class="about-section">
    <div class="about-section-label">Thanks</div>
    <div class="about-ack">
      Special thanks to Google for the Gemini API credits — literally couldn't
      have shipped this without them. Thanks to my research partner Yikai Tang
      for sparking the idea during a Zoom call, and to my friends who drove to
      St. Louis Mountain while I zoned out and brainstormed in the passenger
      seat. Boring times on the road always give me interesting ideas.
    </div>
  </div>

</div>

<div style="max-width:920px; margin:0 auto; padding:0 24px;">
  <div class="footer">
    <span>&copy; 2026 Cherry Picker &middot; Built with LangChain &amp; Gemini</span>
    <span>Made for DBW Lab</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
