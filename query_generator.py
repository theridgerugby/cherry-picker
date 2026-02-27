# query_generator.py — 将用户研究兴趣转化为优化的 arXiv 搜索查询

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import ARXIV_CATEGORY_MAP

# Query generation is a simple, high-frequency task — use a lightweight model
# regardless of what the main pipeline is configured to use.
_query_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


QUERY_SYSTEM_PROMPT = """You are an academic librarian expert in arXiv search syntax.
The user's input may be colloquial or vague. Your job is to produce
a precise arXiv boolean query that maximizes recall of relevant papers
while minimizing noise.

You MUST correctly identify the PRIMARY DISCIPLINE of the topic first,
then select categories accordingly.

Full arXiv category reference:
  --- Computer Science / ML ---
  cs.LG  = Machine Learning
  cs.AI  = Artificial Intelligence
  cs.CV  = Computer Vision
  cs.CL  = Natural Language Processing
  cs.NE  = Neural and Evolutionary Computing
  cs.IR  = Information Retrieval
  cs.RO  = Robotics
  eess.SP = Signal Processing
  stat.ML = Statistics / Machine Learning
  math.OC = Optimization and Control

  --- Physical Sciences ---
  cond-mat.mtrl-sci = Materials Science
  cond-mat.soft     = Soft Condensed Matter
  physics.app-ph    = Applied Physics
  physics.chem-ph   = Chemical Physics
  physics.flu-dyn   = Fluid Dynamics
  quant-ph          = Quantum Physics

  --- Astrophysics ---
  astro-ph.ga = Astrophysics of Galaxies
  astro-ph.co = Cosmology and Nongalactic Astrophysics
  astro-ph.he = High Energy Astrophysical Phenomena
  astro-ph.im = Instrumentation and Methods for Astrophysics

  --- Life Sciences ---
  q-bio.BM = Biomolecules
  q-bio.CB = Cell Behavior
  q-bio.NC = Neurons and Cognition

  --- Engineering / Interdisciplinary ---
  eess.IV = Image and Video Processing
  eess.SY = Systems and Control
  math.NA = Numerical Analysis

DISCIPLINE DETECTION RULES (apply in order):
0. If the topic is astrophysics/astronomy/cosmology (e.g., galaxies, exoplanets,
   black holes, gravitational waves, star formation) -> use astro-ph categories
   (astro-ph.ga, astro-ph.co, astro-ph.he, astro-ph.im)
1. If the topic involves materials, coatings, surfaces, polymers, alloys,
   nanostructures, or chemical properties -> use cond-mat.mtrl-sci and/or physics.app-ph
2. If the topic involves biology, medicine, proteins, cells -> use q-bio categories
3. If the topic involves fluid, heat transfer, aerodynamics -> use physics.flu-dyn
4. If the topic involves robotics, control systems -> use cs.RO and eess.SY
5. If the topic is clearly ML/AI -> use cs.LG, cs.AI, stat.ML
6. If the topic is INTERDISCIPLINARY (e.g. AI for materials, ML for drug discovery),
   include BOTH the domain category AND the ML category

CRITICAL RULE: Never force CS/ML categories onto non-CS topics.
"anti-ice coating" is materials science -> use cond-mat.mtrl-sci, NOT cs.LG.

Query construction rules:
- Prefer at least one cat: filter matching the detected discipline
- If the topic is broad and category filters would sharply reduce recall, use a broader query without cat: filters
- Use ti: for specific technical terms (high precision)
- Use abs: for broader concept terms (higher recall)
- Keep total query under 450 characters
- Do NOT include date ranges

Examples:
Input: "astrophysics signals"
Output: (ti:astrophysics OR abs:cosmology OR abs:"gravitational wave"
  OR abs:galaxy OR abs:exoplanet OR abs:"star formation")
  AND (cat:astro-ph.ga OR cat:astro-ph.co OR cat:astro-ph.he OR cat:astro-ph.im)

Input: "anti-ice coating"
Output: (ti:"anti-ice" OR ti:icephobic OR ti:"ice adhesion"
  OR abs:"superhydrophobic") AND (cat:cond-mat.mtrl-sci OR cat:physics.app-ph)

Input: "machine learning for drug discovery"
Output: (abs:"drug discovery" OR abs:"molecular property")
  AND (cat:cs.LG OR cat:q-bio.BM) AND (ti:graph OR ti:transformer OR ti:generative)

Input: "sparse representation"
Output: (ti:"sparse representation" OR ti:"sparse coding"
  OR ti:"dictionary learning") AND (cat:cs.LG OR cat:eess.SP OR cat:stat.ML)

Input: "climate model inference"
Output: (abs:"climate model" OR ti:"weather prediction" OR ti:forecasting)
  AND (cat:cs.LG OR cat:physics.app-ph OR cat:eess.SY)

Input: "quantum error correction"
Output: (ti:"quantum error" OR ti:"quantum fault" OR abs:"qubit")
  AND (cat:quant-ph OR cat:cs.IT)

Output ONLY the raw query string. No JSON, no explanation, no quotes around
the entire string. Just the query."""


_KNOWN_CATEGORIES = sorted(
    {
        *ARXIV_CATEGORY_MAP.keys(),
        # Keep a few common variants that may appear in LLM output.
        "q-bio",
        "physics.geo-ph",
    },
    key=len,
    reverse=True,
)


def _normalize_category_syntax(query: str) -> str:
    """
    Ensure category codes are consistently prefixed with `cat:`.
    """
    normalized = query
    for code in _KNOWN_CATEGORIES:
        # Replace standalone category code not already prefixed by cat:
        # Example: "... OR stat.ML)" -> "... OR cat:stat.ML)"
        pattern = re.compile(rf"(?<!cat:)(?<![\w.-]){re.escape(code)}(?![\w.-])", re.IGNORECASE)
        normalized = pattern.sub(f"cat:{code}", normalized)
    return normalized


def _trim_query_safely(query: str, limit: int = 450) -> str:
    """
    Trim long query without cutting in the middle of a token if possible.
    """
    if len(query) <= limit:
        return query
    cut = query.rfind(" AND ", 0, limit)
    if cut > 0:
        return query[:cut].strip()
    return query[:limit].strip()


def _rule_based_query_override(raw_input: str) -> str | None:
    """
    Deterministic overrides for topics where narrow phrasing often causes zero recall.
    """
    topic = (raw_input or "").strip().lower()
    anti_ice_markers = [
        "anti-ice",
        "anti ice",
        "anti-icing",
        "anti icing",
        "icephobic",
        "ice adhesion",
    ]
    if any(marker in topic for marker in anti_ice_markers):
        return (
            '(ti:"anti-ice" OR ti:icephobic OR ti:"ice adhesion" '
            'OR abs:"superhydrophobic" OR abs:"anti-icing") '
            "AND (cat:cond-mat.mtrl-sci OR cat:physics.app-ph OR cat:cond-mat.soft)"
        )

    astrophysics_markers = [
        "astrophysics",
        "astronomy",
        "cosmology",
        "galaxy",
        "exoplanet",
        "black hole",
        "gravitational wave",
        "star formation",
    ]
    if any(marker in topic for marker in astrophysics_markers):
        return (
            '(ti:astrophysics OR abs:cosmology OR abs:"gravitational wave" '
            'OR abs:galaxy OR abs:exoplanet OR abs:"star formation") '
            "AND (cat:astro-ph.ga OR cat:astro-ph.co OR cat:astro-ph.he OR cat:astro-ph.im)"
        )
    return None


def _make_display_name(raw_input: str) -> str:
    """Derive a clean human-readable topic name from raw user input."""
    # Strip leading/trailing whitespace and convert to title case,
    # collapsing internal whitespace runs.
    cleaned = re.sub(r"\s+", " ", raw_input.strip())
    return cleaned.title()


def generate_arxiv_query(
    raw_input: str,
    keywords: list[str],
    llm=None,  # ignored — module uses its own lightweight model
) -> dict:
    """
    Generate an optimized arXiv search query from validated user input.

    The `llm` parameter is accepted for API compatibility but ignored;
    this module always uses its own gemini-1.5-flash instance.

    Returns:
        {
          "arxiv_query":  str,
          "display_name": str,
          "sub_topics":   list[str],
          "user_level_detected": "beginner" | "intermediate" | "expert",
        }
    """
    user_msg = f"Research interest: {raw_input}\nPre-extracted keywords: {', '.join(keywords)}"

    override_query = _rule_based_query_override(raw_input)
    if override_query:
        arxiv_query = override_query
        print("[QueryGen] Using rule-based query override.")
    else:
        try:
            response = _query_llm.invoke(
                [
                    SystemMessage(content=QUERY_SYSTEM_PROMPT),
                    HumanMessage(content=user_msg),
                ]
            )
            arxiv_query = response.content.strip()
        except Exception as e:
            # Fallback: build a minimal query from keywords
            print(f"[QueryGen] LLM call failed, using keyword fallback: {e}")
            kw_parts = " OR ".join(f"ti:{kw}" for kw in keywords[:4])
            arxiv_query = kw_parts or f"ti:{raw_input}"

    # Normalize category syntax to avoid malformed category clauses.
    arxiv_query = _normalize_category_syntax(arxiv_query)

    # Hard cap at 450 characters as per the prompt rules
    if len(arxiv_query) > 450:
        arxiv_query = _trim_query_safely(arxiv_query, limit=450)
        print("[QueryGen] Query trimmed to <= 450 characters.")

    display_name = _make_display_name(raw_input)

    print(f"[QueryGen] Query  : {arxiv_query}")
    print(f"[QueryGen] Display: {display_name}")

    return {
        "arxiv_query": arxiv_query,
        "display_name": display_name,
        "sub_topics": keywords[:5],
        "user_level_detected": "intermediate",  # not inferred in this version
    }
