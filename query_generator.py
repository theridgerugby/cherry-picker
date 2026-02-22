# query_generator.py — 将用户研究兴趣转化为优化的 arXiv 搜索查询

import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


# Query generation is a simple, high-frequency task — use a lightweight model
# regardless of what the main pipeline is configured to use.
_query_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)


QUERY_SYSTEM_PROMPT = """You are an academic librarian expert in arXiv search syntax.
The user's input may be colloquial or vague. Your job is to produce
a precise arXiv boolean query that maximizes recall of relevant papers
while minimizing noise.

You MUST use arXiv category operators. Use this category reference:
  cs.LG  = Machine Learning
  cs.AI  = Artificial Intelligence  
  cs.CV  = Computer Vision
  cs.CL  = Natural Language Processing
  cs.NE  = Neural and Evolutionary Computing
  cs.IR  = Information Retrieval
  eess.SP = Signal Processing
  stat.ML = Statistics / Machine Learning
  math.OC = Optimization and Control
  q-bio   = Quantitative Biology (for interdisciplinary topics)

Query construction rules:
- Always include at least one cat: filter
- Use ti: for specific technical terms (high precision)
- Use abs: for broader concept terms (higher recall)
- Combine categories with OR inside parentheses
- Keep total query under 300 characters
- Do NOT include date ranges in the query string
  (date filtering is handled separately by the fetcher)

Examples:
Input: "machine learning"
Output: abs:"machine learning" AND (cat:cs.LG OR cat:cs.AI OR stat.ML)
  AND (ti:efficient OR ti:novel OR ti:robust OR ti:scalable)

Input: "sparse representation"
Output: (ti:"sparse representation" OR ti:"sparse coding"
  OR ti:"dictionary learning") AND (cat:cs.LG OR eess.SP OR stat.ML)

Input: "multimodal architectures that reduce GPU memory"
Output: (ti:multimodal OR abs:multimodal)
  AND (ti:"memory efficient" OR ti:"parameter efficient" OR ti:lightweight)
  AND (cat:cs.LG OR cat:cs.CV OR cat:cs.CL)

Input: "drug discovery using graphs"
Output: (ti:"graph neural" OR ti:GNN OR abs:"molecular graph")
  AND (cat:cs.LG OR q-bio) AND (ti:drug OR ti:molecular OR ti:protein)

Output ONLY the raw query string. No JSON, no explanation, no quotes around
the entire string. Just the query."""


def _make_display_name(raw_input: str) -> str:
    """Derive a clean human-readable topic name from raw user input."""
    # Strip leading/trailing whitespace and convert to title case,
    # collapsing internal whitespace runs.
    cleaned = re.sub(r'\s+', ' ', raw_input.strip())
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
    user_msg = (
        f"Research interest: {raw_input}\n"
        f"Pre-extracted keywords: {', '.join(keywords)}"
    )

    try:
        response = _query_llm.invoke([
            SystemMessage(content=QUERY_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        arxiv_query = response.content.strip()
    except Exception as e:
        # Fallback: build a minimal query from keywords
        print(f"[QueryGen] LLM call failed, using keyword fallback: {e}")
        kw_parts = " OR ".join(f"ti:{kw}" for kw in keywords[:4])
        arxiv_query = kw_parts or f"ti:{raw_input}"

    # Hard cap at 300 characters as per the prompt rules
    if len(arxiv_query) > 300:
        arxiv_query = arxiv_query[:300]
        print("[QueryGen] Query truncated to 300 characters.")

    display_name = _make_display_name(raw_input)

    print(f"[QueryGen] Query  : {arxiv_query}")
    print(f"[QueryGen] Display: {display_name}")

    return {
        "arxiv_query": arxiv_query,
        "display_name": display_name,
        "sub_topics": keywords[:5],
        "user_level_detected": "intermediate",  # not inferred in this version
    }
