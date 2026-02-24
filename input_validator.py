# input_validator.py — 用户研究主题输入验证模块
#
# 在传递给 query generator 之前验证用户输入，拒绝无效输入并返回结构化反馈。

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage


# ── 常量 ──────────────────────────────────────────────────────────────────────

_STOPWORDS = {"the", "a", "an", "is", "are", "what", "how", "why", "i", "me"}

_VALIDATION_SYSTEM_PROMPT = """You are an academic search query validator.
Given a user's research interest input, determine if it is suitable for
searching academic papers on arXiv.

Evaluate the input and return ONLY valid JSON:
{
  "is_academic": true | false,
  "reason": "string (one sentence)",
  "extractable_keywords": ["string"] or [],
  "suggested_refinement": "string or null"
}

Mark is_academic as false if:
- The topic is about sports, entertainment, celebrities, or pop culture
  with no academic angle (e.g. "lebron james nfl stats")
- The input is too vague to extract any technical keywords
  (e.g. "interesting stuff", "science things")
- The input describes a person's personal data or private information

Mark is_academic as true even if the topic seems niche or unusual,
as long as it could plausibly appear in an academic paper."""


# ── Layer 1: Rule-based validation (no LLM, instant) ─────────────────────────

def _rule_too_short(raw: str) -> dict | None:
    if len(raw.strip()) < 3:
        return {
            "is_valid": False,
            "rejection_reason": "Input is too short to be a research topic.",
            "rejection_category": "too_short",
            "user_message": (
                "Your input is too short. Please describe your research "
                "interest in at least a few words."
            ),
            "suggestion": None,
        }
    return None


def _rule_no_letters(raw: str) -> dict | None:
    if re.match(r'^[^a-zA-Z]*$', raw):
        return {
            "is_valid": False,
            "rejection_reason": "Input contains no alphabetic characters.",
            "rejection_category": "nonsense",
            "user_message": (
                "Your input doesn't contain any letters. Please enter a "
                "research topic using words, e.g. 'sparse representation'."
            ),
            "suggestion": None,
        }
    return None


def _rule_no_vowels(raw: str) -> dict | None:
    letters_only = re.sub(r'[^a-zA-Z]', '', raw)
    if letters_only and not re.search(r'[aeiouAEIOU]', letters_only):
        return {
            "is_valid": False,
            "rejection_reason": "Input appears to be gibberish (no vowels).",
            "rejection_category": "nonsense",
            "user_message": (
                "That doesn't look like a real research topic. "
                "Could you try again with actual words?"
            ),
            "suggestion": None,
        }
    return None


def _rule_only_stopwords(raw: str) -> dict | None:
    words = set(re.findall(r'[a-zA-Z]+', raw.lower()))
    if words and words.issubset(_STOPWORDS):
        return {
            "is_valid": False,
            "rejection_reason": "Input contains only common stopwords.",
            "rejection_category": "too_short",
            "user_message": (
                "Your input is too vague — it only contains common words "
                f"like '{', '.join(sorted(words))}'. Please add specific "
                "technical terms."
            ),
            "suggestion": "Try something like 'graph neural networks' or "
                          "'natural language processing'.",
        }
    return None


_LAYER1_RULES = [
    _rule_too_short,
    _rule_no_letters,
    _rule_no_vowels,
    _rule_only_stopwords,
]


# ── Layer 2: LLM-based validation ────────────────────────────────────────────

def _clean_json_response(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def _llm_validate(raw_input: str, llm) -> dict:
    """
    Call the LLM to judge whether the input is academically valid.
    Returns the final validation dict.
    """
    try:
        response = llm.invoke([
            SystemMessage(content=_VALIDATION_SYSTEM_PROMPT),
            HumanMessage(content=f"User input: {raw_input}"),
        ])
        parsed = json.loads(_clean_json_response(response.content))
    except (json.JSONDecodeError, Exception) as e:
        # If LLM call fails, be lenient — let it through
        print(f"[Validator] LLM validation failed, allowing input: {e}")
        return {
            "is_valid": True,
            "rejection_reason": None,
            "rejection_category": None,
            "user_message": "Input accepted (LLM validation skipped).",
            "suggestion": None,
            "extractable_keywords": [],
        }

    is_academic = parsed.get("is_academic", False)
    keywords = parsed.get("extractable_keywords", [])
    if not isinstance(keywords, list):
        keywords = []

    if not is_academic or not keywords:
        reason = parsed.get("reason", "No academic keywords found.")
        suggestion = parsed.get("suggested_refinement")
        return {
            "is_valid": False,
            "rejection_reason": reason,
            "rejection_category": "no_keywords_extractable",
            "user_message": (
                f"We couldn't find academic search terms in '{raw_input}'. "
                f"{reason}"
            ),
            "suggestion": suggestion,
        }

    return {
        "is_valid": True,
        "rejection_reason": None,
        "rejection_category": None,
        "user_message": "Input accepted.",
        "suggestion": None,
        "extractable_keywords": keywords,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def validate_user_input(raw_input: str, llm) -> dict:
    """
    Validate user research topic input in two layers:
      Layer 1 — rule-based (instant, no LLM)
      Layer 2 — LLM-based (only if Layer 1 passes)

    Returns a dict with keys:
      is_valid, rejection_reason, rejection_category, user_message, suggestion
    """
    # Layer 1: fast rule checks
    for rule in _LAYER1_RULES:
        result = rule(raw_input)
        if result is not None:
            return result

    # Layer 2: LLM-based academic validation
    return _llm_validate(raw_input, llm)


def format_rejection_for_ui(validation_result: dict) -> str:
    """
    Format a rejection result as a friendly markdown string for Streamlit.
    Returns empty string if input was valid.
    """
    if validation_result.get("is_valid", True):
        return ""

    msg = validation_result.get("user_message", "Invalid input.")

    lines = [
        "❌ **We couldn't process this topic.**",
        "",
        f"*Reason:* {msg}",
    ]

    return "\n".join(lines)
