"""Gemini Context Caching lifecycle management.

Owns cache creation, retrieval, and cleanup for the parallel agent pipeline.
Called once per user request before parallel agents are spawned.
"""

from __future__ import annotations

import datetime
import json
import os

import google.generativeai as genai

# Minimum content that makes caching worthwhile.
# Gemini requires >= 32,768 tokens to activate cache discount.
# We cache regardless - correctness over cost optimization.
_CACHE_TTL_MINUTES = 15
_FAST_MODEL = "gemini-1.5-flash-001"


def _get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set. Cannot initialize Gemini Context Cache.")
    return key


def build_shared_cache(
    papers: list[dict],
    domain: str,
    model_name: str = _FAST_MODEL,
    ttl_minutes: int = _CACHE_TTL_MINUTES,
) -> genai.caching.CachedContent | None:
    """Upload paper data once as a cached context shared by all parallel agents."""
    genai.configure(api_key=_get_api_key())

    paper_json = json.dumps(papers, ensure_ascii=False, indent=2)
    cache_preamble = (
        "The following is the complete set of extracted research papers for the domain "
        f'"{domain}". All subsequent requests in this session refer to these papers.\n\n'
        f"{paper_json}"
    )

    try:
        cache = genai.caching.CachedContent.create(
            model=model_name,
            contents=[
                {
                    "role": "user",
                    "parts": [{"text": cache_preamble}],
                }
            ],
            ttl=datetime.timedelta(minutes=ttl_minutes),
        )
        print(f"[Cache] Created: {cache.name} | expires in {ttl_minutes}min | model={model_name}")
        return cache
    except Exception as exc:
        print(f"[Cache] Creation failed (will use direct transmission): {exc}")
        return None


def delete_cache(cache: genai.caching.CachedContent | None) -> None:
    """Delete the cache after all agents finish. Safe to call with None."""
    if cache is None:
        return
    try:
        cache.delete()
        print(f"[Cache] Deleted: {cache.name}")
    except Exception as exc:
        print(f"[Cache] Delete failed (will expire naturally): {exc}")


def make_model_from_cache(
    cache: genai.caching.CachedContent,
    system_instruction: str,
) -> genai.GenerativeModel:
    """Build a GenerativeModel that reads from an existing cache."""
    return genai.GenerativeModel.from_cached_content(
        cached_content=cache,
        system_instruction=system_instruction,
    )
