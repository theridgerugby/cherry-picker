"""Central prompt registry for all parallel agents.

Each key maps to the system instruction for one agent role.
Prompts here are role-specific: they do not include paper data.
"""

from __future__ import annotations

from report_generator import (
    GAPS_DOMAIN_SPECIFICITY_RULE_TEMPLATE,
    GAPS_SYSTEM_PROMPT,
    METHODOLOGY_CONTRAST_PROMPT,
    REPORT_SYSTEM_PROMPT,
)
from skills_analyzer import SKILL_EXTRACTION_SYSTEM_PROMPT
from trend_analyzer import TREND_SYSTEM_PROMPT


def get_report_prompt(domain: str, days: int, date: str) -> str:
    """System prompt for the main report generation agent."""
    return REPORT_SYSTEM_PROMPT.format(domain=domain, days=days, date=date)


def get_gaps_prompt(domain: str) -> str:
    """System prompt for the gap analysis agent."""
    domain_rule = GAPS_DOMAIN_SPECIFICITY_RULE_TEMPLATE.format(domain=domain)
    return GAPS_SYSTEM_PROMPT + "\n\n" + domain_rule


def get_matrix_prompt() -> str:
    """System prompt for the methodology matrix agent."""
    return METHODOLOGY_CONTRAST_PROMPT


def get_trend_prompt() -> str:
    """System prompt for the trend analysis agent."""
    return TREND_SYSTEM_PROMPT


def get_skill_prompt() -> str:
    """System prompt for the skill extraction agent."""
    return SKILL_EXTRACTION_SYSTEM_PROMPT
