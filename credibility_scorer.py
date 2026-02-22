# credibility_scorer.py — 论文可信度评分模块

import re
import datetime


# 顶级会议/期刊关键词
_TOP_VENUES = [
    "neurips", "nips", "icml", "iclr", "cvpr", "iccv", "eccv",
    "aaai", "ijcai", "acl", "emnlp", "naacl", "sigir", "kdd",
    "nature", "science", "ieee", "transactions",
    "jmlr", "tmlr", "pami",
]

# 顶级机构关键词
_TOP_INSTITUTIONS = [
    "mit", "stanford", "cmu", "carnegie mellon",
    "google", "deepmind", "microsoft research",
    "oxford", "cambridge", "tsinghua", "pku", "peking university",
    "berkeley", "eth zurich", "meta ai", "openai",
]

# arXiv 类别 → 领域关键词映射
_DOMAIN_CATEGORY_MAP = {
    "sparse representation": ["cs.cv", "cs.lg", "cs.ai", "eess.sp", "stat.ml", "eess.iv"],
    "dictionary learning":   ["cs.lg", "eess.sp", "stat.ml"],
    "compressed sensing":    ["eess.sp", "cs.it", "math.oc"],
    "sparse coding":         ["cs.cv", "cs.lg", "cs.ne"],
}


def score_paper_credibility(paper: dict, target_domain: str = "sparse representation") -> dict:
    """
    对单篇论文进行可信度评分（0-100），并附上评分明细。

    参数:
        paper: 包含论文元数据的字典（需含 title, abstract, authors 等字段）
        target_domain: 目标研究领域（用于类别匹配评分）

    返回:
        原字典的副本，新增 credibility_score, credibility_breakdown, venue_detected 字段
    """
    scored = dict(paper)  # 不修改原字典
    breakdown = {}
    venue_detected = None

    # ── 1. 会议/期刊评分（+30）──────────────────────────────────────────────
    venue_score = 0
    comment = str(paper.get("comment", "") or "").lower()
    journal_ref = str(paper.get("journal_ref", "") or "").lower()
    venue_text = f"{comment} {journal_ref}"

    for venue in _TOP_VENUES:
        if venue in venue_text:
            venue_score = 30
            venue_detected = venue.upper()
            break

    breakdown["venue"] = venue_score

    # ── 2. 作者机构评分（+20）──────────────────────────────────────────────
    institution_score = 0
    # 从 authors (list[str]) 和 comment 中检测机构
    authors_text = " ".join(str(a) for a in paper.get("authors", [])).lower()
    affiliations_text = str(paper.get("affiliations", "") or "").lower()
    institution_text = f"{authors_text} {affiliations_text} {comment}"

    for inst in _TOP_INSTITUTIONS:
        if inst in institution_text:
            institution_score = 20
            break

    breakdown["institution"] = institution_score

    # ── 3. arXiv 类别匹配评分（+15）───────────────────────────────────────
    category_score = 0
    primary_category = str(paper.get("primary_category", "") or "").lower()
    categories = [str(c).lower() for c in paper.get("categories", [])]
    all_cats = [primary_category] + categories

    target_key = target_domain.lower()
    relevant_cats = _DOMAIN_CATEGORY_MAP.get(target_key, [])
    # 也做模糊匹配：如果没有精确命中，尝试部分匹配
    if not relevant_cats:
        for key, cats in _DOMAIN_CATEGORY_MAP.items():
            if key in target_key or target_key in key:
                relevant_cats = cats
                break

    for cat in all_cats:
        if cat in relevant_cats:
            category_score = 15
            break

    breakdown["category_match"] = category_score

    # ── 4. 时效性评分（+20）───────────────────────────────────────────────
    recency_score = 0
    pub_date_str = paper.get("published_date", "")
    if pub_date_str:
        try:
            pub_date = datetime.datetime.strptime(pub_date_str, "%Y-%m-%d")
            days_ago = (datetime.datetime.now() - pub_date).days
            if days_ago <= 30:
                recency_score = 20
            elif days_ago <= 90:
                recency_score = 10
            elif days_ago <= 180:
                recency_score = 5
        except ValueError:
            pass

    breakdown["recency"] = recency_score

    # ── 5. 摘要丰富度评分（+15）───────────────────────────────────────────
    abstract_score = 0
    abstract = str(paper.get("abstract", "") or "")
    if len(abstract) > 1500:
        abstract_score = 15

    breakdown["abstract_length"] = abstract_score

    # ── 汇总 ──────────────────────────────────────────────────────────────
    total = sum(breakdown.values())
    scored["credibility_score"] = total
    scored["credibility_breakdown"] = breakdown
    scored["venue_detected"] = venue_detected

    return scored
