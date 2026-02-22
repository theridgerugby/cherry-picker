# skills_analyzer.py — 技术技能与跨学科知识分析模块

import json
import os
import re
import threading
from collections import Counter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import GEMINI_MODEL, DOMAIN, MIN_PAPERS_FOR_ROADMAP, SKILL_CACHE_PATH


# ── Prompt：技能提取 ──────────────────────────────────────────────────────────

SKILL_EXTRACTION_SYSTEM_PROMPT = """You are a research skill analyst. Given a structured paper summary, extract ALL technical
and disciplinary knowledge required to understand, reproduce, or extend this work.

Be specific and granular. Do not generalize "machine learning" when you can say "sparse autoencoder".
For interdisciplinary requirements, identify the source discipline explicitly.

Output ONLY valid JSON:
{
  "ml_methods": ["string"],
  "math_foundations": ["string"],
  "programming_tools": ["string"],
  "domain_knowledge": ["string"],
  "interdisciplinary": [
    {
      "discipline": "string (e.g. Neuroscience, Psychology, Linguistics, Physics)",
      "specific_knowledge": "string (e.g. working memory models, Gestalt principles)",
      "why_needed": "string (1 sentence)"
    }
  ],
  "prerequisite_papers_or_concepts": ["string"],
  "difficulty_level": "undergraduate | graduate | postdoc"
}"""

SKILL_EXTRACTION_USER_TEMPLATE = """Paper Summary:
Title: {title}
Sub-domain: {sub_domain}
Problem: {problem_statement}
Method: {method_summary}
Keywords: {method_keywords}
Contributions: {contributions}

Extract all required skills now."""


# ── Prompt：学习路线图 ─────────────────────────────────────────────────────────

ROADMAP_SYSTEM_PROMPT = """You are a research mentor. Given a frequency-ranked list of skills required across recent papers
in a research domain, design a practical learning roadmap for someone with a CS undergraduate background.

Output ONLY valid JSON — an array of stages:
[
  {
    "stage": 1,
    "title": "string (e.g. 'Mathematical Foundations')",
    "duration_weeks": int,
    "skills": ["string"],
    "resources_type": ["string (e.g. 'linear algebra textbook', 'PyTorch tutorial')"],
    "milestone": "string (what you can do after this stage)",
    "interdisciplinary_intro": "string or null (if this stage requires cross-discipline knowledge, name it)"
  }
]

Rules:
- Maximum 5 stages
- Be realistic about duration
- Stage 1 should always be prerequisites
- Last stage should be "reading seminal papers in the field"
- Only include skills that appeared in the actual paper data provided"""

ROADMAP_USER_TEMPLATE = """Domain: {domain}
Total papers analyzed: {count}

Skills ranked by frequency (must_have > important > good_to_have):

Must-have skills (appear in >60% of papers):
{must_have}

Important skills (appear in 30-60% of papers):
{important}

Good-to-have skills (appear in <30% of papers):
{good_to_have}

Interdisciplinary requirements:
{interdisciplinary}

Generate the learning roadmap now."""


# ── 技能规范化与去重 ──────────────────────────────────────────────────────────

SKILL_SYNONYMS = {
    "pytorch": "deep learning frameworks",
    "tensorflow": "deep learning frameworks",
    "keras": "deep learning frameworks",
    "jax": "deep learning frameworks",
    "deep learning frameworks (e.g., pytorch, tensorflow)": "deep learning frameworks",
    "deep learning framework": "deep learning frameworks",
    "linear algebra (vectors, matrices, eigenvalues)": "linear algebra",
    "linear algebra (vectors, matrices)": "linear algebra",
    "calculus (multivariate)": "calculus",
    "multivariate calculus": "calculus",
    "numpy": "numpy/scipy",
    "scipy": "numpy/scipy",
    "python": "python programming",
    "python programming language": "python programming",
    "python 3": "python programming",
    "machine learning": "machine learning fundamentals",
    "ml": "machine learning fundamentals",
    "deep learning": "deep learning fundamentals",
    "dl": "deep learning fundamentals",
    "neural networks": "deep learning fundamentals",
    "neural network": "deep learning fundamentals",
    "convolutional neural networks": "convolutional neural networks (cnn)",
    "cnn": "convolutional neural networks (cnn)",
    "cnns": "convolutional neural networks (cnn)",
    "recurrent neural networks": "recurrent neural networks (rnn)",
    "rnn": "recurrent neural networks (rnn)",
    "transformers": "transformer architecture",
    "transformer": "transformer architecture",
    "attention mechanism": "transformer architecture",
    "self-attention": "transformer architecture",
    "gpu programming": "gpu/cuda programming",
    "cuda": "gpu/cuda programming",
    "git": "version control (git)",
    "github": "version control (git)",
}


def normalize_and_deduplicate_skills(skills_list: list[str]) -> list[str]:
    """
    规范化并去重技能列表：
    1. 小写化
    2. 去掉括号内的说明文字（如 "linear algebra (vectors, matrices)" → "linear algebra"）
    3. 应用同义词映射
    4. 集合去重
    """
    seen = set()
    result = []
    for skill in skills_list:
        s = skill.lower().strip()
        # 去掉括号内的补充说明（但先查一次完整版同义词）
        if s in SKILL_SYNONYMS:
            s = SKILL_SYNONYMS[s]
        else:
            s = re.sub(r'\s*\(.*?\)', '', s).strip()
            if s in SKILL_SYNONYMS:
                s = SKILL_SYNONYMS[s]
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result


# ── 1. 技能提取（含缓存） ─────────────────────────────────────────────────────

_cache_lock = threading.Lock()


def _load_skill_cache() -> dict:
    """从磁盘加载技能缓存"""
    if os.path.exists(SKILL_CACHE_PATH):
        try:
            with open(SKILL_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_skill_cache(cache: dict):
    """将技能缓存写入磁盘（线程安全）"""
    with _cache_lock:
        try:
            with open(SKILL_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"[Skills] 缓存写入失败: {e}")


def _clean_json(raw: str) -> str:
    """防御性清洗：去掉 markdown 代码围栏"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def extract_skills_from_paper(paper: dict, llm: ChatGoogleGenerativeAI) -> dict | None:
    """
    用 Gemini 提取单篇论文所需的技术与跨学科技能。
    优先从本地缓存读取，缓存未命中时调用 LLM 并写入缓存。
    返回解析好的 dict，失败时返回 None。
    """
    arxiv_id = paper.get("arxiv_id", "")

    # ── 缓存命中检查 ──
    if arxiv_id:
        cache = _load_skill_cache()
        if arxiv_id in cache:
            print(f"[Skills] 缓存命中: {paper.get('title', '')[:50]}")
            return cache[arxiv_id]

    user_msg = SKILL_EXTRACTION_USER_TEMPLATE.format(
        title=paper.get("title", ""),
        sub_domain=paper.get("sub_domain", ""),
        problem_statement=paper.get("problem_statement", ""),
        method_summary=paper.get("method_summary", ""),
        method_keywords=", ".join(paper.get("method_keywords", [])),
        contributions="; ".join(paper.get("contributions", []) or []),
    )

    try:
        response = llm.invoke([
            SystemMessage(content=SKILL_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = _clean_json(response.content)
        extracted = json.loads(raw)
        extracted["_paper_title"] = paper.get("title", "")

        # ── 写入缓存 ──
        if arxiv_id:
            cache = _load_skill_cache()
            cache[arxiv_id] = extracted
            _save_skill_cache(cache)
            print(f"[Skills] 已缓存: {paper.get('title', '')[:50]}")

        return extracted
    except json.JSONDecodeError as e:
        print(f"[Skills] JSON 解析失败: {paper.get('title', '')[:50]}... 错误: {e}")
        return None
    except Exception as e:
        print(f"[Skills] 技能提取失败: {e}")
        return None


# ── 2. 技能聚合 ───────────────────────────────────────────────────────────────

# 3 cleaner categories (replaces ml_methods/math_foundations/programming_tools/domain_knowledge)
_SKILL_CATEGORIES = ["core_theory", "applied_methods", "engineering_tools"]

# Mapping from LLM output keys → new category
_LLM_KEY_TO_CATEGORY = {
    "math_foundations": "core_theory",
    "ml_methods": "core_theory",
    "domain_knowledge": "applied_methods",
    "programming_tools": "engineering_tools",
}

_MAX_PER_BUCKET = 8   # cap per tier per category
_MAX_TABLE_ROWS = 20  # max total rows rendered in markdown


def aggregate_skills(papers_skills: list[dict]) -> dict:
    """
    合并所有论文的技能提取结果，规范化去重后按出现频率分类。

    返回:
        聚合后的技能字典，含 must_have / important / good_to_have /
        interdisciplinary_summary / dedup_stats
    """
    total = len(papers_skills)
    if total == 0:
        return {
            "must_have": {cat: [] for cat in _SKILL_CATEGORIES},
            "important": {cat: [] for cat in _SKILL_CATEGORIES},
            "good_to_have": {cat: [] for cat in _SKILL_CATEGORIES},
            "interdisciplinary_summary": [],
            "learning_roadmap": [],
            "dedup_stats": {"before": 0, "after": 0},
        }

    # ── 按新类别统计（规范化后）────────────────────────────────────────────────
    category_counters: dict[str, Counter] = {cat: Counter() for cat in _SKILL_CATEGORIES}
    total_raw = 0
    total_normalized_set: set[str] = set()

    for ps in papers_skills:
        for llm_key, target_cat in _LLM_KEY_TO_CATEGORY.items():
            raw_skills = ps.get(llm_key, [])
            total_raw += len(raw_skills)
            normed = normalize_and_deduplicate_skills(raw_skills)
            total_normalized_set.update(normed)
            for skill in normed:
                category_counters[target_cat][skill] += 1

    # ── 按频率分桶（60% / 30% / <30%）────────────────────────────────────────
    must_have: dict[str, list] = {cat: [] for cat in _SKILL_CATEGORIES}
    important: dict[str, list] = {cat: [] for cat in _SKILL_CATEGORIES}
    good_to_have: dict[str, list] = {cat: [] for cat in _SKILL_CATEGORIES}

    for cat in _SKILL_CATEGORIES:
        for skill, count in category_counters[cat].most_common():
            pct = count / total
            entry = {"skill": skill, "frequency": count, "percentage": round(pct * 100)}
            if pct >= 0.6:
                if len(must_have[cat]) < _MAX_PER_BUCKET:
                    must_have[cat].append(entry)
            elif pct >= 0.3:
                if len(important[cat]) < _MAX_PER_BUCKET:
                    important[cat].append(entry)
            else:
                if len(good_to_have[cat]) < _MAX_PER_BUCKET:
                    good_to_have[cat].append(entry)

    # ── 跨学科聚合：合并同学科条目 + 频率 >= 2 过滤 ───────────────────────────
    discipline_data: dict[str, dict] = {}
    for ps in papers_skills:
        for inter in ps.get("interdisciplinary", []):
            disc = inter.get("discipline", "Unknown").strip().title()
            if disc not in discipline_data:
                discipline_data[disc] = {"concepts": set(), "whys": [], "count": 0}
            discipline_data[disc]["count"] += 1
            concept = inter.get("specific_knowledge", "").strip()
            if concept:
                discipline_data[disc]["concepts"].add(concept)
            why = inter.get("why_needed", "").strip()
            if why and why not in discipline_data[disc]["whys"]:
                discipline_data[disc]["whys"].append(why)

    interdisciplinary_summary = []
    for disc, data in sorted(discipline_data.items(), key=lambda x: x[1]["count"], reverse=True):
        if data["count"] < 2:
            continue  # 过滤单次出现的学科
        interdisciplinary_summary.append({
            "discipline": disc,
            "frequency": data["count"],
            "percentage_of_papers": round(data["count"] / total * 100, 1),
            "key_concepts": sorted(data["concepts"]),
            "example_why_needed": data["whys"][0] if data["whys"] else "",
        })

    return {
        "must_have": must_have,
        "important": important,
        "good_to_have": good_to_have,
        "interdisciplinary_summary": interdisciplinary_summary,
        "learning_roadmap": [],
        "dedup_stats": {
            "before": total_raw,
            "after": len(total_normalized_set),
        },
    }


# ── 3. 学习路线图生成 ─────────────────────────────────────────────────────────

def _format_skill_bucket(bucket: dict) -> str:
    """将一个技能分桶格式化为文本"""
    lines = []
    for cat in _SKILL_CATEGORIES:
        items = bucket.get(cat, [])
        if items:
            label = cat.replace("_", " ").title()
            skills_str = ", ".join(
                f"{s['skill']} ({s['frequency']}x)" if isinstance(s, dict) else str(s)
                for s in items
            )
            lines.append(f"  {label}: {skills_str}")
    return "\n".join(lines) if lines else "  (none)"


def generate_learning_roadmap(
    aggregated: dict,
    llm: ChatGoogleGenerativeAI = None,
    paper_count: int = 0,
) -> list[dict] | None:
    """
    基于聚合后的技能数据，调用 Gemini 生成学习路线图。

    参数:
        aggregated: aggregate_skills 的返回值
        llm: LangChain LLM 实例（为空则自动创建）
        paper_count: 实际论文数量，低于 MIN_PAPERS_FOR_ROADMAP 时跳过

    返回:
        路线图 stage 列表，或 None（数据不足）
    """
    if paper_count < MIN_PAPERS_FOR_ROADMAP:
        print(
            f"[Skills] 论文数 ({paper_count}) < 路线图阈值 ({MIN_PAPERS_FOR_ROADMAP})，"
            "跳过路线图生成。"
        )
        return None

    if llm is None:
        llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.3)

    inter_lines = []
    for item in aggregated.get("interdisciplinary_summary", []):
        inter_lines.append(
            f"  {item['discipline']} ({item['frequency']}x, "
            f"{item['percentage_of_papers']}%): {', '.join(item['key_concepts'][:3])}"
        )
    inter_str = "\n".join(inter_lines) if inter_lines else "  (none)"

    user_msg = ROADMAP_USER_TEMPLATE.format(
        domain=DOMAIN,
        count=sum(
            len(aggregated["must_have"].get(c, []))
            + len(aggregated["important"].get(c, []))
            + len(aggregated["good_to_have"].get(c, []))
            for c in _SKILL_CATEGORIES
        ),
        must_have=_format_skill_bucket(aggregated["must_have"]),
        important=_format_skill_bucket(aggregated["important"]),
        good_to_have=_format_skill_bucket(aggregated["good_to_have"]),
        interdisciplinary=inter_str,
    )

    print("[Skills] 正在生成学习路线图...")

    try:
        response = llm.invoke([
            SystemMessage(content=ROADMAP_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = _clean_json(response.content)
        roadmap = json.loads(raw)
        if isinstance(roadmap, list):
            return roadmap
        print("[Skills] 路线图格式不正确，期望数组")
        return []
    except json.JSONDecodeError as e:
        print(f"[Skills] 路线图 JSON 解析失败: {e}")
        return []
    except Exception as e:
        print(f"[Skills] 路线图生成失败: {e}")
        return []


# ── 4. Markdown 渲染 ──────────────────────────────────────────────────────────

_CATEGORY_LABELS = {
    "core_theory": "Core Theory",
    "applied_methods": "Applied Methods",
    "engineering_tools": "Engineering Tools",
}


def render_skills_markdown(aggregated: dict) -> str:
    """
    将聚合后的技能数据渲染为 Markdown 章节。
    最多 20 行技能表 + 分类视觉分隔 + 精简路线图。
    """
    lines = []
    lines.append("## 7. Skills & Learning Roadmap")
    lines.append("")

    # ── 去重统计 ──
    stats = aggregated.get("dedup_stats", {})
    if stats.get("before"):
        lines.append(
            f"*Skills normalized: {stats['before']} raw entries "
            f"→ {stats['after']} unique after deduplication*"
        )
        lines.append("")

    # ── 技能表格（最多 _MAX_TABLE_ROWS 行） ──
    lines.append("### Required Technical Skills")
    lines.append("")
    lines.append("| Skill | Category | Priority | Frequency |")
    lines.append("| --- | --- | --- | --- |")

    # 收集所有 must_have + important 条目，按频率排序
    all_rows = []
    good_to_have_count = 0
    for tier_name, tier_label in [
        ("must_have", "Must-have"),
        ("important", "Important"),
    ]:
        tier = aggregated.get(tier_name, {})
        for cat in _SKILL_CATEGORIES:
            cat_label = _CATEGORY_LABELS.get(cat, cat)
            for item in tier.get(cat, []):
                skill = item["skill"] if isinstance(item, dict) else str(item)
                freq = item.get("frequency", 0) if isinstance(item, dict) else 0
                pct = item.get("percentage", 0) if isinstance(item, dict) else 0
                all_rows.append((freq, skill, cat_label, tier_label, pct))

    # 统计 good_to_have 数量（不显示在表里）
    gth = aggregated.get("good_to_have", {})
    for cat in _SKILL_CATEGORIES:
        good_to_have_count += len(gth.get(cat, []))

    # 按频率降序排列，取前 _MAX_TABLE_ROWS 行
    all_rows.sort(key=lambda r: r[0], reverse=True)
    shown = all_rows[:_MAX_TABLE_ROWS]
    overflow = len(all_rows) - len(shown) + good_to_have_count

    # 按类别分组输出（视觉分隔）
    current_cat = None
    for freq, skill, cat_label, tier_label, pct in shown:
        if cat_label != current_cat:
            current_cat = cat_label
            # 类别分隔行（粗体类别名，合并列）
            lines.append(f"| **{cat_label}** | | | |")
        freq_str = f"{freq}x ({pct}%)" if pct else f"{freq}x"
        lines.append(f"| {skill} | {cat_label} | {tier_label} | {freq_str} |")

    if overflow > 0:
        lines.append(f"| *... and {overflow} additional skills (good-to-have)* | | | |")

    lines.append("")

    # ── 跨学科需求 ──
    inter = aggregated.get("interdisciplinary_summary", [])
    if inter:
        lines.append("### Interdisciplinary Requirements")
        lines.append("")
        for item in inter:
            disc = item.get("discipline", "Unknown")
            freq = item.get("frequency", 0)
            total_pct = item.get("percentage_of_papers", 0)
            concepts = ", ".join(item.get("key_concepts", [])[:5])
            why = item.get("example_why_needed", "")
            lines.append(
                f"> 🧠 **{disc}** (required in {freq} papers, {total_pct}%): "
                f"{concepts}"
            )
            if why:
                lines.append(f"> _{why}_")
            lines.append(">")
        lines.append("")

    # ── 学习路线图（精简版：每阶段最多 5 条） ──
    roadmap = aggregated.get("learning_roadmap", [])
    if roadmap:
        lines.append("### Learning Roadmap")
        lines.append("")
        for stage in roadmap:
            num = stage.get("stage", "?")
            title = stage.get("title", "Untitled")
            weeks = stage.get("duration_weeks", "?")
            milestone = stage.get("milestone", "")
            inter_intro = stage.get("interdisciplinary_intro")

            lines.append(f"**Stage {num}: {title}** (~{weeks} weeks)")

            # 合并 skills + resources 为精简要点（最多 5 条）
            bullets = []
            for s in stage.get("skills", [])[:3]:
                bullets.append(s)
            for r in stage.get("resources_type", [])[:2]:
                bullets.append(f"Resource: {r}")
            for b in bullets[:5]:
                lines.append(f"- {b}")

            if milestone:
                lines.append(f"  ✅ *{milestone}*")
            if inter_intro:
                lines.append(f"  🔗 *{inter_intro}*")
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # 去重测试
    test_input = [
        "PyTorch", "pytorch", "deep learning frameworks (e.g., pytorch, tensorflow)",
        "linear algebra (vectors, matrices)", "linear algebra",
        "Python", "python programming language",
        "NumPy", "SciPy",
    ]
    result = normalize_and_deduplicate_skills(test_input)
    print(f"Dedup test: {len(test_input)} raw → {len(result)} unique")
    print(f"  Result: {result}")
    assert len(result) <= 4, f"Expected <= 4 unique, got {len(result)}: {result}"

    # 聚合测试
    mock_skills = [
        {
            "ml_methods": ["sparse autoencoder", "contrastive learning", "PyTorch"],
            "math_foundations": ["linear algebra", "convex optimization"],
            "programming_tools": ["PyTorch", "Python"],
            "domain_knowledge": ["recommender systems"],
            "interdisciplinary": [
                {"discipline": "Neuroscience", "specific_knowledge": "sparse coding in V1",
                 "why_needed": "Biological inspiration for sparse representation models"}
            ],
        },
        {
            "ml_methods": ["sparse autoencoder", "dictionary learning"],
            "math_foundations": ["linear algebra (vectors, matrices)", "matrix factorization"],
            "programming_tools": ["pytorch", "scikit-learn", "numpy"],
            "domain_knowledge": ["computer vision"],
            "interdisciplinary": [
                {"discipline": "neuroscience", "specific_knowledge": "visual cortex",
                 "why_needed": "Model motivation from biology"}
            ],
        },
        {
            "ml_methods": ["sparse autoencoder"],
            "math_foundations": ["linear algebra", "calculus (multivariate)"],
            "programming_tools": ["tensorflow", "scipy"],
            "domain_knowledge": ["NLP"],
            "interdisciplinary": [
                {"discipline": "Linguistics", "specific_knowledge": "syntax",
                 "why_needed": "Understanding language structure"}
            ],
        },
    ]

    agg = aggregate_skills(mock_skills)
    stats = agg["dedup_stats"]
    print(f"\nAggregation: {stats['before']} raw → {stats['after']} unique")

    # Verify Neuroscience merged (appeared 2x), Linguistics filtered (1x)
    inter = agg["interdisciplinary_summary"]
    disc_names = [i["discipline"] for i in inter]
    print(f"Interdisciplinary (freq>=2): {disc_names}")
    assert "Neuroscience" in disc_names, "Neuroscience should appear (freq=2)"
    assert "Linguistics" not in disc_names, "Linguistics should be filtered (freq=1)"

    print(f"\nCategories: {_SKILL_CATEGORIES}")
    for tier in ["must_have", "important", "good_to_have"]:
        for cat in _SKILL_CATEGORIES:
            items = agg[tier][cat]
            if items:
                print(f"  {tier}/{cat}: {[i['skill'] for i in items]}")

    md = render_skills_markdown(agg)
    row_count = md.count("\n|") - 2  # subtract header + separator
    print(f"\nTable rows (excl header): {row_count} (max {_MAX_TABLE_ROWS})")
    print("\n" + md)
