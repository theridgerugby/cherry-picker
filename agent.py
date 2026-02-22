# agent.py — Prompt 2：ReAct Agent，串联所有工具，主入口文件

import json
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent

from config import GEMINI_MODEL, DOMAIN, MIN_PAPERS_TO_PROCESS, DAYS_BACK
from paper_fetcher import fetch_recent_papers
from paper_extractor import extract_paper_info, store_papers_to_db, load_db
from credibility_scorer import score_paper_credibility
from trend_analyzer import analyze_trends
from skills_analyzer import extract_skills_from_paper, aggregate_skills, generate_learning_roadmap

load_dotenv()

# 全局状态（Agent 运行期间共享）
_papers_raw = []
_papers_extracted = []
_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    return _llm


# ── 工具定义 ──────────────────────────────────────────────────────────────────

@tool
def search_arxiv(query_override: str = "") -> str:
    """
    从 arXiv 拉取最新的 Sparse Representation 相关论文。
    输入：可选的查询词覆盖（留空则用默认配置）。
    输出：论文列表摘要（JSON 字符串）。
    """
    global _papers_raw
    _papers_raw = fetch_recent_papers()
    if not _papers_raw:
        return "没有找到论文，请检查网络或调整搜索关键词。"
    summary = [{"index": i, "arxiv_id": p["arxiv_id"], "title": p["title"], "date": p["published_date"]}
               for i, p in enumerate(_papers_raw)]
    return json.dumps(summary, ensure_ascii=False)


@tool
def extract_and_store_paper(paper_index: str) -> str:
    """
    对指定索引的论文进行结构化提取，并存入向量数据库。
    输入：论文在列表中的 index（整数）。
    输出：提取结果摘要。
    """
    # 兼容 Agent 传入 "paper_index: 0" 或 "0" 等格式
    try:
        idx = int(str(paper_index).split(":")[-1].strip())
    except ValueError:
        return f"错误：无法解析 paper_index 值：{paper_index}"

    global _papers_raw, _papers_extracted
    if not _papers_raw:
        return "错误：请先调用 search_arxiv 获取论文列表。"
    if idx < 0 or idx >= len(_papers_raw):
        return f"错误：index {idx} 超出范围，共 {len(_papers_raw)} 篇。"

    paper = _papers_raw[idx]
    llm = get_llm()
    result = extract_paper_info(paper, llm)

    if result is None:
        return f"提取失败：{paper['title'][:60]}"

    _papers_extracted.append(result)
    return (f"提取成功：{result['title'][:60]}\n"
            f"  子领域：{result.get('sub_domain')}\n"
            f"  工业就绪度：{result.get('industrial_readiness_score')}/5\n"
            f"  理论深度：{result.get('theoretical_depth')}/5\n"
            f"  领域专一性：{result.get('domain_specificity')}/5\n"
            f"  核心贡献：{result.get('contributions', [])}")


@tool
def save_all_to_database(dummy: str = "") -> str:
    """
    将所有已提取的论文一次性存入向量数据库。
    在提取完所有目标论文后调用此工具。
    输入：任意字符串（忽略）。
    输出：存储结果。
    """
    global _papers_extracted
    if not _papers_extracted:
        return "没有已提取的论文，请先调用 extract_and_store_paper。"
    store_papers_to_db(_papers_extracted)
    return f"成功将 {len(_papers_extracted)} 篇论文存入向量数据库。"


@tool
def query_database(question: str) -> str:
    """
    从向量数据库中检索与问题最相关的论文片段，用于辅助报告生成。
    输入：自然语言问题，例如"哪些论文用了字典学习方法"。
    输出：相关论文摘要（JSON）。
    """
    try:
        db = load_db()
        results = db.similarity_search(question, k=3)
        output = []
        for doc in results:
            meta = doc.metadata
            full = json.loads(meta.get("full_json", "{}"))
            output.append({
                "title": full.get("title"),
                "sub_domain": full.get("sub_domain"),
                "contributions": full.get("contributions"),
                "method_summary": full.get("method_summary"),
            })
        return json.dumps(output, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"数据库查询失败：{e}"


@tool
def analyze_industry_trends(dummy: str = "") -> str:
    """
    对数据库中所有论文进行可信度评分并生成行业趋势分析。
    在提取并保存论文后调用此工具，获取宏观/微观趋势洞察。
    输入：任意字符串（忽略）。
    输出：趋势分析 JSON 摘要。
    """
    try:
        db = load_db()
        results = db.similarity_search("sparse representation", k=50)
        papers = []
        seen_ids = set()
        for doc in results:
            meta = doc.metadata
            arxiv_id = meta.get("arxiv_id", "")
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            full = json.loads(meta.get("full_json", "{}"))
            if full:
                papers.append(full)

        if not papers:
            return "错误：数据库为空，请先提取论文。"

        # 可信度评分
        scored_papers = [score_paper_credibility(p, DOMAIN) for p in papers]
        # 趋势分析
        llm = get_llm()
        trend_data = analyze_trends(scored_papers, llm)

        summary_lines = [f"分析了 {len(scored_papers)} 篇论文"]
        macro = trend_data.get("macro_trends", [])
        if macro:
            summary_lines.append(f"发现 {len(macro)} 个宏观趋势：")
            for t in macro:
                summary_lines.append(f"  - {t.get('trend_name')}: {t.get('trajectory')}")
        futures = trend_data.get("future_directions", [])
        if futures:
            summary_lines.append(f"发现 {len(futures)} 个未来方向")

        return json.dumps(trend_data, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"趋势分析失败：{e}"


@tool
def analyze_required_skills(dummy: str = "") -> str:
    """
    分析数据库中所有论文所需的技术技能和跨学科知识，生成学习路线图。
    在 save_all_to_database 之后调用此工具。
    输入：任意字符串（忽略）。
    输出：技能分析与学习路线图 JSON 摘要。
    """
    try:
        db = load_db()
        results = db.similarity_search("sparse representation", k=50)
        papers = []
        seen_ids = set()
        for doc in results:
            meta = doc.metadata
            arxiv_id = meta.get("arxiv_id", "")
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            full = json.loads(meta.get("full_json", "{}"))
            if full:
                papers.append(full)

        if not papers:
            return "错误：数据库为空，请先提取论文。"

        llm = get_llm()
        print(f"[Skills] 正在分析 {len(papers)} 篇论文的技能需求...")

        # 逐篇提取技能
        all_skills = []
        for p in papers:
            skill = extract_skills_from_paper(p, llm)
            if skill:
                all_skills.append(skill)

        if not all_skills:
            return "技能提取失败，未能从任何论文中提取到技能。"

        # 聚合 + 路线图
        aggregated = aggregate_skills(all_skills)
        roadmap = generate_learning_roadmap(aggregated, llm)
        aggregated["learning_roadmap"] = roadmap

        summary_lines = [f"分析了 {len(all_skills)} 篇论文的技能需求"]
        for tier in ["must_have", "important", "good_to_have"]:
            tier_data = aggregated.get(tier, {})
            total_skills = sum(len(v) for v in tier_data.values())
            summary_lines.append(f"  {tier}: {total_skills} 项技能")
        summary_lines.append(f"  跨学科领域: {len(aggregated.get('interdisciplinary_summary', []))} 个")
        summary_lines.append(f"  学习路线图: {len(roadmap)} 个阶段")

        return json.dumps(aggregated, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"技能分析失败：{e}"


AGENT_SYSTEM_PROMPT = f"""You are a research intelligence agent specialized in academic literature analysis.
Your goal is to fetch, analyze, and prepare a set of recent papers on "{DOMAIN}" for report generation.

Rules:
- After fetching papers with search_arxiv, extract AT LEAST {MIN_PAPERS_TO_PROCESS} papers using extract_and_store_paper.
- Prioritize papers with higher domain specificity to {DOMAIN} (domain_specificity 4-5).
- After extracting all target papers, call save_all_to_database.
- After saving, call analyze_industry_trends to generate trend insights.
- After trend analysis, call analyze_required_skills to generate a skills & learning roadmap.
- If a tool returns an error, try a different approach and continue.
- When all steps are complete, reply with a brief summary: how many papers found, how many processed, key themes.

Today\'s task:
Domain: {DOMAIN}
Time range: last {DAYS_BACK} days
Minimum papers to process: {MIN_PAPERS_TO_PROCESS}"""


def run_agent():
    """启动 Agent，执行完整的论文拉取 + 提取 + 存储流程"""
    llm = get_llm()
    tools = [
        search_arxiv, extract_and_store_paper, save_all_to_database,
        query_database, analyze_industry_trends, analyze_required_skills,
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        debug=False,
    )

    print("\n" + "="*60)
    print("🚀 arXiv Agent 启动")
    print(f"   领域：{DOMAIN}")
    print(f"   目标：处理最近 {DAYS_BACK} 天内至少 {MIN_PAPERS_TO_PROCESS} 篇论文")
    print("="*60 + "\n")

    result = agent.invoke({
        "messages": [{"role": "user", "content": f"Start: fetch and process papers on {DOMAIN} from the last {DAYS_BACK} days."}]
    })

    # 取最后一条消息作为输出
    final_output = ""
    if isinstance(result, dict):
        messages = result.get("messages", [])
        if messages:
            last_msg = messages[-1]
            final_output = getattr(last_msg, "content", str(last_msg))

    print("\n" + "="*60)
    print("✅ Agent 完成！")
    print(f"Final Answer: {final_output}")
    print("="*60)
    print("\n下一步：运行 python report_generator.py 生成对比报告")

    return _papers_extracted


if __name__ == "__main__":
    run_agent()
