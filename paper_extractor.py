# paper_extractor.py — Prompt 1：结构化提取论文信息，输出 JSON 存入向量数据库

import json
import os
import re
import urllib.parse
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field, ValidationError, ConfigDict

from config import GEMINI_MODEL, CHROMA_DB_PATH, CHROMA_COLLECTION

load_dotenv()


class MethodologyMatrixSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approach_type: str = Field(
        description="One of: convex_optimization | greedy_algorithm | deep_learning | statistical | hybrid"
    )
    model_architecture: str | None = Field(
        default=None,
        description="Model architecture name or null. Examples: Transformer, CNN, Autoencoder, None.",
    )
    supervision_type: str = Field(
        description="One of: supervised | self_supervised | unsupervised | semi_supervised"
    )
    data_modality: list[str] = Field(
        description="List of modalities, e.g. image, text, graph, time_series, 3D_point_cloud"
    )
    open_source: str = Field(description="One of: yes | no | unknown")
    theoretical_guarantees: str = Field(description="One of: yes | no")
    scalability: str = Field(description="One of: linear | quadratic | unknown")
    key_baseline_compared_to: str | None = Field(
        default=None,
        description="Primary baseline method compared against, or null if unavailable.",
    )


class PaperExtractionSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    authors: list[str]
    published_date: str = Field(description="Date in YYYY-MM-DD format")
    sub_domain: str
    problem_statement: str = Field(description="1-2 sentences")
    method_summary: str = Field(description="2-3 sentences")
    contributions: list[str] = Field(
        min_length=2,
        max_length=4,
        description="List of 2-4 concise strings, each under 20 words."
    )
    limitations: str | None = Field(
        default=None,
        description="Extract only if explicitly mentioned in the abstract; otherwise null.",
    )
    method_keywords: list[str] = Field(
        min_length=3,
        max_length=6,
        description="3-6 technical terms central to the method."
    )
    methodology_matrix: MethodologyMatrixSchema
    industrial_readiness_score: int = Field(
        ge=1,
        le=5,
        description=(
            "Industrial readiness score on a strict 1-5 scale. "
            "1 = Pure theory, no code, no experiments on real data. "
            "3 = Has experiments but on toy datasets, no open-source code. "
            "5 = Open-source code available, tested on production-scale benchmarks. "
            "Use 2 or 4 if evidence is clearly between these anchors."
        ),
    )
    theoretical_depth: int = Field(
        ge=1,
        le=5,
        description=(
            "Theoretical depth score on a strict 1-5 scale. "
            "1 = Heuristic method, no proofs or formal analysis. "
            "3 = Empirical analysis with ablations. "
            "5 = Formal proofs, convergence guarantees, error bounds. "
            "Use 2 or 4 if evidence is clearly between these anchors."
        ),
    )
    domain_specificity: int = Field(
        ge=1,
        le=5,
        description=(
            "Domain specificity score on a strict 1-5 scale for sparse representation focus. "
            "1 = General ML paper, sparse representation is incidental. "
            "3 = Core method uses sparsity but applies to broad domains. "
            "5 = Sparse representation IS the central contribution and primary topic. "
            "Use 2 or 4 if evidence is clearly between these anchors."
        ),
    )


EXTRACTION_JSON_SCHEMA = json.dumps(
    PaperExtractionSchema.model_json_schema(), ensure_ascii=False, indent=2
)

# ── Prompt 1：结构化提取 ──────────────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = """You are a research paper analyst specializing in signal processing and machine learning.
Given the title, abstract, and metadata of an academic paper, extract structured information strictly in JSON format.

Rules:
- Output ONLY valid JSON. No explanation, no markdown fences, no code blocks.
- If a field cannot be determined from the abstract, use null.
- "contributions" must be a list of 2-4 concise strings, each under 20 words.
- "limitations" should be extracted only if explicitly mentioned in the abstract; otherwise null.
- "method_keywords" should be 3-6 technical terms central to the method.
- "industrial_readiness_score", "theoretical_depth", and "domain_specificity" must each be integers from 1 to 5.
- For each score, strictly follow the rubric anchors in the schema description for 1, 3, and 5.
- Never assign a score of 5 unless there is explicit evidence for the 5-level criteria.

Output Schema:
{json_schema}""".format(json_schema=EXTRACTION_JSON_SCHEMA)

EXTRACTION_USER_TEMPLATE = """Title: {title}
Authors: {authors}
Published: {published_date}
Abstract: {abstract}"""


def enrich_open_source_with_paperswithcode(paper_title: str, extracted: dict) -> None:
    """
    If open_source is unknown, query PapersWithCode by title and try to enrich
    with a GitHub URL.
    """
    mm = extracted.get("methodology_matrix")
    if not isinstance(mm, dict):
        return

    open_source = str(mm.get("open_source", "unknown")).strip().lower()
    if open_source != "unknown":
        return

    search_url = (
        "https://paperswithcode.com/search?q_meta=&q_type=&q="
        f"{urllib.parse.quote(paper_title)}"
    )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(search_url, headers=headers, timeout=5)
        if response.status_code != 200 or not response.text:
            return

        html = response.text
        if "github.com" in html:
            match = re.search(r'https://github\.com/[\w\-]+/[\w\-]+', html)
            if match:
                mm["open_source"] = "yes"
                extracted["github_url"] = match.group(0)
                mm["github_url"] = match.group(0)
                return

        mm["open_source"] = "no"
    except requests.RequestException:
        # Keep open_source as unknown on timeout/network failures.
        return


def extract_paper_info(paper: dict, llm: ChatGoogleGenerativeAI) -> dict | None:
    """
    用 Gemini 提取单篇论文的结构化信息。
    返回解析好的 dict，失败时返回 None。
    """
    user_msg = EXTRACTION_USER_TEMPLATE.format(
        title=paper["title"],
        authors=", ".join(paper["authors"][:5]),  # 最多5个作者
        published_date=paper["published_date"],
        abstract=paper["abstract"],
    )

    try:
        response = llm.invoke([
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
        raw = response.content.strip()

        # 防御性清洗：有时模型会加 ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        extracted_raw = json.loads(raw)
        extracted = PaperExtractionSchema.model_validate(extracted_raw).model_dump()
        enrich_open_source_with_paperswithcode(paper["title"], extracted)
        extracted["arxiv_id"] = paper["arxiv_id"]
        extracted["url"] = paper["url"]
        return extracted

    except json.JSONDecodeError as e:
        print(f"[Extractor] JSON 解析失败: {paper['title'][:50]}... 错误: {e}")
        return None
    except ValidationError as e:
        print(f"[Extractor] Schema 校验失败: {paper['title'][:50]}... 错误: {e}")
        return None
    except Exception as e:
        print(f"[Extractor] 提取失败: {e}")
        return None


def store_papers_to_db(papers_json: list[dict]) -> Chroma:
    """
    将结构化论文数据存入 Chroma 向量数据库。
    用 method_summary + problem_statement 作为 embedding 的文本。
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    texts = []
    metadatas = []
    ids = []

    for p in papers_json:
        if p is None:
            continue
        # 拼接用于 embedding 的文本
        text = f"Title: {p.get('title', '')}\n"
        text += f"Problem: {p.get('problem_statement', '')}\n"
        text += f"Method: {p.get('method_summary', '')}\n"
        text += f"Keywords: {', '.join(p.get('method_keywords', []))}"

        texts.append(text)
        metadatas.append({
            "arxiv_id": p.get("arxiv_id", ""),
            "title": p.get("title", ""),
            "published_date": p.get("published_date", ""),
            "sub_domain": p.get("sub_domain", ""),
            "industrial_readiness_score": str(p.get("industrial_readiness_score", "")),
            "theoretical_depth": str(p.get("theoretical_depth", "")),
            "domain_specificity": str(p.get("domain_specificity", "")),
            "url": p.get("url", ""),
            "github_url": p.get("github_url", ""),
            "full_json": json.dumps(p),  # 把完整数据也存进去方便后续取用
        })
        ids.append(p.get("arxiv_id", f"paper_{len(ids)}"))

    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
        persist_directory=CHROMA_DB_PATH,
        collection_name=CHROMA_COLLECTION,
    )
    print(f"[DB] 成功存入 {len(texts)} 篇论文到向量数据库")
    return db


def load_db() -> Chroma:
    """加载已有的向量数据库"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION,
    )


if __name__ == "__main__":
    # 单独测试：python paper_extractor.py
    from paper_fetcher import fetch_recent_papers

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
    papers = fetch_recent_papers()

    print(f"\n开始提取 {len(papers)} 篇论文...\n")
    extracted = []
    for p in papers[:3]:  # 先测试3篇
        result = extract_paper_info(p, llm)
        if result:
            extracted.append(result)
            print(f"✅ {result['title'][:60]}...")
            print(
                "   评分: "
                f"industrial={result.get('industrial_readiness_score')}/5, "
                f"theory={result.get('theoretical_depth')}/5, "
                f"specificity={result.get('domain_specificity')}/5"
            )
            print(f"   贡献: {result.get('contributions', [])}\n")

    if extracted:
        store_papers_to_db(extracted)
