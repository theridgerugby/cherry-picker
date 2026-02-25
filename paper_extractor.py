# paper_extractor.py — Prompt 1：结构化提取论文信息，输出 JSON 存入向量数据库

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field, ValidationError, ConfigDict

from config import GEMINI_MODEL, GEMINI_MODEL_FAST, CHROMA_DB_PATH, CHROMA_COLLECTION
from repo_extractor import extract_repo_candidates_for_paper

load_dotenv()


class MethodologyMatrixSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approach_type: str = Field(
        description="One of: convex_optimization | greedy_algorithm | deep_learning | statistical | hybrid | theoretical"
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
            "Domain specificity score on a strict 1-5 scale for the target research domain. "
            "1 = Primarily about another field, target domain is incidental or absent. "
            "3 = Partially relevant to the target domain, but contribution is broadly scoped. "
            "5 = The target domain is the central contribution and primary topic. "
            "Use 2 or 4 if evidence is clearly between these anchors."
        ),
    )


EXTRACTION_JSON_SCHEMA = json.dumps(
    PaperExtractionSchema.model_json_schema(), ensure_ascii=False, indent=2
)

# ── Prompt 1：结构化提取 ──────────────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT_TEMPLATE = """You are a research paper analyst.
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
- For "domain_specificity": Rate how central the paper is to: {domain}
  1 = This paper is primarily about a different field;
      the topic '{domain}' is incidental or absent.
  3 = The paper applies methods that touch on '{domain}'
      but its core contribution spans broader areas.
  5 = '{domain}' IS the central topic and primary contribution.
  Use 2 or 4 if evidence is clearly between these anchors.
- data_modality must describe what data the METHOD actually operates on, not the paper topic.
  Examples:
  - A paper about galaxy classification using CNNs -> "image"
  - A paper about stock prediction using RNNs -> "time_series"
  - A paper about theoretical physics with no experiments -> "none (theoretical)"
  - A paper about text classification -> "text"
  If the paper is purely theoretical with no experiments, set data_modality to
  ["none (theoretical)"].
- key_baseline_compared_to must be a specific named baseline (e.g. "ResNet-50",
  "GPT-3", "DINO"), not a category description (e.g. "traditional methods",
  "von Neumann computing", "conventional approaches").
  If no specific named baseline is mentioned in the abstract, return null.

Output Schema:
{json_schema}"""

EXTRACTION_USER_TEMPLATE = """Title: {title}
Authors: {authors}
Published: {published_date}
Abstract: {abstract}"""


PREFILTER_SYSTEM_PROMPT = """You are a domain relevance classifier. Given a paper title and abstract,
score its relevance to the target domain.

Target domain: {domain}

Return ONLY valid JSON:
{{
  "relevance_score": 1-10,
  "is_core_domain": true | false,
  "rejection_reason": "string or null",
  "detected_actual_domain": "string"
}}

Scoring rules:
- 8-10: Core paper directly advancing {domain} research
- 5-7: Adjacent field with clear {domain} application
- 1-4: Tangentially related or primarily about a different field

Mark is_core_domain=false if:
- The paper is primarily philosophy or science studies
- The paper is a general ML tool applied to {domain} as a minor use case
- The paper's primary contribution is to a different field
- The title mentions {domain} but the method is domain-agnostic"""

PREFILTER_USER_TEMPLATE = """Title: {title}
Abstract: {abstract}"""


def _clean_json_response(raw: str) -> str:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


VALID_APPROACHES = [
    "convex_optimization",
    "greedy_algorithm",
    "deep_learning",
    "statistical",
    "hybrid",
    "theoretical",
]


def _classify_domain_relevance(paper: dict, domain: str, llm_fast) -> dict:
    default = {
        "relevance_score": 1,
        "is_core_domain": False,
        "rejection_reason": "classification_failed",
        "detected_actual_domain": "unknown",
    }

    user_prompt = PREFILTER_USER_TEMPLATE.format(
        title=paper.get("title", ""),
        abstract=paper.get("abstract", ""),
    )

    try:
        response = llm_fast.invoke([
            SystemMessage(content=PREFILTER_SYSTEM_PROMPT.format(domain=domain)),
            HumanMessage(content=user_prompt),
        ])
        parsed = json.loads(_clean_json_response(response.content))
        score = parsed.get("relevance_score", 1)
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 1
        score = max(1, min(10, score))

        is_core_domain = bool(parsed.get("is_core_domain", False))
        rejection_reason = parsed.get("rejection_reason")
        if rejection_reason is not None:
            rejection_reason = str(rejection_reason).strip() or None
        detected_actual_domain = str(
            parsed.get("detected_actual_domain", "unknown")
        ).strip() or "unknown"

        return {
            "relevance_score": score,
            "is_core_domain": is_core_domain,
            "rejection_reason": rejection_reason,
            "detected_actual_domain": detected_actual_domain,
        }
    except Exception as e:
        print(f"[Prefilter] relevance classification failed: {e}")
        return default


def prefilter_papers(papers: list[dict], domain: str, llm_fast) -> list[dict]:
    """
    Map phase: fast domain relevance scoring before full extraction.
    Runs one Gemini-flash classification call per paper in parallel.
    """
    if not papers:
        return []

    domain_name = str(domain or "").strip() or "unknown domain"
    classifier = llm_fast or ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_FAST,
        temperature=0,
    )
    enriched: list[dict | None] = [None] * len(papers)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_classify_domain_relevance, paper, domain_name, classifier): idx
            for idx, paper in enumerate(papers)
        }
        for future in as_completed(futures):
            idx = futures[future]
            paper = papers[idx]
            try:
                relevance = future.result()
            except Exception as e:
                print(f"[Prefilter] worker failed for '{paper.get('title', '')[:60]}': {e}")
                relevance = {
                    "relevance_score": 1,
                    "is_core_domain": False,
                    "rejection_reason": "worker_failed",
                    "detected_actual_domain": "unknown",
                }
            merged = dict(paper)
            merged.update(relevance)
            enriched[idx] = merged

    return [p for p in enriched if p is not None]


def _strip_disallowed_url_fields(payload: dict) -> dict:
    """
    Remove URL-like fields the LLM must never generate.
    GitHub URLs are discovered only via verified external enrichment.
    """
    if not isinstance(payload, dict):
        return payload

    sanitized = dict(payload)
    sanitized.pop("github_url", None)

    mm = sanitized.get("methodology_matrix")
    if isinstance(mm, dict):
        mm_sanitized = dict(mm)
        mm_sanitized.pop("github_url", None)
        sanitized["methodology_matrix"] = mm_sanitized

    return sanitized


def extract_paper_info(
    paper: dict,
    llm: ChatGoogleGenerativeAI,
    domain: str = "the target research domain",
) -> dict | None:
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
    domain_for_prompt = (domain or "the target research domain").strip()
    if not domain_for_prompt:
        domain_for_prompt = "the target research domain"
    system_prompt = EXTRACTION_SYSTEM_PROMPT_TEMPLATE.format(
        domain=domain_for_prompt,
        json_schema=EXTRACTION_JSON_SCHEMA,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_msg),
        ])
        extracted_raw = _strip_disallowed_url_fields(
            json.loads(_clean_json_response(response.content))
        )
        extracted = PaperExtractionSchema.model_validate(extracted_raw).model_dump()

        # Guardrail: enforce controlled approach labels to avoid hallucinated categories.
        mm = extracted.get("methodology_matrix")
        if isinstance(mm, dict):
            approach = str(mm.get("approach_type", "")).strip()
            if approach not in VALID_APPROACHES:
                mm["approach_type"] = "hybrid"

        # Deterministic repository extraction pipeline:
        # PWC(arXiv id) -> arXiv metadata -> PDF/Tex -> normalize/validate -> score/rank.
        repo_result = extract_repo_candidates_for_paper(paper, top_k=3)
        top_candidates = repo_result.get("top_candidates", []) or []
        all_candidates = repo_result.get("all_candidates", []) or []

        extracted["github_url"] = None
        extracted["github_url_validated"] = False
        extracted["paper_id"] = repo_result.get("paper_id") or paper.get("arxiv_id")
        extracted["repo_candidates"] = all_candidates
        extracted["repo_top_candidates"] = top_candidates
        extracted["repo_low_confidence_candidates"] = (
            repo_result.get("low_confidence_candidates", []) or []
        )

        best_verified = next((c for c in top_candidates if c.get("verified")), None)
        if best_verified is None:
            best_verified = next((c for c in all_candidates if c.get("verified")), None)

        if best_verified:
            extracted["github_url"] = best_verified.get("repo_url")
            extracted["github_url_validated"] = True
            if isinstance(mm, dict):
                mm["open_source"] = "yes"
        elif all_candidates:
            if isinstance(mm, dict):
                # Evidence exists but link is not verified.
                mm["open_source"] = "yes"
        elif isinstance(mm, dict):
            # No deterministic evidence found.
            mm["open_source"] = "unknown"

        extracted["arxiv_id"] = paper["arxiv_id"]
        extracted["url"] = paper["url"]
        if "is_core_domain" in paper:
            extracted["is_core_domain"] = bool(paper.get("is_core_domain"))
        if "relevance_score" in paper:
            extracted["relevance_score"] = paper.get("relevance_score")
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
        result = extract_paper_info(p, llm, domain="the target research domain")
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
