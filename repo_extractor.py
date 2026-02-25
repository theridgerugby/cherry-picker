"""Deterministic repository extraction pipeline for arXiv papers.

Input:
  paper with at least arXiv id / abs url / pdf url

Output:
  paper_id -> ranked repo candidates with source, confidence, validation status
"""

from __future__ import annotations

import io
import math
import re
import tarfile
from urllib.parse import urlparse

import arxiv
import requests

GITHUB_URL_PATTERN = re.compile(
    r"(https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?|"
    r"github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?)",
    re.IGNORECASE,
)

URL_TRIM_CHARS = ".,;:!?)]}\"'`<>"

TITLE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "using",
    "towards",
    "toward",
    "based",
    "study",
    "analysis",
    "approach",
    "method",
    "model",
    "models",
    "learning",
    "deep",
    "neural",
    "network",
    "networks",
    "data",
    "paper",
}

SOURCE_BASE_SCORE = {
    "paperswithcode_api": 0.90,
    "arxiv_metadata_comment": 0.72,
    "arxiv_metadata_summary": 0.65,
    "arxiv_metadata_abstract": 0.62,
    "pdf_text": 0.52,
    "tex_source": 0.56,
}


def _strip_version(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    return re.sub(r"v\d+$", "", arxiv_id.strip(), flags=re.IGNORECASE)


def _extract_arxiv_id_from_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    if "/abs/" in path:
        return path.split("/abs/")[-1]
    if "/pdf/" in path:
        return path.split("/pdf/")[-1].replace(".pdf", "")
    return path.split("/")[-1]


def _build_pdf_url(arxiv_id: str) -> str:
    base_id = _strip_version(arxiv_id)
    return f"https://arxiv.org/pdf/{base_id}.pdf" if base_id else ""


def _significant_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return {t for t in tokens if len(t) >= 4 and t not in TITLE_STOPWORDS}


def _extract_github_urls(text: str) -> list[str]:
    if not text:
        return []
    matches = []
    for raw in GITHUB_URL_PATTERN.findall(text):
        candidate = raw.strip().rstrip(URL_TRIM_CHARS)
        if candidate.lower().startswith("github.com/"):
            candidate = "https://" + candidate
        matches.append(candidate)
    # Keep insertion order while de-duplicating.
    return list(dict.fromkeys(matches))


def _canonicalize_repo_url(url: str) -> str | None:
    if not url:
        return None
    candidate = str(url).strip().rstrip(URL_TRIM_CHARS)
    if candidate.lower().startswith("github.com/"):
        candidate = "https://" + candidate
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"}:
        return None
    if "github.com" not in parsed.netloc.lower():
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None

    owner = parts[0]
    repo = parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    if not re.match(r"^[A-Za-z0-9_.-]+$", owner):
        return None
    if not re.match(r"^[A-Za-z0-9_.-]+$", repo):
        return None
    return f"https://github.com/{owner}/{repo}"


def _verify_repo_url(url: str) -> tuple[bool, str]:
    try:
        response = requests.head(
            url,
            timeout=6,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        status = response.status_code
        if status == 405:
            response = requests.get(
                url,
                timeout=8,
                allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            status = response.status_code
        if status == 200:
            return True, "http_200"
        return False, f"http_{status}"
    except requests.RequestException as exc:
        return False, f"request_error:{type(exc).__name__}"


def get_repos_from_pwc(arxiv_id: str) -> list[dict]:
    """Step 1: query PapersWithCode client API for repositories."""
    try:
        from paperswithcode import PapersWithCodeClient
    except Exception:
        return []

    clean_id = _strip_version(arxiv_id)
    if not clean_id:
        return []

    try:
        client = PapersWithCodeClient()
        papers = client.paper_list(arxiv_id=clean_id, items_per_page=5)
        results = list(getattr(papers, "results", []) or [])
        if not results:
            return []

        # Prefer an exact arXiv id match if available.
        selected = results[0]
        for paper in results:
            paper_arxiv_id = _strip_version(str(getattr(paper, "arxiv_id", "") or ""))
            if paper_arxiv_id and paper_arxiv_id == clean_id:
                selected = paper
                break

        repos = client.paper_repository_list(selected.id, items_per_page=50)
        repo_results = list(getattr(repos, "results", []) or [])
        extracted = []
        for repo in repo_results:
            extracted.append(
                {
                    "source": "paperswithcode_api",
                    "paper_id": str(getattr(selected, "id", "")),
                    "repo_url": str(getattr(repo, "url", "") or ""),
                    "owner": str(getattr(repo, "owner", "") or ""),
                    "name": str(getattr(repo, "name", "") or ""),
                    "stars": int(getattr(repo, "stars", 0) or 0),
                    "framework": str(getattr(repo, "framework", "") or ""),
                    "is_official": getattr(repo, "is_official", None),
                }
            )
        return extracted
    except Exception as exc:
        print(f"[RepoExtractor] PWC API failed for {arxiv_id}: {exc}")
        return []


def parse_arxiv_metadata_for_urls(arxiv_id: str) -> list[dict]:
    """Step 2: parse GitHub URLs from arXiv metadata (comment/summary)."""
    clean_id = _strip_version(arxiv_id)
    if not clean_id:
        return []

    try:
        client = arxiv.Client()
        search = arxiv.Search(id_list=[clean_id], max_results=1)
        result = next(client.results(search), None)
        if result is None:
            return []
    except Exception as exc:
        print(f"[RepoExtractor] arXiv metadata fetch failed for {arxiv_id}: {exc}")
        return []

    candidates: list[dict] = []
    fields = {
        "arxiv_metadata_comment": str(getattr(result, "comment", "") or ""),
        "arxiv_metadata_summary": str(getattr(result, "summary", "") or ""),
    }
    for source, text in fields.items():
        for url in _extract_github_urls(text):
            candidates.append(
                {
                    "source": source,
                    "paper_id": clean_id,
                    "repo_url": url,
                    "is_official": None,
                    "stars": 0,
                }
            )
    return candidates


def _extract_urls_from_pdf(pdf_url: str) -> list[str]:
    if not pdf_url:
        return []
    try:
        response = requests.get(
            pdf_url,
            timeout=(6, 20),
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if response.status_code != 200 or not response.content:
            return []
        # Fast heuristic: extract from decoded binary text.
        text = response.content[:5_000_000].decode("latin-1", errors="ignore")
        return _extract_github_urls(text)
    except requests.RequestException:
        return []


def _extract_urls_from_tex_source(arxiv_id: str) -> list[str]:
    clean_id = _strip_version(arxiv_id)
    if not clean_id:
        return []
    url = f"https://arxiv.org/e-print/{clean_id}"
    try:
        response = requests.get(
            url,
            timeout=(6, 20),
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if response.status_code != 200 or not response.content:
            return []
        # Cap source payload to avoid extreme memory usage.
        payload = response.content[:15_000_000]
        fileobj = io.BytesIO(payload)
        with tarfile.open(fileobj=fileobj, mode="r:*") as archive:
            urls: list[str] = []
            scanned = 0
            for member in archive.getmembers():
                if scanned >= 25:
                    break
                if not member.isfile():
                    continue
                lower_name = member.name.lower()
                if not lower_name.endswith((".tex", ".txt", ".md", ".rst", ".bbl")):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                scanned += 1
                chunk = extracted.read(300_000).decode("utf-8", errors="ignore")
                urls.extend(_extract_github_urls(chunk))
            return list(dict.fromkeys(urls))
    except Exception:
        return []


def extract_urls_from_pdf_or_tex(paper: dict) -> list[dict]:
    """Step 3: parse URLs from PDF bytes or TeX source."""
    arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
    pdf_url = str(paper.get("pdf_url", "") or "").strip() or _build_pdf_url(arxiv_id)

    candidates: list[dict] = []
    for url in _extract_urls_from_pdf(pdf_url):
        candidates.append(
            {
                "source": "pdf_text",
                "paper_id": _strip_version(arxiv_id),
                "repo_url": url,
                "is_official": None,
                "stars": 0,
            }
        )

    if not candidates:
        for url in _extract_urls_from_tex_source(arxiv_id):
            candidates.append(
                {
                    "source": "tex_source",
                    "paper_id": _strip_version(arxiv_id),
                    "repo_url": url,
                    "is_official": None,
                    "stars": 0,
                }
            )
    return candidates


def normalize_dedupe_validate(candidates: list[dict]) -> list[dict]:
    """Step 4: normalize URLs, de-duplicate candidates, and verify existence."""
    merged: dict[str, dict] = {}
    for raw in candidates:
        canonical = _canonicalize_repo_url(str(raw.get("repo_url", "") or ""))
        if not canonical:
            continue
        current = merged.get(canonical)
        if current is None:
            current = {
                "repo_url": canonical,
                "paper_id": str(raw.get("paper_id", "") or ""),
                "sources": [],
                "source": str(raw.get("source", "") or "unknown"),
                "is_official": raw.get("is_official"),
                "stars": int(raw.get("stars", 0) or 0),
                "framework": str(raw.get("framework", "") or ""),
                "raw_urls": [],
            }
            merged[canonical] = current

        source = str(raw.get("source", "") or "unknown")
        if source and source not in current["sources"]:
            current["sources"].append(source)
        current["source"] = (
            source
            if SOURCE_BASE_SCORE.get(source, 0) >= SOURCE_BASE_SCORE.get(current["source"], 0)
            else current["source"]
        )
        current["stars"] = max(current["stars"], int(raw.get("stars", 0) or 0))
        if raw.get("is_official") is True:
            current["is_official"] = True
        if raw.get("framework") and not current["framework"]:
            current["framework"] = str(raw.get("framework"))

        raw_url = str(raw.get("repo_url", "") or "")
        if raw_url and raw_url not in current["raw_urls"]:
            current["raw_urls"].append(raw_url)

    normalized = list(merged.values())
    for item in normalized:
        verified, status = _verify_repo_url(item["repo_url"])
        item["verified"] = verified
        item["verification_status"] = status
    return normalized


def _score_candidate(candidate: dict, paper_title: str) -> float:
    base = SOURCE_BASE_SCORE.get(str(candidate.get("source", "")), 0.45)
    sources = candidate.get("sources", [])
    multi_source_bonus = min(0.08, max(0, len(sources) - 1) * 0.04)
    verified_bonus = 0.12 if candidate.get("verified") else -0.12
    official_bonus = 0.08 if candidate.get("is_official") is True else 0.0

    stars = int(candidate.get("stars", 0) or 0)
    stars_bonus = 0.0
    if stars > 0:
        stars_bonus = min(0.06, math.log10(stars + 1) * 0.02)

    title_tokens = _significant_tokens(paper_title)
    repo_tokens = _significant_tokens(candidate.get("repo_url", ""))
    overlap = 0.0
    if title_tokens and repo_tokens:
        overlap = len(title_tokens & repo_tokens) / max(1, min(len(title_tokens), len(repo_tokens)))
    overlap_bonus = overlap * 0.08

    score = base + multi_source_bonus + verified_bonus + official_bonus + stars_bonus + overlap_bonus
    return round(max(0.0, min(1.0, score)), 4)


def score_and_rank(candidates: list[dict], paper_title: str) -> list[dict]:
    """Step 5: score and rank candidates."""
    ranked = []
    for candidate in candidates:
        enriched = dict(candidate)
        enriched["confidence"] = _score_candidate(enriched, paper_title)
        ranked.append(enriched)

    ranked.sort(
        key=lambda c: (
            c.get("confidence", 0.0),
            1 if c.get("verified") else 0,
            int(c.get("stars", 0) or 0),
        ),
        reverse=True,
    )
    return ranked


def extract_repo_candidates_for_paper(paper: dict, top_k: int = 3) -> dict:
    """
    End-to-end deterministic extraction pipeline:
      PWC(arxiv_id) -> arXiv metadata -> PDF/Tex -> normalize/validate -> rank
    """
    arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
    if not arxiv_id:
        arxiv_id = _extract_arxiv_id_from_url(str(paper.get("url", "") or ""))
    paper_id = _strip_version(arxiv_id)
    title = str(paper.get("title", "") or "")

    raw_candidates: list[dict] = []
    pwc_candidates = get_repos_from_pwc(paper_id)
    raw_candidates.extend(pwc_candidates)

    if not raw_candidates:
        raw_candidates.extend(parse_arxiv_metadata_for_urls(paper_id))

    if not raw_candidates:
        raw_candidates.extend(extract_urls_from_pdf_or_tex(paper))

    normalized = normalize_dedupe_validate(raw_candidates)
    ranked = score_and_rank(normalized, title)

    high_conf = [c for c in ranked if c.get("confidence", 0.0) >= 0.60]
    top_candidates = high_conf[:top_k] if high_conf else ranked[:top_k]
    top_urls = {c.get("repo_url") for c in top_candidates}
    low_confidence = [c for c in ranked if c.get("repo_url") not in top_urls]

    return {
        "paper_id": paper_id,
        "top_candidates": top_candidates,
        "low_confidence_candidates": low_confidence,
        "all_candidates": ranked,
        "pipeline": {
            "pwc_hits": len(pwc_candidates),
            "raw_candidate_count": len(raw_candidates),
            "normalized_candidate_count": len(normalized),
        },
    }


def build_repo_candidate_map(papers: list[dict], top_k: int = 3) -> dict[str, list[dict]]:
    """
    Batch helper:
      Input  -> list of arXiv papers
      Output -> paper_id -> [repo candidates]
    """
    mapping: dict[str, list[dict]] = {}
    for paper in papers:
        result = extract_repo_candidates_for_paper(paper, top_k=top_k)
        paper_id = str(result.get("paper_id", "") or "").strip()
        if not paper_id:
            continue
        mapping[paper_id] = result.get("all_candidates", []) or []
    return mapping
