# Cherry Picker System Architecture & Research Process Report

This document provides an in-depth analysis of the **Cherry Picker** research application, exploring its core functionality, intricate research workflows, and the specifics of its task scheduling and execution pipeline.

## 1. System Overview & Core Functionality

Cherry Picker is a sophisticated, AI-agentic web application (built with Streamlit) designed to assist researchers in navigating the immense flow of newly published academic literature on arXiv. Rather than just returning raw search results, it intelligently fetches, filters, parses, evaluates, and synthesizes complex academic content into coherent, structured markdown reports using Google's generative AI models (Gemini series).

The tool significantly cuts down the manual work of literature review by orchestrating several domain-specific scripts that work in unison to perform the "research process." 

## 2. In-Depth Look at the Research System

The entire research pipeline takes an ambiguous natural language search term and transforms it into structured knowledge across multiple stages:

### A. Input Processing & Query Generation
- **Validation**: User queries are strictly validated via `input_validator.py`, which leverages a lightweight LLM (`gemini-2.5-flash`) to ensure that queries are concrete research topics rather than generic keywords.
- **Query Translation (`query_generator.py`)**: Converts colloquial language into a highly optimized, arXiv-compliant boolean string. It uses discipline-detection rules (CS, Biology, Physics, Engineering) to apply specific `cat:` (category) and `ti:`/`abs:` filters. Rule-based overrrides exist for tricky domains (like "anti-ice").

### B. Adaptive Fetching (`paper_fetcher.py`)
- The system attempts to fetch papers dynamically by expanding its search time-window (e.g., starting at 14 days and stepping up to 60 days for "latest") until it meets a `min_papers` quota.
- **Fallback Recall Expansion**: If the search yields zero results due to excessively narrow category filters, the system progressively relaxes the query, stripping `cat:` assertions until a foundational baseline of papers is met.

### C. Domain Relevance Filtering & Diversity
- Because arXiv queries can be noisy, `prefilter_papers` (via `paper_extractor.py`) performs a map-phase analysis on the fetched abstracts. It evaluates whether the underlying domain is primarily driven by the user's topic or if the word was only mentioned incidentally, scoring it from 1 to 10.
- If too many documents are retrieved (max limit of 30 enforced in `app.py`), the system trims the list to those exhibiting maximum textual *diversity* by parsing tokens and maximizing the Jaccard distance between selected papers.

### D. Deep Extraction & Credibility Scoring
- **Full JSON Extraction (`paper_extractor.py`)**: For the filtered candidates, Gemini structures the free-text abstract into a rigid schema, detecting the exact methodologies (Theoretical, ML, Physical Experiments), data modalities, limitations, and standardizing metrics on a strictly defined 1-5 scale (Industrial Readiness, Theoretical Depth, and Domain Specificity). 
- **Repository Detection**: Integrated natively with the extraction process is an attempt to automatically identify and validate GitHub code repositories from `repo_extractor.py`.
- **Credibility Scorer (`credibility_scorer.py`)**: Each structured paper receives a 0-100 grade taking into account the venue prestige (if extracted), institutional backing of authors, temporal recency, and abstract textual richness.

### E. LLM-Assisted Synthesis & Reporting (`report_generator.py`)
This final stage unifies the isolated findings:
- Extracted schemas are embedded and pushed into a local offline vector database (ChromaDB) for persistence. 
- Three synthesis prompts reconstruct the knowledge:
    1. A base timeline narrative grouping trends into themes and ranking a prescribed "Reading Order" augmented with exact BibTeX formats.
    2. A structured Matrix evaluating methodology choices.
    3. An *Extrapolated Gaps* analyzer that uses cross-paper reasoning to identify blind spots left unsolved even assuming all the proposed solutions succeeded.

---

## 3. The Task Scheduling Flow & Development Process

The application's execution is synchronous at the user-facing level (Streamlit operates top-down), but heavily parallels network-bound and LLM-bound operations for efficiency. 

### A. How Tasks Are Scheduled
There is no heavyweight external task scheduler like Celery or Airflow. Instead, concurrency is handled directly within the Python runtime using standard library utilities:
- **`concurrent.futures.ThreadPoolExecutor`**: At critical bottlenecks where independent LLM calls are needed per document, threads are dynamically provisioned.
    - **Prefiltering**: Maps across up to 10 worker threads (`max_workers=10`) classifying paper domain relevance rapidly with a fast-tier LLM.
    - **Extraction phase (`app.py`)**: Uses `ThreadPoolExecutor(max_workers=10)` to asynchronously request structured JSON from Gemini for up to 30 diverse papers. Using `as_completed(futures)`, the main thread retrieves results as they arrive, simultaneously incrementing a visual loading bar (via Streamlit's `st.empty()` manipulation) to keep the user informed.
    - **Parallel Report Agents (`cached_parallel_agents.py`)**: After extraction, complex reporting is broken into distinct components (Base Report, Extrapolated Gaps, Methodology Matrix). These three tasks are orchestrated synchronously in parallel (`max_workers=3`).

### B. Shared Context Caching
To optimize API latency and cost, the parallel reporting agents do not repetitively embed the same input context payload. Instead, they share an upfront Context Cache via `build_shared_cache()` (in `cache_manager.py`). The orchestration spawns parallel generation duties but references the cached pointer globally, deleting the cache aggressively once the futures return to prevent memory leakage on Google's cache servers.

### C. Robust Error Handling Context
The threading development process favors resilience. Since independent papers might trigger JSON `ValidationError` instances, LLM hallucinations, or simple network `429` ratelimits, the thread closures explicitly wrap extraction steps in `try-except` blocks.
- Failed extractions return `None` rather than bringing down the thread pool.
- The aggregator filters `None` values and processes partial batches gracefully, meaning the scheduling flow effectively promises "best-effort extraction in bounded time."

### D. Iterative Streamlit State Handling
All the executed results are appended to a global `st.session_state` map in `app.py`. This provides cache-busting logic. The interface instantly pivots from displaying a "loading hint" state into showing rich markdown with the results localized entirely in the browser runtime, avoiding re-calculation unless a strictly new analysis is triggered.

## Summary

The **Cherry Picker** orchestration emphasizes adaptive latency reduction. It heavily leans on rule-based fallbacks during fetch, leverages lightweight threading semantics for LLM map-reduce tasks to guarantee responsiveness, and stitches disjointed sub-agents into a seamless, deterministic narrative for end-users relying on rapid context consumption.
