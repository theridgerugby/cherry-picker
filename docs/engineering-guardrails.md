# Engineering Guardrails

This project uses hard quality gates so feature work converges instead of growing unchecked.
Checks run on changed Python files in each PR (ratchet strategy), so legacy debt
does not block all delivery while new debt is prevented.

## Metrics (run on every PR)

1. Duplicate code: Pylint `R0801` must be zero.
2. Complexity: Radon cyclomatic complexity must stay under `12` per function.
3. Maintainability: Radon Maintainability Index must stay at or above `65`.
4. Size limits:
   - max file length: `300` lines
   - max function length: `60` lines

All checks are enforced by:
- local pre-commit hook
- GitHub Actions CI workflow

## Toolchain

- Ruff lint + autofix
- Ruff formatter
- Radon complexity + MI checks
- Pylint duplicate-code check

## Local Setup

```bash
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
python scripts/quality_gate.py
```

## Required PR Rhythm

- Every 3-5 feature PRs, create 1 refactor-only PR.
- Refactor-only PRs cannot add product features.
- Refactor-only PRs must reduce at least one of:
  - duplicate-code findings
  - complexity violations
  - oversized files/functions

## Agent Prompt Templates

### 1) Refactor-only mode

You are in Refactor Only mode.
Goal: reduce duplication, lower complexity, shorten files, keep behavior unchanged.

Rules:
1. No new features.
2. No copy-paste logic into new locations.
3. Prefer extract function and extract module.
4. Delete duplicates and centralize reusable helpers.
5. Changes must pass existing tests and quality gates.
6. Output a refactor plan, changed files, and per-file summary.

Finally suggest quality guardrails to tighten if needed.

### 2) Duplicate-first analysis mode

Scan the repository and identify the top 3 duplicate logic patterns.
For each pattern provide:
1. Files/functions where it appears.
2. Why it is duplicate.
3. Proposed shared interface.
4. Post-refactor call example.

Then implement only pattern #1 with minimal surface change.

### 3) Tooling convergence mode

Add and enforce:
1. Ruff lint + format
2. pre-commit hooks
3. CI gate for PRs
4. Radon and/or Pylint duplicate-code thresholds

Provide all config files and exact setup commands.
