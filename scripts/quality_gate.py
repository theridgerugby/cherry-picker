"""Project quality gate for duplication, complexity, and maintainability.

Default behavior is ratcheted:
- inspect changed Python files
- allow existing baseline debt
- fail only on *new* violations
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from radon.complexity import cc_visit
from radon.metrics import mi_visit


@dataclass
class Thresholds:
    max_file_lines: int = 300
    max_function_lines: int = 60
    max_complexity: int = 12
    min_mi: float = 65.0


@dataclass
class Violation:
    key: str
    message: str


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def tracked_python_files(explicit_files: list[str] | None = None) -> list[Path]:
    if explicit_files:
        candidates = [Path(p) for p in explicit_files]
        return [p for p in candidates if p.suffix == ".py" and p.exists()]

    result = _run(["git", "ls-files", "*.py"])
    if result.returncode != 0:
        print(result.stderr.strip() or "Unable to list tracked Python files.")
        sys.exit(2)

    files = []
    for line in result.stdout.splitlines():
        path = Path(line.strip())
        if path.exists():
            files.append(path)
    return files


def changed_python_files() -> list[Path]:
    paths: set[str] = set()

    for cmd in (
        ["git", "diff", "--name-only", "--cached", "--", "*.py"],
        ["git", "diff", "--name-only", "--", "*.py"],
        ["git", "ls-files", "--others", "--exclude-standard", "*.py"],
    ):
        result = _run(cmd)
        if result.returncode == 0:
            paths.update(line.strip() for line in result.stdout.splitlines() if line.strip())

    return [Path(p) for p in sorted(paths) if Path(p).suffix == ".py" and Path(p).exists()]


def _attach_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._parent = parent


def _qualified_function_name(node: ast.AST) -> str:
    name = getattr(node, "name", "<unknown>")
    parts = [name]
    current = getattr(node, "_parent", None)
    while current is not None:
        if isinstance(current, ast.ClassDef):
            parts.append(current.name)
        current = getattr(current, "_parent", None)
    return ".".join(reversed(parts))


def check_file_and_function_lengths(files: list[Path], t: Thresholds) -> list[Violation]:
    violations: list[Violation] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            violations.append(
                Violation(
                    key=f"read_error|{path}",
                    message=f"{path}: cannot read file ({exc})",
                )
            )
            continue

        lines = text.splitlines()
        if len(lines) > t.max_file_lines:
            violations.append(
                Violation(
                    key=f"file_lines|{path}",
                    message=f"{path}: {len(lines)} lines exceeds max_file_lines={t.max_file_lines}",
                )
            )

        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            violations.append(
                Violation(
                    key=f"syntax_error|{path}",
                    message=f"{path}: syntax error ({exc})",
                )
            )
            continue

        _attach_parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            end_lineno = getattr(node, "end_lineno", None)
            if end_lineno is None:
                continue
            span = end_lineno - node.lineno + 1
            if span > t.max_function_lines:
                qname = _qualified_function_name(node)
                violations.append(
                    Violation(
                        key=f"function_lines|{path}|{qname}",
                        message=(
                            f"{path}:{node.lineno} function {qname} has {span} lines "
                            f"(max {t.max_function_lines})"
                        ),
                    )
                )
    return violations


def check_radon(files: list[Path], t: Thresholds) -> list[Violation]:
    violations: list[Violation] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            violations.append(
                Violation(
                    key=f"radon_read_error|{path}",
                    message=f"{path}: cannot read for radon ({exc})",
                )
            )
            continue

        blocks = cc_visit(text)
        for block in blocks:
            if block.complexity > t.max_complexity:
                violations.append(
                    Violation(
                        key=f"complexity|{path}|{block.name}",
                        message=(
                            f"{path}:{block.lineno} complexity={block.complexity} "
                            f"exceeds max_complexity={t.max_complexity} ({block.name})"
                        ),
                    )
                )

        mi_score = mi_visit(text, multi=True)
        if mi_score < t.min_mi:
            violations.append(
                Violation(
                    key=f"mi|{path}",
                    message=(
                        f"{path}: maintainability_index={mi_score:.2f} "
                        f"below min_mi={t.min_mi:.2f}"
                    ),
                )
            )
    return violations


def _normalize_duplicate_line(line: str) -> str:
    normalized = line.strip()
    normalized = re.sub(r":[0-9]+", ":#", normalized)
    normalized = re.sub(r"\b[0-9]+\b", "#", normalized)
    return normalized


def check_duplicate_code(files: list[Path]) -> list[Violation]:
    if not files:
        return []

    cmd = [
        "pylint",
        "--score=n",
        "--disable=all",
        "--enable=duplicate-code",
        *[str(f) for f in files],
    ]
    result = _run(cmd)

    issues: list[Violation] = []
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    lines = output.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "R0801" not in line:
            i += 1
            continue

        j = i + 1
        block: list[str] = []
        while j < len(lines) and "R0801" not in lines[j]:
            block.append(lines[j])
            j += 1

        file_markers = []
        snippet_line = ""
        for b in block:
            marker = re.match(r"^==([^:]+):\[\d+:\d+\]$", b.strip())
            if marker:
                file_markers.append(marker.group(1))
                continue
            if not snippet_line and b.strip() and not b.strip().endswith("(duplicate-code)"):
                snippet_line = b.strip()

        if len(file_markers) >= 2:
            f1, f2 = sorted(file_markers[:2])
            snippet_key = re.sub(r"\s+", " ", snippet_line)[:80]
            key = f"duplicate_code|{f1}|{f2}|{snippet_key}"
            msg = f"{line.strip()} [{f1} <-> {f2}]"
        else:
            key = f"duplicate_code|{_normalize_duplicate_line(line)}"
            msg = line.strip()

        issues.append(Violation(key=key, message=msg))
        i = j

    # De-duplicate exact keys from noisy pylint output.
    deduped: dict[str, Violation] = {}
    for issue in issues:
        deduped[issue.key] = issue
    issues = list(deduped.values())
    return issues


def load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("violation_keys", []))
    except (json.JSONDecodeError, OSError):
        return set()


def save_baseline(path: Path, violations: list[Violation], thresholds: Thresholds) -> None:
    payload = {
        "version": 1,
        "thresholds": {
            "max_file_lines": thresholds.max_file_lines,
            "max_function_lines": thresholds.max_function_lines,
            "max_complexity": thresholds.max_complexity,
            "min_mi": thresholds.min_mi,
        },
        "violation_keys": sorted({v.key for v in violations}),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quality gate checks.")
    parser.add_argument("--max-file-lines", type=int, default=300)
    parser.add_argument("--max-function-lines", type=int, default=60)
    parser.add_argument("--max-complexity", type=int, default=12)
    parser.add_argument("--min-mi", type=float, default=65.0)
    parser.add_argument("--scope", choices=["changed", "all"], default="changed")
    parser.add_argument("--baseline-file", default=".quality-baseline.json")
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--ignore-baseline", action="store_true")
    parser.add_argument("files", nargs="*")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = Thresholds(
        max_file_lines=args.max_file_lines,
        max_function_lines=args.max_function_lines,
        max_complexity=args.max_complexity,
        min_mi=args.min_mi,
    )

    if args.files:
        files = tracked_python_files(args.files)
    elif args.scope == "all":
        files = tracked_python_files()
    else:
        files = changed_python_files()

    if not files:
        print("No Python files selected for quality gate.")
        print("Tip: use --scope all to run a full repository audit.")
        return 0

    violations: list[Violation] = []
    violations.extend(check_file_and_function_lengths(files, thresholds))
    violations.extend(check_radon(files, thresholds))

    # Run duplicate-code on full tracked Python files for stable results.
    # Running pylint duplicate check on subsets can produce noisy pair changes.
    duplicate_target_files = tracked_python_files()
    violations.extend(check_duplicate_code(duplicate_target_files))

    baseline_path = Path(args.baseline_file)
    if args.update_baseline:
        save_baseline(baseline_path, violations, thresholds)
        print(f"Baseline updated at {baseline_path} with {len({v.key for v in violations})} keys.")
        return 0

    active_violations = violations
    if not args.ignore_baseline:
        baseline_keys = load_baseline(baseline_path)
        if baseline_keys:
            active_violations = [v for v in violations if v.key not in baseline_keys]
            suppressed = len(violations) - len(active_violations)
            if suppressed:
                print(f"Suppressed {suppressed} baseline violation(s) from {baseline_path}.")

    if active_violations:
        print("Quality gate failed with the following NEW issues:")
        for issue in active_violations:
            print(f" - {issue.message}")
        print("\nIf intentional, refresh baseline with:")
        print("  python scripts/quality_gate.py --scope all --update-baseline")
        return 1

    print("Quality gate passed.")
    print(
        f"Checked {len(files)} file(s) | scope={args.scope} | "
        f"max_file_lines={thresholds.max_file_lines}, "
        f"max_function_lines={thresholds.max_function_lines}, "
        f"max_complexity={thresholds.max_complexity}, "
        f"min_mi={thresholds.min_mi}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
