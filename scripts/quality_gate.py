"""Project quality gate for duplication, complexity, and maintainability."""

from __future__ import annotations

import argparse
import ast
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


def check_file_and_function_lengths(files: list[Path], t: Thresholds) -> list[str]:
    violations: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            violations.append(f"{path}: cannot read file ({exc})")
            continue

        lines = text.splitlines()
        if len(lines) > t.max_file_lines:
            violations.append(
                f"{path}: {len(lines)} lines exceeds max_file_lines={t.max_file_lines}"
            )

        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            violations.append(f"{path}: syntax error ({exc})")
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
                    f"{path}:{node.lineno} function {qname} has {span} lines "
                    f"(max {t.max_function_lines})"
                )
    return violations


def check_radon(files: list[Path], t: Thresholds) -> list[str]:
    violations: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            violations.append(f"{path}: cannot read for radon ({exc})")
            continue

        blocks = cc_visit(text)
        for block in blocks:
            if block.complexity > t.max_complexity:
                violations.append(
                    f"{path}:{block.lineno} complexity={block.complexity} "
                    f"exceeds max_complexity={t.max_complexity} ({block.name})"
                )

        mi_score = mi_visit(text, multi=True)
        if mi_score < t.min_mi:
            violations.append(
                f"{path}: maintainability_index={mi_score:.2f} "
                f"below min_mi={t.min_mi:.2f}"
            )
    return violations


def check_duplicate_code(files: list[Path]) -> list[str]:
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

    issues = []
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    for line in output.splitlines():
        if "R0801" in line:
            issues.append(line.strip())
    return issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quality gate checks.")
    parser.add_argument("--max-file-lines", type=int, default=300)
    parser.add_argument("--max-function-lines", type=int, default=60)
    parser.add_argument("--max-complexity", type=int, default=12)
    parser.add_argument("--min-mi", type=float, default=65.0)
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

    files = tracked_python_files(args.files)
    if not files:
        print("No tracked Python files found.")
        return 0

    violations: list[str] = []
    violations.extend(check_file_and_function_lengths(files, thresholds))
    violations.extend(check_radon(files, thresholds))

    duplicate_issues = check_duplicate_code(files)
    violations.extend(duplicate_issues)

    if violations:
        print("Quality gate failed with the following issues:")
        for issue in violations:
            print(f" - {issue}")
        return 1

    print("Quality gate passed.")
    print(
        f"Checked {len(files)} files | "
        f"max_file_lines={thresholds.max_file_lines}, "
        f"max_function_lines={thresholds.max_function_lines}, "
        f"max_complexity={thresholds.max_complexity}, "
        f"min_mi={thresholds.min_mi}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
