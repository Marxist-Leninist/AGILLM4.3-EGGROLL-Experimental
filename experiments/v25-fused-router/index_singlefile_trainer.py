#!/usr/bin/env python3
"""Create a compact, reviewable symbol/context index for a huge trainer file.

GitHub's Contents API omits inline content for files over 1 MiB. This script
runs inside the repository checkout and produces small deterministic reports
containing the exact router, expert, MoE, loss and EGGROLL integration
regions needed by the v25 experiment.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

KEYWORDS = (
    "eggroll",
    "router",
    "routing",
    "expert",
    "moe",
    "mixtureofexperts",
    "mixture_of_experts",
    "topk",
    "top_k",
    "diffusionblock",
    "diffusion_block",
    "local_loss",
    "block_loss",
    "guard",
    "search_crop",
    "route_skew",
    "load_balance",
)

# Wider terms are useful only inside a definition already selected by a
# strong keyword. They are not used for global windows because words such as
# "loss" occur approximately everywhere in a trainer, much like paperwork.
SECONDARY_KEYWORDS = (
    "loss",
    "forward",
    "activation",
    "hidden",
    "gate",
    "dispatch",
    "argmax",
    "softmax",
)


@dataclass(frozen=True)
class Definition:
    qualified_name: str
    kind: str
    start_line: int
    end_line: int
    strong_keywords: tuple[str, ...]
    secondary_keywords: tuple[str, ...]

    @property
    def length(self) -> int:
        return self.end_line - self.start_line + 1


class DefinitionCollector(ast.NodeVisitor):
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.stack: list[str] = []
        self.definitions: list[Definition] = []

    def _record(self, node: ast.AST, name: str, kind: str) -> None:
        start = int(getattr(node, "lineno", 1))
        end = int(getattr(node, "end_lineno", start))
        source = "".join(self.lines[start - 1 : end]).lower()
        qualified = ".".join((*self.stack, name))
        strong = tuple(
            keyword
            for keyword in KEYWORDS
            if keyword in source or keyword in name.lower()
        )
        secondary = tuple(
            keyword for keyword in SECONDARY_KEYWORDS if keyword in source
        )
        self.definitions.append(
            Definition(
                qualified_name=qualified,
                kind=kind,
                start_line=start,
                end_line=end,
                strong_keywords=strong,
                secondary_keywords=secondary,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, node.name, "class")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node, node.name, "function")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node, node.name, "async_function")
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path, required=True)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=180)
    parser.add_argument("--max-lines", type=int, default=7000)
    return parser.parse_args()


def keyword_hits(
    line: str,
    keywords: Iterable[str] = KEYWORDS,
) -> tuple[str, ...]:
    lowered = line.lower()
    return tuple(keyword for keyword in keywords if keyword in lowered)


def coalesce_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end + 1:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def make_windows(
    lines: list[str],
    *,
    context: int,
    max_windows: int,
    max_lines: int,
) -> list[tuple[int, int, tuple[str, ...]]]:
    raw: list[tuple[int, int]] = []
    for index, line in enumerate(lines, start=1):
        if keyword_hits(line):
            raw.append(
                (max(1, index - context), min(len(lines), index + context))
            )
    merged = coalesce_ranges(raw)

    output: list[tuple[int, int, tuple[str, ...]]] = []
    used_lines = 0
    for start, end in merged:
        if len(output) >= max_windows or used_lines >= max_lines:
            break
        if used_lines + (end - start + 1) > max_lines:
            end = start + max_lines - used_lines - 1
        text = "".join(lines[start - 1 : end])
        hits = keyword_hits(text)
        output.append((start, end, hits))
        used_lines += end - start + 1
    return output


def fenced_source(lines: list[str], start: int, end: int) -> str:
    numbered = "".join(
        f"{line_number:>7}: {lines[line_number - 1]}"
        for line_number in range(start, end + 1)
    )
    return f"```python\n{numbered}```\n"


def relevant_definition_windows(
    definition: Definition,
    lines: list[str],
    *,
    context: int = 6,
    max_lines: int = 160,
) -> list[tuple[int, int]]:
    if definition.length <= max_lines:
        return [(definition.start_line, definition.end_line)]

    candidate_ranges: list[tuple[int, int]] = []
    definition_keywords = (*KEYWORDS, *SECONDARY_KEYWORDS)
    for line_number in range(definition.start_line, definition.end_line + 1):
        if keyword_hits(lines[line_number - 1], definition_keywords):
            candidate_ranges.append(
                (
                    max(definition.start_line, line_number - context),
                    min(definition.end_line, line_number + context),
                )
            )
    merged = coalesce_ranges(candidate_ranges)
    selected: list[tuple[int, int]] = []
    used = 0
    for start, end in merged:
        if used >= max_lines:
            break
        if used + (end - start + 1) > max_lines:
            end = start + max_lines - used - 1
        selected.append((start, end))
        used += end - start + 1
    return selected


def main() -> None:
    args = parse_args()
    raw = args.input.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    digest = hashlib.sha256(raw).hexdigest()

    parse_error: str | None = None
    definitions: list[Definition] = []
    try:
        tree = ast.parse(text, filename=str(args.input))
        collector = DefinitionCollector(lines)
        collector.visit(tree)
        definitions = collector.definitions
    except SyntaxError as error:
        parse_error = (
            f"{error.msg} at line {error.lineno}, column {error.offset}"
        )

    relevant = [
        definition for definition in definitions if definition.strong_keywords
    ]
    relevant.sort(key=lambda item: (item.start_line, item.end_line))
    windows = make_windows(
        lines,
        context=args.context,
        max_windows=args.max_windows,
        max_lines=args.max_lines,
    )

    report = {
        "schema": "agillm43.eggroll.v25.singlefile_index.v1",
        "input": str(args.input),
        "sha256": digest,
        "bytes": len(raw),
        "lines": len(lines),
        "ast_parse_error": parse_error,
        "definition_count": len(definitions),
        "relevant_definition_count": len(relevant),
        "definitions": [
            asdict(definition) | {"length": definition.length}
            for definition in definitions
        ],
        "relevant_definitions": [
            asdict(definition) | {"length": definition.length}
            for definition in relevant
        ],
        "global_windows": [
            {"start_line": start, "end_line": end, "keywords": hits}
            for start, end, hits in windows
        ],
    }
    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    markdown: list[str] = [
        "# AGILLM 4.3 single-file trainer integration index\n\n",
        f"- Source: `{args.input}`\n",
        f"- SHA-256: `{digest}`\n",
        f"- Size: {len(raw):,} bytes\n",
        f"- Lines: {len(lines):,}\n",
        f"- AST definitions: {len(definitions):,}\n",
        f"- Strongly relevant definitions: {len(relevant):,}\n",
        (
            f"- AST parse error: `{parse_error}`\n"
            if parse_error
            else "- AST parse: successful\n"
        ),
        "\n## Relevant definitions\n\n",
    ]
    if not relevant:
        markdown.append("No strongly matching definitions were found.\n")
    for definition in relevant:
        markdown.append(
            f"### `{definition.qualified_name}` ({definition.kind}, "
            f"lines {definition.start_line}-{definition.end_line})\n\n"
        )
        markdown.append(
            "Strong keywords: "
            + ", ".join(
                f"`{keyword}`" for keyword in definition.strong_keywords
            )
            + "\n\n"
        )
        for start, end in relevant_definition_windows(definition, lines):
            markdown.append(fenced_source(lines, start, end))
            markdown.append("\n")

    markdown.append("\n## Coalesced global keyword windows\n\n")
    markdown.append(
        "These windows are generated independently of AST nesting so "
        "module-level configuration and call sites remain visible.\n\n"
    )
    for index, (start, end, hits) in enumerate(windows, start=1):
        markdown.append(
            f"### Window {index}: lines {start}-{end} "
            f"({', '.join(hits)})\n\n"
        )
        markdown.append(fenced_source(lines, start, end))
        markdown.append("\n")

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("".join(markdown))


if __name__ == "__main__":
    main()
