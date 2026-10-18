"""Export the after-workshop reading as HTML: two separate pages and one combined page.

Uses the executed copies written by scripts/execute_answer_notebooks.py, so run that
first. Output goes to artifacts/reading/. The pages contain simulator truth and the
reference answers to every task: share them only after the session.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
EXECUTED = ROOT / "artifacts" / "causal_study_20261015" / "executed"
OUT = ROOT / "artifacts" / "reading"

PARTS = [
    ("part-1", "Part 1 · The complete analysis", "causal_cafe_complete",
     "Everything behind the reveal: diagnostics, T-Learner vs. S-Learner vs. boosting, "
     "the budget policy, and all of it checked against the simulator's truth."),
    ("part-2", "Part 2 · Beyond the T-Learner: Causal Forest and more data", "causal_forest_appendix",
     "A pre-run study of 240 fits: when a Causal Forest helps, when more data helps, "
     "and why neither fixes a hidden confounder."),
]

# Machine-specific paths must never reach a page handed to participants.
LOCAL_PATH = re.compile(r"[A-Za-z]:\\\\?(Users|Club)[^\s\"'<]*")


def load(stem: str) -> nbformat.NotebookNode:
    path = EXECUTED / f"{stem}.executed.ipynb"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run scripts/execute_answer_notebooks.py first")
    return nbformat.read(path, as_version=4)


def cover(parts: list[tuple[str, str, str, str]]) -> nbformat.NotebookNode:
    lines = [
        "# Causal Cafe Lab · After-workshop reading",
        "",
        "Read this after the session. It uses the simulator's ground truth, which no real "
        "company has, and contains reference answers to all five tasks. Every check against "
        "truth is a teaching check on synthetic data, not evidence about a real campaign.",
        "",
        "**Contents**",
        "",
    ]
    lines += [f"{i}. [{title}](#{anchor}): {blurb}" for i, (anchor, title, _, blurb) in enumerate(parts, 1)]
    return nbformat.v4.new_markdown_cell("\n".join(lines))


def divider(anchor: str, title: str, blurb: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(
        f'<a id="{anchor}"></a>\n\n---\n\n# {title}\n\n{blurb}\n\n[Back to contents](#top)'
    )


def export(notebook: nbformat.NotebookNode, target: Path, title: str) -> None:
    exporter = HTMLExporter(template_name="lab")
    body, _ = exporter.from_notebook_node(notebook, resources={"metadata": {"name": title}})
    leaked = LOCAL_PATH.findall(body)
    if leaked:
        raise ValueError(f"{target.name} contains local paths: {sorted(set(leaked))[:3]}")
    target.write_text(body, encoding="utf-8")
    print(f"{target.relative_to(ROOT).as_posix()}  ({target.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notebooks = {stem: load(stem) for _, _, stem, _ in PARTS}

    for _, title, stem, _ in PARTS:
        export(notebooks[stem], OUT / f"{stem}.html", title)

    combined = copy.deepcopy(notebooks[PARTS[0][2]])
    first = cover(PARTS)
    first.source = '<a id="top"></a>\n\n' + first.source
    cells = [first]
    for anchor, title, stem, blurb in PARTS:
        cells.append(divider(anchor, title, blurb))
        cells.extend(copy.deepcopy(notebooks[stem].cells))
    combined.cells = cells
    nbformat.validate(combined)
    export(combined, OUT / "causal_cafe_reading.html", "Causal Cafe Lab · After-workshop reading")


if __name__ == "__main__":
    main()
