"""Synchronize notebook sources while retaining unchanged-cell outputs and IDs."""

from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import jupytext
import nbformat
from jupytext.combine import combine_inputs_with_outputs
from scripts.build_answer_notebook import build_answers


def main():
    (ROOT/"solutions/workshop_answers.py").write_text(
        build_answers((ROOT/"workshop.py").read_text(encoding="utf-8")),encoding="utf-8")
    for stem in ["workshop","solutions/workshop_answers","solutions/causal_cafe_complete","solutions/causal_forest_appendix"]:
        source=ROOT/f"{stem}.py"
        target=source.with_suffix(".ipynb")
        notebook=jupytext.read(source)
        if target.exists():
            notebook=combine_inputs_with_outputs(notebook,nbformat.read(target,as_version=4))
        nbformat.validate(notebook)
        jupytext.write(notebook,target)
        print("Synchronized:",target.relative_to(ROOT))


if __name__=="__main__":
    main()
