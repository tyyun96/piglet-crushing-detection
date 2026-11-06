"""Builds the analysis notebook from nb_src.py (jupytext percent format), inlining the code sections of core.py and the
video registry so that the notebook is self-contained.

usage: python build_notebook.py [OUT.ipynb] [--lang ko|en] [--appendix]
  --lang      language of the text (default ko): markdown cells tagged lang-ko / lang-en are kept only for their language and
              LANG is set in the setup cell (code-cell text goes through tr(ko, en)); the code itself is identical
  --appendix  append every other script of analysis/ (preprocessing, annotation tools) as read-only code listings, so that all
              code of the project can be read in one notebook
"""
import argparse
import re
from pathlib import Path

import jupytext
from nbformat.v4 import new_markdown_cell

HERE = Path(__file__).resolve().parent

APPENDIX = [  # (title, files) - read-only listings
    ("A. Preprocessing (run once, outputs in cache/ and meta/)", [
        "prep_02_detect.py", "prep_03_ocr_overlay.py", "prep_04_align.py", "prep_05_train_lolo_detectors.py", "prep_06_train_yolov5.py",
        "v5_val.py", "prep_11_previous_study_dataset.py", "prep_12_train_previous_recipe.py", "prep_13_mot_prefill.py"]),
    ("B. Review material and annotation tools (Jupyter notebooks used by the observer)", [
        "review_clips.py", "prep_07_render_review_clips.py", "prep_08_render_annotation_segments.py", "labeler_src.py", "annotate_src.py",
        "prep_09_adjudication_items.py", "adjudicate_src.py", "prep_10_retest_items.py", "retest_src.py", "mot_src.py"]),
]
APPENDIX_INTRO = {
    "ko": "# 부록 — 나머지 코드 전체 (읽기용)\n\n이 노트북이 읽는 `cache/`·`meta/`·주석 파일을 만든 코드다. 실행되지 않는 목록이며, 실제 실행은 "
          "`analysis/` 의 같은 파일로 한다(`analysis/README.md` 의 순서).",
    "en": "# Appendix — all remaining code (read-only)\n\nThe code that produced the `cache/`, `meta/` and annotation files this notebook "
          "reads. These listings are not executed; run the same files in `analysis/` (order in the README).",
}


def build(out, lang="ko", appendix=False):
    core = (HERE / "core.py").read_text()
    sections = {}
    for m in re.finditer(r"# %% \[([\w-]+)\]\n(.*?)(?=\n# %% \[|\Z)", core, flags=re.S):
        sections[m.group(1)] = m.group(2).strip("\n")
    reg = (HERE / "video_registry.py").read_text().strip("\n")
    src = (HERE / "nb_src.py").read_text()
    src = src.replace("# @@REGISTRY", reg)
    src = re.sub(r"# @@CORE:([\w-]+)", lambda m: sections[m.group(1)], src)
    assert "@@" not in src, "unresolved placeholder"
    assert src.count('LANG = "ko"') == 1, "language line not found"
    src = src.replace('LANG = "ko"', f'LANG = "{lang}"')
    nb = jupytext.reads(src, fmt="py:percent")
    cells = []
    for c in nb.cells:
        tags = c.metadata.get("tags", [])
        langs = [t for t in tags if t.startswith("lang-")]
        if langs and f"lang-{lang}" not in langs:
            continue
        rest = [t for t in tags if not t.startswith("lang-")]
        if rest:
            c.metadata["tags"] = rest
        else:
            c.metadata.pop("tags", None)
        cells.append(c)
    if appendix:
        cells.append(new_markdown_cell(APPENDIX_INTRO[lang]))
        for title, files in APPENDIX:
            cells.append(new_markdown_cell(f"## {title}"))
            for f in files:
                code = (HERE / f).read_text().rstrip("\n")
                cells.append(new_markdown_cell(f"### `analysis/{f}`\n\n```python\n{code}\n```"))
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python (crushing .venv)", "language": "python", "name": "crushing-venv"}
    jupytext.write(nb, out, fmt="ipynb")
    return len(cells)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=str(HERE.parent / "analysis.ipynb"))
    ap.add_argument("--lang", default="en", choices=["ko", "en"])
    ap.add_argument("--appendix", action="store_true")
    a = ap.parse_args()
    print("written", a.out, build(a.out, a.lang, a.appendix), "cells")
