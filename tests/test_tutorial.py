"""The tutorial notebook (examples/pbsolve_tutorial.ipynb) stays consistent and runs.

The structural checks need only the standard library and run everywhere. The
execution test needs nbformat, nbclient, ipykernel and matplotlib and is
skipped without them.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "pbsolve_tutorial.ipynb"

# Results the tutorial states; its saved outputs must show them.
CLAIMS = [
    "All tests passed.",  # pbsolve test
    "charges and radii identical to pdb2pqr: True",  # section 10
    "identical results: True",  # section 11, batched = one by one
]


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _cells(nb: dict, kind: str) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == kind]


def _text(outputs: list) -> str:
    parts = []
    for o in outputs:
        if o.get("output_type") == "stream":
            parts.append("".join(o["text"]) if isinstance(o["text"], list) else o["text"])
        for key in ("text/plain", "text/markdown"):
            value = o.get("data", {}).get(key)
            if value is not None:
                parts.append("".join(value) if isinstance(value, list) else value)
    return "\n".join(parts)


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_sections_are_numbered_listed_and_dated(notebook):
    intro = _source(notebook["cells"][0])
    headings = []
    for cell in _cells(notebook, "markdown"):
        src = _source(cell)
        lines = src.splitlines()
        if lines and lines[0].startswith("## "):
            headings.append(lines[0][3:])
            assert re.search(r"_Last updated: \d{4}-\d{2}-\d{2}", src), f"section without a date: {lines[0]}"
    numbered = [h for h in headings if h[0].isdigit()]
    assert [int(h.split(".")[0]) for h in numbered] == list(range(1, len(numbered) + 1))
    for h in numbered:
        number, title = h.split(". ", 1)
        assert f"{number}. {title}" in intro, f"section missing from the contents: {h}"
    assert "Setup" in headings


def test_changelog_covers_every_section_date(notebook):
    intro = _source(notebook["cells"][0])
    logged = set(re.findall(r"^\| (\d{4}-\d{2}-\d{2}) \|", intro, flags=re.M))
    assert logged, "the tutorial changelog table is empty"
    dated = {
        m
        for cell in _cells(notebook, "markdown")
        for m in re.findall(r"_Last updated: (\d{4}-\d{2}-\d{2})", _source(cell))
    }
    assert dated <= logged, f"sections updated on dates missing from the changelog: {sorted(dated - logged)}"


def test_code_cells_are_valid_python(notebook):
    for cell in _cells(notebook, "code"):
        ast.parse(_source(cell))


def test_referenced_test_files_exist(notebook):
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    names = re.findall(r'DATA / "([^"]+)"', code)
    assert names
    for name in names:
        assert (ROOT / "tests" / "data" / name).is_file(), name


def test_saved_outputs_are_complete_and_show_the_stated_results(notebook):
    code = _cells(notebook, "code")
    assert all(c.get("execution_count") for c in code), "store the notebook executed (all cells run)"
    counts = [c["execution_count"] for c in code]
    assert counts == sorted(counts), "cells were not run from top to bottom"
    assert not [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    text = "\n".join(_text(c.get("outputs", [])) for c in code)
    for claim in CLAIMS:
        assert claim in text, claim


def test_notebook_runs():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    pytest.importorskip("matplotlib")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(
        nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK.parent)}}
    )
    client.execute()
    text = "\n".join(_text(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
    for claim in CLAIMS:
        assert claim in text, claim
    if "pbsolve vs APBS" in text:
        assert "binding verdict: PASS" in text and "verdict: FAIL" not in text
