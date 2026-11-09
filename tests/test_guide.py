"""The user guide (docs/guide) stays true.

* Every ```python block of a page runs, in order, in one namespace per page.
  Blocks written as ```{code-block} python are illustrations with
  placeholder files (or needing APBS, OpenMM) and are not run.
* The option tables of the command-line page list exactly the options each
  command accepts.

The examples run in a temporary directory that links ``tests`` from the
repository, so their paths work and their output files land there. To keep
the suite fast (CI runners are slow), ``pbsolve.Solver`` builds coarse
automatic grids here: a fine spacing of at least 1 Å and ``fadd=12.0``
unless an example sets ``fadd``. The examples still exercise every call they
show; only the numbers they print differ from a run at full resolution.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import pbsolve

REPO = Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "guide"
PAGES = sorted(p.stem for p in GUIDE.glob("*.md"))

FENCE = re.compile(r"^```(\S*)[^\n]*\n(.*?)^```\s*$", re.M | re.S)


def python_blocks(page: str) -> list[str]:
    """The ```python blocks of a guide page, in order."""
    text = (GUIDE / f"{page}.md").read_text(encoding="utf-8")
    return [body for lang, body in FENCE.findall(text) if lang == "python"]


COARSEST_SPACING = 1.0  # Å


def coarse_solver(**kwargs):
    """``pbsolve.Solver`` with coarse automatic grids (see the module docstring)."""
    kwargs["grid_spacing"] = max(kwargs.get("grid_spacing", COARSEST_SPACING), COARSEST_SPACING)
    kwargs.setdefault("fadd", 12.0)
    return SOLVER(**kwargs)


SOLVER = pbsolve.Solver


@pytest.mark.parametrize("page", PAGES)
def test_the_examples_of_the_guide_run(page, tmp_path, monkeypatch):
    blocks = python_blocks(page)
    if not blocks:
        pytest.skip(f"{page}.md has no runnable examples")
    (tmp_path / "tests").symlink_to(REPO / "tests", target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pbsolve, "Solver", coarse_solver)
    namespace: dict = {"__name__": f"guide_{page}"}
    for n, block in enumerate(blocks, 1):
        try:
            exec(compile(block, f"docs/guide/{page}.md (python block {n})", "exec"), namespace)
        except Exception as err:  # pragma: no cover - the message is the point
            raise AssertionError(f"docs/guide/{page}.md, python block {n} failed: {err!r}\n{block}") from err


def test_the_guide_has_runnable_examples():
    # Guards against the fence pattern silently matching nothing.
    assert len(python_blocks("model")) >= 3
    assert sum(len(python_blocks(p)) for p in PAGES) >= 20


# ------------------------------------------------------------------ CLI page

COMMANDS = ["info", "test", "solvation", "convert", "mmpbsa", "prep", "apbs compare", "apbs run"]
OPTION = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")


def cli_options(command: str) -> set[str]:
    """Options `pbsolve COMMAND --help` lists (both spellings, e.g. -I and --include)."""
    env = dict(os.environ, COLUMNS="200")
    out = subprocess.run(
        [sys.executable, "-m", "pbsolve", *command.split(), "--help"],
        capture_output=True, text=True, check=True, env=env, encoding="utf-8",
    ).stdout
    options = set()
    for line in out.splitlines():
        if line.startswith("  -"):  # an entry of the options list: "  -I INCLUDE, --include INCLUDE   help"
            spec = re.split(r"\s{2,}", line.strip())[0]
            options |= set(OPTION.findall(spec))
    return options - {"-h", "--help"}


def documented_options() -> dict[str, set[str]]:
    """Options the command-line page documents for every command."""
    text = (GUIDE / "cli.md").read_text(encoding="utf-8")
    sections = dict(
        (m.group(1).strip(), m.group(2)) for m in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    )

    def table_options(body: str) -> set[str]:
        rows = [line.split("|")[1] for line in body.splitlines() if line.startswith("| `")]
        return {opt for cell in rows for opt in re.findall(r"`(--?[A-Za-z][\w-]*)`", cell)}

    groups = {"#input-options": table_options(sections["Input options"]),
              "#solver-options": table_options(sections["Solver options"])}
    out = {}
    for command in COMMANDS:
        body = sections[f"`pbsolve {command}`"]
        options = table_options(body)
        for anchor, group in groups.items():
            # A section that takes a whole group says so in its text, not in a table row.
            prose = "\n".join(line for line in body.splitlines() if not line.startswith("|"))
            if f"]({anchor})" in prose:
                options |= group
        out[command] = options
    return out


@pytest.mark.parametrize("command", COMMANDS)
def test_the_command_line_page_lists_every_option(command):
    documented = documented_options()[command]
    actual = cli_options(command)
    assert actual - documented == set(), f"`pbsolve {command}` options missing in docs/guide/cli.md"
    assert documented - actual == set(), f"docs/guide/cli.md lists options `pbsolve {command}` does not have"
