"""Documentation of the public API.

The compiled module is documented twice: in ``python/pbsolve/_core.pyi``
(read by the documentation site and by editors) and in the extension itself
(``help()``, from the ``///`` comments of ``crates/pbsolve-py/src/lib.rs``).
These tests keep the two identical, the stub complete, and every public name
of the Python package documented.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from pbsolve import _core

PACKAGE = Path(__file__).resolve().parents[1] / "python" / "pbsolve"
STUB = PACKAGE / "_core.pyi"

# Modules on the documentation site (docs/conf.py ignores __main__,
# _console and test_installation).
MODULES = ["__init__", "apbs", "dx", "io", "mmpbsa", "openmm_bridge", "prep", "radii"]


def _docstring_after(body: list[ast.stmt], i: int) -> str | None:
    """The string literal right after ``body[i]`` (an attribute docstring), if any."""
    nxt = body[i + 1] if i + 1 < len(body) else None
    if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) and isinstance(nxt.value.value, str):
        return inspect.cleandoc(nxt.value.value)
    return None


def _public_entries(tree: ast.Module, attributes: bool) -> list[tuple[str, str | None]]:
    """(qualified name, docstring) of the public functions, classes and members."""
    out: list[tuple[str, str | None]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            out.append((node.name, ast.get_docstring(node)))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            out.append((node.name, ast.get_docstring(node)))
            for i, member in enumerate(node.body):
                if isinstance(member, ast.FunctionDef) and not member.name.startswith("_"):
                    out.append((f"{node.name}.{member.name}", ast.get_docstring(member)))
                elif attributes and isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                    out.append((f"{node.name}.{member.target.id}", _docstring_after(node.body, i)))
    return out


STUB_ENTRIES = _public_entries(ast.parse(STUB.read_text(encoding="utf-8")), attributes=True)
STUB_DOCS = dict(STUB_ENTRIES)


def _runtime_doc(qualname: str) -> str:
    """Docstring of a name of the compiled module as ``help()`` shows it."""
    owner, _, member = qualname.partition(".")
    obj = getattr(_core, owner)
    if member:
        # The descriptor itself: properties and fields keep their own __doc__.
        obj = inspect.getattr_static(obj, member)
    return inspect.cleandoc(obj.__doc__ or "")


def _runtime_names() -> set[str]:
    """Public names of the compiled module, members of its classes included."""
    names = set()
    for name, obj in vars(_core).items():
        if name.startswith("_"):
            continue
        names.add(name)
        if inspect.isclass(obj) and obj.__module__ == "pbsolve":
            names |= {f"{name}.{m}" for m in vars(obj) if not m.startswith("_")}
    return names


@pytest.mark.parametrize("name", [n for n, _ in STUB_ENTRIES])
def test_every_public_name_of_the_stub_is_documented(name):
    assert STUB_DOCS[name], f"{name} has no docstring in _core.pyi"


@pytest.mark.parametrize("name", [n for n, _ in STUB_ENTRIES])
def test_help_shows_the_docstring_of_the_stub(name):
    assert _runtime_doc(name) == STUB_DOCS[name], (
        f"the docstring of {name} differs between _core.pyi and crates/pbsolve-py/src/lib.rs"
    )


def test_the_stub_lists_every_public_name_of_the_extension():
    stub = set(STUB_DOCS)
    runtime = _runtime_names()
    assert runtime - stub == set(), "names of the extension missing in _core.pyi"
    assert stub - runtime == set(), "names in _core.pyi the extension does not have"


@pytest.mark.parametrize("module", MODULES)
def test_every_public_name_of_the_python_modules_is_documented(module):
    tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
    assert ast.get_docstring(tree), f"pbsolve/{module}.py has no module docstring"
    missing = [name for name, doc in _public_entries(tree, attributes=False) if not doc]
    assert missing == [], f"undocumented in pbsolve/{module}.py"


def _property_docs(tree: ast.Module) -> list[tuple[str, str]]:
    """(qualified name, docstring) of the documented properties and attributes of public classes."""
    out = []
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and not node.name.startswith("_")):
            continue
        for i, member in enumerate(node.body):
            doc = None
            if isinstance(member, ast.FunctionDef) and any(
                isinstance(d, ast.Name) and d.id == "property" for d in member.decorator_list
            ):
                doc = ast.get_docstring(member)
            elif isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                doc = _docstring_after(node.body, i)
            if doc:
                name = member.name if isinstance(member, ast.FunctionDef) else member.target.id
                out.append((f"{node.name}.{name}", doc))
    return out


PROPERTY_DOCS = [
    (f"{module}.{name}", doc)
    for module in ["_core.pyi"] + [f"{m}.py" for m in MODULES]
    for name, doc in _property_docs(ast.parse((PACKAGE / module).read_text(encoding="utf-8")))
]


@pytest.mark.parametrize("name,doc", PROPERTY_DOCS, ids=[n for n, _ in PROPERTY_DOCS])
def test_property_summaries_are_not_read_as_a_type(name, doc):
    # Napoleon reads the first line "X: Y" of a property or attribute
    # docstring as type X and description Y, which garbles it on the site.
    assert ": " not in doc.split("\n")[0], f"{name}: no ': ' in the first line"


def test_the_checks_see_the_api():
    # Guards against the parsing above silently finding nothing.
    assert {"Solver", "Solver.polar_solvation", "PolarResult.energy", "GridSpec.levels", "read_pqr"} <= set(STUB_DOCS)
    assert "Solver.polar_solvation" in _runtime_names()
