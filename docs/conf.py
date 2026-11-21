"""Sphinx configuration of the pbsolve documentation.

The same build serves Read the Docs (``.readthedocs.yaml``) and GitHub Pages
(``.github/workflows/docs.yml``). Build locally::

    pip install -r docs/requirements.txt
    sphinx-build -W --keep-going -b html docs docs/_build/html

Nothing is executed and pbsolve need not be installed: the API reference is
read from the sources (sphinx-autoapi, which also reads the stubs of the
compiled module, ``_core.pyi``), and the tutorial notebook is shown with the
outputs stored in ``examples/pbsolve_tutorial.ipynb``.
"""

import shutil
import tomllib
from pathlib import Path

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent

with open(ROOT / "Cargo.toml", "rb") as f:
    release = tomllib.load(f)["workspace"]["package"]["version"]

project = "pbsolve"
author = "pbsolve contributors"
copyright = "2026, pbsolve contributors"
version = release

# The tutorial lives next to the examples, where it is run and tested; the
# documentation shows a copy (not tracked, see .gitignore).
shutil.copy2(ROOT / "examples" / "pbsolve_tutorial.ipynb", DOCS / "tutorial.ipynb")

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_wagtail_theme",
]

# docs/*.md (VALIDATION, ARCHITECTURE, ...) are pages as they are, so the
# files people read on GitHub and the site stay one source.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# Markdown (MyST) and notebooks; reStructuredText for the generated API pages.
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3
nb_execution_mode = "off"
# Markdown outputs of the notebook (its tables) need MyST, not CommonMark.
nb_render_markdown_format = "myst"

# Python API from the sources.
autoapi_type = "python"
autoapi_dirs = [str(ROOT / "python" / "pbsolve")]
autoapi_file_patterns = ["*.pyi", "*.py"]
autoapi_root = "api"
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_ignore = ["*/__main__.py", "*/_console.py", "*/test_installation.py"]
# Docstrings are reStructuredText; those with parameters use NumPy sections
# (Parameters, Returns, Raises), the same text help() shows.
napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = "sphinx_wagtail_theme"
html_title = f"pbsolve {release}"
html_static_path = ["_static"]
html_favicon = "_static/pbsolve-mark.svg"
REPO = "https://github.com/Bio-Otto/pbsolve"
html_theme_options = {
    # The logo carries the name (787.5 x 298, paths relative to _static).
    "project_name": "",
    "logo": "pbsolve-logo.svg",
    "logo_alt": "pbsolve",
    "logo_height": 53,
    "logo_width": 140,
    # No "Edit on GitHub": the tutorial copy and the API pages have no source file there.
    "github_url": "",
    "header_links": f"GitHub|{REPO}",
    "footer_links": f"Source|{REPO}, Issues|{REPO}/issues, Changelog|{REPO}/blob/main/CHANGELOG.md",
}
