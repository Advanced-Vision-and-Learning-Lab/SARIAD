"""Configuration file for the Sphinx documentation builder.

https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]   # the repository root, where the SARIAD package lives
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_docs  # noqa: E402

gen_docs.generate()  # dataset/model tables come from the code

project = "SARIAD"
author = "Texas A&M's Advanced Vision and Learning Lab"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
]

# The SARATR-X code is a git submodule that may not be checked out where the docs are built.
autodoc_mock_imports = ["SARIAD.models.image.SARATRX.SARATRX"]

myst_enable_extensions = ["colon_fence", "linkify", "substitution", "tasklist", "deflist", "fieldlist", "amsmath", "dollarmath"]
myst_heading_anchors = 3

exclude_patterns = ["_build", "_generated/README.md", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]
templates_path: list[str] = []

copybutton_exclude = ".linenos, .gp, .go"
autosectionlabel_prefix_document = True
suppress_warnings = ["autosectionlabel.*"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_theme_options = {"logo": {"text": "SARIAD"}}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "lightning": ("https://lightning.ai/docs/pytorch/stable/", None),
    "anomalib": ("https://anomalib.readthedocs.io/en/latest/", None),
}
