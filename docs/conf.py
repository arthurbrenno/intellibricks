# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "IntelliBricks"
copyright = "2025, Arthur Brenno"
author = "Arthur Brenno"
release = "0.4.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions: list[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_rtd_theme",  # If you prefer Read the Docs theme
    "myst_parser",  # Markdown support
]

templates_path = ["_templates"]
pygments_style = "default"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon settings (for docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "msgspec": ("https://jcrist.github.io/msgspec/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"  # 'sphinx_rtd_theme' # 'furo'
html_theme_options = {
    "light_logo": "logo-light.svg",
    "dark_logo": "logo-dark.svg",
    "navigation_depth": 4,
    "sidebar_hide_name": False,
}
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
html_js_files = [
    "custom.js",
]
html_favicon = "_static/favicon.ico"  # Optional favicon

# -- Options for MyST Parser (Markdown Support) --------------------------
myst_enable_extensions = [
    "amsmath",
    "attrs_block",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
