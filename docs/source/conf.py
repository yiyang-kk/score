# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
print(os.path.abspath("../.."))

from scoring import __version__

# -- Project information -----------------------------------------------------

project = "Python Scoring Workflow"
copyright = "2020, Data Science A-Team"
author = "Data Science A-Team"

# The full version, including alpha/beta/rc tags
release = __version__
version = __version__  # version shown on the sidebar, too


## Little hack to add download links to the index file
file_vesion = release.replace(".","_")
download_lines = [
    f'''* Download all released workflows here :download:`workflows_{file_vesion}.zip <_static/workflows_{file_vesion}.zip>`\n''',
    f'''* Download scoring library package here :download:`scoring_{file_vesion}.zip <_static/scoring_{file_vesion}.zip>`\n''',
    f'''* Download demo data here :download:`demo_data_{file_vesion}.zip <_static/demo_data_{file_vesion}.zip>`\n'''
]
with open("index.rst", "a") as f:
    f.writelines(download_lines)


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.napoleon",
              'sphinx.ext.autosummary',  # should be useful for long docstrings
              'recommonmark',
              "sphinx_rtd_theme",  # better than the original theme
              ]

source_parsers = {
   '.md': 'recommonmark.parser.CommonMarkParser',
}

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

#---sphinx-themes-----
html_theme = "sphinx_rtd_theme"

# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_logo = r'HC_RnD_logo.png'
html_favicon = r'favicon-32x32.png'

html_theme_options = {
    'display_version': True,
    'prev_next_buttons_location': 'both',
    'style_nav_header_background': 'rgb(200, 16, 46)',  # DO NOT CHANGE, THIS IS THE OFFICIAL HC RGB :-D
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': False,
    'navigation_depth': 5,
    'titles_only': True
}
html_show_sourcelink = False  # disabling the 'View page source' in top right corner

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", ]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

smartquotes = True  # converts quotes and dashes to typographically correct entities
