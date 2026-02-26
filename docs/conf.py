# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like this:
# 
#import os
#import sys
#sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------
project = 'gnc-toolkit'
author = 'Batuhan Akkova'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]  

# The theme to use for HTML and HTML Help pages (see Sphinx docs for all options).
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML pages.
theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

