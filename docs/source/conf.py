# Copyright 2021-2022
#
# This file is part of HiPACE++.
#
# Authors: Axel Huebl, MaxThevenet, Severin Diederichs
# License: BSD-3-Clause-LBNL

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
import os, sys, urllib.request, subprocess
sys.path.insert(0, os.path.abspath('../../src/'))


# -- Project information -----------------------------------------------------

project = 'HiPACE++'
copyright = '2021, Severin Diederichs, Axel Huebl, Remi Lehe, Andrew Myers, Alexander Sinn, Maxence Thevenet, Weiqun Zhang'
author = 'Severin Diederichs, Axel Huebl, Remi Lehe, Andrew Myers, Alexander Sinn, Maxence Thevenet, Weiqun Zhang'
version = u'24.06'
release = u'24.06'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Setup the breathe extension
breathe_projects = {
    "HiPACE++": "../doxyxml/"
}
breathe_default_project = "HiPACE++"

# Tell sphinx what the primary language being documented is.
primary_domain = 'cpp'
# Tell sphinx what the pygments highlight language should be.
highlight_language = 'cpp'

# Download AMReX Doxygen Tagfile to interlink Doxygen docs
url = 'https://amrex-codes.github.io/amrex/docs_xml/doxygen/amrex-doxygen-web.tag.xml'
urllib.request.urlretrieve(url, '../amrex-doxygen-web.tag.xml')

# Build Doxygen
subprocess.call('cd ../; doxygen; mkdir -p source/_static; cp -r doxyhtml source/_static/', shell=True)
