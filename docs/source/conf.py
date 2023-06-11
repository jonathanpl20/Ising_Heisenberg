# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))




project = 'Modelo de Ising y Heisenberg'
copyright = '2023, Luis Miguel Galvis y Jonathan Posada'
author = 'Luis Miguel Galvis y Jonathan Posada'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
]

autodoc_typehints = "signature"
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
    "Boolean": "bool",
    "BooleanOrArrayLike": "BooleanOrArrayLike",
    "BooleanOrNDArray": "BooleanOrNDArray",
    "DType": "DType",
    "DTypeBoolean": "DTypeBoolean",
    "DTypeComplex": "DTypeComplex",
    "DTypeFloating": "DTypeFloating",
    "DTypeInteger": "DTypeInteger",
    "DTypeNumber": "DTypeNumber",
    "Floating": "float",
    "FloatingOrArrayLike": "FloatingOrArrayLike",
    "FloatingOrNDArray": "FloatingOrNDArray",
    "Integer": "int",
    "IntegerOrArrayLike": "IntegerOrArrayLike",
    "IntegerOrNDArray": "IntegerOrNDArray",
    "NestedSequence": "NestedSequence",
    "Number": "Number",
    "NumberOrArrayLike": "NumberOrArrayLike",
    "NumberOrNDArray": "NumberOrNDArray",
    "StrOrArrayLike": "StrOrArrayLike",
    "StrOrNDArray": "StrOrNDArray",
}

autosummary_generate = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']


# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'



language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


#html_theme = 'sphinx_rtd_theme'
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_logo = '../../docs/build/html/_static/logo.jpg'

html_theme_options = {
    #"logo": {
    #    "text": "ising & Heisenberg",
    #},

    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.sfu.ca/dodgelab/thztools",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fab fa-github-square",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ]

}

