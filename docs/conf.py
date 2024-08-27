import toml

import audeer


config = toml.load(audeer.path("..", "pyproject.toml"))

# Project -----------------------------------------------------------------
project = config["project"]["name"]
author = ", ".join(author["name"] for author in config["project"]["authors"])
copyright = "audEERING GmbH"
version = audeer.git_repo_version()
title = "Documentation"


# General -----------------------------------------------------------------
master_doc = "index"
extensions = []
source_suffix = ".rst"
exclude_patterns = [
    "api-src",
    "build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]
pygments_style = None
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # support for Google-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",  # for "copy to clipboard" buttons
    "sphinxcontrib.bibtex",
    "sphinx_apipages",
]


bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "alpha"


# Ignore package dependencies during building the docs
autodoc_mock_imports = [
    "tqdm",
]
autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
}

# Reference with :ref:`data-header:Database`
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Do not copy prompot output
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Disable Gitlab as we need to sign in
linkcheck_ignore = [
    "https://gitlab.audeering.com",
    r".*evaluationdashboard.com/index.php/2012/09/22/*",
    "https://pypi.org/project/audpsychometric/",  # no package published yet
]


# HTML --------------------------------------------------------------------
html_theme = "sphinx_audeering_theme"
html_theme_options = {
    "display_version": True,
    "logo_only": False,
}
html_title = title


# -- Intersphinx ------------------------------------------------
intersphinx_mapping = {
    "audmetric": ("https://audeering.github.io/audmetric/", None),
    "matplotlib": ("http://matplotlib.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "sklearn": ("http://scikit-learn.org/stable", None),
    "python": ("https://docs.python.org/3/", None),
}
