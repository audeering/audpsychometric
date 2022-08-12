import configparser
import os
import subprocess
from datetime import date

import audeer

config = configparser.ConfigParser()
config.read(os.path.join('..', 'setup.cfg'))

# Project -----------------------------------------------------------------
author = config['metadata']['author']
copyright = f'2020-{date.today().year} audEERING GmbH'
project = config['metadata']['name']
version = audeer.git_repo_version()
title = f'{project} Documentation'


# General -----------------------------------------------------------------
master_doc = 'index'
extensions = []
source_suffix = '.rst'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
pygments_style = None
extensions = [
    'jupyter_sphinx',  # executing code blocks
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # support for Google-style docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',  # for "copy to clipboard" buttons
    'sphinxcontrib.bibtex'
]


bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'alpha'


# Ignore package dependencies during building the docs
autodoc_mock_imports = [
    'tqdm',
]

# Reference with :ref:`data-header:Database`
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Do not copy prompot output
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# Disable Gitlab as we need to sign in
linkcheck_ignore = [
    'https://gitlab.audeering.com',
    r'.*evaluationdashboard.com/index.php/2012/09/22/*'
]


# HTML --------------------------------------------------------------------
html_theme = 'sphinx_audeering_theme'
html_theme_options = {
    'display_version': True,
    'logo_only': False,
}
html_title = title


# -- Intersphinx ------------------------------------------------
intersphinx_mapping = {
    'audmetric': ('https://audeering.github.io/audmetric/', None),
    'matplotlib': ('http://matplotlib.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'sklearn': ('http://scikit-learn.org/stable', None),
    'statsmodels': ('http://www.statsmodels.org/stable/', None),
}
