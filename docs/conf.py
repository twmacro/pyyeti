#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# pyYeti documentation build configuration file, created by
# sphinx-quickstart on Wed Dec 30 13:08:12 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# import pkg_resources
import sys
import os
import shlex
import matplotlib as mpl
import sphinx
from pyyeti import __version__


mpl.interactive(False)
mpl.use("Agg")

# Color cycle by Matthew A. Petroff
#
#  https://github.com/matplotlib/matplotlib/issues/9460
#  https://github.com/mpetroff/accessible-color-cycles
#
# The final results of the present analysis are located in the
# aesthetic-models/top-cycles.json file. The top six-color color cycle
# is:
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
    "color", ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
)
mpl.rcParams["axes.grid"] = True

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.3.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.intersphinx",
    "sphinx_toggleprompt",
    "sphinx_copybutton",
    # bug in anaconda package??:
    #  https://github.com/ContinuumIO/anaconda-issues/issues/1430
    # could do: conda update ipython -c conda-forge
    # 'IPython.sphinxext.ipython_console_highlighting',
]

highlight_language = "python3"

# Autosummary setting:
autosummary_generate = True
# autodoc_default_flags = ['no-inherited-members']
# autodoc_default_flags = ['members', 'undoc-members',
#                          'show-inheritance', 'inherited-members']
numpydoc_show_inherited_class_members = False

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = False

# Include the example source for plots in API docs
plot_include_source = True
# plot_formats = [("png", 90)]
# plot_rcparams = {'figure.figsize':[8, 6]}
plot_formats = [
    ("png", 100),  # pngs for html building
    ("pdf", 100),  # pdfs for latex building
]
plot_html_show_formats = False
plot_html_show_source_link = False

# Defaults to 0 if not provided. Use 25 so toggleprompt and copybutton
# work together
toggleprompt_offset_right = 25

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "pyYeti"
copyright = "2015-2023, Tim Widrick"
author = "Tim Widrick"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# # The short X.Y version.
# version = ""
# # The full version, including alpha/beta/rc tags.
# release = ""
#
# try:
#     release = pkg_resources.get_distribution("pyyeti").version
# except pkg_resources.DistributionNotFound:
#     print(
#         "To build the documentation, the distribution information "
#         "of pyYeti must be available. Either install the package "
#         'or run "python setup.py develop"'
#     )
#     sys.exit(1)
# del pkg_resources
# version = ".".join(release.split(".")[:2])
#
# print(f"{release=}, {version=}")
# sys.exit(1)

release = __version__
version = ".".join(release.split(".")[:2])


class cd:
    def __init__(self, newdir):
        self.olddir = os.getcwd()
        self.newdir = newdir

    def __enter__(self):
        os.chdir(self.newdir)

    def __exit__(self, *args):
        os.chdir(self.olddir)


with cd("tutorials"):
    sys.path.append("tools")
    from nb_to_doc import convert_nb
    import glob

    nbnames = glob.glob("*.ipynb")
    for nbname in nbnames:
        convert_nb(nbname)

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'alabaster'
html_theme = "nature"
# html_theme = 'sphinx_rtd_theme'
# html_theme = 'scipy'
# html_theme = 'classic'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = ['../../../numpy-1.10.1/doc/scipy-sphinx-theme/_theme/']
# html_theme_path = ['/home/loads/twidrick/numpy-1.10.1/doc/scipy-sphinx-theme/']
# html_theme_path = ['/home/loads/twidrick/numpy-1.10.1/doc/']

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

### html_static_path = ["_static"]
###
###
### def setup(app):
###     # app.add_javascript("copybutton.js")
###     app.add_js_file("copybutton.js")


# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
# html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'h', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'r', 'sv', 'tr'
# html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
# html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
# html_search_scorer = 'scorer.js'


# Output file base name for HTML help builder.
htmlhelp_basename = "pyyetidoc"

# -- Options for LaTeX output ---------------------------------------------
USE_PDFLATEX = True

if USE_PDFLATEX:
    latex_elements = {
        # The paper size ('letterpaper' or 'a4paper').
        "papersize": "letterpaper",
        # The font size ('10pt', '11pt' or '12pt').
        # 'pointsize': '10pt',
        # Additional stuff for the LaTeX preamble.
        "preamble": r"""
\usepackage{enumitem}
\setlistdepth{999}
\usepackage[LGR,T1]{fontenc}
\usepackage{textgreek}
""",
        # "fncychap": r"\usepackage[Bjornstrup]{fncychap}",
        "fncychap": r"\usepackage[Bjarne]{fncychap}",
        # Latex figure (float) alignment
        # 'figure_align': 'htbp',
    }

else:
    latex_engine = "xelatex"
    latex_elements = {
        "papersize": "letterpaper",
        "fontpkg": r"""
\setmainfont{DejaVu Serif}
\setsansfont{DejaVu Sans}
\setmonofont{DejaVu Sans Mono}
""",
        "preamble": r"""
\usepackage[titles]{tocloft}
\cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
\setlength{\cftchapnumwidth}{0.75cm}
\setlength{\cftsecindent}{\cftchapnumwidth}
\setlength{\cftsecnumwidth}{1.25cm}
\usepackage{enumitem}
\setlistdepth{999}
""",
        "fncychap": r"\usepackage[Bjornstrup]{fncychap}",
        "printindex": r"\footnotesize\raggedright\printindex",
    }
    # latex_show_urls = "footnote"

latex_use_xindy = False

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "pyyeti-{}.tex".format(release),
        "pyYeti Documentation",
        "Tim Widrick",
        "manual",
    )
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "pyYeti", "pyYeti Documentation", [author], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "pyYeti",
        "pyYeti Documentation",
        author,
        "pyYeti",
        "A structural dynamics toolbox.",
        "Miscellaneous",
    )
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False


# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# The basename for the epub file. It defaults to the project name.
# epub_basename = project

# The HTML theme for the epub output. Since the default themes are not optimized
# for small screen space, using the same theme for HTML and epub output is
# usually not wise. This defaults to 'epub', a theme designed to save visual
# space.
# epub_theme = 'epub'

# The language of the text. It defaults to the language option
# or 'en' if the language is not set.
# epub_language = ''

# The scheme of the identifier. Typical schemes are ISBN or URL.
# epub_scheme = ''

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
# epub_identifier = ''

# A unique identification for the text.
# epub_uid = ''

# A tuple containing the cover image and cover page html template filenames.
# epub_cover = ()

# A sequence of (type, uri, title) tuples for the guide element of content.opf.
# epub_guide = ()

# HTML files that should be inserted before the pages created by sphinx.
# The format is a list of tuples containing the path and title.
# epub_pre_files = []

# HTML files shat should be inserted after the pages created by sphinx.
# The format is a list of tuples containing the path and title.
# epub_post_files = []

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]

# The depth of the table of contents in toc.ncx.
# epub_tocdepth = 3

# Allow duplicate toc entries.
# epub_tocdup = True

# Choose between 'default' and 'includehidden'.
# epub_tocscope = 'default'

# Fix unsupported image types using the Pillow.
# epub_fix_images = False

# Scale large images.
# epub_max_image_width = 0

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# epub_show_urls = 'inline'

# If false, no index is generated.
# epub_use_index = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "h5py": ("https://docs.h5py.org/en/latest/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
}
