# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = sphinx_doc_test
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile


clean:
	rm -rf $(BUILDDIR)/*
	rm -rf modules/generated/*
	rm -fr modules/nastran/generated/*
	rm -rf tutorials/*.rst
	rm -rf tutorials/temp_*
	rm -rf tutorials/.ipynb_checkpoints
	rm -rf tutorials/*_files

cleanish:
	rm -rf $(BUILDDIR)/*
	rm -rf modules/generated/*
	rm -fr modules/nastran/generated/*

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
