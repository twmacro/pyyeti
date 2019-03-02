# Comparing results

import matplotlib as mpl

mpl.interactive(0)
mpl.use("Agg")

import numpy as np
from pyyeti import cla
from pyyeti.pp import PP

# Load both sets of results and report percent differences:
results = cla.load("results.pgz")
lvc = cla.load("contractor_results_no_srs.pgz")

results["extreme"].rptpct(lvc, names=("LSP", "Contractor"))
