import matplotlib as mpl

mpl.interactive(0)
mpl.use("Agg")

import numpy as np
from pyyeti import cla
from pyyeti.pp import PP

event = "Envelope"

# load data in desired order:
results = cla.DR_Results()
results.merge(
    (
        cla.load(fn)
        for fn in [
            "../toes/results.pgz",
            "../owlab/results.pgz",
            "../toeco/results.pgz",
        ]
    ),
    {"OWLab": "O&W Lab"},
)

results.strip_hists()
results.form_extreme(event, doappend=2)

# save overall results:
cla.save("results.pgz", results)

# write extrema reports:
results["extreme"].rpttab(excel=event.lower())
results["extreme"].srs_plots(Q=10, showall=True)

# group results together to facilitate investigation:
Grouped_Results = cla.DR_Results()

# put these in the order you want:
groups = [("Time Domain", ("TOES", "TOECO")), ("Freq Domain", ("O&W Lab",))]

for key, names in groups:
    Grouped_Results[key] = cla.DR_Results()
    for name in names:
        Grouped_Results[key][name] = results[name]

Grouped_Results.form_extreme()

# plot the two groups:
Grouped_Results["extreme"].srs_plots(direc="grouped_srs", Q=10, showall=True)

# plot just time domain srs:
Grouped_Results["Time Domain"]["extreme"].srs_plots(
    direc="timedomain_srs", Q=10, showall=True
)
