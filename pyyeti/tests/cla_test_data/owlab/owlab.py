# Simulate event and recover responses
from types import SimpleNamespace
import numpy as np
from scipy.io import matlab
import scipy.interpolate as interp

import matplotlib as mpl

mpl.interactive(0)
mpl.use("Agg")
import matplotlib.pyplot as plt

from pyyeti import n2p, op2, stats, ode, cla
from pyyeti.pp import PP

# event name:
event = "OWLab"

if 1:
    # load data recovery data:
    sc = cla.load("../cla_params.pgz")
    cla.PrintCLAInfo(sc["mission"], event)

    # load nastran data:
    nas = op2.rdnas2cam("../nas2cam")

    # form ulvs for some SEs:
    SC = 101
    n2p.addulvs(nas, SC)

    # prepare spacecraft data recovery matrices
    DR = cla.DR_Event()
    DR.add(nas, sc["drdefs"])

    # initialize results (ext, mnc, mxc for all drms)
    results = DR.prepare_results(sc["mission"], event)

    # set rfmodes:
    rfmodes = nas["rfmodes"][0]

    # setup modal mass, damping and stiffness
    m = None  # None means identity
    k = nas["lambda"][0]
    k[: nas["nrb"]] = 0.0
    b = 2 * 0.02 * np.sqrt(k)
    mbk = (m, b, k)

    # form force transform:
    T = n2p.formdrm(nas, 0, [[22, 123]])[0]

    # random part:
    freq = cla.freq3_augment(np.arange(25.0, 45.1, 0.5), nas["lambda"][0])
    #                 freq     x      y      z
    rnd = [
        np.array(
            [
                [1.0, 90.0, 110.0, 110.0],
                [30.0, 90.0, 110.0, 110.0],
                [31.0, 200.0, 400.0, 400.0],
                [40.0, 200.0, 400.0, 400.0],
                [41.0, 90.0, 110.0, 110.0],
                [50.0, 90.0, 110.0, 110.0],
            ]
        ),
        np.array(
            [
                [1.0, 90.0, 110.0, 110.0],
                [20.0, 90.0, 110.0, 110.0],
                [21.0, 200.0, 400.0, 400.0],
                [30.0, 200.0, 400.0, 400.0],
                [31.0, 90.0, 110.0, 110.0],
                [50.0, 90.0, 110.0, 110.0],
            ]
        ),
    ]

    for j, ff in enumerate(rnd):
        caseid = "{} {:2d}".format(event, j + 1)
        print("Running {} case {}".format(event, j + 1))
        F = interp.interp1d(ff[:, 0], ff[:, 1:].T, axis=1, fill_value=0.0)(freq)
        results.solvepsd(nas, caseid, DR, *mbk, F, T, freq)
        results.psd_data_recovery(caseid, DR, len(rnd), j)

    # save results:
    cla.save("results.pgz", results)

    # make some srs plots and tab files:
    # results.rptext()
    # results.rpttab()
    # results.rpttab(excel=event.lower())
    # results.srs_plots()
    results.srs_plots(Q=10, direc="srs_cases", showall=True, plot=plt.semilogy)
    # results.resp_plots()
