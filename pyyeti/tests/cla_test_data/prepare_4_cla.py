import numpy as np
import matplotlib as mpl

mpl.interactive(0)
mpl.use("Agg")
from pyyeti import cla


def scatm(sol, nas, Vars, se):
    return Vars[se]["atm"] @ sol.a


def net_ifltm(sol, nas, Vars, se):
    return Vars[se]["net_ifltm"] @ sol.a


def net_ifatm(sol, nas, Vars, se):
    return Vars[se]["net_ifatm"] @ sol.a


def get_xyr():
    # return the xr, yr, and rr indexes for the "cglf" data recovery
    # ... see :func:`cla.DR_Defs.add`
    xr = np.array([1, 3, 6, 8])  # 'x' row(s)
    yr = xr + 1  # 'y' row(s)
    rr = np.arange(4) + 10  # rss  rows
    return xr, yr, rr


def cglf(sol, nas, Vars, se):
    resp = Vars[se]["cglf"] @ sol.a
    xr, yr, rr = get_xyr()
    resp[rr] = np.sqrt(resp[xr] ** 2 + resp[yr] ** 2)
    return resp


def cglf_psd(sol, nas, Vars, se, freq, forcepsd, drmres, case, i):
    resp = Vars[se]["cglf"] @ sol.a
    cla.PSD_consistent_rss(resp, *get_xyr(), freq, forcepsd, drmres, case, i)


if __name__ == "__main__":
    import os
    import inspect
    import re
    import matplotlib.pyplot as plt
    from pyyeti import op2, n2p, nastran, op4, cb
    from pyyeti.pp import PP

    se = 101
    uset, coords = nastran.bulk2uset("outboard.asm")
    dct = op4.read("outboard.op4")
    maa = dct["mxx"]
    kaa = dct["kxx"]
    atm = dct["mug1"]
    #    ltm = dct['mef1']
    pch = "outboard.pch"

    def getlabels(lbl, id_dof):
        return ["{} {:4d}-{:1d}".format(lbl, g, i) for g, i in id_dof]

    atm_labels = getlabels("Grid", nastran.rddtipch(pch, "tug1"))
    #    ltm_labels = getlabels('CBAR', nastran.rddtipch(pch, 'tef1'))
    iflabels = getlabels("Grid", uset[:, :2].astype(int))

    # setup CLA parameters:
    mission = "Micro Space Station"

    nb = uset.shape[0]
    nq = maa.shape[0] - nb
    bset = np.arange(nb)
    qset = np.arange(nq) + nb
    ref = [600.0, 150.0, 150.0]
    g = 9806.65
    net = cb.mk_net_drms(maa, kaa, bset, uset=uset, ref=ref, g=g)

    # run cbcheck:
    chk = cb.cbcheck(
        "outboard_cbcheck.out", maa, kaa, bset, bref=np.arange(6), uset=uset, uref=ref
    )

    # define some defaults for data recovery:
    defaults = dict(
        se=se, uf_reds=(1, 1, 1.25, 1), srsfrq=np.arange(0.1, 50.1, 0.1), srsQs=(10, 33)
    )

    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = "scatm"
        desc = "Outboard Internal Accelerations"
        units = "mm/sec^2, rad/sec^2"
        labels = atm_labels[:12]
        drms = {"atm": atm[:12]}
        prog = re.compile(" [2]-[1]")
        histpv = [i for i, s in enumerate(atm_labels) if prog.search(s)]
        srspv = histpv
        srsopts = dict(eqsine=1, ic="steady")
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "cglf"
        desc = "S/C CG Load Factors"
        units = "G"
        labels = net.cglf_labels
        drms = {"cglf": net.cglfa}
        histpv = slice(1)
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "net_ifatm"
        desc = "NET S/C Interface Accelerations"
        units = "g, rad/sec^2"
        labels = net.ifatm_labels[:3]
        drms = {"net_ifatm": net.ifatm[:3]}
        srsopts = dict(eqsine=1, ic="steady")
        histpv = "all"
        srspv = np.array([True, False, True])
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "net_ifltm"
        desc = "NET I/F Loads"
        units = "mN, mN-mm"
        labels = net.ifltm_labels[:6]
        drms = {"net_ifltm": net.ifltma[:6]}
        drdefs.add(**locals())

    # add a 0rb version of the NET ifatm:
    drdefs.add_0rb("net_ifatm")

    # make excel summary file for checking:
    drdefs.excel_summary()

    # save data to gzipped pickle file:
    sc = dict(mission=mission, drdefs=drdefs)
    cla.save("cla_params.pgz", sc)
