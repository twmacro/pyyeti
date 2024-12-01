import numpy as np
from pyyeti import cla


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
