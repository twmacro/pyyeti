import os
import itertools
import shutil
import inspect
import re
import warnings
import copy
from types import SimpleNamespace
from glob import glob
from io import StringIO
import numpy as np
from scipy.io import matlab
import scipy.interpolate as interp
import scipy.linalg as la
import matplotlib.pyplot as plt
from pyyeti import cla, cb, ode, stats, locate, ytools
from pyyeti import nastran, srs
from pyyeti.nastran import op2, n2p, op4
import pytest


def test_magpct():
    with pytest.raises(ValueError):
        cla.magpct([1, 2], [1, 2, 3])
    pds = cla.magpct([1, 2], [1, 3], filterval=4)
    assert len(pds) == 1
    assert np.allclose(pds[0], [0, -33.3333333])

    pds = cla.magpct([1, 2], [1, 3], filterval=4, symlogy=False)
    assert len(pds) == 1
    assert len(pds[0]) == 2

    pds = cla.magpct([1, 2], [1, 3], filterval=4, plot_all=False)
    assert len(pds) == 1
    assert len(pds[0]) == 0

    pds = cla.magpct([1, 2], [0, 0], filterval=4)
    assert len(pds) == 1
    assert len(pds[0]) == 0

    pds = cla.magpct([1, 2], [0, 3], filterval=[2, 2])
    assert np.allclose(pds[0], [-33.3333333])

    pds = cla.magpct([[1, 1], [2, 2]], [[0, 0], [3, 3]], filterval=[2, 2])
    assert len(pds) == 2
    assert np.allclose(pds[0], [-33.3333333])
    assert np.allclose(pds[1], [-33.3333333])


def ATM():
    pass


def LTM():
    pass


# return these labels as ndarrays for testing:
def _get_labels0(rows, name):
    return np.array(["{} Row  {:6d}".format(name, i + 1) for i in range(rows[name])])


def _get_labels1(rows, name):
    return np.array(
        ["{} Row  {:6d}".format(name, i + 1) for i in range(0, 2 * rows[name], 2)]
    )


def _get_labels2(rows, name):
    # word = itertools.cycle(['Item', 'Row', 'Id'])
    word = itertools.cycle(["Item"])
    return np.array(
        [
            "{} {:4} {:6d}".format(name, w, i + 1)
            for w, i in zip(word, range(rows[name]))
        ]
    )


def _get_minmax(drm, eventnumber, cyclenumber):
    ext = {
        "LTM": np.array(
            [
                [2.72481567, -2.89079134],
                [2.25786, -2.88626652],
                [3.02440516, -2.80780524],
                [2.53286749, -3.40485914],
                [2.28348523, -3.53863051],
                [3.82729032, -2.61684849],
                [3.35335482, -2.60736874],
                [2.86110496, -2.56407221],
                [2.14606204, -2.55517801],
                [2.54651205, -3.01547524],
                [2.31767096, -2.47119804],
                [2.18782636, -2.50638871],
                [2.64771791, -2.90906464],
                [3.87022179, -2.8447158],
                [3.13803533, -2.96040968],
                [2.19274763, -2.1466145],
                [2.35224123, -2.2461871],
                [2.37220776, -2.37927315],
                [2.70107313, -2.55167378],
                [2.43641342, -2.53973724],
                [3.19988018, -2.27876702],
                [3.26828777, -2.99453974],
                [2.63198951, -2.54630802],
                [2.90049869, -2.70155806],
                [2.06576135, -3.01145668],
                [2.50973189, -2.57272325],
                [2.5291785, -2.87873901],
                [2.5534714, -2.40617426],
                [2.75582, -1.96866783],
            ]
        ),
        "ATM": np.array(
            [
                [4.15547381, -2.60250299],
                [3.30988464, -2.95335224],
                [2.52136841, -2.15885709],
                [2.71879804, -2.4792219],
                [2.40233936, -3.12799065],
                [3.28859809, -2.962606],
                [2.11816761, -2.4080584],
                [3.15167173, -3.01657837],
                [2.41730971, -2.533919],
                [3.29167757, -2.13105438],
                [2.27611906, -3.46433397],
                [2.4100566, -3.3943848],
                [2.63918211, -2.68209126],
                [2.55784324, -2.29710417],
                [3.05160678, -2.46384131],
                [2.61573592, -2.30890182],
                [2.70690245, -2.69287401],
                [1.99385389, -2.36857087],
                [2.27205095, -2.89722068],
                [2.65968896, -3.38645715],
                [2.54024118, -2.35912789],
                [2.62673628, -3.07818987],
                [2.49945891, -2.56637166],
                [2.95143805, -2.34052105],
                [3.35468889, -2.43842187],
                [2.23664468, -2.7788623],
                [3.02078059, -2.84829591],
                [2.69653637, -2.16359541],
                [3.18788459, -2.56054783],
                [3.03810484, -2.23800354],
                [2.60597387, -2.57964111],
                [2.6155941, -2.50413382],
                [2.70912049, -2.87191784],
                [2.58207062, -2.9524317],
            ]
        ),
    }
    addon = 0.2
    curext = ext[drm].copy()
    if eventnumber == 1:
        curext[::3] = curext[::3] - addon
        curext[1::3] = curext[1::3] + 2 * addon
        curext[2::3] = curext[2::3] - addon
    elif eventnumber == 2:
        curext[::3] = curext[::3] - 2 * addon
        curext[1::3] = curext[1::3] + addon
        curext[2::3] = curext[2::3] + addon
    addon = 0.03 * cyclenumber
    curext[::4] = curext[::4] - addon
    curext[1::4] = curext[1::4] + 2 * addon
    curext[2::4] = curext[2::4] - 2 * addon
    curext[3::4] = curext[3::4] + addon
    return curext


def get_fake_cla_results(ext_name, _get_labels, cyclenumber):
    # make up some CLA results:
    events = ("Liftoff", "Transonics", "MECO")
    rows = {"ATM": 34, "LTM": 29}
    ext_results = {i: {} for i in rows}
    for i, event in enumerate(events):
        for drm, nrows in rows.items():
            ext_results[drm][event] = _get_minmax(drm, i, cyclenumber)

    # setup CLA parameters:
    mission = "Rocket / Spacecraft CLA"
    duf = 1.2
    suf = 1.0

    # defaults for data recovery
    defaults = dict(se=0, uf_reds=(1, 1, duf, suf), drfile=".")

    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        desc = "S/C Internal Accelerations"
        units = "m/sec^2, rad/sec^2"
        labels = _get_labels(rows, name)
        drfunc = "no-func"
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "LTM"
        desc = "S/C Internal Loads"
        units = "N, N-m"
        labels = _get_labels(rows, name)
        drfunc = "no-func"
        drdefs.add(**locals())

    # for checking, make a pandas DataFrame to summarize data
    # recovery definitions (but skip the excel file for this
    # demo)
    df = drdefs.excel_summary(None)

    # prepare results data structure:
    DR = cla.DR_Event()
    DR.add(None, drdefs)
    results = cla.DR_Results()
    for event in events:
        results[event] = DR.prepare_results(mission, event)
        for drm in rows:
            results[event].add_maxmin(drm, ext_results[drm][event], event)

    # Done with setup; now we can use the standard cla tools:
    results.form_extreme(ext_name)

    # Test some features of add_maxmin:
    res2 = cla.DR_Results()
    for event in events:
        res2[event] = DR.prepare_results(mission, event)

    ext = ext_results["ATM"]["Liftoff"]
    r = ext.shape[0]
    maxcase = ["LO {}".format(i + 1) for i in range(r)]
    mincase = "LO Min"
    res2["Liftoff"].add_maxmin("ATM", ext, maxcase, mincase, ext, "Time")
    assert res2["Liftoff"]["ATM"].maxcase == maxcase
    assert res2["Liftoff"]["ATM"].mincase == r * [mincase]

    res2["Liftoff"].add_maxmin("LTM", ext, maxcase, r * [mincase], ext, "Time")
    assert res2["Liftoff"]["LTM"].maxcase == maxcase
    assert res2["Liftoff"]["LTM"].mincase == r * [mincase]

    res2 = cla.DR_Results()
    for event in events:
        res2[event] = DR.prepare_results(mission, event)
        for drm in rows:
            res2[event].add_maxmin(
                drm, ext_results[drm][event], event, domain=event + drm
            )
    res2.form_extreme(ext_name)

    assert results["extreme"]["ATM"].domain is None
    assert res2["extreme"]["ATM"].domain == "X-Value"

    res2 = cla.DR_Results()
    for event in events:
        res2[event] = DR.prepare_results(mission, event)
        for drm in rows:
            res2[event].add_maxmin(drm, ext_results[drm][event], event, domain="Time")
    res2.form_extreme(ext_name)
    assert res2["extreme"]["ATM"].domain == "Time"

    return results


def compresults(f1, f2):
    # - some values in "f2" were spot checked, not all
    # - this routine simply compares lines, skipping the date line
    with open(f1) as F1, open(f2) as F2:
        for l1, l2 in zip(F1, F2):
            if not l1.startswith("Date:"):
                assert l1 == l2


def test_form_extreme():
    results = cla.DR_Results()
    # use not-exactly-matching row labels for testing
    results.merge(
        (
            get_fake_cla_results("FLAC", _get_labels0, 0),
            get_fake_cla_results("VLC", _get_labels1, 1),
            get_fake_cla_results("PostVLC", _get_labels2, 2),
        ),
        {"FLAC": "FDLC", "PostVLC": "VLC2"},
    )

    # Add non-unique label to mess up the expanding of results
    # for form_extreme:
    events = ("Liftoff", "Transonics", "MECO")
    lbl_keep = results["FDLC"]["Liftoff"]["ATM"].drminfo.labels[1]
    for e in events:
        lbls = results["FDLC"][e]["ATM"].drminfo.labels
        lbls[1] = lbls[0]
    with pytest.raises(ValueError):
        results.form_extreme()

    # change it back to being correct:
    for e in events:
        lbls = results["FDLC"][e]["ATM"].drminfo.labels
        lbls[1] = lbl_keep

    results.form_extreme()
    assert repr(results).startswith("DR_Results ")
    assert str(results).endswith("with 4 keys: ['FDLC', 'VLC', 'VLC2', 'extreme']")
    try:
        # results['extreme'].rpttab(direc='./temp_tab', excel='results')
        results["extreme"].rpttab(direc="./temp_tab", excel=True)
        assert os.path.exists("./temp_tab/ATM.xlsx")
        assert os.path.exists("./temp_tab/LTM.xlsx")

        results["extreme"].rpttab(direc="./temp_tab")
        results["FDLC"]["extreme"].rpttab(direc="./temp_fdlc_tab")
        # check results:
        compresults("./temp_tab/ATM.tab", "pyyeti/tests/cla_test_data/fake_cla/ATM.tab")
        compresults("./temp_tab/LTM.tab", "pyyeti/tests/cla_test_data/fake_cla/LTM.tab")
        compresults(
            "./temp_fdlc_tab/ATM.tab",
            "pyyeti/tests/cla_test_data/fake_cla/ATM_fdlc.tab",
        )
        compresults(
            "./temp_fdlc_tab/LTM.tab",
            "pyyeti/tests/cla_test_data/fake_cla/LTM_fdlc.tab",
        )
    finally:
        shutil.rmtree("./temp_tab", ignore_errors=True)
        shutil.rmtree("./temp_fdlc_tab", ignore_errors=True)

    # test the different "doappend" options:
    maxcase = results["extreme"]["ATM"].maxcase[:]
    mincase = results["extreme"]["ATM"].mincase[:]

    # doappend = 1 (keep lowest with higher):
    results.form_extreme(doappend=1)

    def _newcase(s):
        i, j = s.split(",")
        return "{},{},{}".format(i, j, j)

    mxc = [_newcase(s) for s in maxcase]
    mnc = [_newcase(s) for s in mincase]
    assert mxc == results["extreme"]["ATM"].maxcase
    assert mnc == results["extreme"]["ATM"].mincase

    # doappend = 3 (keep only lowest):
    results.form_extreme(doappend=3)
    mxc = [s.split(",")[1] for s in maxcase]
    mnc = [s.split(",")[1] for s in mincase]
    assert mxc == results["extreme"]["ATM"].maxcase
    assert mnc == results["extreme"]["ATM"].mincase

    cases = ["VLC2", "FDLC", "VLC"]
    results.form_extreme(case_order=cases)
    assert results["extreme"]["ATM"].cases == cases


def test_rptpct1_align_by_label():
    results = cla.DR_Results()
    # use not-exactly-matching row labels for testing
    results.merge(
        (
            get_fake_cla_results("FLAC", _get_labels0, 0),
            get_fake_cla_results("VLC", _get_labels1, 1),
        ),
        {"FLAC": "FDLC"},
    )

    # results["VLC"]["Liftoff"]["ATM"].drminfo.labels:
    # ['ATM Row       1',
    #  'ATM Row       3',
    #  'ATM Row       5',
    # ...
    #  'ATM Row      65',
    #  'ATM Row      67']  # 34 elements

    # results["FDLC"]["Liftoff"]["ATM"].drminfo.labels
    # ['ATM Row       1',
    #  'ATM Row       2',
    #  'ATM Row       3',
    # ...
    #  'ATM Row      33',
    #  'ATM Row      34']  # 34 elements

    with StringIO() as f:
        mx_pct = cla.rptpct1(
            results["VLC"]["Liftoff"]["ATM"], results["FDLC"]["Liftoff"]["ATM"], f
        )["mx"]["pct"]
        assert len(mx_pct) == 17

        mx_pct = cla.rptpct1(
            results["VLC"]["Liftoff"]["ATM"],
            results["FDLC"]["Liftoff"]["ATM"],
            f,
            align_by_label=False,
        )["mx"]["pct"]
        assert len(mx_pct) == 34

        # test ignorepv with label align:
        mx_pct = cla.rptpct1(
            results["VLC"]["Liftoff"]["ATM"],
            results["FDLC"]["Liftoff"]["ATM"],
            f,
            ignorepv=np.arange(10),
            # 7 rows get compared: 21, 23, ... 33
        )["mx"]["pct"]
        assert len(mx_pct) == 7

        mx_pct = cla.rptpct1(
            results["VLC"]["Liftoff"]["ATM"],
            results["FDLC"]["Liftoff"]["ATM"],
            f,
            ignorepv=np.arange(10, 34),
            # 10 rows get compared: 1, 3, ... 19
        )["mx"]["pct"]
        assert len(mx_pct) == 10

        # test filterval with label align:
        filterval = np.empty(34)
        filterval[:] = 2.7
        mx_mag = cla.rptpct1(
            results["VLC"]["Liftoff"]["ATM"],
            results["FDLC"]["Liftoff"]["ATM"],
            f,
            filterval=filterval,
            ignorepv=np.arange(10),
            # 7 rows get compared: 21, 23, ... 33
        )["mx"]["mag"][0]
        assert len(mx_mag) == 7


# run a cla:


class cd:
    def __init__(self, newdir):
        self.olddir = os.getcwd()
        self.newdir = newdir

    def __enter__(self):
        os.chdir(self.newdir)

    def __exit__(self, *args):
        os.chdir(self.olddir)


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


def prepare_4_cla(pth):
    se = 101
    uset, coords = nastran.bulk2uset(pth + "outboard.asm")
    dct = op4.read(pth + "outboard.op4")
    maa = dct["mxx"]
    kaa = dct["kxx"]
    atm = dct["mug1"]
    pch = pth + "outboard.pch"

    def getlabels(lbl, id_dof):
        return ["{} {:4d}-{:1d}".format(lbl, g, i) for g, i in id_dof]

    atm_labels = getlabels("Grid", nastran.rddtipch(pch, "tug1"))

    # setup CLA parameters:
    mission = "Micro Space Station"

    nb = uset.shape[0]
    bset = np.arange(nb)
    ref = [600.0, 150.0, 150.0]
    g = 9806.65
    net = cb.mk_net_drms(maa, kaa, bset, uset=uset, ref=ref, g=g)

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
        drms = {name: atm[:12]}
        drfunc = f"Vars[se]['{name}'] @ sol.a"
        prog = re.compile(" [2]-[1]")
        histpv = [i for i, s in enumerate(atm_labels) if prog.search(s)]
        srspv = np.arange(4)
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
        srspv = 0
        srsQs = 10
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "net_ifatm"
        desc = "NET S/C Interface Accelerations"
        units = "g, rad/sec^2"
        labels = net.ifatm_labels[:3]
        drms = {name: net.ifatm[:3]}
        drfunc = f"Vars[se]['{name}'] @ sol.a"
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
        drms = {name: net.ifltma[:6]}
        drfunc = f"Vars[se]['{name}'] @ sol.a"
        drdefs.add(**locals())

    # add a 0rb version of the NET ifatm:
    drdefs.add_0rb("net_ifatm")

    # make excel summary file for checking:
    df = drdefs.excel_summary()

    # save data to gzipped pickle file:
    sc = dict(mission=mission, drdefs=drdefs)
    cla.save("cla_params.pgz", sc)


def toes(pth):
    os.mkdir("toes")
    with cd("toes"):
        pth = "../" + pth
        event = "TOES"
        # load data recovery data:
        sc = cla.load("../cla_params.pgz")
        cla.PrintCLAInfo(sc["mission"], event)

        # load nastran data:
        nas = op2.rdnas2cam(pth + "nas2cam")

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

        # load in forcing functions:
        mat = pth + "toes/toes_ffns.mat"
        toes = matlab.loadmat(mat, squeeze_me=True, struct_as_record=False)
        toes["ffns"] = toes["ffns"][:3, ::2]
        toes["sr"] = toes["sr"] / 2
        toes["t"] = toes["t"][::2]

        # form force transform:
        T = n2p.formdrm(nas, 0, [[8, 12], [24, 13]])[0].T

        # do pre-calcs and loop over all cases:
        ts = ode.SolveUnc(*mbk, 1 / toes["sr"], rf=rfmodes)
        LC = toes["ffns"].shape[0]
        t = toes["t"]
        for j, force in enumerate(toes["ffns"]):
            print("Running {} case {}".format(event, j + 1))
            genforce = T @ ([[1], [0.1], [1], [0.1]] * force[None, :])
            # solve equations of motion
            sol = ts.tsolve(genforce, static_ic=1)
            sol.t = t
            sol = DR.apply_uf(sol, *mbk, nas["nrb"], rfmodes)
            caseid = "{} {:2d}".format(event, j + 1)
            results.time_data_recovery(sol, nas["nrb"], caseid, DR, LC, j, verbose=3)

        results.calc_stat_ext(stats.ksingle(0.99, 0.90, LC))

        # save results:
        cla.save("results.pgz", results)

        # make some srs plots and tab files:
        results.rptext()
        results.rpttab()
        results.rpttab(excel="toes")
        results.srs_plots()
        results.srs_plots(fmt="png")
        results.resp_plots(legend_args={"ncol": 2}, layout=(2, 1))
        assert os.path.exists("resp_plots/TOES_hist.pdf")


def owlab(pth):
    os.mkdir("owlab")
    with cd("owlab"):
        pth = "../" + pth
        # event name:
        event = "OWLab"

        # load data recovery data:
        sc = cla.load("../cla_params.pgz")
        cla.PrintCLAInfo(sc["mission"], event)

        # load nastran data:
        nas = op2.rdnas2cam(pth + "nas2cam")

        # form ulvs for some SEs:
        SC = 101
        n2p.addulvs(nas, SC)

        # prepare spacecraft data recovery matrices
        DR = cla.DR_Event()
        DR.add(nas, sc["drdefs"])

        # initialize results (ext, mnc, mxc for all drms)
        results = DR.prepare_results(sc["mission"], event)

        # set rfmodes:
        # rfmodes = nas["rfmodes"][0]

        # setup modal mass, damping and stiffness
        m = None  # None means identity
        k = nas["lambda"][0]
        k[: nas["nrb"]] = 0.0
        b = 2 * 0.02 * np.sqrt(k)
        mbk = (m, b, k)

        # form force transform:
        T = n2p.formdrm(nas, 0, [[22, 123]])[0].T

        # random part:
        freq = cla.freq3_augment(np.arange(25.0, 45.1, 0.5), nas["lambda"][0])
        #          freq     x      y      z
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

        fs = ode.SolveUnc(*mbk)
        for j, ff in enumerate(rnd):
            caseid = "{} {:2d}".format(event, j + 1)
            print("Running {} case {}".format(event, j + 1))
            F = interp.interp1d(ff[:, 0], ff[:, 1:].T, axis=1, fill_value=0.0)(freq)
            results.solvepsd(nas, caseid, DR, fs, F, T, freq)
            with pytest.warns(RuntimeWarning, match=r"Integ.*freq.*inacc.*result"):
                results.psd_data_recovery(caseid, DR, len(rnd), j, verbose=3)

        # save results:
        cla.save("results.pgz", results)
        results.srs_plots(Q=10, direc="srs_cases", showall=True, plot="semilogy")
        results.resp_plots(direc="srs_cases")
        assert os.path.exists("srs_cases/OWLab_srs.pdf")
        assert os.path.exists("srs_cases/OWLab_psd.pdf")
        with pytest.raises(TypeError):
            results.srs_plots(
                Q=10,
                direc="srs_cases",
                showall=True,
                plot=plt.semilogy,  # has to be a string in v1.0.8
            )

        # for testing:
        results2 = DR.prepare_results(sc["mission"], event)
        verbose = True  # for testing
        freq2 = freq.copy()
        for j, ff in enumerate(rnd):
            caseid = "{} {:2d}".format(event, j + 1)
            print("Running {} case {}".format(event, j + 1))
            F = interp.interp1d(ff[:, 0], ff[:, 1:].T, axis=1, fill_value=0.0)(freq2)
            if j == 0:
                results2.solvepsd(
                    nas, caseid, DR, fs, F, T, freq2, verbose=verbose, incrb="av"
                )
                verbose = not verbose
                freq2 = +freq2  # make copy
                freq2[-1] = 49.7  # to cause error on next 'solvepsd'
                with pytest.warns(RuntimeWarning, match=r"Integ.*freq.*inacc.*result"):
                    results2.psd_data_recovery(caseid, DR, len(rnd), j, resp_time=20)
            else:
                with pytest.raises(ValueError):
                    results2.solvepsd(
                        nas,
                        caseid,
                        DR,
                        fs,
                        F,
                        T,
                        freq2,
                        verbose=verbose,
                    )

        # test for incompatibly sized ValueError:
        T = T[:, :2]  # chop off last column
        results3 = DR.prepare_results(sc["mission"], event)
        for j, ff in enumerate(rnd):
            caseid = "{} {:2d}".format(event, j + 1)
            print("Running {} case {}".format(event, j + 1))
            F = interp.interp1d(ff[:, 0], ff[:, 1:].T, axis=1, fill_value=0.0)(freq)
            with pytest.raises(ValueError):
                results3.solvepsd(nas, caseid, DR, fs, F, T, freq)
            break

        # compare srs results using 3.0 peak factor and frequency
        # dependent peak factor:
        frq = results["scatm"].srs.frq
        pf = np.sqrt(2 * np.log(frq * 20))
        for Q in results["scatm"].srs.srs:
            srs1 = results["scatm"].srs.srs[Q][0]  # (cases, dof, frq)
            srs2 = results2["scatm"].srs.srs[Q][0]
            assert np.allclose(srs1 * pf, srs2 * 3)


def get_drdefs(nas, sc):
    drdefs = cla.DR_Def(sc["drdefs"].defaults)

    @cla.DR_Def.addcat
    def _():
        se = 0
        name = "alphajoint"
        desc = "Alpha-Joint Acceleration"
        units = "mm/sec^2, rad/sec^2"
        labels = ["Alpha-Joint {:2s}".format(i) for i in "X,Y,Z,RX,RY,RZ".split(",")]
        drms = {name: n2p.formdrm(nas, 0, 33)[0]}
        drfunc = f"Vars[se]['{name}'] @ sol.a"
        srsopts = dict(eqsine=1, ic="steady")
        histpv = 1  # second row
        srspv = [1]
        drdefs.add(**locals())

    return drdefs


def toeco(pth):
    os.mkdir("toeco")
    with cd("toeco"):
        pth = "../" + pth
        # event name:
        event = "TOECO"

        # load data recovery data:
        sc = cla.load("../cla_params.pgz")
        cla.PrintCLAInfo(sc["mission"], event)

        # load nastran data:
        nas = op2.rdnas2cam(pth + "nas2cam")

        # form ulvs for some SEs:
        SC = 101
        n2p.addulvs(nas, SC)
        drdefs = get_drdefs(nas, sc)

        # prepare spacecraft data recovery matrices
        DR = cla.DR_Event()
        DR.add(nas, sc["drdefs"])
        DR.add(nas, drdefs)

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

        # load in forcing functions:
        mat = pth + "toeco/toeco_ffns.mat"
        toeco = matlab.loadmat(mat, squeeze_me=True, struct_as_record=False)
        toeco["ffns"] = toeco["ffns"][:2, ::2]
        toeco["sr"] = toeco["sr"] / 2
        toeco["t"] = toeco["t"][::2]

        # form force transform:
        T = n2p.formdrm(nas, 0, [[8, 12], [24, 13]])[0].T

        # do pre-calcs and loop over all cases:
        ts = ode.SolveUnc(*mbk, 1 / toeco["sr"], rf=rfmodes)
        LC = toeco["ffns"].shape[0]
        t = toeco["t"]
        for j, force in enumerate(toeco["ffns"]):
            print("Running {} case {}".format(event, j + 1))
            genforce = T @ ([[1], [0.1], [1], [0.1]] * force[None, :])
            # solve equations of motion
            sol = ts.tsolve(genforce, static_ic=1)
            sol.t = t
            sol = DR.apply_uf(sol, *mbk, nas["nrb"], rfmodes)
            caseid = "{} {:2d}".format(event, j + 1)
            results.time_data_recovery(sol, nas["nrb"], caseid, DR, LC, j)

        # save results:
        cla.save("results.pgz", results)


def summarize(pth):
    os.mkdir("summary")
    with cd("summary"):
        pth = "../" + pth
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
        results["extreme"].srs_plots(Q=10, showall=True, fmt="png")

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
        Grouped_Results["extreme"].srs_plots(
            direc="grouped_srs",
            Q=10,
            showall=True,
            tight_layout_args=dict(
                rect=(0.1, 0.1, 0.9, 0.9), pad=3.0, w_pad=2.0, h_pad=2.0
            ),
        )

        # plot just time domain srs:
        Grouped_Results["Time Domain"]["extreme"].srs_plots(
            direc="timedomain_srs", Q=10, showall=True
        )


def compare(pth):
    with cd("summary"):
        pth = "../" + pth
        # Load both sets of results and report percent differences:
        results = cla.load("results.pgz")
        lvc = cla.load(pth + "summary/contractor_results_no_srs.pgz")

        # to check for warning message, add a category not in lvc:
        results["extreme"]["ifa2"] = results["extreme"]["net_ifatm"]

        plt.close("all")
        regex = re.compile(r"Some comp.*skipped.*ifa2", re.DOTALL)
        with pytest.warns(RuntimeWarning, match=regex):
            results["extreme"].rptpct(lvc, names=("LSP", "Contractor"))

        plt.close("all")
        # modify lvc['cglf'] for testing:
        lvc["cglf"].ext[0, 0] = 0.57  # cause a -7.3% diff
        lvc["cglf"].ext[5, 0] = 0.449  # cause a 17.7% diff
        with pytest.warns(RuntimeWarning, match=r"Some compar.*skipped"):
            results["extreme"].rptpct(
                lvc, names=("LSP", "Contractor"), direc="absmax_compare", doabsmax=True
            )

        plt.close("all")
        results["extreme"].rptpct(
            lvc,
            names=("LSP", "Contractor"),
            # drms=['alphajoint', 'cglf', 'net_ifatm_0rb',
            #       'net_ifatm', 'net_ifltm', 'scatm'],
            drms=["alphajoint"],
            direc="absmax_compare_2",
            doabsmax=True,
        )

        # test magpct filterval options; when a filtered value %diff
        # is large
        # 1) magpct_options['filterval'] = 'filterval'
        #    magpct_options['symlogy'] = True
        #      - filterval is None
        #      - filterval is 10
        #      - filterval is 1d array
        # 2) magpct_options['filterval'] = 'filterval'
        #    magpct_options['symlogy'] = False
        #      - filterval is None
        #      - filterval is 10
        #      - filterval is 1d array
        # 3) magpct_options['filterval'] = None

        # first, modify scatm results for testing:
        lsp = results["extreme"]
        del lsp["scatm"].drminfo.filterval
        lsp["scatm"].ext[4, :] = [2e-7, -2e-7]
        lvc["scatm"].ext[4, :] = [1e-11, -1e-11]

        lsp["scatm"].ext[5, :] = [1.12, -0.39]  # ~5% exceedances
        lvc["scatm"].ext[5, :] = [1.074, -0.345]

        for ms in (True, False):
            for mf, fv in (
                ("same", None),
                ("same", 10.0),
                ("filterval", 10.0 + np.zeros(12)),
                (None, None),
            ):
                try:
                    _fv = fv[0]
                except (TypeError, IndexError):
                    _fv = fv
                    direc = "scatm_msymlog_{}_mfilterval_{}_fv_{}s".format(ms, mf, _fv)
                else:
                    direc = "scatm_msymlog_{}_mfilterval_{}_fv_{}a".format(ms, mf, _fv)
                magpct_options = {"filterval": mf, "symlogy": ms}
                lsp.rptpct(
                    lvc,
                    names=("LSP", "Contractor"),
                    drms=["scatm"],
                    direc=direc,
                    magpct_options=magpct_options,
                    filterval=fv,
                )
        # test for some exceptions:
        magpct_options = {"filterval": mf, "symlogy": ms}
        with pytest.raises(IndexError):
            lsp.rptpct(
                lvc,
                names=("LSP", "Contractor"),
                drms=["scatm"],
                direc="junk",
                magpct_options=magpct_options,
                filterval=[1, 1, 2, 3],
            )
        with pytest.raises(IndexError):
            lsp.rptpct(
                lvc,
                names=("LSP", "Contractor"),
                drms=["scatm"],
                direc="junk",
                magpct_options=magpct_options,
                filterval=np.ones((3, 4)),
            )
        magpct_options["filterval"] = "bad string"
        with pytest.raises(ValueError):
            lsp.rptpct(
                lvc,
                names=("LSP", "Contractor"),
                drms=["scatm"],
                direc="junk",
                magpct_options=magpct_options,
                filterval=1.0,
            )


def confirm():
    for direc, cnt in (("compare", 3), ("absmax_compare", 1)):
        cmp_files = glob("summary/{}/*.cmp".format(direc))
        assert len(cmp_files) == 6
        png_files = glob("summary/{}/*.png".format(direc))
        assert len(png_files) == 12
        for n in cmp_files:
            with open(n) as f:
                count = 0
                for line in f:
                    if "% Diff Statistics:" in line:
                        count += 1
                        p = line.index(" = [")
                        stats = np.array(
                            [float(i) for i in line[p + 4 : -2].split(",")]
                        )
                        if direc == "absmax_compare" and "cglf" in n:
                            claval = 0.5284092590377919
                            mnval = (claval / 0.57 - 1) * 100
                            mxval = (claval / 0.449 - 1) * 100
                            mean = (mnval + mxval) / 14
                            std = np.sqrt(
                                (
                                    (mnval - mean) ** 2
                                    + (mxval - mean) ** 2
                                    + 12 * (0 - mean) ** 2
                                )
                                / 13
                            )
                            sbe = np.r_[
                                np.round([mnval, mxval], 2), np.round([mean, std], 4)
                            ]
                            assert np.allclose(stats, sbe)
                        else:
                            assert np.all(stats == 0.0)
            assert count == cnt

    cmp_files = glob("summary/absmax_compare_2/*.cmp".format(direc))
    assert len(cmp_files) == 1
    png_files = glob("summary/absmax_compare_2/*.png".format(direc))
    assert len(png_files) == 2
    for n in cmp_files:
        n0 = n.replace("compare_2/", "compare/")
        with open(n) as f, open(n0) as f0:
            for line, line0 in zip(f, f0):
                assert line == line0


def check_split():
    for event, dirname, do_stats in (("TOES", "toes", 1), ("O&W Lab", "owlab", 0)):
        for res in (
            cla.load(f"{dirname}/results.pgz"),
            cla.load("summary/results.pgz")[event],
        ):
            sp = res.split()

            mg = cla.DR_Results()
            mg.merge(sp.values())
            LC = len(mg)
            mg.form_extreme()

            with pytest.raises(TypeError):
                mg.split()

            if do_stats:
                mg["extreme"].calc_stat_ext(stats.ksingle(0.99, 0.90, LC))

            cat = "cglf"
            for cat in mg["extreme"]:
                assert np.all(mg["extreme"][cat].ext == res[cat].ext)
                assert np.all(mg["extreme"][cat].mx == res[cat].mx)
                assert np.all(mg["extreme"][cat].mn == res[cat].mn)
                assert np.all(mg["extreme"][cat].ext_x == res[cat].ext_x)
                assert np.all(mg["extreme"][cat].mx_x == res[cat].mx_x)
                assert np.all(mg["extreme"][cat].mn_x == res[cat].mn_x)
                assert mg["extreme"][cat].maxcase == res[cat].maxcase
                assert mg["extreme"][cat].mincase == res[cat].mincase

                event = res[cat].event
                for j in range(LC):
                    name = f"{event}  {j+1}"
                    if hasattr(sp[name][cat], "hist"):
                        assert np.all(sp[name][cat].hist == res[cat].hist[[j]])
                        assert np.all(sp[name][cat].time == res[cat].time)

                    if hasattr(sp[name][cat], "psd"):
                        assert np.all(sp[name][cat].psd == res[cat].psd[[j]])
                        assert np.all(sp[name][cat].freq == res[cat].freq)

                    if hasattr(sp[name][cat], "srs"):
                        for q in res[cat].srs.srs:
                            assert np.all(
                                sp[name][cat].srs.srs[q] == res[cat].srs.srs[q][[j]]
                            )
                            assert np.all(
                                sp[name][cat].srs.ext[q] == res[cat].srs.srs[q][[j]]
                            )
                        assert np.all(sp[name][cat].srs.frq == res[cat].srs.frq)


def do_srs_plots():
    plt.close("all")
    with cd("summary"):
        results = cla.load("results.pgz")
        with pytest.warns(RuntimeWarning, match="no Q="):
            results["extreme"].srs_plots(
                Q=33,
                showall=True,
                direc="srs2",
                # drms=['net_ifltm', 'cglf'])
                drms=["cglf"],
            )

        with pytest.warns(RuntimeWarning, match="no SRS data"):
            results["extreme"].srs_plots(
                Q=33, showall=True, direc="srs2", drms=["net_ifltm", "cglf"]
            )

        with pytest.raises(ValueError):
            results["extreme"].srs_plots(
                Q=[10, 33],
                showall=True,
                direc="srs2",
            )

        results["extreme"].srs_plots(
            event="EXTREME",
            Q=10,
            showall=True,
            direc="srs2",
            drms=["cglf"],
            showboth=True,
            layout=(1, 1),
            legend_args={"ncol": 2},
        )
        assert os.path.exists("srs2/EXTREME_srs.pdf")

        results["extreme"].srs_plots(
            Q=10, showall=True, direc="srs3", layout=(2, 1), onepdf=False
        )
        files = [
            "alphajoint_eqsine_all.pdf",
            "cglf_srs_all.pdf",
            "net_ifatm_0rb_eqsine_all.pdf",
            "net_ifatm_eqsine_all.pdf",
            "scatm_eqsine_all_0.pdf",
            "scatm_eqsine_all_1.pdf",
        ]
        for f in files:
            assert os.path.exists("srs3/" + f)

        results["extreme"].srs_plots(
            Q=10, showall=True, direc="srs3", layout=(2, 1), onepdf="srs3_onepdf_file"
        )
        assert os.path.exists("srs3/srs3_onepdf_file.pdf")


def do_time_plots():
    plt.close("all")
    with cd("toeco"):
        # Load both sets of results and report percent differences:
        results = cla.load("results.pgz")
        with pytest.raises(ValueError):
            results.resp_plots(
                direc="time2",
                cases=["TOECO  1", "bad case name"],
            )


def test_transfer_orbit_cla():
    try:
        if os.path.exists("temp_cla"):
            shutil.rmtree("./temp_cla", ignore_errors=True)
        os.mkdir("temp_cla")
        pth = "../pyyeti/tests/cla_test_data/"
        with cd("temp_cla"):
            plt.close("all")
            prepare_4_cla(pth)
            toes(pth)
            owlab(pth)
            toeco(pth)
            summarize(pth)
            compare(pth)
            confirm()
            check_split()
            do_srs_plots()
            do_time_plots()
    finally:
        # pass
        shutil.rmtree("./temp_cla", ignore_errors=True)


def test_maxmin():
    with pytest.raises(ValueError):
        cla.maxmin(np.ones((2, 2)), np.ones((5)))


def test_extrema_1():
    mm = SimpleNamespace(ext=np.ones((5, 3)))
    with pytest.raises(ValueError):
        cla.extrema([], mm, "test")

    rows = 5
    curext = SimpleNamespace(
        ext=None,
        ext_x=None,
        maxcase=None,
        mincase=None,
        mx=np.empty((rows, 2)),
        mn=np.empty((rows, 2)),
        mx_x=np.empty((rows, 2)),
        mn_x=np.empty((rows, 2)),
    )
    mm = SimpleNamespace(ext=np.ones((rows, 1)), ext_x=np.zeros((rows, 1)))
    maxcase = "Case 1"
    mincase = None
    casenum = 0
    cla.extrema(curext, mm, maxcase, mincase, casenum)

    mm = SimpleNamespace(ext=np.arange(rows)[:, None], ext_x=np.ones((rows, 1)))
    maxcase = "Case 2"
    mincase = None
    casenum = 1
    cla.extrema(curext, mm, maxcase, mincase, casenum)

    assert np.all(
        curext.ext
        == np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    )

    assert np.all(
        curext.ext_x
        == np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    )
    assert curext.maxcase == ["Case 1", "Case 1", "Case 2", "Case 2", "Case 2"]
    assert curext.mincase == ["Case 2", "Case 1", "Case 1", "Case 1", "Case 1"]

    assert np.all(
        curext.mx
        == np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    )

    assert np.all(
        curext.mn
        == np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    )

    assert np.all(
        curext.mx_x
        == np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    )

    assert np.all(
        curext.mn_x
        == np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    )


def test_extrema_2():
    rows = 5
    curext = SimpleNamespace(
        ext=None,
        ext_x=None,
        maxcase=None,
        mincase=None,
        mx=np.empty((rows, 2)),
        mn=np.empty((rows, 2)),
        mx_x=np.empty((rows, 2)),
        mn_x=np.empty((rows, 2)),
    )
    mm = SimpleNamespace(ext=np.ones((rows, 1)), ext_x=None)
    maxcase = "Case 1"
    mincase = None
    casenum = 0
    cla.extrema(curext, mm, maxcase, mincase, casenum)

    mm = SimpleNamespace(ext=np.arange(rows)[:, None], ext_x=None)
    maxcase = "Case 2"
    mincase = None
    casenum = 1
    cla.extrema(curext, mm, maxcase, mincase, casenum)

    assert np.all(
        curext.ext
        == np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]])
    )

    assert curext.ext_x is None
    assert curext.maxcase == ["Case 1", "Case 1", "Case 2", "Case 2", "Case 2"]
    assert curext.mincase == ["Case 2", "Case 1", "Case 1", "Case 1", "Case 1"]

    assert np.all(
        curext.mx
        == np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    )

    assert np.all(
        curext.mn
        == np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    )
    assert np.isnan(curext.mx_x).sum() == 10
    assert np.isnan(curext.mn_x).sum() == 10


def test_addcat():
    def _():
        name = "ATM"
        labels = 12

    # doesn't call DR_Def.add:
    with pytest.raises(RuntimeError):
        cla.DR_Def.addcat(_)

    defaults = dict(
        # se = 0,
        uf_reds=(1, None, 1, None),
        drfile=".",
        srsQs=10,
    )
    drdefs = cla.DR_Def(defaults)

    drm = np.ones((4, 4))

    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        labels = 12
        drms = {"drm": drm}
        drdefs.add(**locals())

    repr_str = repr(drdefs)
    assert "with 1 categories" in repr_str

    assert drdefs["ATM"].se == 0
    assert np.allclose(drdefs["ATM"].uf_reds, (1, 1, 1, 1))

    def _():
        name = "LTM"
        labels = 12
        drms = {"drm": drm}
        curfile = os.path.realpath(inspect.stack()[0][1])
        drfile = os.path.split(curfile)[1]
        drfunc = "ATM"
        drdefs.add(**locals())

    with pytest.warns(RuntimeWarning, match='"drm" already'):
        cla.DR_Def.addcat(_)

    assert drdefs["LTM"].drfile == drdefs["ATM"].drfile

    def _():
        name = "ATM45"
        labels = 12
        drfile = "no such file"
        drdefs.add(**locals())

    with pytest.warns(RuntimeWarning, match=r"ATM45.*could not open"):
        cla.DR_Def.addcat(_)

    def _():
        name = "DTM"
        labels = 12
        drms = {"drm": 0 + drm}
        drfunc = "ATM"
        drdefs.add(**locals())

    # uses a different "drm":
    with pytest.raises(ValueError):
        cla.DR_Def.addcat(_)

    def _():
        name = "DTM"
        labels = 2
        drdefs.add(**locals())

    # already defined data recovery category:
    with pytest.raises(ValueError):
        cla.DR_Def.addcat(_)

    def _():
        name = "SDTM"
        labels = 12
        drms = {"sdrm": 1}
        desc = defaults
        drfunc = "ATM"
        drdefs.add(**locals())

    # `desc` not in defaults:
    with pytest.raises(ValueError):
        cla.DR_Def.addcat(_)

    def _():
        name = "TDTM"
        labels = 12
        drms = {"tdrm": 1}
        filterval = [0.003, 0.004]
        drfunc = "ATM"
        drdefs.add(**locals())

    # length of `filterval` does not match length of labels:
    with pytest.raises(ValueError):
        cla.DR_Def.addcat(_)

    # this length does match, so no error:
    @cla.DR_Def.addcat
    def _():
        name = "TDTM2"
        labels = 2
        drms = {"tdrm2": 1}
        filterval = [0.003, 0.004]
        drfunc = "ATM"
        drdefs.add(**locals())

    # a good bool `histpv`
    @cla.DR_Def.addcat
    def _():
        name = "ATM2"
        labels = 4
        drms = {"atm2": 1}
        histpv = [True, False, False, True]
        drfunc = "ATM"
        drdefs.add(**locals())

    # a bad bool `histpv`
    def _():
        name = "ATM3"
        labels = 4
        drms = {"atm3": 1}
        histpv = [True, False, False]
        drfunc = "ATM"
        drdefs.add(**locals())

    # length of `histpv` does not match length of labels:
    with pytest.raises(ValueError):
        cla.DR_Def.addcat(_)

    # a good integer `histpv`
    @cla.DR_Def.addcat
    def _():
        name = "ATM4"
        labels = 4
        drms = {"atm4": 1}
        histpv = [0, 3]
        drfunc = "ATM"
        srsfrq = np.arange(1.0, 10.0)
        drdefs.add(**locals())

    # a bad integer `histpv`
    def _():
        name = "ATM5"
        labels = 4
        drms = {"atm5": 1}
        histpv = [1, 4]
        drfunc = "ATM"
        drdefs.add(**locals())

    # `histpv` exceeds dimensions:
    with pytest.raises(ValueError):
        cla.DR_Def.addcat(_)

    # a bad type for `histpv`
    def _():
        name = "ATM6"
        labels = 4
        drms = {"atm6": 1}
        histpv = {}
        # drfunc = 'ATM' ... so that a warning message is triggered
        drdefs.add(**locals())

    # `histpv` is bad type:
    with pytest.raises(TypeError, match="`histpv` input not understood"):
        with pytest.warns(RuntimeWarning, match='function "ATM6" not found'):
            cla.DR_Def.addcat(_)

    # overlapping drms and nondrms names:
    def _():
        name = "ATM7"
        labels = 4
        drms = {"atm7": 1}
        nondrms = {"atm7": 1}
        drfunc = "ATM"
        drdefs.add(**locals())

    # overlapping names in `drms` and `nondrms`
    with pytest.raises(ValueError):
        cla.DR_Def.addcat(_)

    drdefs.copycat("ATM", "_dummy", uf_reds=(0, 1, 1.0, 1))

    assert drdefs["ATM_dummy"].labels == drdefs["ATM"].labels

    # modify category that doesn't exist
    with pytest.raises(ValueError):
        drdefs.copycat("notexist", "_2", uf_reds=(0, 1, 1.0, 1))

    # modify parameter that doesn't exist
    with pytest.raises(ValueError):
        drdefs.copycat("ATM", "_2", notexist=1)

    drdefs.copycat("ATM", ["ATM_2"], uf_reds=(0, 1, 1.0, 1))

    assert drdefs["ATM_2"].labels == drdefs["ATM"].labels

    drdefs.copycat("ATM", ["ATM_3"], uf_reds=(0, None, None, None))

    assert np.allclose(drdefs["ATM_3"].uf_reds, (0, 1, 1, 1))

    # atm_2 already exists:
    with pytest.raises(ValueError):
        drdefs.copycat("ATM", "_2")

    # add a 0rb version of non-existent category:
    with pytest.raises(ValueError):
        drdefs.add_0rb("net_ifatm")


def test_addcat_2():
    defaults = dict(se=0, uf_reds=(1, 1, 1, 1), drfile=".", srsQs=10)
    drdefs = cla.DR_Def(defaults)
    # error because there no categories
    with pytest.raises(RuntimeError):
        drdefs.excel_summary(None)


def test_addcat_3():
    defaults = dict(se=0, uf_reds=(1, 1, 1, 1), drfile=".", srsQs=10)
    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        desc = "First set of accelerations"
        labels = 4
        drms = {"drm": 1}
        misc = np.arange(10)
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "ATM2"
        desc = "Second set of accelerations"
        labels = 4
        drms = {"drm2": 1}
        drfunc = "ATM"
        misc = np.arange(10)
        drdefs.add(**locals())

    df = drdefs.excel_summary(None)
    assert df["ATM2"]["misc"] == "-"

    drdefs.add_0rb("ATM", "ATM2")

    assert "ATM_0rb" in drdefs
    assert "ATM2_0rb" in drdefs
    assert drdefs["ATM_0rb"].desc == "First set of accelerations w/o RB"
    assert drdefs["ATM2_0rb"].desc == "Second set of accelerations w/o RB"
    assert drdefs["ATM2_0rb"].desc == drdefs["ATM2"].desc + " w/o RB"


def test_addcat_4():
    defaults = dict(
        se=0,
        uf_reds=(1, 1, 1, 1),
        srsQs=10,
        srsfrq=0.0,
        srsconv=2.0,
        srsopts={"scale_by_Q_only": True, "eqsine": True, "ic": "steady"},
    )

    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        desc = "First set of accelerations"
        labels = 4
        drfunc = "blah * 4"
        srsQs = 10
        drdefs.add(**locals())

    assert drdefs["ATM"].srsopts == {
        "scale_by_Q_only": True,
        "eqsine": True,
        "ic": "steady",
    }
    assert drdefs["ATM"].srsconv == 2.0

    def _():
        name = "ATM2"
        desc = "First set of accelerations"
        labels = 4
        drfunc = "blah *"
        drdefs.add(**locals())

    with pytest.warns(RuntimeWarning, match=r"ATM2.*failed to compile string"):
        cla.DR_Def.addcat(_)

    assert drdefs["ATM2"].srsQs is None


def test_event_add():
    #    ext_name = 'FLAC'
    _get_labels = _get_labels0
    cyclenumber = 0

    # make up some CLA results:
    events = ("Liftoff", "Transonics", "MECO")
    rows = {"ATM": 34, "LTM": 29}
    ext_results = {i: {} for i in rows}
    for i, event in enumerate(events):
        for drm, nrows in rows.items():
            ext_results[drm][event] = _get_minmax(drm, i, cyclenumber)

    # setup CLA parameters:
    mission = "Rocket / Spacecraft CLA"
    duf = 1.2
    suf = 1.0

    # defaults for data recovery
    defaults = dict(se=10, uf_reds=(1, 1, duf, 1), drfile=".")

    drdefs = cla.DR_Def(defaults)
    drdefs2 = cla.DR_Def(defaults)
    drdefs3 = cla.DR_Def(defaults)
    drdefs4 = cla.DR_Def(defaults)

    uset = n2p.addgrid(None, 1, "b", 0, [0, 0, 0], 0)
    nas = {"ulvs": {10: np.ones((1, 1), float)}, "uset": {10: uset[:1]}}

    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        desc = "S/C Internal Accelerations"
        units = "m/sec^2, rad/sec^2"
        labels = _get_labels(rows, name)
        drms = {"atm": np.ones((4, 1), float)}
        nondrms = {"var": "any value"}
        drdefs.add(**locals())

    # so that "ATM data recovery category alread defined" error
    # gets triggered in DR_Event.add:
    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        desc = "S/C Internal Accelerations"
        units = "m/sec^2, rad/sec^2"
        labels = _get_labels(rows, name)
        drdefs2.add(**locals())

    # so that nondrms "atm is already in Vars[10]'" error
    # gets triggered in DR_Event.add:
    @cla.DR_Def.addcat
    def _():
        name = "ATM3"
        desc = "S/C Internal Accelerations"
        units = "m/sec^2, rad/sec^2"
        labels = _get_labels(rows, "ATM")
        drfunc = "ATM"
        drms = {"atm": np.ones((4, 1), float)}
        drdefs3.add(**locals())

    # so that nondrms "atm is already in Vars[10]'" error
    # gets triggered in DR_Event.add:
    @cla.DR_Def.addcat
    def _():
        name = "ATM4"
        desc = "S/C Internal Accelerations"
        units = "m/sec^2, rad/sec^2"
        labels = _get_labels(rows, "ATM")
        drfunc = "ATM"
        nondrms = {"var": "any value"}
        drdefs4.add(**locals())

    # for checking, make a pandas DataFrame to summarize data
    # recovery definitions (but skip the excel file for this
    # demo)
    df = drdefs.excel_summary(None)
    # prepare results data structure:
    DR = cla.DR_Event()

    # test for DR.add:
    DR.add(None, None)
    assert len(DR.Info) == 0

    DR.add(nas, drdefs, uf_reds=(2, 2, 2, 2))
    with pytest.raises(ValueError):
        DR.add(nas, drdefs2)
    with pytest.raises(ValueError):
        DR.add(nas, drdefs3)
    with pytest.raises(ValueError):
        DR.add(nas, drdefs4)

    # for testing apply_uf:
    sol = SimpleNamespace()
    sol.a = np.ones((1, 10), float)
    sol.v = sol.a + 1.0
    sol.d = sol.a + 2.0
    sol.pg = np.array([45])
    SOL1 = DR.apply_uf(sol, None, np.ones(1, float), np.ones(1, float) + 1.0, 0, None)
    SOL2 = DR.apply_uf(
        sol,
        np.ones(1, float),
        np.ones((1, 1), float),
        np.ones((1, 1), float) + 1.0,
        0,
        None,
    )
    SOL3 = DR.apply_uf(
        sol,
        np.ones((1, 1), float),
        np.ones((1, 1), float),
        np.ones((1, 1), float) + 1.0,
        0,
        None,
    )

    for k, d1 in SOL1.items():  # loop over uf_reds
        d2 = SOL2[k]
        d3 = SOL3[k]
        for k, v1 in d1.__dict__.items():  # loop over a, v, d, ...
            v2 = getattr(d2, k)
            v3 = getattr(d3, k)
            assert np.all(v2 == v1)
            assert np.all(v3 == v1)

    assert SOL1[(2, 2, 2, 2)].pg == 90

    SOL = DR.frf_apply_uf(sol, 0)

    assert np.all(SOL[(2, 2, 2, 2)].a == 4 * sol.a)
    assert np.all(SOL[(2, 2, 2, 2)].v == 4 * sol.v)
    assert np.all(SOL[(2, 2, 2, 2)].d == 4 * sol.d)
    assert np.all(SOL[(2, 2, 2, 2)].pg == 2 * sol.pg)


def test_apply_uf():
    sol = SimpleNamespace()
    sol.a = np.ones((4, 10))
    sol.v = sol.a + 1.0
    sol.d = sol.a + 2.0

    reds_2 = [(1.1, 1.2, 1.3, 1.4), (2.1, 2.2, 2.3, 2.4)]

    defaults = dict(
        se=0,
        uf_reds=(1, 1, 1, 1),
    )

    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        desc = "First set of accelerations"
        labels = 4
        drfunc = "blah * 4"
        uf_reds = reds_2[0]
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "ATM2"
        desc = "Second set of accelerations"
        labels = 4
        drfunc = "blah * 4"
        uf_reds = reds_2[1]
        drdefs.add(**locals())

    DR = cla.DR_Event()
    DR.add(None, drdefs)

    for shape in ((4, 4), (4,)):
        if len(shape) == 1:
            m = None
        else:
            m = np.ones(shape)
        b = np.ones(shape)
        k = b + 1
        if k.ndim > 1:
            k[2, 2] += 1.0
            k[3, 3] += 2.0
        nrb = 2
        rf = np.array([False, False, False, True])

        sol1 = DR.apply_uf(sol, m, b, k, nrb, 3)
        sol2 = DR.apply_uf(sol, m, b, k, nrb, rf)

        for reds in reds_2:
            for attr in sol1[reds].__dict__:
                assert np.allclose(getattr(sol1[reds], attr), getattr(sol2[reds], attr))

            assert np.allclose(sol1[reds].a[:nrb], 1.0 * reds[0] * reds[3])  # rb
            assert np.allclose(sol1[reds].a[nrb], 1.0 * reds[1] * reds[2])  # el
            assert np.allclose(sol1[reds].a[rf], 0.0)  # rf

            assert np.allclose(sol1[reds].v[:nrb], 2.0 * reds[0] * reds[3])  # rb
            assert np.allclose(sol1[reds].v[nrb], 2.0 * reds[1] * reds[2])  # el
            assert np.allclose(sol1[reds].v[rf], 0.0)  # rf

            e = [nrb]
            if k.ndim == 1:
                f_e = sol.a[e] + b[e, None] * sol.v[e] + k[e, None] * sol.d[e]
                d_static = f_e / k[e, None]
            else:
                ee = np.ix_(e, e)
                f_e = m[ee] @ sol.a[e] + b[ee] @ sol.v[e] + k[ee] @ sol.d[e]
                d_static = la.solve(k[ee], f_e)

            d_dynamic = sol.d[nrb] - d_static
            ds = d_static * reds[1] * reds[3]
            dd = d_dynamic * reds[1] * reds[2]
            assert np.allclose(sol1[reds].d[:nrb], 0.0)  # rb
            assert np.allclose(sol1[reds].d_static[nrb], ds)  # el
            assert np.allclose(sol1[reds].d_dynamic[nrb], dd)  # el
            assert np.allclose(sol1[reds].d[nrb], ds + dd)  # el

            ds = sol.d[rf] * reds[1] * reds[3]
            assert np.allclose(sol1[reds].d_static[rf], ds)  # rf
            assert np.allclose(sol1[reds].d_dynamic[rf], 0.0)  # rf
            assert np.allclose(sol1[reds].d[rf], ds)  # rf


def test_apply_uf_rb():
    sol = SimpleNamespace()
    sol.a = np.ones((4, 10))
    sol.v = sol.a + 1.0
    sol.d = sol.a + 2.0

    reds = (1.1, 1.2, 1.3, 1.4)

    defaults = dict(
        se=0,
        uf_reds=reds,
    )

    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        desc = "First set of accelerations"
        labels = 4
        drfunc = "blah * 4"
        drdefs.add(**locals())

    DR = cla.DR_Event()
    DR.add(None, drdefs)

    for shape in ((4, 4), (4,)):
        m = np.ones(shape)
        b = np.ones(shape)
        k = b * 0.0
        nrb = 4
        rf = np.array([False, False, False, False])

        sol1 = DR.apply_uf(sol, m, b, k, nrb, None)
        sol2 = DR.apply_uf(sol, m, b, k, nrb, rf)

        for attr in sol1[reds].__dict__:
            assert np.allclose(getattr(sol1[reds], attr), getattr(sol2[reds], attr))

        assert np.allclose(sol1[reds].a, 1.0 * reds[0] * reds[3])  # rb
        assert np.allclose(sol1[reds].v, 2.0 * reds[0] * reds[3])  # rb
        assert np.allclose(sol1[reds].d, 0.0)  # rb


def test_apply_uf_el():
    sol = SimpleNamespace()
    sol.a = np.ones((4, 10))
    sol.v = sol.a + 1.0
    sol.d = sol.a + 2.0

    # reds = (1, 1, 1, 1)
    reds = (1.1, 1.2, 1.3, 1.4)

    defaults = dict(
        se=0,
        uf_reds=reds,
    )

    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        desc = "First set of accelerations"
        labels = 4
        drfunc = "blah * 4"
        drdefs.add(**locals())

    DR = cla.DR_Event()
    DR.add(None, drdefs)

    for shape in ((4, 4), (4,)):
        m = np.ones(shape)
        b = np.ones(shape)
        k = np.random.randn(*shape)
        nrb = 0
        rf = np.array([False, False, False, False])

        sol1 = DR.apply_uf(sol, m, b, k, nrb, None)
        sol2 = DR.apply_uf(sol, m, b, k, nrb, rf)

        for attr in sol1[reds].__dict__:
            assert np.allclose(getattr(sol1[reds], attr), getattr(sol2[reds], attr))

        assert np.allclose(sol1[reds].a, 1.0 * reds[1] * reds[2])  # el
        assert np.allclose(sol1[reds].v, 2.0 * reds[1] * reds[2])  # el

        if k.ndim == 2:
            f = m @ sol.a + b @ sol.v + k @ sol.d
            d_static = la.solve(k, f)
        else:
            f = m[:, None] * sol.a + b[:, None] * sol.v + k[:, None] * sol.d
            d_static = f / k[:, None]

        d_dynamic = sol.d - d_static
        ds = d_static * reds[1] * reds[3]
        dd = d_dynamic * reds[1] * reds[2]
        assert np.allclose(sol1[reds].d_static, ds)  # el
        assert np.allclose(sol1[reds].d_dynamic, dd)  # el
        assert np.allclose(sol1[reds].d, ds + dd)  # el


def test_apply_uf_rf():
    sol = SimpleNamespace()
    sol.a = np.ones((4, 10))
    sol.v = sol.a + 1.0
    sol.d = sol.a + 2.0

    # reds = (1, 1, 1, 1)
    reds = (1.1, 1.2, 1.3, 1.4)

    defaults = dict(
        se=0,
        uf_reds=reds,
    )

    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        desc = "First set of accelerations"
        labels = 4
        drfunc = "blah * 4"
        drdefs.add(**locals())

    DR = cla.DR_Event()
    DR.add(None, drdefs)

    for shape in ((4, 4), (4,)):
        m = np.ones(shape)
        b = np.ones(shape)
        k = np.random.randn(*shape)
        nrb = 0
        rf = np.array([True, True, True, True])

        sol1 = DR.apply_uf(sol, m, b, k, nrb, np.arange(4))
        sol2 = DR.apply_uf(sol, m, b, k, nrb, rf)

        for attr in sol1[reds].__dict__:
            assert np.allclose(getattr(sol1[reds], attr), getattr(sol2[reds], attr))

        assert np.allclose(sol1[reds].a[rf], 0.0)  # rf
        assert np.allclose(sol1[reds].v[rf], 0.0)  # rf

        ds = sol.d * reds[1] * reds[3]
        assert np.allclose(sol1[reds].d_static[rf], ds)  # rf
        assert np.allclose(sol1[reds].d_dynamic[rf], 0.0)  # rf
        assert np.allclose(sol1[reds].d[rf], ds)  # rf


def test_event_add_uf_reds_update():
    _get_labels = _get_labels0
    rows = {"ATM": 34}

    # defaults for data recovery
    defaults = dict(se=10, uf_reds=(2, 2, 2, 2), drfile=".")

    drdefs = cla.DR_Def(defaults)

    uset = n2p.addgrid(None, 1, "b", 0, [0, 0, 0], 0)
    nas = {"ulvs": {10: np.ones((1, 1), float)}, "uset": {10: uset.iloc[:1]}}

    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        desc = "S/C Internal Accelerations"
        units = "m/sec^2, rad/sec^2"
        labels = _get_labels(rows, name)
        drms = {"atm": np.ones((4, 1), float)}
        nondrms = {"var": "any value"}
        drdefs.add(**locals())

    DR = cla.DR_Event()
    DR.add(nas, drdefs, uf_reds=(3, None, 4, None))

    # In [8]: PP(DR);
    # <class 'pyyeti.cla.DR_Event'>[n=3]
    #     .Info   : <class 'dict'>[n=1]
    #     .UF_reds: [n=1]: [[n=4]: (3, 2, 4, 2)]
    #     .Vars   : <class 'dict'>[n=1]

    assert DR.UF_reds[0] == (3, 2, 4, 2)

    DR = cla.DR_Event()
    DR.add(nas, drdefs, uf_reds=(3, 4, None, 6), method="multiply")
    assert DR.UF_reds[0] == (6, 8, 2, 12)

    DR = cla.DR_Event()
    DR.add(nas, drdefs, uf_reds=(None, 1, 4, 7), method=lambda o, n: o + n)
    assert DR.UF_reds[0] == (2, 3, 6, 9)

    DR = cla.DR_Event()
    with pytest.raises(ValueError):
        DR.add(
            nas,
            drdefs,
            uf_reds=(None, 1, 4, 7),
            method="bad method string",
        )


def test_DR_Results_init():
    _get_labels = _get_labels0
    rows = {"ATM": 34}

    # defaults for data recovery
    defaults = dict(se=10)

    drdefs = cla.DR_Def(defaults)

    uset = n2p.addgrid(None, 1, "b", 0, [0, 0, 0], 0)
    nas = {"ulvs": {10: np.ones((1, 1), float)}, "uset": {10: uset.iloc[:1]}}

    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        desc = "S/C Internal Accelerations"
        units = "m/sec^2, rad/sec^2"
        labels = _get_labels(rows, name)
        drms = {"atm": np.ones((4, 1), float)}
        nondrms = {"var": "any value"}
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "ATM2"
        labels = _get_labels(rows, "ATM")
        drfunc = "1"
        drdefs.add(**locals())

    DR = cla.DR_Event()
    DR.add(nas, drdefs)

    results = cla.DR_Results()
    results.init(DR.Info, "Mission", "Event")

    assert "ATM" in results
    assert "ATM2" in results

    results = cla.DR_Results()
    results.init(DR.Info, "Mission", "Event", cats=("ATM2",))

    assert "ATM" not in results
    assert "ATM2" in results


def test_merge():
    results = cla.DR_Results()
    r1 = get_fake_cla_results("FLAC", _get_labels0, 0)
    r2 = get_fake_cla_results("VLC", _get_labels1, 1)
    r3 = get_fake_cla_results("PostVLC", _get_labels2, 2)
    del r3["extreme"]
    results.merge((r1, r2, r3), {"FLAC": "FDLC", "PostVLC": "VLC2"})

    results.form_extreme()
    assert repr(results).startswith("DR_Results ")
    assert str(results).endswith(
        "with 4 keys: ['FDLC', 'VLC', 'Liftoff, Transonics, MECO', 'extreme']"
    )

    results = cla.DR_Results()
    r1 = {"FLAC": "this is a bad entry"}
    with pytest.raises(TypeError):
        results.merge((r1, r2))

    # ValueError: event with name {event} already exists!
    with pytest.raises(ValueError):
        results.merge((r2, r2))


def mass_spring_system():
    r"""
                |--> x1       |--> x2        |--> x3


             |----|    k1   |----|    k2   |----|
          f  |    |--\/\/\--|    |--\/\/\--|    |
        ====>| m1 |         | m2 |         | m3 |
             |    |---| |---|    |---| |---|    |
             |----|    c1   |----|    c2   |----|
               |                             |
               |             k3              |
               |-----------\/\/\-------------|
               |                             |
               |------------| |--------------|
                             c3

    m1 = 2 kg
    m2 = 4 kg
    m3 = 6 kg

    k1 = 12000 N/m
    k2 = 16000 N/m
    k3 = 10000 N/m

    c1 = 70 N s/m
    c2 = 75 N s/m
    c3 = 30 N s/m

    h = 0.001
    t = np.arange(0, 1.0, h)
    f = np.zeros((3, len(t)))
    f[0, 20:250] = 10.0  # N

    """
    m1 = 2.0
    m2 = 4.0
    m3 = 6.0
    k1 = 12000.0
    k2 = 16000.0
    k3 = 10000.0
    c1 = 70.0
    c2 = 75.0
    c3 = 30.0
    mass = np.diag([m1, m2, m3])
    stiff = np.array([[k1 + k3, -k1, -k3], [-k1, k1 + k2, -k2], [-k3, -k2, k2 + k3]])
    damp = np.array([[c1 + c3, -c1, -c3], [-c1, c1 + c2, -c2], [-c3, -c2, c2 + c3]])
    # drm for subtracting 1 from 2, 2 from 3, 1 from 3:
    sub = np.array([[-1.0, 1.0, 0], [0.0, -1.0, 1.0], [-1.0, 0, 1.0]])
    drms1 = {
        "springdrm": [[k1], [k2], [k3]] * sub,
        "damperdrm": [[c1], [c2], [c3]] * sub,
    }

    # define some defaults for data recovery:
    uf_reds = (1, 1, 1, 1)
    defaults = dict(se=0, uf_reds=uf_reds)
    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = "kc_forces"
        desc = "Spring & Damper Forces"
        units = "N"
        labels = [
            "{} {}".format(j, i + 1) for j in ("Spring", "Damper") for i in range(3)
        ]
        # force will be positive for tension
        drms = drms1
        drfunc = """np.vstack((Vars[se]['springdrm'] @ sol.d,
                               Vars[se]['damperdrm'] @ sol.v))"""
        histpv = "all"
        drdefs.add(**locals())

    # prepare spacecraft data recovery matrices
    DR = cla.DR_Event()
    DR.add(None, drdefs)

    return mass, damp, stiff, drms1, uf_reds, defaults, DR


def test_case_defined():
    (mass, damp, stiff, drms1, uf_reds, defaults, DR) = mass_spring_system()

    # define forces:
    h = 0.001
    t = np.arange(0, 1.0, h)
    f = np.zeros((3, len(t)))
    f[0, 20:250] = 10.0

    # setup solver:
    # ts = ode.SolveExp2(mass, damp, stiff, h)
    ts = ode.SolveUnc(mass, damp, stiff, h, pre_eig=True)
    sol = {uf_reds: ts.tsolve(f)}

    # initialize results (ext, mnc, mxc for all drms)
    event = "Case 1"
    results = DR.prepare_results("Spring & Damper Forces", event)

    # perform data recovery:
    results.time_data_recovery(sol, None, event, DR, 1, 0)

    assert np.allclose(
        results["kc_forces"].ext,
        np.array(
            [
                [1.71124021, -5.94610295],
                [1.10707637, -1.99361428],
                [1.89895824, -5.99096572],
                [2.01946488, -2.01871227],
                [0.46376154, -0.45142869],
                [0.96937744, -0.96687706],
            ]
        ),
    )

    # test for some errors:
    results = DR.prepare_results("Spring & Damper Forces", event)
    results.time_data_recovery(sol, None, event, DR, 2, 0)
    with pytest.raises(ValueError):
        results.time_data_recovery(sol, None, event, DR, 2, 1)

    # mess the labels up:
    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = "kc_forces"
        desc = "Spring & Damper Forces"
        units = "N"
        labels = ["one", "two"]
        drms = drms1
        drfunc = """np.vstack((Vars[se]['springdrm'] @ sol.d,
                               Vars[se]['damperdrm'] @ sol.v))"""
        drdefs.add(**locals())

    # prepare spacecraft data recovery matrices
    DR = cla.DR_Event()
    DR.add(None, drdefs)

    # initialize results (ext, mnc, mxc for all drms)
    results = DR.prepare_results("Spring & Damper Forces", event)

    # perform data recovery:
    with pytest.raises(ValueError):
        results.time_data_recovery(sol, None, event, DR, 1, 0)


def test_PSD_consistent():
    # resp:
    #   0   0.
    #   1   0.
    #     2   1.
    #     3   0.
    #   4   0.
    #   5   1.
    #     6   1.
    #     7   1.
    #   8   1.
    #   9   0.5
    #     10  0.5
    #     11  1.
    freq = np.arange(1.0, 6.0)
    resp = np.zeros((12, 5))
    forcepsd = np.ones((1, 5))
    resp[[2, 5, 6, 7, 8, 11]] = 1.0
    resp[[9, 10]] = 0.5
    xr = np.arange(0, 12, 2)
    yr = xr + 1
    rr = None
    drmres = SimpleNamespace(_psd={})
    case = "test"
    cla.PSD_consistent_rss(resp, xr, yr, rr, freq, forcepsd, drmres, case, 0)
    sbe = np.zeros((6, 5))
    sbe[1:3] = 1.0
    sbe[3] = 2.0
    sbe[4:] = 1.0 + 0.5**2
    assert np.allclose(drmres._psd[case], sbe)


def test_PSD_consistent2():
    s = []
    c = []
    Varx = []
    Vary = []
    Covar = []
    for varx in [1, 3, 5]:
        for vary in [1, 3, 5]:
            for covar in [0, 3, -3]:
                A = np.array([[varx, covar], [covar, vary]])
                lam, phi = la.eigh(A)
                theta = np.arctan2(phi[1, 1], phi[0, 1])

                s.append(np.sin(theta))
                c.append(np.cos(theta))

                Varx.append(varx)
                Vary.append(vary)
                Covar.append(covar)

    s2, c2 = ytools._calc_covariance_sine_cosine(
        np.array(Varx), np.array(Vary), np.array(Covar)
    )
    s = np.array(s)
    c = np.array(c)

    # The total response is c * x + s * y, which is then squared and
    # mulitplied by the PSD. So, signs can differ and, when x == y,
    # it's arbitrary whether c = 1 or s = 1. So, to be equivalent, we
    # can just check this (letting x = y = 1):

    assert np.allclose(abs(s2 + c2), abs(s + c))


def _comp_rpt(s, sbe):
    for i, j in zip(s, sbe):
        if not j.startswith("Date:"):
            assert i == j


def test_rptext1():
    (mass, damp, stiff, drms1, uf_reds, defaults, DR) = mass_spring_system()

    # define forces:
    h = 0.01
    t = np.arange(0, 1.0, h)
    f = np.zeros((3, len(t)))
    f[0, 2:25] = 10.0

    # setup solver:
    ts = ode.SolveUnc(mass, damp, stiff, h, pre_eig=True)
    sol = {uf_reds: ts.tsolve(f)}

    # initialize results (ext, mnc, mxc for all drms)
    event = "Case 1"
    results = DR.prepare_results("Spring & Damper Forces", event)

    # perform data recovery:
    results.time_data_recovery(sol, None, event, DR, 1, 0)

    with StringIO() as f:
        cla.rptext1(results["kc_forces"], f)
        s = f.getvalue().split("\n")
    sbe = [
        "M A X / M I N  S U M M A R Y",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Date:        21-Jan-2017",
        "",
        "  Row    Description       Maximum       Time    Case      "
        "   Minimum       Time    Case",
        "-------  -----------    -------------  --------  ------    "
        "-------------  --------  ------",
        "      1  Spring 1         1.51883e+00     0.270  Case 1    "
        " -5.75316e+00     0.040  Case 1",
        "      2  Spring 2         1.04111e+00     0.280  Case 1    "
        " -1.93144e+00     0.050  Case 1",
        "      3  Spring 3         1.59375e+00     0.280  Case 1    "
        " -5.68091e+00     0.050  Case 1",
        "      4  Damper 1         1.76099e+00     0.260  Case 1    "
        " -1.76088e+00     0.030  Case 1",
        "      5  Damper 2         4.23522e-01     0.270  Case 1    "
        " -4.11612e-01     0.040  Case 1",
        "      6  Damper 3         8.93351e-01     0.260  Case 1    "
        " -8.90131e-01     0.030  Case 1",
        "",
    ]
    _comp_rpt(s, sbe)

    lbls = results["kc_forces"].drminfo.labels[:]
    results["kc_forces"].drminfo.labels = lbls[:-1]
    with StringIO() as f:
        with pytest.raises(ValueError):
            cla.rptext1(results["kc_forces"], f)

    results["kc_forces"].drminfo.labels = lbls
    del results["kc_forces"].domain

    with StringIO() as f:
        cla.rptext1(results["kc_forces"], f, doabsmax=True, perpage=3)
        s = f.getvalue().split("\n")
    sbe = [
        "M A X / M I N  S U M M A R Y",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Date:        21-Jan-2017",
        "",
        "  Row    Description       Maximum     X-Value   Case",
        "-------  -----------    -------------  --------  ------",
        "      1  Spring 1        -5.75316e+00     0.270  Case 1",
        "      2  Spring 2        -1.93144e+00     0.280  Case 1",
        "      3  Spring 3        -5.68091e+00     0.280  Case 1",
        "\x0cM A X / M I N  S U M M A R Y",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Date:        21-Jan-2017",
        "",
        "  Row    Description       Maximum     X-Value   Case",
        "-------  -----------    -------------  --------  ------",
        "      4  Damper 1         1.76099e+00     0.260  Case 1",
        "      5  Damper 2         4.23522e-01     0.270  Case 1",
        "      6  Damper 3         8.93351e-01     0.260  Case 1",
        "",
    ]
    _comp_rpt(s, sbe)

    results["kc_forces"].ext = results["kc_forces"].ext[:, :1]
    results["kc_forces"].ext_x = results["kc_forces"].ext_x[:, :1]
    with StringIO() as f:
        cla.rptext1(results["kc_forces"], f)
        s = f.getvalue().split("\n")
    sbe = [
        "M A X / M I N  S U M M A R Y",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Date:        22-Jan-2017",
        "",
        "  Row    Description       Maximum     X-Value   Case",
        "-------  -----------    -------------  --------  ------",
        "      1  Spring 1         1.51883e+00     0.270  Case 1",
        "      2  Spring 2         1.04111e+00     0.280  Case 1",
        "      3  Spring 3         1.59375e+00     0.280  Case 1",
        "      4  Damper 1         1.76099e+00     0.260  Case 1",
        "      5  Damper 2         4.23522e-01     0.270  Case 1",
        "      6  Damper 3         8.93351e-01     0.260  Case 1",
        "",
    ]
    _comp_rpt(s, sbe)

    results["kc_forces"].ext = results["kc_forces"].ext[:, 0]
    results["kc_forces"].ext_x = results["kc_forces"].ext_x[:, 0]
    with StringIO() as f:
        cla.rptext1(results["kc_forces"], f)
        s = f.getvalue().split("\n")
    _comp_rpt(s, sbe)


def test_get_numform():
    from pyyeti.cla._utilities import _get_numform

    assert _get_numform(0.0) == "{:13.0f}"
    assert _get_numform(np.array([1e12, 1e4])) == "{:13.6e}"
    assert _get_numform(np.array([1e8, 1e4])) == "{:13.1f}"
    assert _get_numform(np.array([1e10, 1e5])) == "{:13.0f}"


def test_rpttab1():
    (mass, damp, stiff, drms1, uf_reds, defaults, DR) = mass_spring_system()

    # define forces:
    h = 0.01
    t = np.arange(0, 1.0, h)
    f = {0: np.zeros((3, len(t))), 1: np.zeros((3, len(t)))}
    f[0][0, 2:25] = 10.0
    f[1][0, 2:25] = np.arange(23.0)
    f[1][0, 25:48] = np.arange(22.0, -1.0, -1.0)

    # initialize results (ext, mnc, mxc for all drms)
    results = DR.prepare_results("Spring & Damper Forces", "Steps")
    # setup solver:
    ts = ode.SolveUnc(mass, damp, stiff, h, pre_eig=True)
    for i in range(2):
        sol = {uf_reds: ts.tsolve(f[i])}
        case = "FFN {}".format(i)
        # perform data recovery:
        results.time_data_recovery(sol, None, case, DR, 2, i)

    with StringIO() as f:
        cla.rpttab1(results["kc_forces"], f, "Title")
        s = f.getvalue().split("\n")
    sbe = [
        "Title",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Date:        22-Jan-2017",
        "",
        "Maximum Responses",
        "",
        " Row     Description       FFN 0         FFN 1        Maximum         Case",
        "====== =============== ============= ============= ============= =============",
        "     1 Spring 1         1.518830e+00  1.858805e-01  1.518830e+00 FFN 0",
        "     2 Spring 2         1.041112e+00  1.335839e-01  1.041112e+00 FFN 0",
        "     3 Spring 3         1.593752e+00  2.383903e-01  1.593752e+00 FFN 0",
        "     4 Damper 1         1.760988e+00  3.984896e-01  1.760988e+00 FFN 0",
        "     5 Damper 2         4.235220e-01  1.315805e-01  4.235220e-01 FFN 0",
        "     6 Damper 3         8.933514e-01  2.115629e-01  8.933514e-01 FFN 0",
        "",
        "",
        "Minimum Responses",
        "",
        " Row     Description       FFN 0         FFN 1        Minimum         Case",
        "====== =============== ============= ============= ============= =============",
        "     1 Spring 1        -5.753157e+00 -9.464095e+00 -9.464095e+00 FFN 1",
        "     2 Spring 2        -1.931440e+00 -2.116412e+00 -2.116412e+00 FFN 1",
        "     3 Spring 3        -5.680914e+00 -9.182364e+00 -9.182364e+00 FFN 1",
        "     4 Damper 1        -1.760881e+00 -3.428864e-01 -1.760881e+00 FFN 0",
        "     5 Damper 2        -4.116117e-01 -9.167583e-02 -4.116117e-01 FFN 0",
        "     6 Damper 3        -8.901312e-01 -1.797676e-01 -8.901312e-01 FFN 0",
        "",
        "",
        "Abs-Max Responses",
        "",
        " Row     Description       FFN 0         FFN 1        Abs-Max         Case",
        "====== =============== ============= ============= ============= =============",
        "     1 Spring 1        -5.753157e+00 -9.464095e+00 -9.464095e+00 FFN 1",
        "     2 Spring 2        -1.931440e+00 -2.116412e+00 -2.116412e+00 FFN 1",
        "     3 Spring 3        -5.680914e+00 -9.182364e+00 -9.182364e+00 FFN 1",
        "     4 Damper 1         1.760988e+00  3.984896e-01  1.760988e+00 FFN 0",
        "     5 Damper 2         4.235220e-01  1.315805e-01  4.235220e-01 FFN 0",
        "     6 Damper 3         8.933514e-01  2.115629e-01  8.933514e-01 FFN 0",
        "",
        "",
        "Extrema Count",
        "Filter: 1e-06",
        "",
        "         Description       FFN 0         FFN 1",
        "       =============== ============= =============",
        "       Maxima Count                6             0",
        "       Minima Count                3             3",
        "       Max+Min Count               9             3",
        "       Abs-Max Count               3             3",
        "",
        "         Description       FFN 0         FFN 1",
        "       =============== ============= =============",
        "       Maxima Percent          100.0           0.0",
        "       Minima Percent           50.0          50.0",
        "       Max+Min Percent          75.0          25.0",
        "       Abs-Max Percent          50.0          50.0",
        "",
    ]
    _comp_rpt(s, sbe)

    results["kc_forces"].maxcase = None
    results["kc_forces"].mincase = None
    with StringIO() as f:
        cla.rpttab1(results["kc_forces"], f, "Title")
        s = f.getvalue().split("\n")
    sbe2 = sbe[:]
    for i in range(len(sbe2)):
        if len(sbe2[i]) > 60 and "FFN " in sbe2[i][60:]:
            sbe2[i] = sbe2[i].replace("FFN 0", "N/A").replace("FFN 1", "N/A")
    _comp_rpt(s, sbe2)

    results["kc_forces"].ext[:] = 0.0
    results["kc_forces"].mn[:] = 0.0
    results["kc_forces"].mx[:] = 0.0
    with StringIO() as f:
        cla.rpttab1(results["kc_forces"], f, "Title")
        s = f.getvalue().split("\n")
    sbe = [
        "Title",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Date:        22-Jan-2017",
        "",
        "Maximum Responses",
        "",
        " Row     Description       FFN 0         FFN 1        Maximum         Case",
        "====== =============== ============= ============= ============= =============",
        "     1 Spring 1         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     2 Spring 2         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     3 Spring 3         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     4 Damper 1         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     5 Damper 2         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     6 Damper 3         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "",
        "",
        "Minimum Responses",
        "",
        " Row     Description       FFN 0         FFN 1        Minimum         Case",
        "====== =============== ============= ============= ============= =============",
        "     1 Spring 1         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     2 Spring 2         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     3 Spring 3         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     4 Damper 1         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     5 Damper 2         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     6 Damper 3         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "",
        "",
        "Abs-Max Responses",
        "",
        " Row     Description       FFN 0         FFN 1        Abs-Max         Case",
        "====== =============== ============= ============= ============= =============",
        "     1 Spring 1         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     2 Spring 2         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     3 Spring 3         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     4 Damper 1         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     5 Damper 2         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "     6 Damper 3         0.000000e+00  0.000000e+00  0.000000e+00 zero row",
        "",
        "",
        "Extrema Count",
        "Filter: 1e-06",
        "",
        "         Description       FFN 0         FFN 1",
        "       =============== ============= =============",
        "       Maxima Count                0             0",
        "       Minima Count                0             0",
        "       Max+Min Count               0             0",
        "       Abs-Max Count               0             0",
        "",
        "         Description       FFN 0         FFN 1",
        "       =============== ============= =============",
        "       Maxima Percent            0.0           0.0",
        "       Minima Percent            0.0           0.0",
        "       Max+Min Percent           0.0           0.0",
        "       Abs-Max Percent           0.0           0.0",
        "",
    ]
    _comp_rpt(s, sbe)

    lbls = results["kc_forces"].drminfo.labels[:]
    results["kc_forces"].drminfo.labels = lbls[:-1]
    with StringIO() as f:
        with pytest.raises(ValueError):
            cla.rpttab1(results["kc_forces"], f, "Title")

    results["kc_forces"].drminfo.labels = lbls
    with pytest.raises(ValueError):
        cla.rpttab1(results["kc_forces"], "t.xlsx", "Title")


def test_rptpct1():
    ext1 = [[120.0, -8.0], [8.0, -120.0]]
    ext2 = [[115.0, -5.0], [10.0, -125.0]]
    opts = {
        "domagpct": False,
        "dohistogram": False,
        "filterval": np.array([5.0, 1000.0]),
    }
    with StringIO() as f:
        dct = cla.rptpct1(ext1, ext2, f, **opts)
        s = f.getvalue().split("\n")
    sbe = [
        "PERCENT DIFFERENCE REPORT",
        "",
        "Description: No description provided",
        "Uncertainty: Not specified",
        "Units:       Not specified",
        "Filter:      <defined row-by-row>",
        "Notes:       % Diff = +/- abs(Self-Reference)/max(abs(Reference(max,min)))*100",
        "             Sign set such that positive % differences indicate exceedances",
        "Date:        22-Jan-2017",
        "",
        "                             Self        Reference                    Self    "
        "    Reference                    Self        Reference",
        "  Row    Description       Maximum        Maximum      % Diff       Minimum   "
        "     Minimum      % Diff       Abs-Max        Abs-Max      % Diff",
        "-------  -----------    -------------  -------------  -------    -------------"
        "  -------------  -------    -------------  -------------  -------",
        "      1  Row      1         120.00000      115.00000     4.35         -8.00000"
        "       -5.00000     2.61        120.00000      115.00000     4.35",
        "      2  Row      2           8.00000       10.00000  n/a           -120.00000"
        "     -125.00000  n/a            120.00000      125.00000  n/a    ",
        "",
        "",
        "",
        "    No description provided - Maximum Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         4.00          1    100.00",
        "",
        "    0.0% of values are within 1%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [4.35, 4.35, 4.3478, 0.0000]",
        "",
        "",
        "    No description provided - Minimum Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         3.00          1    100.00",
        "",
        "    0.0% of values are within 1%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.61, 2.61, 2.6087, 0.0000]",
        "",
        "",
        "    No description provided - Abs-Max Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         4.00          1    100.00",
        "",
        "    0.0% of values are within 1%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [4.35, 4.35, 4.3478, 0.0000]",
        "",
    ]
    _comp_rpt(s, sbe)


def test_rptpct1_2():
    (mass, damp, stiff, drms1, uf_reds, defaults, DR) = mass_spring_system()

    # define forces:
    h = 0.01
    t = np.arange(0, 1.0, h)
    f = {0: np.zeros((3, len(t))), 1: np.zeros((3, len(t)))}
    f[0][0, 2:25] = 10.0
    f[1][0, 2:25] = 10.0
    f[1][0, 3:25:3] = 9.5

    # initialize results
    results = cla.DR_Results()
    # setup solver:
    ts = ode.SolveUnc(mass, damp, stiff, h, pre_eig=True)
    for i in range(2):
        sol = {uf_reds: ts.tsolve(f[i])}
        case = "FFN {}".format(i)
        # perform data recovery:
        results[case] = DR.prepare_results("Spring & Damper Forces", "Steps")
        results[case].time_data_recovery(sol, None, case, DR, 1, 0)

    opts = {"domagpct": False, "dohistogram": False, "filterval": 0.3 * np.ones(6)}
    drminfo = results["FFN 0"]["kc_forces"].drminfo
    drminfo.labels = drminfo.labels[:]
    drminfo.labels[2] = "SPRING 3"
    with StringIO() as f:
        dct = cla.rptpct1(
            results["FFN 0"]["kc_forces"], results["FFN 1"]["kc_forces"], f, **opts
        )
        s = f.getvalue().split("\n")
    sbe = [
        "PERCENT DIFFERENCE REPORT",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Filter:      <defined row-by-row>",
        "Notes:       % Diff = +/- abs(Self-Reference)/max(abs(Reference(max,min)))*100",
        "             Sign set such that positive % differences indicate exceedances",
        "Date:        23-Jan-2017",
        "",
        "                             Self        Reference                    Self        Reference                    Self        Reference",
        "  Row    Description       Maximum        Maximum      % Diff       Minimum        Minimum      % Diff       Abs-Max        Abs-Max      % Diff",
        "-------  -----------    -------------  -------------  -------    -------------  -------------  -------    -------------  -------------  -------",
        "      1  Spring 1            1.518830       1.509860     0.16        -5.753157      -5.603161     2.68         5.753157       5.603161     2.68",
        "      2  Spring 2            1.041112       1.031179     0.53        -1.931440      -1.887905     2.31         1.931440       1.887905     2.31",
        "      4  Damper 1            1.760988       1.714232     2.73        -1.760881      -1.697650     3.69         1.760988       1.714232     2.73",
        "      5  Damper 2            0.423522       0.415255     1.99        -0.411612      -0.399434     2.93         0.423522       0.415255     1.99",
        "      6  Damper 3            0.893351       0.873861     2.23        -0.890131      -0.861630     3.26         0.893351       0.873861     2.23",
        "",
        "",
        "",
        "    Spring & Damper Forces - Maximum Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         0.00          1     20.00",
        "         1.00          1     20.00",
        "         2.00          2     40.00",
        "         3.00          1     20.00",
        "",
        "    40.0% of values are within 1%",
        "    80.0% of values are within 2%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [0.16, 2.73, 1.5270, 1.1204]",
        "",
        "",
        "    Spring & Damper Forces - Minimum Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         2.00          1     20.00",
        "         3.00          3     60.00",
        "         4.00          1     20.00",
        "",
        "    0.0% of values are within 1%",
        "    20.0% of values are within 2%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.31, 3.69, 2.9731, 0.5314]",
        "",
        "",
        "    Spring & Damper Forces - Abs-Max Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         2.00          3     60.00",
        "         3.00          2     40.00",
        "",
        "    0.0% of values are within 1%",
        "    60.0% of values are within 2%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [1.99, 2.73, 2.3863, 0.3115]",
        "",
    ]
    _comp_rpt(s, sbe)

    opts = {
        "domagpct": False,
        "dohistogram": False,
        "filterval": 0.3 * np.ones(1),
        "use_range": False,
    }
    with StringIO() as f:
        dct = cla.rptpct1(
            results["FFN 0"]["kc_forces"], results["FFN 1"]["kc_forces"], f, **opts
        )
        s = f.getvalue().split("\n")
    sbe = [
        "PERCENT DIFFERENCE REPORT",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Filter:      0.3",
        "Notes:       % Diff = +/- abs((Self-Reference)/Reference)*100",
        "             Sign set such that positive % differences indicate exceedances",
        "Date:        30-Jan-2017",
        "",
        "                             Self        Reference                    Self    "
        "    Reference                    Self        Reference",
        "  Row    Description       Maximum        Maximum      % Diff       Minimum   "
        "     Minimum      % Diff       Abs-Max        Abs-Max      % Diff",
        "-------  -----------    -------------  -------------  -------    -------------"
        "  -------------  -------    -------------  -------------  -------",
        "      1  Spring 1            1.518830       1.509860     0.59        -5.753157"
        "      -5.603161     2.68         5.753157       5.603161     2.68",
        "      2  Spring 2            1.041112       1.031179     0.96        -1.931440"
        "      -1.887905     2.31         1.931440       1.887905     2.31",
        "      4  Damper 1            1.760988       1.714232     2.73        -1.760881"
        "      -1.697650     3.72         1.760988       1.714232     2.73",
        "      5  Damper 2            0.423522       0.415255     1.99        -0.411612"
        "      -0.399434     3.05         0.423522       0.415255     1.99",
        "      6  Damper 3            0.893351       0.873861     2.23        -0.890131"
        "      -0.861630     3.31         0.893351       0.873861     2.23",
        "",
        "",
        "",
        "    Spring & Damper Forces - Maximum Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         1.00          2     40.00",
        "         2.00          2     40.00",
        "         3.00          1     20.00",
        "",
        "    40.0% of values are within 1%",
        "    80.0% of values are within 2%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [0.59, 2.73, 1.7012, 0.8927]",
        "",
        "",
        "    Spring & Damper Forces - Minimum Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         2.00          1     20.00",
        "         3.00          3     60.00",
        "         4.00          1     20.00",
        "",
        "    0.0% of values are within 1%",
        "    20.0% of values are within 2%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.31, 3.72, 3.0128, 0.5494]",
        "",
        "",
        "    Spring & Damper Forces - Abs-Max Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         2.00          3     60.00",
        "         3.00          2     40.00",
        "",
        "    0.0% of values are within 1%",
        "    60.0% of values are within 2%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [1.99, 2.73, 2.3863, 0.3115]",
        "",
    ]
    _comp_rpt(s, sbe)

    opts = {"domagpct": False, "dohistogram": False, "prtbad": 2.5, "flagbad": 2.7}
    with StringIO() as f:
        dct = cla.rptpct1(
            results["FFN 0"]["kc_forces"], results["FFN 1"]["kc_forces"], f, **opts
        )
        s = f.getvalue().split("\n")
    sbe = [
        "PERCENT DIFFERENCE REPORT",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Filter:      1e-06",
        "Notes:       % Diff = +/- abs(Self-Reference)/max(abs(Reference(max,min)))*100",
        "             Sign set such that positive % differences indicate exceedances",
        "             Printing rows where abs(% Diff) > 2.5%",
        "             Flagging (*) rows where abs(% Diff) > 2.7%",
        "Date:        30-Jan-2017",
        "",
        "                             Self        Reference                     Self        Reference                     Self        Reference",
        "  Row    Description       Maximum        Maximum      % Diff        Minimum        Minimum      % Diff        Abs-Max        Abs-Max      % Diff",
        "-------  -----------    -------------  -------------  --------    -------------  -------------  --------    -------------  -------------  --------",
        "      1  Spring 1            1.518830       1.509860     0.16         -5.753157      -5.603161     2.68          5.753157       5.603161     2.68 ",
        "      4  Damper 1            1.760988       1.714232     2.73*        -1.760881      -1.697650     3.69*         1.760988       1.714232     2.73*",
        "      5  Damper 2            0.423522       0.415255     1.99         -0.411612      -0.399434     2.93*         0.423522       0.415255     1.99 ",
        "      6  Damper 3            0.893351       0.873861     2.23         -0.890131      -0.861630     3.26*         0.893351       0.873861     2.23 ",
        "",
        "",
        "",
        "    Spring & Damper Forces - Maximum Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         0.00          1     20.00",
        "         1.00          1     20.00",
        "         2.00          2     40.00",
        "         3.00          1     20.00",
        "",
        "    40.0% of values are within 1%",
        "    80.0% of values are within 2%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [0.16, 2.73, 1.5270, 1.1204]",
        "",
        "",
        "    Spring & Damper Forces - Minimum Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         2.00          1     20.00",
        "         3.00          3     60.00",
        "         4.00          1     20.00",
        "",
        "    0.0% of values are within 1%",
        "    20.0% of values are within 2%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.31, 3.69, 2.9731, 0.5314]",
        "",
        "",
        "    Spring & Damper Forces - Abs-Max Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         2.00          3     60.00",
        "         3.00          2     40.00",
        "",
        "    0.0% of values are within 1%",
        "    60.0% of values are within 2%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [1.99, 2.73, 2.3863, 0.3115]",
        "",
    ]
    _comp_rpt(s, sbe)

    opts = {"domagpct": False, "dohistogram": False, "prtbadh": 2.5, "flagbadh": 2.7}
    with StringIO() as f:
        dct = cla.rptpct1(
            results["FFN 0"]["kc_forces"], results["FFN 1"]["kc_forces"], f, **opts
        )
        s = f.getvalue().split("\n")
    sbe[8] = "             Printing rows where % Diff > 2.5%"
    sbe[9] = "             Flagging (*) rows where % Diff > 2.7%"
    _comp_rpt(s, sbe)

    opts = {"domagpct": False, "dohistogram": False, "prtbadl": 2.0, "flagbadl": 2.2}
    with StringIO() as f:
        dct = cla.rptpct1(
            results["FFN 0"]["kc_forces"], results["FFN 1"]["kc_forces"], f, **opts
        )
        s = f.getvalue().split("\n")
    sbe[8] = "             Printing rows where % Diff < 2.0%"
    sbe[9] = "             Flagging (*) rows where % Diff < 2.2%"
    sbe[12:19] = [
        "                             Self        Reference                     Self        Reference                    Self        Reference",
        "  Row    Description       Maximum        Maximum      % Diff        Minimum        Minimum      % Diff       Abs-Max        Abs-Max      % Diff",
        "-------  -----------    -------------  -------------  --------    -------------  -------------  -------    -------------  -------------  --------",
        "      1  Spring 1            1.518830       1.509860     0.16*        -5.753157      -5.603161     2.68         5.753157       5.603161     2.68 ",
        "      2  Spring 2            1.041112       1.031179     0.53*        -1.931440      -1.887905     2.31         1.931440       1.887905     2.31 ",
        "      5  Damper 2            0.423522       0.415255     1.99*        -0.411612      -0.399434     2.93         0.423522       0.415255     1.99*",
    ]
    _comp_rpt(s, sbe)

    opts = {
        "domagpct": False,
        "dohistogram": False,
        "prtbadl": 2.0,
        "flagbadl": 2.2,
        "doabsmax": True,
    }
    with StringIO() as f:
        dct = cla.rptpct1(
            results["FFN 0"]["kc_forces"], results["FFN 1"]["kc_forces"], f, **opts
        )
        s = f.getvalue().split("\n")
    sbe = [
        "PERCENT DIFFERENCE REPORT",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Filter:      1e-06",
        "Notes:       % Diff = +/- abs((Self-Reference)/Reference)*100",
        "             Sign set such that positive % differences indicate exceedances",
        "             Printing rows where % Diff < 2.0%",
        "             Flagging (*) rows where % Diff < 2.2%",
        "Date:        30-Jan-2017",
        "",
        "                             Self           Self           Self        Reference",
        "  Row    Description       Maximum        Minimum        Abs-Max        Abs-Max      % Diff",
        "-------  -----------    -------------  -------------  -------------  -------------  --------",
        "      5  Damper 2            0.423522      -0.411612       0.423522       0.415255     1.99*",
        "",
        "",
        "",
        "    Spring & Damper Forces - Abs-Max Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         2.00          3     60.00",
        "         3.00          2     40.00",
        "",
        "    0.0% of values are within 1%",
        "    60.0% of values are within 2%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [1.99, 2.73, 2.3863, 0.3115]",
        "",
    ]
    _comp_rpt(s, sbe)

    opts = {
        "domagpct": False,
        "dohistogram": False,
        "prtbadl": 2.0,
        "flagbadl": 2.2,
        "shortabsmax": True,
    }
    with StringIO() as f:
        dct = cla.rptpct1(
            results["FFN 0"]["kc_forces"], results["FFN 1"]["kc_forces"], f, **opts
        )
        s = f.getvalue().split("\n")
    sbe = [
        "PERCENT DIFFERENCE REPORT",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Filter:      1e-06",
        "Notes:       % Diff = +/- abs((Self-Reference)/Reference)*100",
        "             Sign set such that positive % differences indicate exceedances",
        "             Printing rows where % Diff < 2.0%",
        "             Flagging (*) rows where % Diff < 2.2%",
        "Date:        30-Jan-2017",
        "",
        "                             Self        Reference",
        "  Row    Description       Abs-Max        Abs-Max      % Diff",
        "-------  -----------    -------------  -------------  --------",
        "      5  Damper 2            0.423522       0.415255     1.99*",
        "",
        "",
        "",
        "    Spring & Damper Forces - Abs-Max Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         2.00          3     60.00",
        "         3.00          2     40.00",
        "",
        "    0.0% of values are within 1%",
        "    60.0% of values are within 2%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [1.99, 2.73, 2.3863, 0.3115]",
        "",
    ]
    _comp_rpt(s, sbe)

    with StringIO() as f:
        # mxmn2 has different number of rows:
        with pytest.raises(ValueError):
            cla.rptpct1(
                results["FFN 0"]["kc_forces"],
                results["FFN 1"]["kc_forces"].ext[:4],
                f,
                **opts,
            )

    drminfo0 = results["FFN 0"]["kc_forces"].drminfo
    drminfo1 = results["FFN 1"]["kc_forces"].drminfo
    drminfo0.labels = drminfo1.labels[:4]
    with StringIO() as f:
        # labels is wrong length
        with pytest.raises(ValueError):
            cla.rptpct1(
                results["FFN 0"]["kc_forces"],
                results["FFN 1"]["kc_forces"].ext,
                f,
                **opts,
            )

    opts = {
        "domagpct": False,
        "dohistogram": False,
        "shortabsmax": True,
        "ignorepv": np.array([False, False, True, False, True, True]),
        "roundvals": 3,
        "perpage": 3,
    }
    drminfo0.labels = drminfo1.labels[:]
    with StringIO() as f:
        cla.rptpct1(
            results["FFN 0"]["kc_forces"], results["FFN 1"]["kc_forces"].ext, f, **opts
        )
        s = f.getvalue().split("\n")
    sbe = [
        "PERCENT DIFFERENCE REPORT",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Filter:      1e-06",
        "Notes:       % Diff = +/- abs((Self-Reference)/Reference)*100",
        "             Sign set such that positive % differences indicate exceedances",
        "Date:        30-Jan-2017",
        "",
        "                             Self        Reference",
        "  Row    Description       Abs-Max        Abs-Max      % Diff",
        "-------  -----------    -------------  -------------  -------",
        "      1  Spring 1            5.753000       5.603000     2.68",
        "      2  Spring 2            1.931000       1.888000     2.28",
        "      3  Spring 3            5.681000       5.572000  n/a    ",
        "",
        "PERCENT DIFFERENCE REPORT",
        "",
        "Description: Spring & Damper Forces",
        "Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]",
        "Units:       N",
        "Filter:      1e-06",
        "Notes:       % Diff = +/- abs((Self-Reference)/Reference)*100",
        "             Sign set such that positive % differences indicate exceedances",
        "Date:        30-Jan-2017",
        "",
        "                             Self        Reference",
        "  Row    Description       Abs-Max        Abs-Max      % Diff",
        "-------  -----------    -------------  -------------  -------",
        "      4  Damper 1            1.761000       1.714000     2.74",
        "      5  Damper 2            0.424000       0.415000  n/a    ",
        "      6  Damper 3            0.893000       0.874000  n/a    ",
        "",
        "",
        "",
        "    Spring & Damper Forces - Abs-Max Comparison Histogram",
        "",
        "      % Diff      Count    Percent",
        "     --------   --------   -------",
        "         2.00          1     33.33",
        "         3.00          2     66.67",
        "",
        "    0.0% of values are within 1%",
        "    33.3% of values are within 2%",
        "    100.0% of values are within 5%",
        "",
        "    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.28, 2.74, 2.5656, 0.2516]",
        "",
    ]
    _comp_rpt(s, sbe)


def test_frf_data_recovery():
    (mass, damp, stiff, drms1, uf_reds, defaults, DR) = mass_spring_system()

    # define forces:
    frq = np.arange(10, 25, 0.1)
    f = np.zeros((3, len(frq)))
    f[0] = 1.0

    # setup solver:
    fs = ode.SolveUnc(mass, damp, stiff, pre_eig=True)
    sol = fs.fsolve(f, frq)

    elforces = np.vstack((drms1["springdrm"] @ sol.d, drms1["damperdrm"] @ sol.v))

    accels = sol.a
    Q = 20
    fsh = srs.srs_frf(elforces.T, frq, frq, Q)
    ash = srs.srs_frf(accels.T, frq, frq, Q) / Q

    # now, solve with DR tools:
    # define some defaults for data recovery:
    # - accept default (1, 1, 1, 1) for uf_reds
    defaults = dict(se=0, srsfrq=frq, srsQs=(Q, 50))
    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = "kc_forces"
        desc = "Spring & Damper Forces"
        units = "N"
        labels = [
            "{} {}".format(j, i + 1) for j in ("Spring", "Damper") for i in range(3)
        ]
        # force will be positive for tension
        drms = drms1
        drfunc = """np.vstack((Vars[se]['springdrm'] @ sol.d,
                               Vars[se]['damperdrm'] @ sol.v))"""
        histpv = "all"
        srspv = "all"
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "accels"
        desc = "Accelerations"
        units = "m/sec^2"
        labels = [f"Accel {i}" for i in range(3)]
        drfunc = """sol.a"""
        histpv = "all"
        srspv = "all"
        srsopts = dict(eqsine=1)
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "accels2"
        desc = "Accelerations"
        units = "m/sec^2"
        labels = [f"Accel {i}" for i in range(3)]
        drfunc = """sol.a"""
        srspv = 2
        srsopts = dict(eqsine=1)
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "noaccels"
        desc = "No Accelerations"
        units = "m/sec^2"
        labels = [f"Accel {i}" for i in range(3)]
        drfunc = """sol.a"""
        active = "no"
        srspv = 2
        srsopts = dict(eqsine=1)
        drdefs.add(**locals())

    # prepare spacecraft data recovery matrices
    DR = cla.DR_Event()
    DR.add(None, drdefs)
    assert "with 3 categories" in repr(DR)

    # initialize results (ext, mnc, mxc for all drms)
    event = "Case 1"
    results = DR.prepare_results("Spring & Damper", event)

    sol = {uf_reds: sol}

    # perform data recovery:
    results.frf_data_recovery(sol, None, event, DR, 1, 0, verbose=3)

    assert np.allclose(elforces, results["kc_forces"].frf[0])
    assert np.allclose(fsh.T, results["kc_forces"].srs.srs[20][0])
    assert np.allclose(accels, results["accels"].frf[0])
    assert np.allclose(ash.T, results["accels"].srs.srs[20][0])

    assert np.allclose(
        results["accels"].srs.srs[20][0, 2], results["accels2"].srs.srs[20][0, 0]
    )

    # to exercise some code (not checking anything other than that it
    # runs), write tab, plots:
    try:
        direc = "temp_frf"
        results.rptext(direc=direc)
        results.resp_plots(direc=direc)
        assert os.path.exists("temp_frf/Case 1_frf.pdf")
        assert os.path.exists("temp_frf/kc_forces.ext")
        assert os.path.exists("temp_frf/accels.ext")
        assert os.path.exists("temp_frf/accels2.ext")
        assert not os.path.exists("temp_frf/noaccels.ext")
    finally:
        # pass
        shutil.rmtree("./temp_frf", ignore_errors=True)

    try:
        direc = "temp_frf"
        results.rptext(direc=direc, drms=["accels", "kc_forces"])
        assert not os.path.exists("temp_frf/Case 1_frf.pdf")
        assert os.path.exists("temp_frf/kc_forces.ext")
        assert os.path.exists("temp_frf/accels.ext")
        assert not os.path.exists("temp_frf/accels2.ext")
        assert not os.path.exists("temp_frf/noaccels.ext")

        results.rpttab(direc=direc, drms=["accels", "accels2"])
        assert not os.path.exists("temp_frf/kc_forces.tab")
        assert os.path.exists("temp_frf/accels.tab")
        assert os.path.exists("temp_frf/accels2.tab")
        assert not os.path.exists("temp_frf/noaccels.tab")
    finally:
        # pass
        shutil.rmtree("./temp_frf", ignore_errors=True)


def test_reldisp_dtm():
    INBOARD = 0
    OUTBOARD = 101

    event = "TOES"
    pth = "pyyeti/tests/cla_test_data/"

    # load nastran data:
    nas = op2.rdnas2cam(pth + "nas2cam")

    # form ulvs for some SEs:
    SC = 101
    n2p.addulvs(nas, SC)

    # read in more data for OUTBOARD:
    if "tug1" not in nas:
        nas["tug1"] = {}
    nas["tug1"][OUTBOARD] = nastran.rddtipch(pth + "outboard.pch")

    if "extse" not in nas:
        nas["extse"] = {}
    nas["extse"][OUTBOARD] = nastran.op4.read(pth + "outboard.op4")

    nodepairs = [[OUTBOARD, 3, OUTBOARD, 10], [INBOARD, 11, OUTBOARD, 18]]

    reldtm, L, rellabels = cla.relative_displacement_dtm(nas, nodepairs)
    assert np.allclose(L, 300.0 * np.sqrt(2))
    assert rellabels == ["SE101,10 - SE101,3", "SE101,18 - SE0,11"]

    with pytest.raises(ValueError):
        cla.relative_displacement_dtm(nas, [[SC, 11, 0, 11]])

    # notes:
    # - element 66 runs between 3 & 10
    # - element 72 runs between 11 & 18
    # form axial force recovery for 66 & 72:
    tef1 = nastran.rddtipch(pth + "outboard.pch", "tef1")

    # pv = n2p.mkdofpv(tef1, "p", [[66, 7], [72, 7]])[0]
    pv = locate.mat_intersect(tef1, [[66, 7], [72, 7]], keep=2)[0]

    ltm = nas["extse"][OUTBOARD]["mef1"][pv] @ nas["ulvs"][OUTBOARD]

    # convert the axial load ltm to a delta-x ltm:
    # - axial load is positive for tension (grid b moves away)
    E = 6.894e7
    A = 12.566
    ltm *= L[:, None] / (E * A)

    # add the above items to the data recovery:
    drdefs = cla.DR_Def({"se": 0})

    @cla.DR_Def.addcat
    def _():
        name = "reldisp1"
        desc = "Relative Displacements"
        units = "mm"
        labels = rellabels
        drms = {name: reldtm}
        drfunc = f"Vars[se]['{name}'] @ sol.d"
        histpv = "all"
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "reldisp2"
        desc = "Relative Displacements"
        units = "mm"
        labels = ["LTM: OB 3 - OB 10", "LTM: IN 10 - OB 18"]
        drms = {name: ltm}
        drfunc = f"Vars[se]['{name}'] @ sol.d"
        histpv = "all"
        drdefs.add(**locals())

    # prepare spacecraft data recovery matrices
    DR = cla.DR_Event()
    DR.add(nas, drdefs)

    # initialize results (ext, mnc, mxc for all drms)
    results = DR.prepare_results("micro space station", event)

    # set rfmodes:
    rfmodes = nas["rfmodes"][0]

    # setup modal mass, damping and stiffness
    m = None  # None means identity
    k = nas["lambda"][0]
    k[: nas["nrb"]] = 0.0
    b = 2 * 0.02 * np.sqrt(k)
    mbk = (m, b, k)

    # load in forcing functions:
    mat = pth + "toes/toes_ffns.mat"
    toes = matlab.loadmat(mat, squeeze_me=True, struct_as_record=False)
    toes["ffns"] = toes["ffns"][:1, ::2]
    toes["sr"] = toes["sr"] / 2
    toes["t"] = toes["t"][::2]

    # form force transform:
    T = n2p.formdrm(nas, 0, [[8, 12], [24, 13]])[0].T

    # do pre-calcs and loop over all cases:
    ts = ode.SolveUnc(*mbk, 1 / toes["sr"], rf=rfmodes)
    LC = toes["ffns"].shape[0]
    t = toes["t"]
    for j, force in enumerate(toes["ffns"]):
        # print('Running {} case {}'.format(event, j + 1))
        genforce = T @ ([[1], [0.1], [1], [0.1]] * force[None, :])
        # solve equations of motion
        sol = ts.tsolve(genforce, static_ic=1)
        sol.t = t
        sol = DR.apply_uf(sol, *mbk, nas["nrb"], rfmodes)
        caseid = "{} {:2d}".format(event, j + 1)
        results.time_data_recovery(sol, nas["nrb"], caseid, DR, LC, j)

    h1 = results["reldisp1"].hist
    h2 = results["reldisp2"].hist

    assert abs(h1 - h2).max() / abs(h1).max() < 0.001


def test_set_dr_order():
    drdefs0 = cla.DR_Def()
    for name, nrows in (("atm0", 12), ("ltm0", 30), ("dtm0", 9)):
        drdefs0.add(name=name, labels=nrows, drfunc="no-func")

    drdefs1 = cla.DR_Def()
    for name, nrows in (("atm1", 12), ("ltm1", 30), ("dtm1", 9)):
        drdefs1.add(name=name, labels=nrows, drfunc="no-func")

    DR = cla.DR_Event()
    DR.add(None, drdefs0)
    DR.add(None, drdefs1)

    # order must be as defined:
    r = repr(DR)
    assert "['atm0', 'ltm0', 'dtm0', 'atm1', 'ltm1', 'dtm1']" in r

    # ensure that ltm1, dtm0, atm1 are recovered in that order, the
    # others are okay in current order:

    # case 1: put ltm1, dtm0, atm1 first:
    DR.set_dr_order(("ltm1", "dtm0", "atm1"), where="first")
    r = repr(DR)
    assert "['ltm1', 'dtm0', 'atm1', 'atm0', 'ltm0', 'dtm1']" in r

    # case 2: put ltm1, dtm0, atm1 last:
    DR.set_dr_order(("ltm1", "dtm0", "atm1"), where="last")
    r = repr(DR)
    assert "['atm0', 'ltm0', 'dtm1', 'ltm1', 'dtm0', 'atm1']" in r

    # check for proper errors:
    with pytest.raises(ValueError):
        DR.set_dr_order(("scatm",), "first")
    with pytest.raises(ValueError):
        DR.set_dr_order(("atm0",), "bad where")

    # check a couple corner cases:
    DR.set_dr_order([], where="first")
    r = repr(DR)
    assert "['atm0', 'ltm0', 'dtm1', 'ltm1', 'dtm0', 'atm1']" in r

    DR.set_dr_order([], where="last")
    r = repr(DR)
    assert "['atm0', 'ltm0', 'dtm1', 'ltm1', 'dtm0', 'atm1']" in r

    DR.set_dr_order(["ltm0"], where="last")
    r = repr(DR)
    assert "['atm0', 'dtm1', 'ltm1', 'dtm0', 'atm1', 'ltm0']" in r

    DR.set_dr_order(["atm1"], where="first")
    r = repr(DR)
    assert "['atm1', 'atm0', 'dtm1', 'ltm1', 'dtm0', 'ltm0']" in r


def get_dr_defs(se, cats, drms_, nondrms_):
    # setup CLA parameters:
    # define some defaults for data recovery:
    defaults = dict(
        se=se, uf_reds=(1, 1, 1.25, 1), srsfrq=np.arange(0.1, 50.1, 0.1), srsQs=(10, 33)
    )

    drdefs = cla.DR_Def(defaults)

    for i in range(len(cats)):

        @cla.DR_Def.addcat
        def _():
            name = cats[i]
            desc = "description"
            labels = 12
            drms = {drms_[i]: i}
            nondrms = {nondrms_[i]: 10 + i}
            drfunc = f"Vars[se]['{name}'] @ sol.a"
            if not name.startswith("ltm"):
                srsopts = dict(eqsine=1, ic="steady")
            drdefs.add(**locals())

    return drdefs


def test_dr_def_merge():
    names = ["atm", "ltm", "dtm"]
    cats = ["atm0", "ltm0", "dtm0", "atm1", "ltm1", "dtm1", "atm2", "ltm2", "dtm2"]
    drms = [name + "_d" for name in cats]
    nondrms = [name + "_n" for name in cats]

    drdefs = [
        get_dr_defs(
            se,
            cats[3 * i : 3 * i + 3],
            drms[3 * i : 3 * i + 3],
            nondrms[3 * i : 3 * i + 3],
        )
        for i, se in zip(range(3), (101, 102, 102))
    ]

    drdefs_merge = cla.DR_Def.merge(drdefs[0], drdefs[1], drdefs[2])
    drdefs_add = drdefs[0] + drdefs[1] + drdefs[2]

    for _ in range(2):
        # se 101 has the 0 drms and se 102 has the 1 & 2 drms:
        for drdefs_ in (drdefs_merge, drdefs_add):
            for se, addon in ((101, [0]), (102, [1, 2])):
                cur_drms = drdefs_["_vars"].drms[se]
                cur_nondrms = drdefs_["_vars"].nondrms[se]
                for i, name in enumerate(names):
                    for j in addon:
                        cur_name = name + f"{j}_d"
                        assert cur_name in cur_drms
                        assert cur_drms[cur_name] == i

                        cur_name = name + f"{j}_n"
                        assert cur_name in cur_nondrms
                        assert cur_nondrms[cur_name] == 10 + i

        assert list(drdefs_merge) == ["_vars", *cats]
        assert list(drdefs_add) == ["_vars", *cats]

        try:
            pname = "test_drdefs_pickle.p"
            cla.save(pname, drdefs_add)
            drdefs_add = cla.load(pname)
        finally:
            # pass
            os.remove(pname)

    drdefs_add = drdefs[2] + drdefs[0] + drdefs[1]
    new_cats = [name + str(i) for i in (2, 0, 1) for name in ["atm", "ltm", "dtm"]]
    assert list(drdefs_add) == ["_vars", *new_cats]

    assert list(cla.DR_Def.merge(drdefs[1])) == ["_vars", "atm1", "ltm1", "dtm1"]

    # check for some errors:
    cats2 = ["atm0", "ltm0", "dtm0", "atm0", "ltm1", "dtm1", "atm2", "ltm1", "dtm2"]
    drms2 = [name + "_d" for name in cats2]
    nondrms2 = [name + "_n" for name in cats2]

    drdefs = [
        get_dr_defs(
            se,
            cats2[3 * i : 3 * i + 3],
            drms[3 * i : 3 * i + 3],
            nondrms[3 * i : 3 * i + 3],
        )
        for i, se in zip(range(3), (101, 102, 102))
    ]

    # should raise ValueError for duplicate categories:
    with pytest.raises(
        ValueError, match=r"there were duplicate categories:\n.*atm0.*ltm1"
    ):
        cla.DR_Def.merge(
            drdefs[0],
            drdefs[1],
            drdefs[2],
        )

    drdefs = [
        get_dr_defs(
            se,
            cats[3 * i : 3 * i + 3],
            drms2[3 * i : 3 * i + 3],
            nondrms[3 * i : 3 * i + 3],
        )
        for i, se in zip(range(3), (101, 102, 102))
    ]

    # should raise ValueError for duplicate drms:
    with pytest.raises(
        ValueError,
        match=(r'there were duplicate "drms" names. By SE:\n102:.*ltm1_d.*'),
    ):
        cla.DR_Def.merge(
            drdefs[0],
            drdefs[1],
            drdefs[2],
        )

    drdefs = [
        get_dr_defs(
            se,
            cats[3 * i : 3 * i + 3],
            drms[3 * i : 3 * i + 3],
            nondrms2[3 * i : 3 * i + 3],
        )
        for i, se in zip(range(3), (101, 102, 102))
    ]

    # should raise ValueError for duplicate nondrms:
    with pytest.raises(
        ValueError,
        match=(r'there were duplicate "nondrms" names. By SE:\n102:.*ltm1_n.*'),
    ):
        cla.DR_Def.merge(
            drdefs[0],
            drdefs[1],
            drdefs[2],
        )


def test_drdef_importer():
    # setup CLA parameters:
    # define some defaults for data recovery:
    drdefs = cla.DR_Def()

    @cla.DR_Def.addcat
    def _():
        name = "atm"
        labels = 12
        drfile = "drfuncs.py"
        drfunc = "atm"
        with warnings.catch_warnings(record=True) as w:
            drdefs.add(**locals())
            assert len(w) == 0

    @cla.DR_Def.addcat
    def _():
        name = "ltm"
        labels = 12
        drfile = "ltmdrfuncs/drfuncs.py"
        drfunc = "ltm"
        with warnings.catch_warnings(record=True) as w:
            drdefs.add(**locals())
            assert len(w) == 0


def test_dr_def_amend():
    # define some defaults for data recovery:
    defaults = dict(
        se=90, uf_reds=(1, 1, 1.25, 1), srsfrq=np.arange(0.1, 50.1, 0.1), srsQs=(10, 33)
    )

    drdefs = cla.DR_Def(defaults)

    atm = np.arange(12 * 12).reshape(12, -1)

    @cla.DR_Def.addcat
    def _():
        name = "scatm"
        desc = "Outboard Internal Accelerations"
        units = "mm/sec^2, rad/sec^2"
        labels = [f"Row {i+1}" for i in range(12)]
        drms = {name: atm}
        drfunc = f"Vars[se]['{name}'] @ sol.a"
        drdefs.add(**locals())

    # PP(drdefs["scatm"])

    cat = drdefs["scatm"]
    assert cat.srsfrq is None
    assert cat.srsQs is None
    assert cat.srslabels is None
    assert cat.srsopts is None
    assert cat.srsunits is None
    assert cat.srspv is None
    assert cat.filterval == 1e-6
    assert (drdefs["_vars"].drms[90]["scatm"] == atm).all()

    drdefs.amend(name="scatm", srspv=[0, 4], filterval=1.2)

    # PP(drdefs["scatm"])

    cat = drdefs["scatm"]
    assert (defaults["srsfrq"] == cat.srsfrq).all()
    assert defaults["srsQs"] == cat.srsQs
    assert cat.srslabels == ["Row 1", "Row 5"]
    assert cat.srsopts == {}
    assert cat.srsunits == cat.units
    assert (cat.srspv == [0, 4]).all()
    assert cat.filterval == 1.2

    with pytest.raises(ValueError):
        drdefs.amend(**dict(name="scatm", drms={"scatm": 1}))

    drdefs.amend(name="scatm", drms={"scatm": 1}, overwrite_drms=True)

    assert drdefs["_vars"].drms[90]["scatm"] == 1


def grounded_mass_spring_system():
    r"""
                |--> x1       |--> x2        |--> x3


             |----|    k1   |----|    k2   |----|    k4   |/
          f  |    |--\/\/\--|    |--\/\/\--|    |--\/\/\--|/
        ====>| m1 |         | m2 |         | m3 |         |/
             |    |---| |---|    |---| |---|    |---| |---|/
             |----|    c1   |----|    c2   |----|    c4   |/
               |                             |
               |             k3              |
               |-----------\/\/\-------------|
               |                             |
               |------------| |--------------|
                             c3

    m1 = 2 kg
    m2 = 4 kg
    m3 = 6 kg

    k1 = 12000 N/m
    k2 = 16000 N/m
    k3 = 10000 N/m
    k4 = 15000 N/m

    c1 = 70 N s/m
    c2 = 75 N s/m
    c3 = 30 N s/m
    c4 = 50 N s/m

    h = 0.001
    t = np.arange(0, 1.0, h)
    f = np.zeros((3, len(t)))
    f[0, 20:250] = 10.0  # N

    """
    m1 = 2.0
    m2 = 4.0
    m3 = 6.0
    k1 = 12000.0
    k2 = 16000.0
    k3 = 10000.0
    k4 = 15000.0
    c1 = 70.0
    c2 = 75.0
    c3 = 30.0
    c4 = 50.0
    mass = np.diag([m1, m2, m3])
    stiff = np.array(
        [[k1 + k3, -k1, -k3], [-k1, k1 + k2, -k2], [-k3, -k2, k2 + k3 + k4]]
    )
    damp = np.array(
        [[c1 + c3, -c1, -c3], [-c1, c1 + c2, -c2], [-c3, -c2, c2 + c3 + c4]]
    )

    # force will be positive for tension

    # drm for subtracting 1 from 2, 2 from 3, 1 from 3:
    sub = np.array([[-1.0, 1.0, 0], [0.0, -1.0, 1.0], [-1.0, 0, 1.0], [0.0, 0.0, -1.0]])

    # ltm for forces in springs:
    #    fs = ltm @ x
    ltm = [[k1], [k2], [k3], [k4]] * sub

    # form the mode-acce versions:
    #    fs = ltmf @ f + ltma @ xdd + ltmv @ xd
    # ltmf = ltm @ inv(k)
    # ltma = -ltm @ inv(k) @ m
    # ltmv = -ltm @ inv(k) @ c
    ltmf = la.solve(stiff.T, ltm.T).T
    ltma = -ltmf @ mass
    ltmv = -ltmf @ damp

    drms1 = {"ltm": ltm, "ltmf": ltmf, "ltma": ltma, "ltmv": ltmv}

    # define some defaults for data recovery:
    uf_reds = (1, 1, 1, 1)
    defaults = dict(se=0, uf_reds=uf_reds)
    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = "fs_md"
        desc = "Spring Forces - Mode Displacement"
        units = "N"
        labels = [f"Spring {i}" for i in range(4)]
        # force will be positive for tension
        drms = drms1
        drfunc = "Vars[se]['ltm'] @ sol.d"
        histpv = "all"
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = "fs_ma"
        desc = "Spring Forces - Mode Acceleration"
        units = "N"
        labels = [f"Spring {i}" for i in range(4)]
        # force will be positive for tension
        drfunc = """(Vars[se]['ltmf'] @ sol.pg +
                     Vars[se]['ltma'] @ sol.a +
                     Vars[se]['ltmv'] @ sol.v)"""
        histpv = "all"
        drdefs.add(**locals())

    # prepare spacecraft data recovery matrices
    DR = cla.DR_Event()
    DR.add(None, drdefs)

    return mass, damp, stiff, drms1, uf_reds, defaults, DR


def test_solvepsd_ltmf():
    mass, damp, stiff, drms, uf_reds, defaults, DR = grounded_mass_spring_system()

    # define forces:
    h = 0.001
    t = np.arange(0, 1.0, h)
    f = np.zeros((3, len(t)))
    f[0, 20:250] = 10.0

    # setup solver:
    # ts = ode.SolveExp2(mass, damp, stiff, h)
    ts = ode.SolveUnc(mass, damp, stiff, h, pre_eig=True)
    sol = ts.tsolve(f)
    sol.pg = f
    sol2 = {uf_reds: sol}

    # initialize results (ext, mnc, mxc for all drms)
    event = "Case 1"
    results = DR.prepare_results("Spring & Damper Forces", event)

    # perform data recovery:
    results.time_data_recovery(sol2, None, event, DR, 1, 0)

    # ensure the two methods are equivalent:
    assert np.allclose(results["fs_md"].ext, results["fs_ma"].ext)

    # --------------------------------------------
    # now run a PSD style event with two psd's to ensure that solvepsd
    # handles the forces (sol.pg) properly:
    # >>> lam, phi = la.eigh(stiff, mass)
    # >>> np.sqrt(lam) / 2 / np.pi
    # array([  5.21246336,  15.89585871,  18.68656157])

    freq = np.arange(0.1, 30.0, 0.1)
    n = freq.shape[0]

    # - fpsd[0] on dof 2
    # - fpsd[1] on dof 1
    fpv = np.array([2, 1])
    fpsd = np.zeros((2, n))
    fpsd[0] = np.interp(freq, [2.0, 12.0, 20.0, 25.0], [0.0, 10.0, 10.0, 0.0])
    fpsd[1] = np.interp(freq, [2.0, 12.0, 20.0, 25.0], [0.0, 12.0, 12.0, 0.0])

    # "ltmf" recovery matrix must match PSD force (order and size)
    DR.Vars[0]["ltmf_keep"] = DR.Vars[0]["ltmf"].copy()
    DR.Vars[0]["ltmf"] = DR.Vars[0]["ltmf_keep"][:, fpv]

    t_frc = np.eye(stiff.shape[0])[fpv].T
    caseid = "PSDTest"
    nas = {"nrb": 0}
    results = DR.prepare_results("Spring & Damper Forces", event)
    results.solvepsd(nas, caseid, DR, ts, fpsd, t_frc, freq, use_apply_uf=True)
    results.psd_data_recovery(caseid, DR, 1, 0)

    # --------------------------------------------
    # now run the two psds separately and rss the resulting rms values
    # ... should match (taking sqrt of the sum of PSDs is the same as
    # rss'ing the rms values)
    res_separate = cla.DR_Results()
    for i, fpsdi in enumerate(fpsd):
        caseid = f"PSD Case {i}"
        res_separate[caseid] = DR.prepare_results("Spring & Damper Forces", event)
        DR.Vars[0]["ltmf"] = DR.Vars[0]["ltmf_keep"][:, fpv[[i]]]

        t_frc = np.eye(stiff.shape[0])[[fpv[i]]].T
        res_separate[caseid].solvepsd(
            nas, caseid, DR, ts, fpsdi[None, :], t_frc, freq, use_apply_uf=False
        )
        res_separate[caseid].psd_data_recovery(caseid, DR, 1, 0)

    # ensure the two methods AND approaches are equivalent:
    rss_md = np.sqrt(
        res_separate["PSD Case 0"]["fs_md"].ext ** 2
        + res_separate["PSD Case 1"]["fs_md"].ext ** 2
    )
    rss_md[:, 1] *= -1
    assert np.allclose(results["fs_md"].ext, rss_md)

    rss_ma = np.sqrt(
        res_separate["PSD Case 0"]["fs_ma"].ext ** 2
        + res_separate["PSD Case 1"]["fs_ma"].ext ** 2
    )
    rss_ma[:, 1] *= -1
    assert np.allclose(results["fs_ma"].ext, rss_ma)

    assert np.allclose(rss_md, rss_ma)

    # --------------------------------------------
    # rerun with a zero psd force for more testing:
    fpsd = np.zeros((3, n))
    # should give same answer as above:
    fpsd[fpv[0]] = np.interp(freq, [2.0, 12.0, 20.0, 25.0], [0.0, 10.0, 10.0, 0.0])
    fpsd[fpv[1]] = np.interp(freq, [2.0, 12.0, 20.0, 25.0], [0.0, 12.0, 12.0, 0.0])

    # "ltmf" recovery matrix must match PSD force (order and size)
    DR.Vars[0]["ltmf"] = DR.Vars[0]["ltmf_keep"]

    t_frc = np.eye(stiff.shape[0])
    caseid = "PSDTest"

    results2 = DR.prepare_results("Spring & Damper Forces", event)
    with pytest.warns(RuntimeWarning, match="There are 1 zero forces"):
        results2.solvepsd(nas, caseid, DR, ts, fpsd, t_frc, freq)

    results2.psd_data_recovery(caseid, DR, 1, 0)

    assert np.allclose(results["fs_md"].ext, results2["fs_md"].ext)
    assert np.allclose(results["fs_md"].ext, results2["fs_ma"].ext)

    # as one last test, tell it to trim forces (mode-acce won't work
    # because of sizes, so delete that one first)
    del DR.Info["fs_ma"]

    results2 = DR.prepare_results("Spring & Damper Forces", event)
    results2.solvepsd(nas, caseid, DR, ts, fpsd, t_frc, freq, allow_force_trimming=True)
    results2.psd_data_recovery(caseid, DR, 1, 0)

    assert np.allclose(results["fs_md"].ext, results2["fs_md"].ext)


def test_solvepsd_rf_disp_only():
    mass, damp, stiff, drms, uf_reds, defaults, DR = grounded_mass_spring_system()
    freq = np.arange(0.1, 30.0, 0.1)
    n = freq.shape[0]

    # - fpsd[0] on dof 2
    # - fpsd[1] on dof 1
    fpv = np.array([2, 1])
    fpsd = np.zeros((2, n))
    fpsd[0] = np.interp(freq, [2.0, 12.0, 20.0, 25.0], [0.0, 10.0, 10.0, 0.0])
    fpsd[1] = np.interp(freq, [2.0, 12.0, 20.0, 25.0], [0.0, 12.0, 12.0, 0.0])

    # "ltmf" recovery matrix must match PSD force (order and size)
    DR.Vars[0]["ltmf_keep"] = DR.Vars[0]["ltmf"].copy()
    DR.Vars[0]["ltmf"] = DR.Vars[0]["ltmf_keep"][:, fpv]

    t_frc = np.eye(stiff.shape[0])[fpv].T
    caseid = "PSDTest"
    nas = {"nrb": 0}
    event = "Case 1"
    fs = ode.SolveUnc(mass, damp, stiff, pre_eig=True, rf=2)

    results = DR.prepare_results("Spring & Damper Forces", event)
    results.solvepsd(nas, caseid, DR, fs, fpsd, t_frc, freq)
    results.psd_data_recovery(caseid, DR, 1, 0)

    results_rfd = DR.prepare_results("Spring & Damper Forces", event)
    results_rfd.solvepsd(nas, caseid, DR, fs, fpsd, t_frc, freq, rf_disp_only=True)
    results_rfd.psd_data_recovery(caseid, DR, 1, 0)

    # We're not actually checking for correct values here, just that
    # the rf_disp_only argument was used. The values computed with
    # rf_disp_only are checked in test_ode.py in a few places.
    assert np.allclose(results["fs_md"].ext, results_rfd["fs_md"].ext)
    assert not np.allclose(results["fs_ma"].ext, results_rfd["fs_ma"].ext)


def comp_all_na():
    # make up some "external source" CLA results:
    mission = "Rocket / Spacecraft VLC"
    event = "Liftoff"
    t = np.arange(200) / 200
    nrows = 12
    resp = np.random.randn(nrows, len(t))
    mxmn = cla.maxmin(resp, t)
    ext_results = mxmn.ext
    ext_results[:, 1] = 0.0  # set min to 0.0

    drdefs = cla.DR_Def()

    @cla.DR_Def.addcat
    def _():
        name = "ATM"
        desc = "S/C Internal Accelerations"
        units = "m/sec^2, rad/sec^2"
        labels = [f"{name} Row {i+1:6d}" for i in range(nrows)]
        drfunc = "no-func"
        drdefs.add(**locals())

    # prepare results data structure:
    DR = cla.DR_Event()
    DR.add(None, drdefs)
    results = DR.prepare_results(mission, event)
    results.add_maxmin("ATM", ext_results, event)

    # results 2:
    ext2 = ext_results + 0.1
    ext2[:, 1] = 0.0

    results.rptpct({"ATM": ext2}, dohistogram=False, domagpct=False)


def test_rptpct_all_na():
    try:
        if os.path.exists("temp_cla2"):
            shutil.rmtree("./temp_cla2", ignore_errors=True)
        os.mkdir("temp_cla2")
        with cd("temp_cla2"):
            comp_all_na()
    finally:
        # pass
        shutil.rmtree("./temp_cla2", ignore_errors=True)


def test_delete_data():
    res = cla.DR_Results()
    res["NonBase"] = cla.DR_Results()
    res["NonBase"]["Base1"] = cla.DR_Results()
    res["NonBase"]["Base1"]["ATM"] = SimpleNamespace()
    res["NonBase"]["Base1"]["LTM"] = SimpleNamespace()
    res["Base2"] = cla.DR_Results()
    res["Base2"]["ATM"] = SimpleNamespace()
    res["Base2"]["LTM"] = SimpleNamespace()
    res["empty"] = cla.DR_Results()  # for testing

    # set some values to test delete_data:
    res["NonBase"]["Base1"]["ATM"].ab = "base1_atm_2"
    res["NonBase"]["Base1"]["LTM"].ba = SimpleNamespace(cc="base1_ltm_3")

    res["Base2"]["ATM"].ab = "base2_atm_1"
    res["Base2"]["LTM"].ba = SimpleNamespace(cc="base2_ltm_2")

    res1 = copy.deepcopy(res)
    res1.delete_data(attributes="ab")
    assert not hasattr(res1["NonBase"]["Base1"]["ATM"], "ab")
    assert hasattr(res1["NonBase"]["Base1"]["LTM"], "ba")
    assert not hasattr(res1["Base2"]["ATM"], "ab")
    assert hasattr(res1["Base2"]["LTM"], "ba")

    res1 = copy.deepcopy(res)
    res1.delete_data(attributes=("ab", "ba"), pathfunc=lambda path: len(path) > 2)
    assert not hasattr(res1["NonBase"]["Base1"]["ATM"], "ab")
    assert not hasattr(res1["NonBase"]["Base1"]["LTM"], "ba")
    assert hasattr(res1["Base2"]["ATM"], "ab")
    assert hasattr(res1["Base2"]["LTM"], "ba")

    res1 = copy.deepcopy(res)
    res1.delete_data(attributes=set(["ab", "ba"]), pathfunc=lambda path: len(path) < 3)
    assert hasattr(res1["NonBase"]["Base1"]["ATM"], "ab")
    assert hasattr(res1["NonBase"]["Base1"]["LTM"], "ba")
    assert not hasattr(res1["Base2"]["ATM"], "ab")
    assert not hasattr(res1["Base2"]["LTM"], "ba")

    res1 = copy.deepcopy(res)
    res1.delete_data(attributes="ba.cc", pathfunc=lambda path: len(path) > 2)
    assert hasattr(res1["NonBase"]["Base1"]["ATM"], "ab")
    assert hasattr(res1["NonBase"]["Base1"]["LTM"], "ba")
    assert not hasattr(res1["NonBase"]["Base1"]["LTM"].ba, "cc")
    assert hasattr(res1["Base2"]["ATM"], "ab")
    assert hasattr(res1["Base2"]["LTM"], "ba")
    assert hasattr(res1["Base2"]["LTM"].ba, "cc")
