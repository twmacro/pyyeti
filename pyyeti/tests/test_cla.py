import os
from subprocess import run

import sys
import itertools
import shutil
import inspect
import re
import warnings
from types import SimpleNamespace
from glob import glob
from io import StringIO
import numpy as np
from scipy.io import matlab
from scipy import linalg
import scipy.interpolate as interp
from nose.tools import *
import matplotlib as mpl
mpl.interactive(0)
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyyeti import cla, op2, n2p, nastran, op4, cb, ode, stats


def ATM():
    pass


def LTM():
    pass


def _get_labels0(rows, name):
    return ['{} Row  {:6d}'.format(name, i+1)
            for i in range(rows[name])]


def _get_labels1(rows, name):
    return ['{} Row  {:6d}'.format(name, i+1)
            for i in range(0, 2*rows[name], 2)]


def _get_labels2(rows, name):
    # word = itertools.cycle(['Item', 'Row', 'Id'])
    word = itertools.cycle(['Item'])
    return ['{} {:4} {:6d}'.format(name, w, i+1)
            for w, i in zip(word, range(rows[name]))]


def _get_minmax(drm, eventnumber, cyclenumber):
    ext = {'LTM': np.array([[ 2.72481567, -2.89079134],
                            [ 2.25786   , -2.88626652],
                            [ 3.02440516, -2.80780524],
                            [ 2.53286749, -3.40485914],
                            [ 2.28348523, -3.53863051],
                            [ 3.82729032, -2.61684849],
                            [ 3.35335482, -2.60736874],
                            [ 2.86110496, -2.56407221],
                            [ 2.14606204, -2.55517801],
                            [ 2.54651205, -3.01547524],
                            [ 2.31767096, -2.47119804],
                            [ 2.18782636, -2.50638871],
                            [ 2.64771791, -2.90906464],
                            [ 3.87022179, -2.8447158 ],
                            [ 3.13803533, -2.96040968],
                            [ 2.19274763, -2.1466145 ],
                            [ 2.35224123, -2.2461871 ],
                            [ 2.37220776, -2.37927315],
                            [ 2.70107313, -2.55167378],
                            [ 2.43641342, -2.53973724],
                            [ 3.19988018, -2.27876702],
                            [ 3.26828777, -2.99453974],
                            [ 2.63198951, -2.54630802],
                            [ 2.90049869, -2.70155806],
                            [ 2.06576135, -3.01145668],
                            [ 2.50973189, -2.57272325],
                            [ 2.5291785 , -2.87873901],
                            [ 2.5534714 , -2.40617426],
                            [ 2.75582   , -1.96866783]]),
           'ATM': np.array([[ 4.15547381, -2.60250299],
                            [ 3.30988464, -2.95335224],
                            [ 2.52136841, -2.15885709],
                            [ 2.71879804, -2.4792219 ],
                            [ 2.40233936, -3.12799065],
                            [ 3.28859809, -2.962606  ],
                            [ 2.11816761, -2.4080584 ],
                            [ 3.15167173, -3.01657837],
                            [ 2.41730971, -2.533919  ],
                            [ 3.29167757, -2.13105438],
                            [ 2.27611906, -3.46433397],
                            [ 2.4100566 , -3.3943848 ],
                            [ 2.63918211, -2.68209126],
                            [ 2.55784324, -2.29710417],
                            [ 3.05160678, -2.46384131],
                            [ 2.61573592, -2.30890182],
                            [ 2.70690245, -2.69287401],
                            [ 1.99385389, -2.36857087],
                            [ 2.27205095, -2.89722068],
                            [ 2.65968896, -3.38645715],
                            [ 2.54024118, -2.35912789],
                            [ 2.62673628, -3.07818987],
                            [ 2.49945891, -2.56637166],
                            [ 2.95143805, -2.34052105],
                            [ 3.35468889, -2.43842187],
                            [ 2.23664468, -2.7788623 ],
                            [ 3.02078059, -2.84829591],
                            [ 2.69653637, -2.16359541],
                            [ 3.18788459, -2.56054783],
                            [ 3.03810484, -2.23800354],
                            [ 2.60597387, -2.57964111],
                            [ 2.6155941 , -2.50413382],
                            [ 2.70912049, -2.87191784],
                            [ 2.58207062, -2.9524317 ]])}
    addon = 0.2
    curext = ext[drm].copy()
    if eventnumber == 1:
        curext[::3] = curext[::3] - addon
        curext[1::3] = curext[1::3] + 2*addon
        curext[2::3] = curext[2::3] - addon
    elif eventnumber == 2:
        curext[::3] = curext[::3] - 2*addon
        curext[1::3] = curext[1::3] + addon
        curext[2::3] = curext[2::3] + addon
    addon = 0.03 * cyclenumber
    curext[::4] = curext[::4] - addon
    curext[1::4] = curext[1::4] + 2*addon
    curext[2::4] = curext[2::4] - 2*addon
    curext[3::4] = curext[3::4] + addon
    return curext


def get_fake_cla_results(ext_name, _get_labels, cyclenumber):
    # make up some CLA results:
    events = ('Liftoff', 'Transonics', 'MECO')
    rows = {'ATM': 34, 'LTM': 29}
    ext_results = {i: {} for i in rows}
    for i, event in enumerate(events):
        for drm, nrows in rows.items():
            ext_results[drm][event] = _get_minmax(
                drm, i, cyclenumber)

    # setup CLA parameters:
    mission = "Rocket / Spacecraft CLA"
    duf = 1.2
    suf = 1.0

    # defaults for data recovery
    defaults = dict(
        se = 0,
        uf_reds = (1, 1, duf, 1),
        drfile = '.',
        )

    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = 'ATM'
        desc = 'S/C Internal Accelerations'
        units = 'm/sec^2, rad/sec^2'
        labels = _get_labels(rows, name)
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = 'LTM'
        desc = 'S/C Internal Loads'
        units = 'N, N-m'
        labels = _get_labels(rows, name)
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
            results[event].add_maxmin(
                drm, ext_results[drm][event], event)

    # Done with setup; now we can use the standard cla tools:
    results.form_extreme(ext_name)

    # Test some features of add_maxmin:
    res2 = cla.DR_Results()
    for event in events:
        res2[event] = DR.prepare_results(mission, event)

    ext = ext_results['ATM']['Liftoff']
    r = ext.shape[0]
    maxcase = ['LO {}'.format(i+1) for i in range(r)]
    mincase = 'LO Min'
    res2['Liftoff'].add_maxmin('ATM', ext, maxcase, mincase,
                               ext, 'Time')
    assert res2['Liftoff']['ATM'].maxcase == maxcase
    assert res2['Liftoff']['ATM'].mincase == r*[mincase]

    res2['Liftoff'].add_maxmin('LTM', ext, maxcase, r*[mincase],
                               ext, 'Time')
    assert res2['Liftoff']['LTM'].maxcase == maxcase
    assert res2['Liftoff']['LTM'].mincase == r*[mincase]

    res2 = cla.DR_Results()
    for event in events:
        res2[event] = DR.prepare_results(mission, event)
        for drm in rows:
            res2[event].add_maxmin(
                drm, ext_results[drm][event], event,
                domain=event+drm)
    res2.form_extreme(ext_name)

    assert results['extreme']['ATM'].domain is None
    assert res2['extreme']['ATM'].domain == 'X-Value'

    res2 = cla.DR_Results()
    for event in events:
        res2[event] = DR.prepare_results(mission, event)
        for drm in rows:
            res2[event].add_maxmin(
                drm, ext_results[drm][event], event,
                domain='Time')
    res2.form_extreme(ext_name)
    assert res2['extreme']['ATM'].domain == 'Time'

    return results


def compresults(f1, f2):
    # - some values in "f2" were spot checked, not all
    # - this routine simply compares lines, skipping the date line
    with open(f1) as F1, open(f2) as F2:
        for l1, l2 in zip(F1, F2):
            if not l1.startswith('Date:'):
                assert l1 == l2


def test_form_extreme():
    results = cla.DR_Results()
    results.merge(
        (get_fake_cla_results('FLAC', _get_labels0, 0),
         get_fake_cla_results('VLC', _get_labels1, 1),
         get_fake_cla_results('PostVLC', _get_labels2, 2)),
        {'FLAC': 'FDLC',
         'PostVLC': 'VLC2'})

    # Add non-unique label to mess up the expanding of results
    # for form_extreme:
    events = ('Liftoff', 'Transonics', 'MECO')
    lbl_keep = results['FDLC']['Liftoff']['ATM'].drminfo.labels[1]
    for e in events:
        lbls = results['FDLC'][e]['ATM'].drminfo.labels
        lbls[1] = lbls[0]
    assert_raises(ValueError, results.form_extreme)

    # change it back to being correct:
    for e in events:
        lbls = results['FDLC'][e]['ATM'].drminfo.labels
        lbls[1] = lbl_keep

    results.form_extreme()
    assert repr(results).startswith('<pyyeti.cla.DR_Results object')
    assert str(results) == ('DR_Results with 4 categories: '
                            "['FDLC', 'VLC', 'VLC2', 'extreme']")
    try:
        # results['extreme'].rpttab(direc='./temp_tab', excel='results')
        results['extreme'].rpttab(direc='./temp_tab', excel=True)
        assert os.path.exists('./temp_tab/ATM.xlsx')
        assert os.path.exists('./temp_tab/LTM.xlsx')

        results['extreme'].rpttab(direc='./temp_tab')
        results['FDLC']['extreme'].rpttab(direc='./temp_fdlc_tab')
        # check results:
        compresults('./temp_tab/ATM.tab',
                    'pyyeti/tests/cla_test_data/fake_cla/ATM.tab')
        compresults('./temp_tab/LTM.tab',
                    'pyyeti/tests/cla_test_data/fake_cla/LTM.tab')
        compresults('./temp_fdlc_tab/ATM.tab',
                    'pyyeti/tests/cla_test_data/fake_cla/ATM_fdlc.tab')
        compresults('./temp_fdlc_tab/LTM.tab',
                    'pyyeti/tests/cla_test_data/fake_cla/LTM_fdlc.tab')
    finally:
        shutil.rmtree('./temp_tab', ignore_errors=True)
        shutil.rmtree('./temp_fdlc_tab', ignore_errors=True)

    # test the different "doappend" options:
    maxcase = results['extreme']['ATM'].maxcase[:]
    mincase = results['extreme']['ATM'].mincase[:]

    # doappend = 1 (keep lowest with higher):
    results.form_extreme(doappend=1)

    def _newcase(s):
        i, j = s.split(',')
        return '{},{},{}'.format(i, j, j)
    mxc = [_newcase(s) for s in maxcase]
    mnc = [_newcase(s) for s in mincase]
    assert mxc == results['extreme']['ATM'].maxcase
    assert mnc == results['extreme']['ATM'].mincase

    # doappend = 3 (keep only lowest):
    results.form_extreme(doappend=3)
    mxc = [s.split(',')[1] for s in maxcase]
    mnc = [s.split(',')[1] for s in mincase]
    assert mxc == results['extreme']['ATM'].maxcase
    assert mnc == results['extreme']['ATM'].mincase

    cases = ['VLC2', 'FDLC', 'VLC']
    results.form_extreme(case_order=cases)
    assert results['extreme']['ATM'].cases == cases


# run a cla:

class cd():
    def __init__(self, newdir):
        self.olddir = os.getcwd()
        self.newdir = newdir

    def __enter__(self):
        os.chdir(self.newdir)

    def __exit__(self, *args):
        os.chdir(self.olddir)

def scatm(sol, nas, Vars, se):
    return Vars[se]['atm'] @ sol.a

def net_ifltm(sol, nas, Vars, se):
    return Vars[se]['net_ifltm'] @ sol.a

def net_ifatm(sol, nas, Vars, se):
    return Vars[se]['net_ifatm'] @ sol.a

def get_xyr():
    # return the xr, yr, and rr indexes for the "cglf" data recovery
    # ... see :func:`cla.DR_Defs.add`
    xr = np.array([1, 3, 6, 8])   # 'x' row(s)
    yr = xr + 1                   # 'y' row(s)
    rr = np.arange(4) + 10        # rss  rows
    return xr, yr, rr

def cglf(sol, nas, Vars, se):
    resp = Vars[se]['cglf'] @ sol.a
    xr, yr, rr = get_xyr()
    resp[rr] = np.sqrt(resp[xr]**2 + resp[yr]**2)
    return resp

def cglf_psd(sol, nas, Vars, se, freq, forcepsd,
             drmres, case, i):
    resp = Vars[se]['cglf'] @ sol.a
    cla.PSD_consistent_rss(
        resp, *get_xyr(), freq, forcepsd,
        drmres, case, i)


def prepare_4_cla(pth):
    se = 101
    uset, coords = nastran.bulk2uset(pth+'outboard.asm')
    dct = op4.read(pth+'outboard.op4')
    maa = dct['mxx']
    kaa = dct['kxx']
    atm = dct['mug1']
    pch = pth+'outboard.pch'

    def getlabels(lbl, id_dof):
        return ['{} {:4d}-{:1d}'.format(lbl, g, i)
                for g, i in id_dof]

    atm_labels = getlabels('Grid', nastran.rddtipch(pch, 'tug1'))
    iflabels = getlabels('Grid', uset[:, :2].astype(int))

    # setup CLA parameters:
    mission = 'Micro Space Station'

    nb = uset.shape[0]
    nq = maa.shape[0] - nb
    bset = np.arange(nb)
    qset = np.arange(nq) + nb
    ref = [600., 150., 150.]
    g = 9806.65
    net = cb.mk_net_drms(maa, kaa, bset, uset=uset, ref=ref, g=g)

    # define some defaults for data recovery:
    defaults = dict(
        se = se,
        uf_reds = (1, 1, 1.25, 1),
        srsfrq = np.arange(.1, 50.1, .1),
        srsQs = (10, 33),
        )

    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = 'scatm'
        desc = 'Outboard Internal Accelerations'
        units = 'mm/sec^2, rad/sec^2'
        labels = atm_labels[:12]
        drms = {'atm': atm[:12]}
        prog = re.compile(' [2]-[1]')
        histpv = [i for i, s in enumerate(atm_labels)
                  if prog.search(s)]
        srspv = histpv
        srsopts = dict(eqsine=1, ic='steady')
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = 'cglf'
        desc = 'S/C CG Load Factors'
        units = 'G'
        labels = net.cglf_labels
        drms = {'cglf': net.cglfa}
        histpv = slice(1)
        srspv = 0
        srsQs = 10
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = 'net_ifatm'
        desc = 'NET S/C Interface Accelerations'
        units = 'g, rad/sec^2'
        labels = net.ifatm_labels[:3]
        drms = {'net_ifatm': net.ifatm[:3]}
        srsopts = dict(eqsine=1, ic='steady')
        histpv = 'all'
        srspv = np.array([True, False, True])
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = 'net_ifltm'
        desc = 'NET I/F Loads'
        units = 'mN, mN-mm'
        labels = net.ifltm_labels[:6]
        drms = {'net_ifltm': net.ifltma[:6]}
        drdefs.add(**locals())

    # add a 0rb version of the NET ifatm:
    drdefs.add_0rb('net_ifatm')

    # make excel summary file for checking:
    df = drdefs.excel_summary()

    # save data to gzipped pickle file:
    sc = dict(mission=mission, drdefs=drdefs)
    cla.save('cla_params.pgz', sc)


def toes(pth):
    os.mkdir('toes')
    with cd('toes'):
        pth = '../' + pth
        event = 'TOES'
        # load data recovery data:
        sc = cla.load('../cla_params.pgz')
        cla.PrintCLAInfo(sc['mission'], event)

        # load nastran data:
        nas = op2.rdnas2cam(pth+'nas2cam')

        # form ulvs for some SEs:
        SC = 101
        n2p.addulvs(nas, SC)

        # prepare spacecraft data recovery matrices
        DR = cla.DR_Event()
        DR.add(nas, sc['drdefs'])

        # initialize results (ext, mnc, mxc for all drms)
        results = DR.prepare_results(sc['mission'], event)

        # set rfmodes:
        rfmodes = nas['rfmodes'][0]

        # setup modal mass, damping and stiffness
        m = None   # None means identity
        k = nas['lambda'][0]
        k[:nas['nrb']] = 0.0
        b = 2*0.02*np.sqrt(k)
        mbk = (m, b, k)

        # load in forcing functions:
        mat = pth + 'toes/toes_ffns.mat'
        toes = matlab.loadmat(mat, squeeze_me=True,
                              struct_as_record=False)
        toes['ffns'] = toes['ffns'][:3, ::2]
        toes['sr'] = toes['sr']/2
        toes['t'] = toes['t'][::2]

        # form force transform:
        T = n2p.formdrm(nas, 0, [[8, 12], [24, 13]])[0].T

        # do pre-calcs and loop over all cases:
        ts = ode.SolveUnc(*mbk, 1/toes['sr'], rf=rfmodes)
        LC = toes['ffns'].shape[0]
        t = toes['t']
        for j, force in enumerate(toes['ffns']):
            print('Running {} case {}'.format(event, j+1))
            genforce = T @ ([[1], [0.1], [1], [0.1]] * force[None, :])
            # solve equations of motion
            sol = ts.tsolve(genforce, static_ic=1)
            sol.t = t
            sol = DR.apply_uf(sol, *mbk, nas['nrb'], rfmodes)
            caseid = '{} {:2d}'.format(event, j+1)
            results.time_data_recovery(sol, nas['nrb'],
                                       caseid, DR, LC, j)

        results.form_stat_ext(stats.ksingle(.99, .90, LC))

        # save results:
        cla.save('results.pgz', results)

        # make some srs plots and tab files:
        results.rptext()
        results.rpttab()
        results.rpttab(excel='toes')
        results.srs_plots()
        results.srs_plots(fmt='png')
        results.resp_plots()


def owlab(pth):
    os.mkdir('owlab')
    with cd('owlab'):
        pth = '../' + pth
        # event name:
        event = 'OWLab'

        # load data recovery data:
        sc = cla.load('../cla_params.pgz')
        cla.PrintCLAInfo(sc['mission'], event)

        # load nastran data:
        nas = op2.rdnas2cam(pth+'nas2cam')

        # form ulvs for some SEs:
        SC = 101
        n2p.addulvs(nas, SC)

        # prepare spacecraft data recovery matrices
        DR = cla.DR_Event()
        DR.add(nas, sc['drdefs'])

        # initialize results (ext, mnc, mxc for all drms)
        results = DR.prepare_results(sc['mission'], event)

        # set rfmodes:
        rfmodes = nas['rfmodes'][0]

        # setup modal mass, damping and stiffness
        m = None   # None means identity
        k = nas['lambda'][0]
        k[:nas['nrb']] = 0.0
        b = 2*0.02*np.sqrt(k)
        mbk = (m, b, k)

        # form force transform:
        T = n2p.formdrm(nas, 0, [[22, 123]])[0]

        # random part:
        freq = cla.freq3_augment(np.arange(25.0, 45.1, 0.5),
                                 nas['lambda'][0])
        #                 freq     x      y      z
        rnd = [np.array([[ 1.0,  90.0, 110.0, 110.0],
                         [30.0,  90.0, 110.0, 110.0],
                         [31.0, 200.0, 400.0, 400.0],
                         [40.0, 200.0, 400.0, 400.0],
                         [41.0,  90.0, 110.0, 110.0],
                         [50.0,  90.0, 110.0, 110.0]]),
               np.array([[ 1.0,  90.0, 110.0, 110.0],
                         [20.0,  90.0, 110.0, 110.0],
                         [21.0, 200.0, 400.0, 400.0],
                         [30.0, 200.0, 400.0, 400.0],
                         [31.0,  90.0, 110.0, 110.0],
                         [50.0,  90.0, 110.0, 110.0]])]

        for j, ff in enumerate(rnd):
            caseid = '{} {:2d}'.format(event, j+1)
            print('Running {} case {}'.format(event, j+1))
            F = interp.interp1d(ff[:, 0], ff[:, 1:].T,
                                axis=1, fill_value=0.0)(freq)
            results.solvepsd(nas, caseid, DR, *mbk, F, T, freq)
            results.psd_data_recovery(caseid, DR, len(rnd), j)

        # save results:
        cla.save('results.pgz', results)
        results.srs_plots(Q=10, direc='srs_cases', showall=True,
                          plot=plt.semilogy)

        # for testing:
        results = DR.prepare_results(sc['mission'], event)
        verbose = True  # for testing
        for j, ff in enumerate(rnd):
            caseid = '{} {:2d}'.format(event, j+1)
            print('Running {} case {}'.format(event, j+1))
            F = interp.interp1d(ff[:, 0], ff[:, 1:].T,
                                axis=1, fill_value=0.0)(freq)
            if j == 0:
                results.solvepsd(nas, caseid, DR, *mbk, F, T, freq,
                                 verbose=verbose)
                verbose = not verbose
                freq = +freq  # make copy
                freq[-1] = 49.7  # to cause error on next 'solvepsd'
                results.psd_data_recovery(caseid, DR, len(rnd), j)
            else:
                assert_raises(ValueError, results.solvepsd, nas,
                              caseid, DR, *mbk, F, T, freq,
                              verbose=verbose)


def alphajoint(sol, nas, Vars, se):
    return Vars[se]['alphadrm'] @ sol.a


def get_drdefs(nas, sc):
    drdefs = cla.DR_Def(sc['drdefs'].defaults)
    @cla.DR_Def.addcat
    def _():
        se = 0
        name = 'alphajoint'
        desc = 'Alpha-Joint Acceleration'
        units = 'mm/sec^2, rad/sec^2'
        labels = ['Alpha-Joint {:2s}'.format(i)
                  for i in 'X,Y,Z,RX,RY,RZ'.split(',')]
        drms = {'alphadrm': n2p.formdrm(nas, 0, 33)[0]}
        srsopts = dict(eqsine=1, ic='steady')
        histpv = 1  # second row
        srspv = [1]
        drdefs.add(**locals())
    return drdefs


def toeco(pth):
    os.mkdir('toeco')
    with cd('toeco'):
        pth = '../' + pth
        # event name:
        event = 'TOECO'

        # load data recovery data:
        sc = cla.load('../cla_params.pgz')
        cla.PrintCLAInfo(sc['mission'], event)

        # load nastran data:
        nas = op2.rdnas2cam(pth+'nas2cam')

        # form ulvs for some SEs:
        SC = 101
        n2p.addulvs(nas, SC)
        drdefs = get_drdefs(nas, sc)

        # prepare spacecraft data recovery matrices
        DR = cla.DR_Event()
        DR.add(nas, sc['drdefs'])
        DR.add(nas, drdefs)

        # initialize results (ext, mnc, mxc for all drms)
        results = DR.prepare_results(sc['mission'], event)

        # set rfmodes:
        rfmodes = nas['rfmodes'][0]

        # setup modal mass, damping and stiffness
        m = None   # None means identity
        k = nas['lambda'][0]
        k[:nas['nrb']] = 0.0
        b = 2*0.02*np.sqrt(k)
        mbk = (m, b, k)

        # load in forcing functions:
        mat = pth + 'toeco/toeco_ffns.mat'
        toeco = matlab.loadmat(mat, squeeze_me=True,
                               struct_as_record=False)
        toeco['ffns'] = toeco['ffns'][:2, ::2]
        toeco['sr'] = toeco['sr']/2
        toeco['t'] = toeco['t'][::2]

        # form force transform:
        T = n2p.formdrm(nas, 0, [[8, 12], [24, 13]])[0].T

        # do pre-calcs and loop over all cases:
        ts = ode.SolveUnc(*mbk, 1/toeco['sr'], rf=rfmodes)
        LC = toeco['ffns'].shape[0]
        t = toeco['t']
        for j, force in enumerate(toeco['ffns']):
            print('Running {} case {}'.format(event, j+1))
            genforce = T @ ([[1], [0.1], [1], [0.1]] * force[None, :])
            # solve equations of motion
            sol = ts.tsolve(genforce, static_ic=1)
            sol.t = t
            sol = DR.apply_uf(sol, *mbk, nas['nrb'], rfmodes)
            caseid = '{} {:2d}'.format(event, j+1)
            results.time_data_recovery(sol, nas['nrb'],
                                       caseid, DR, LC, j)

        # save results:
        cla.save('results.pgz', results)


def summarize(pth):
    os.mkdir('summary')
    with cd('summary'):
        pth = '../' + pth
        event = 'Envelope'

        # load data in desired order:
        results = cla.DR_Results()
        results.merge(
            (cla.load(fn) for fn in ['../toes/results.pgz',
                                     '../owlab/results.pgz',
                                     '../toeco/results.pgz']),
            {'OWLab': 'O&W Lab'}
        )

        results.strip_hists()
        results.form_extreme(event, doappend=2)

        # save overall results:
        cla.save('results.pgz', results)

        # write extrema reports:
        results['extreme'].rpttab(excel=event.lower())
        results['extreme'].srs_plots(Q=10, showall=True)

        # group results together to facilitate investigation:
        Grouped_Results = cla.DR_Results()

        # put these in the order you want:
        groups = [
            ('Time Domain', ('TOES', 'TOECO')),
            ('Freq Domain', ('O&W Lab',)),
        ]

        for key, names in groups:
            Grouped_Results[key] = cla.DR_Results()
            for name in names:
                Grouped_Results[key][name] = results[name]

        Grouped_Results.form_extreme()

        # plot the two groups:
        Grouped_Results['extreme'].srs_plots(
            direc='grouped_srs', Q=10, showall=True)

        # plot just time domain srs:
        Grouped_Results['Time Domain']['extreme'].srs_plots(
            direc='timedomain_srs', Q=10, showall=True)


def compare(pth):
    import sys
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}

    with cd('summary'):
        pth = '../' + pth
        # Load both sets of results and report percent differences:
        results = cla.load('results.pgz')
        lvc = cla.load(pth+'summary/contractor_results_no_srs.pgz')

        # to check for warning message, add a category not in lvc:
        results['extreme']['ifa2'] = results['extreme']['net_ifatm']

        plt.close('all')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results['extreme'].rptpct(
                lvc, names=('LSP', 'Contractor'))
            assert issubclass(w[-1].category, RuntimeWarning)
            assert 'Some comparisons' in str(w[-1].message)
            assert 'ifa2' in str(w[-1].message)

        plt.close('all')
        # modify lvc['cglf'] for testing:
        lvc['cglf'].ext[0, 0] = 0.57   # cause a -7.3% diff
        lvc['cglf'].ext[5, 0] = 0.449  # cause a 17.7% diff
        results['extreme'].rptpct(
            lvc, names=('LSP', 'Contractor'),
            direc='absmax_compare', doabsmax=True)


def confirm():
    for (direc, cnt) in (('compare', 3),
                         ('absmax_compare', 1)):
        cmp_files = glob('summary/{}/*.cmp'.format(direc))
        assert len(cmp_files) == 6
        png_files = glob('summary/{}/*.png'.format(direc))
        assert len(png_files) == 12
        for n in cmp_files:
            with open(n) as f:
                count = 0
                for line in f:
                    if '% Diff Statistics:' in line:
                        count += 1
                        p = line.index(' = [')
                        stats = np.array(
                            [float(i)
                             for i in line[p+4:-2].split(',')])
                        if direc == 'absmax_compare' and 'cglf' in n:
                            mean = (18-7)/14
                            std = np.sqrt(((18-mean)**2 + (-7-mean)**2
                                           + 12*(0-mean)**2)/13)
                            sbe = np.round([-7.0, 18.0, mean, std], 4)
                            assert np.allclose(stats, sbe)
                        else:
                            assert np.all(stats == 0.0)
            assert count == cnt


def do_srs_plots():
    with cd('summary'):
        # Load both sets of results and report percent differences:
        results = cla.load('results.pgz')
        with assert_warns(RuntimeWarning) as cm:
            results['extreme'].srs_plots(
                Q=33, showall=True, direc='srs2',
                # drms=['net_ifltm', 'cglf'])
                drms=['cglf'])
        the_warning = str(cm.warning)
        print(the_warning)
        assert 0 == the_warning.find('no Q=')

        with assert_warns(RuntimeWarning) as cm:
            results['extreme'].srs_plots(
                Q=33, showall=True, direc='srs2',
                drms=['net_ifltm', 'cglf'])
        the_warning = str(cm.warning)
        print(the_warning)
        assert 0 == the_warning.find('no SRS data')

        assert_raises(
            ValueError, results['extreme'].srs_plots,
            Q=[10, 33], showall=True, direc='srs2')

        results['extreme'].srs_plots(
            event='EXTREME',
            Q=10, showall=True, direc='srs2',
            drms=['cglf'], showboth=True)
        assert os.path.exists('srs2/EXTREME.pdf')


def do_time_plots():
    with cd('toeco'):
        # Load both sets of results and report percent differences:
        results = cla.load('results.pgz')
        assert_raises(
            ValueError, results.resp_plots,
            direc='time2', cases=['TOECO  1', 'bad case name'])


def test_transfer_orbit_cla():
    try:
        if os.path.exists('temp_cla'):
            shutil.rmtree('./temp_cla', ignore_errors=True)
        os.mkdir('temp_cla')
        pth = '../pyyeti/tests/cla_test_data/'
        with cd('temp_cla'):
            plt.close('all')
            prepare_4_cla(pth)
            toes(pth)
            owlab(pth)
            toeco(pth)
            summarize(pth)
            compare(pth)
            confirm()
            do_srs_plots()
            do_time_plots()
    finally:
        shutil.rmtree('./temp_cla', ignore_errors=True)


def test_maxmin():
    assert_raises(ValueError, cla.maxmin, np.ones((2, 2)),
                  np.ones((5)))

def test_extrema_1():
    mm = SimpleNamespace(ext=np.ones((5, 3)))
    assert_raises(ValueError, cla.extrema, [], mm, 'test')

    rows = 5
    curext = SimpleNamespace(ext=None, exttime=None,
                             maxcase=None, mincase=None,
                             mx=np.empty((rows, 2)),
                             mn=np.empty((rows, 2)),
                             maxtime=np.empty((rows, 2)),
                             mintime=np.empty((rows, 2)))
    mm = SimpleNamespace(ext=np.ones((rows, 1)),
                         exttime=np.zeros((rows, 1)))
    maxcase = 'Case 1'
    mincase = None
    casenum = 0
    cla.extrema(curext, mm, maxcase, mincase, casenum)

    mm = SimpleNamespace(ext=np.arange(rows)[:, None],
                         exttime=np.ones((rows, 1)))
    maxcase = 'Case 2'
    mincase = None
    casenum = 1
    cla.extrema(curext, mm, maxcase, mincase, casenum)

    assert np.all(curext.ext == np.array([[ 1.,  0.],
                                          [ 1.,  1.],
                                          [ 2.,  1.],
                                          [ 3.,  1.],
                                          [ 4.,  1.]]))

    assert np.all(curext.exttime == np.array([[ 0.,  1.],
                                              [ 0.,  0.],
                                              [ 1.,  0.],
                                              [ 1.,  0.],
                                              [ 1.,  0.]]))
    assert curext.maxcase == ['Case 1', 'Case 1',
                              'Case 2', 'Case 2', 'Case 2']
    assert curext.mincase == ['Case 2', 'Case 1',
                              'Case 1', 'Case 1', 'Case 1']

    assert np.all(curext.mx == np.array([[ 1.,  0.],
                                         [ 1.,  1.],
                                         [ 1.,  2.],
                                         [ 1.,  3.],
                                         [ 1.,  4.]]))

    assert np.all(curext.mn == np.array([[ 1.,  0.],
                                         [ 1.,  1.],
                                         [ 1.,  2.],
                                         [ 1.,  3.],
                                         [ 1.,  4.]]))

    assert np.all(curext.maxtime == np.array([[ 0.,  1.],
                                              [ 0.,  1.],
                                              [ 0.,  1.],
                                              [ 0.,  1.],
                                              [ 0.,  1.]]))

    assert np.all(curext.mintime == np.array([[ 0.,  1.],
                                              [ 0.,  1.],
                                              [ 0.,  1.],
                                              [ 0.,  1.],
                                              [ 0.,  1.]]))


def test_extrema_2():
    rows = 5
    curext = SimpleNamespace(ext=None, exttime=None,
                             maxcase=None, mincase=None,
                             mx=np.empty((rows, 2)),
                             mn=np.empty((rows, 2)),
                             maxtime=np.empty((rows, 2)),
                             mintime=np.empty((rows, 2)))
    mm = SimpleNamespace(ext=np.ones((rows, 1)),
                         exttime=None)
    maxcase = 'Case 1'
    mincase = None
    casenum = 0
    cla.extrema(curext, mm, maxcase, mincase, casenum)

    mm = SimpleNamespace(ext=np.arange(rows)[:, None],
                         exttime=None)
    maxcase = 'Case 2'
    mincase = None
    casenum = 1
    cla.extrema(curext, mm, maxcase, mincase, casenum)

    assert np.all(curext.ext == np.array([[ 1.,  0.],
                                          [ 1.,  1.],
                                          [ 2.,  1.],
                                          [ 3.,  1.],
                                          [ 4.,  1.]]))

    assert curext.exttime is None
    assert curext.maxcase == ['Case 1', 'Case 1',
                              'Case 2', 'Case 2', 'Case 2']
    assert curext.mincase == ['Case 2', 'Case 1',
                              'Case 1', 'Case 1', 'Case 1']

    assert np.all(curext.mx == np.array([[ 1.,  0.],
                                         [ 1.,  1.],
                                         [ 1.,  2.],
                                         [ 1.,  3.],
                                         [ 1.,  4.]]))

    assert np.all(curext.mn == np.array([[ 1.,  0.],
                                         [ 1.,  1.],
                                         [ 1.,  2.],
                                         [ 1.,  3.],
                                         [ 1.,  4.]]))
    assert np.isnan(curext.maxtime).sum() == 10
    assert np.isnan(curext.mintime).sum() == 10


def test_addcat():
    def _():
        name = 'ATM'
        labels = 12
    # doesn't call DR_Def.add:
    assert_raises(RuntimeError, cla.DR_Def.addcat, _)

    defaults = dict(
        se = 0,
        uf_reds = (1, 1, 1, 1),
        drfile = '.',
        srsQs = 10,
        )
    drdefs = cla.DR_Def(defaults)

    drm = np.ones((4, 4))
    @cla.DR_Def.addcat
    def _():
        name = 'ATM'
        labels = 12
        drms = {'drm': drm}
        drdefs.add(**locals())

    import sys
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}

    def _():
        name = 'LTM'
        labels = 12
        drms = {'drm': drm}
        curfile = os.path.realpath(inspect.stack()[0][1])
        drfile = os.path.split(curfile)[1]
        drfunc = 'ATM'
        drdefs.add(**locals())
    with assert_warns(RuntimeWarning) as cm:
        cla.DR_Def.addcat(_)
    the_warning = str(cm.warning)
    assert 0 == the_warning.find('"drm" already')

    assert drdefs.dr_def['LTM'].drfile == drdefs.dr_def['ATM'].drfile

    def _():
        name = 'DTM'
        labels = 12
        drms = {'drm': 0+drm}
        drfunc = 'ATM'
        drdefs.add(**locals())
    # uses a different "drm":
    assert_raises(ValueError, cla.DR_Def.addcat, _)

    def _():
        name = 'DTM'
        labels = 2
        drdefs.add(**locals())
    # already defined data recovery category:
    assert_raises(ValueError, cla.DR_Def.addcat, _)

    def _():
        name = 'SDTM'
        labels = 12
        drms = {'sdrm': 1}
        desc = defaults
        drfunc = 'ATM'
        drdefs.add(**locals())
    # `desc` not in defaults:
    assert_raises(ValueError, cla.DR_Def.addcat, _)

    def _():
        name = 'TDTM'
        labels = 12
        drms = {'tdrm': 1}
        filterval = [0.003, 0.004]
        drfunc = 'ATM'
        drdefs.add(**locals())
    # length of `filterval` does not match length of labels:
    assert_raises(ValueError, cla.DR_Def.addcat, _)

    # this length does match, so no error:
    @cla.DR_Def.addcat
    def _():
        name = 'TDTM2'
        labels = 2
        drms = {'tdrm2': 1}
        filterval = [0.003, 0.004]
        drfunc = 'ATM'
        drdefs.add(**locals())

    # a good bool `histpv`
    @cla.DR_Def.addcat
    def _():
        name = 'ATM2'
        labels = 4
        drms = {'atm2': 1}
        histpv = [True, False, False, True]
        drfunc = 'ATM'
        drdefs.add(**locals())

    # a bad bool `histpv`
    def _():
        name = 'ATM3'
        labels = 4
        drms = {'atm3': 1}
        histpv = [True, False, False]
        drfunc = 'ATM'
        drdefs.add(**locals())
    # length of `histpv` does not match length of labels:
    assert_raises(ValueError, cla.DR_Def.addcat, _)

    # a good integer `histpv`
    @cla.DR_Def.addcat
    def _():
        name = 'ATM4'
        labels = 4
        drms = {'atm4': 1}
        histpv = [0, 3]
        drfunc = 'ATM'
        srsfrq = np.arange(1.0, 10.0)
        drdefs.add(**locals())

    # a bad integer `histpv`
    def _():
        name = 'ATM5'
        labels = 4
        drms = {'atm5': 1}
        histpv = [1, 4]
        drfunc = 'ATM'
        drdefs.add(**locals())
    # `histpv` exceeds dimensions:
    assert_raises(ValueError, cla.DR_Def.addcat, _)

    # a bad type for `histpv`
    def _():
        name = 'ATM6'
        labels = 4
        drms = {'atm6': 1}
        histpv = {}
        # drfunc = 'ATM' ... so that an error message is triggered
        drdefs.add(**locals())
    # `histpv` is bad type:
    assert_raises(TypeError, cla.DR_Def.addcat, _)

    # overlapping drms and nondrms names:
    def _():
        name = 'ATM7'
        labels = 4
        drms = {'atm7': 1}
        nondrms = {'atm7': 1}
        drfunc = 'ATM'
        drdefs.add(**locals())
    # overlapping names in `drms` and `nondrms`
    assert_raises(ValueError, cla.DR_Def.addcat, _)

    drdefs.copycat('ATM', '_dummy',
                   uf_reds=(0, 1, 1., 1))

    assert drdefs.dr_def['ATM_dummy'].labels == drdefs.dr_def['ATM'].labels

    # modify category that doesn't exist
    assert_raises(ValueError, drdefs.copycat, 'notexist', '_2',
                  uf_reds=(0, 1, 1., 1))

    # modify parameter that doesn't exist
    assert_raises(ValueError, drdefs.copycat, 'ATM', '_2',
                  notexist=1)

    drdefs.copycat('ATM', ['ATM_2'],
                   uf_reds=(0, 1, 1., 1))

    assert drdefs.dr_def['ATM_2'].labels == drdefs.dr_def['ATM'].labels

    # atm_2 already exists:
    assert_raises(ValueError, drdefs.copycat, 'ATM', '_2')

    # add a 0rb version of non-existent category:
    assert_raises(ValueError, drdefs.add_0rb, 'net_ifatm')


def test_addcat_2():
    defaults = dict(
        se = 0,
        uf_reds = (1, 1, 1, 1),
        drfile = '.',
        srsQs = 10,
        )
    drdefs = cla.DR_Def(defaults)
    assert_raises(RuntimeError, drdefs.excel_summary, None)


def test_addcat_3():
    defaults = dict(
        se = 0,
        uf_reds = (1, 1, 1, 1),
        drfile = '.',
        srsQs = 10,
        )
    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = 'ATM'
        labels = 4
        drms = {'drm': 1}
        misc = np.arange(10)
        drdefs.add(**locals())

    @cla.DR_Def.addcat
    def _():
        name = 'ATM2'
        labels = 4
        drms = {'drm2': 1}
        drfunc = 'ATM'
        misc = np.arange(10)
        drdefs.add(**locals())

    df = drdefs.excel_summary(None)
    assert df['ATM2']['misc'] == '-'


def test_event_add():
    #    ext_name = 'FLAC'
    _get_labels = _get_labels0
    cyclenumber = 0

    # make up some CLA results:
    events = ('Liftoff', 'Transonics', 'MECO')
    rows = {'ATM': 34, 'LTM': 29}
    ext_results = {i: {} for i in rows}
    for i, event in enumerate(events):
        for drm, nrows in rows.items():
            ext_results[drm][event] = _get_minmax(
                drm, i, cyclenumber)

    # setup CLA parameters:
    mission = "Rocket / Spacecraft CLA"
    duf = 1.2
    suf = 1.0

    # defaults for data recovery
    defaults = dict(
        se = 10,
        uf_reds = (1, 1, duf, 1),
        drfile = '.',
        )

    drdefs = cla.DR_Def(defaults)
    drdefs2 = cla.DR_Def(defaults)
    drdefs3 = cla.DR_Def(defaults)
    drdefs4 = cla.DR_Def(defaults)

    uset = n2p.addgrid(None, 1, 'b', 0, [0, 0, 0], 0)
    nas = {
        'ulvs': {10: np.ones((1, 1), float)},
        'uset': {10: uset[:1]},
    }

    @cla.DR_Def.addcat
    def _():
        name = 'ATM'
        desc = 'S/C Internal Accelerations'
        units = 'm/sec^2, rad/sec^2'
        labels = _get_labels(rows, name)
        drms = {'atm': np.ones((4, 1), float)}
        nondrms = {'var': 'any value'}
        drdefs.add(**locals())

    # so that "ATM data recovery category alread defined" error
    # gets triggered in DR_Event.add:
    @cla.DR_Def.addcat
    def _():
        name = 'ATM'
        desc = 'S/C Internal Accelerations'
        units = 'm/sec^2, rad/sec^2'
        labels = _get_labels(rows, name)
        drdefs2.add(**locals())

    # so that nondrms "atm is already in Vars[10]'" error
    # gets triggered in DR_Event.add:
    @cla.DR_Def.addcat
    def _():
        name = 'ATM3'
        desc = 'S/C Internal Accelerations'
        units = 'm/sec^2, rad/sec^2'
        labels = _get_labels(rows, 'ATM')
        drfunc = 'ATM'
        drms = {'atm': np.ones((4, 1), float)}
        drdefs3.add(**locals())

    # so that nondrms "atm is already in Vars[10]'" error
    # gets triggered in DR_Event.add:
    @cla.DR_Def.addcat
    def _():
        name = 'ATM4'
        desc = 'S/C Internal Accelerations'
        units = 'm/sec^2, rad/sec^2'
        labels = _get_labels(rows, 'ATM')
        drfunc = 'ATM'
        nondrms = {'var': 'any value'}
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
    assert_raises(ValueError, DR.add, nas, drdefs2)
    assert_raises(ValueError, DR.add, nas, drdefs3)
    assert_raises(ValueError, DR.add, nas, drdefs4)

    # for testing apply_uf:
    sol = SimpleNamespace()
    sol.a = np.ones((1, 10), float)
    sol.v = sol.a + 1.0
    sol.d = sol.a + 2.0
    sol.pg = np.array([45])
    SOL1 = DR.apply_uf(sol, None, np.ones(1, float),
                       np.ones(1, float)+1.0, 0, None)
    SOL2 = DR.apply_uf(sol, np.ones(1, float), np.ones((1, 1), float),
                       np.ones((1, 1), float)+1.0, 0, None)
    SOL3 = DR.apply_uf(sol, np.ones((1, 1), float),
                       np.ones((1, 1), float),
                       np.ones((1, 1), float)+1.0, 0, None)

    for k, d1 in SOL1.items():   # loop over uf_reds
        d2 = SOL2[k]
        d3 = SOL3[k]
        for k, v1 in d1.__dict__.items():  # loop over a, v, d, ...
            v2 = getattr(d2, k)
            v3 = getattr(d3, k)
            assert np.all(v2 == v1)
            assert np.all(v3 == v1)

    assert SOL1[(2, 2, 2, 2)].pg == 90

    SOL = DR.frf_apply_uf(sol, 0)

    assert np.all(SOL[(2, 2, 2, 2)].a == 4*sol.a)
    assert np.all(SOL[(2, 2, 2, 2)].v == 4*sol.v)
    assert np.all(SOL[(2, 2, 2, 2)].d == 4*sol.d)
    assert np.all(SOL[(2, 2, 2, 2)].pg == 2*sol.pg)


def test_merge():
    results = cla.DR_Results()
    r1 = get_fake_cla_results('FLAC', _get_labels0, 0)
    r2 = get_fake_cla_results('VLC', _get_labels1, 1)
    r3 = get_fake_cla_results('PostVLC', _get_labels2, 2)
    del r3['extreme']
    results.merge((r1, r2, r3),
        {'FLAC': 'FDLC',
         'PostVLC': 'VLC2'})

    results.form_extreme()
    assert repr(results).startswith('<pyyeti.cla.DR_Results object')
    assert str(results) == ('DR_Results with 4 categories: '
                            "['FDLC', 'VLC', "
                            "'Liftoff, Transonics, MECO', 'extreme']")

    results['newentry'] = 'should cause type error'
    assert_raises(TypeError, results.strip_hists)

    results = cla.DR_Results()
    r1 = {'FLAC': 'this is a bad entry'}
    assert_raises(TypeError, results.merge, (r1, r2))


def kc_forces(sol, nas, Vars, se):
    return np.vstack((Vars[se]['springdrm'] @ sol.d,
                      Vars[se]['damperdrm'] @ sol.v))


def mass_spring_system():
    """

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
    m1 = 2.
    m2 = 4.
    m3 = 6.
    k1 = 12000.
    k2 = 16000.
    k3 = 10000.
    c1 = 70.
    c2 = 75.
    c3 = 30.
    mass = np.diag([m1, m2, m3])
    stiff = np.array([[k1+k3, -k1, -k3],
                      [-k1, k1+k2, -k2],
                      [-k3, -k2, k2+k3]])
    damp = np.array([[c1+c3, -c1, -c3],
                     [-c1, c1+c2, -c2],
                     [-c3, -c2, c2+c3]])
    # drm for subtracting 1 from 2, 2 from 3, 1 from 3:
    sub = np.array([[-1., 1., 0],
                    [0., -1., 1.],
                    [-1., 0, 1.]])
    drms1 = {'springdrm': [[k1], [k2], [k3]] * sub,
             'damperdrm': [[c1], [c2], [c3]] * sub}

    # define some defaults for data recovery:
    uf_reds = (1, 1, 1, 1)
    defaults = dict(se=0, uf_reds=uf_reds)
    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = 'kc_forces'
        desc = 'Spring & Damper Forces'
        units = 'N'
        labels = ['{} {}'.format(j, i+1)
                  for j in ('Spring', 'Damper')
                  for i in range(3)]
        # force will be positive for tension
        drms = drms1
        histpv = 'all'
        drdefs.add(**locals())

    # prepare spacecraft data recovery matrices
    DR = cla.DR_Event()
    DR.add(None, drdefs)

    return mass, damp, stiff, drms1, uf_reds, defaults, DR


def test_case_defined():
    (mass, damp, stiff,
     drms1, uf_reds, defaults, DR) = mass_spring_system()

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
    event = 'Case 1'
    results = DR.prepare_results('Spring & Damper Forces', event)

    # perform data recovery:
    results.time_data_recovery(sol, None, event, DR, 1, 0)

    assert np.allclose(results['kc_forces'].ext,
                       np.array([[ 1.71124021, -5.94610295],
                                 [ 1.10707637, -1.99361428],
                                 [ 1.89895824, -5.99096572],
                                 [ 2.01946488, -2.01871227],
                                 [ 0.46376154, -0.45142869],
                                 [ 0.96937744, -0.96687706]]))

    # test for some errors:
    results = DR.prepare_results('Spring & Damper Forces', event)
    results.time_data_recovery(sol, None, event, DR, 2, 0)
    assert_raises(ValueError, results.time_data_recovery, sol, None,
                  event, DR, 2, 1)

    # mess the labels up:
    drdefs = cla.DR_Def(defaults)

    @cla.DR_Def.addcat
    def _():
        name = 'kc_forces'
        desc = 'Spring & Damper Forces'
        units = 'N'
        labels = ['one', 'two']
        drms = drms1
        drdefs.add(**locals())

    # prepare spacecraft data recovery matrices
    DR = cla.DR_Event()
    DR.add(None, drdefs)

    # initialize results (ext, mnc, mxc for all drms)
    results = DR.prepare_results('Spring & Damper Forces', event)

    # perform data recovery:
    assert_raises(ValueError, results.time_data_recovery, sol, None,
                  event, DR, 1, 0)


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
    freq = np.arange(1., 6.)
    resp = np.zeros((12, 5))
    forcepsd = np.ones((1, 5))
    resp[[2, 5, 6, 7, 8, 11]] = 1.0
    resp[[9, 10]] = 0.5
    xr = np.arange(0, 12, 2)
    yr = xr + 1
    rr = None
    drmres = SimpleNamespace(_psd={})
    case = 'test'
    cla.PSD_consistent_rss(resp, xr, yr, rr, freq, forcepsd, drmres,
                           case, 0)
    sbe = np.zeros((6, 5))
    sbe[1:3] = 1.0
    sbe[3] = 2.
    sbe[4:] = 1.0 + 0.5**2
    assert np.allclose(drmres._psd[case], sbe)


def _comp_rpt(s, sbe):
    for i, j in zip(s, sbe):
        if not j.startswith('Date:'):
            assert i == j


def test_rptext1():
    (mass, damp, stiff,
     drms1, uf_reds, defaults, DR) = mass_spring_system()

    # define forces:
    h = 0.01
    t = np.arange(0, 1.0, h)
    f = np.zeros((3, len(t)))
    f[0, 2:25] = 10.0

    # setup solver:
    ts = ode.SolveUnc(mass, damp, stiff, h, pre_eig=True)
    sol = {uf_reds: ts.tsolve(f)}

    # initialize results (ext, mnc, mxc for all drms)
    event = 'Case 1'
    results = DR.prepare_results('Spring & Damper Forces', event)

    # perform data recovery:
    results.time_data_recovery(sol, None, event, DR, 1, 0)

    with StringIO() as f:
        cla.rptext1(results['kc_forces'], f)
        s = f.getvalue().split('\n')
    sbe = [
        'M A X / M I N  S U M M A R Y',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Date:        21-Jan-2017',
        '',
        '  Row    Description       Maximum       Time    Case      '
        '   Minimum       Time    Case',
        '-------  -----------    -------------  --------  ------    '
        '-------------  --------  ------',
        '      1  Spring 1         1.51883e+00     0.270  Case 1    '
        ' -5.75316e+00     0.040  Case 1',
        '      2  Spring 2         1.04111e+00     0.280  Case 1    '
        ' -1.93144e+00     0.050  Case 1',
        '      3  Spring 3         1.59375e+00     0.280  Case 1    '
        ' -5.68091e+00     0.050  Case 1',
        '      4  Damper 1         1.76099e+00     0.260  Case 1    '
        ' -1.76088e+00     0.030  Case 1',
        '      5  Damper 2         4.23522e-01     0.270  Case 1    '
        ' -4.11612e-01     0.040  Case 1',
        '      6  Damper 3         8.93351e-01     0.260  Case 1    '
        ' -8.90131e-01     0.030  Case 1',
        '']
    _comp_rpt(s, sbe)

    lbls = results['kc_forces'].drminfo.labels[:]
    results['kc_forces'].drminfo.labels = lbls[:-1]
    with StringIO() as f:
        assert_raises(ValueError, cla.rptext1, results['kc_forces'], f)

    results['kc_forces'].drminfo.labels = lbls
    del results['kc_forces'].domain

    with StringIO() as f:
        cla.rptext1(results['kc_forces'], f, doabsmax=True, perpage=3)
        s = f.getvalue().split('\n')
    sbe = [
        'M A X / M I N  S U M M A R Y',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Date:        21-Jan-2017',
        '',
        '  Row    Description       Maximum     X-Value   Case',
        '-------  -----------    -------------  --------  ------',
        '      1  Spring 1        -5.75316e+00     0.270  Case 1',
        '      2  Spring 2        -1.93144e+00     0.280  Case 1',
        '      3  Spring 3        -5.68091e+00     0.280  Case 1',
        '\x0cM A X / M I N  S U M M A R Y',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Date:        21-Jan-2017',
        '',
        '  Row    Description       Maximum     X-Value   Case',
        '-------  -----------    -------------  --------  ------',
        '      4  Damper 1         1.76099e+00     0.260  Case 1',
        '      5  Damper 2         4.23522e-01     0.270  Case 1',
        '      6  Damper 3         8.93351e-01     0.260  Case 1',
        '']
    _comp_rpt(s, sbe)

    results['kc_forces'].ext = results['kc_forces'].ext[:, :1]
    results['kc_forces'].exttime = results['kc_forces'].exttime[:, :1]
    with StringIO() as f:
        cla.rptext1(results['kc_forces'], f)
        s = f.getvalue().split('\n')
    sbe = [
        'M A X / M I N  S U M M A R Y',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Date:        22-Jan-2017',
        '',
        '  Row    Description       Maximum     X-Value   Case',
        '-------  -----------    -------------  --------  ------',
        '      1  Spring 1         1.51883e+00     0.270  Case 1',
        '      2  Spring 2         1.04111e+00     0.280  Case 1',
        '      3  Spring 3         1.59375e+00     0.280  Case 1',
        '      4  Damper 1         1.76099e+00     0.260  Case 1',
        '      5  Damper 2         4.23522e-01     0.270  Case 1',
        '      6  Damper 3         8.93351e-01     0.260  Case 1',
        '']
    _comp_rpt(s, sbe)

    results['kc_forces'].ext = results['kc_forces'].ext[:, 0]
    results['kc_forces'].exttime = results['kc_forces'].exttime[:, 0]
    with StringIO() as f:
        cla.rptext1(results['kc_forces'], f)
        s = f.getvalue().split('\n')
    _comp_rpt(s, sbe)


def test_get_numform():
    assert cla._get_numform(0.0) == '{:13.0f}'
    assert cla._get_numform(np.array([1e12, 1e4])) == '{:13.6e}'
    assert cla._get_numform(np.array([1e8, 1e4])) == '{:13.1f}'
    assert cla._get_numform(np.array([1e10, 1e5])) == '{:13.0f}'


def test_rpttab1():
    (mass, damp, stiff,
     drms1, uf_reds, defaults, DR) = mass_spring_system()

    # define forces:
    h = 0.01
    t = np.arange(0, 1.0, h)
    f = {0: np.zeros((3, len(t))),
         1: np.zeros((3, len(t)))}
    f[0][0, 2:25] = 10.0
    f[1][0, 2:25] = np.arange(23.0)
    f[1][0, 25:48] = np.arange(22.0, -1.0, -1.0)

    # initialize results (ext, mnc, mxc for all drms)
    results = DR.prepare_results('Spring & Damper Forces', 'Steps')
    # setup solver:
    ts = ode.SolveUnc(mass, damp, stiff, h, pre_eig=True)
    for i in range(2):
        sol = {uf_reds: ts.tsolve(f[i])}
        case = 'FFN {}'.format(i)
        # perform data recovery:
        results.time_data_recovery(sol, None, case, DR, 2, i)

    with StringIO() as f:
        cla.rpttab1(results['kc_forces'], f, 'Title')
        s = f.getvalue().split('\n')
    sbe = [
        'Title',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Date:        22-Jan-2017',
        '',
        'Maximum Responses',
        '',
        ' Row     Description       FFN 0         FFN 1        Maximum         Case',
        '====== =============== ============= ============= ============= =============',
        '     1 Spring 1         1.518830e+00  1.858805e-01  1.518830e+00 FFN 0',
        '     2 Spring 2         1.041112e+00  1.335839e-01  1.041112e+00 FFN 0',
        '     3 Spring 3         1.593752e+00  2.383903e-01  1.593752e+00 FFN 0',
        '     4 Damper 1         1.760988e+00  3.984896e-01  1.760988e+00 FFN 0',
        '     5 Damper 2         4.235220e-01  1.315805e-01  4.235220e-01 FFN 0',
        '     6 Damper 3         8.933514e-01  2.115629e-01  8.933514e-01 FFN 0',
        '',
        '',
        'Minimum Responses',
        '',
        ' Row     Description       FFN 0         FFN 1        Minimum         Case',
        '====== =============== ============= ============= ============= =============',
        '     1 Spring 1        -5.753157e+00 -9.464095e+00 -9.464095e+00 FFN 1',
        '     2 Spring 2        -1.931440e+00 -2.116412e+00 -2.116412e+00 FFN 1',
        '     3 Spring 3        -5.680914e+00 -9.182364e+00 -9.182364e+00 FFN 1',
        '     4 Damper 1        -1.760881e+00 -3.428864e-01 -1.760881e+00 FFN 0',
        '     5 Damper 2        -4.116117e-01 -9.167583e-02 -4.116117e-01 FFN 0',
        '     6 Damper 3        -8.901312e-01 -1.797676e-01 -8.901312e-01 FFN 0',
        '',
        '',
        'Abs-Max Responses',
        '',
        ' Row     Description       FFN 0         FFN 1        Abs-Max         Case',
        '====== =============== ============= ============= ============= =============',
        '     1 Spring 1        -5.753157e+00 -9.464095e+00 -9.464095e+00 FFN 1',
        '     2 Spring 2        -1.931440e+00 -2.116412e+00 -2.116412e+00 FFN 1',
        '     3 Spring 3        -5.680914e+00 -9.182364e+00 -9.182364e+00 FFN 1',
        '     4 Damper 1         1.760988e+00  3.984896e-01  1.760988e+00 FFN 0',
        '     5 Damper 2         4.235220e-01  1.315805e-01  4.235220e-01 FFN 0',
        '     6 Damper 3         8.933514e-01  2.115629e-01  8.933514e-01 FFN 0',
        '',
        '',
        'Extrema Count',
        'Filter: 1e-06',
        '',
        '         Description       FFN 0         FFN 1',
        '       =============== ============= =============',
        '       Maxima Count                6             0',
        '       Minima Count                3             3',
        '       Max+Min Count               9             3',
        '       Abs-Max Count               3             3',
        '',
        '         Description       FFN 0         FFN 1',
        '       =============== ============= =============',
        '       Maxima Percent          100.0           0.0',
        '       Minima Percent           50.0          50.0',
        '       Max+Min Percent          75.0          25.0',
        '       Abs-Max Percent          50.0          50.0',
        '']
    _comp_rpt(s, sbe)

    results['kc_forces'].maxcase = None
    results['kc_forces'].mincase = None
    with StringIO() as f:
        cla.rpttab1(results['kc_forces'], f, 'Title')
        s = f.getvalue().split('\n')
    sbe2 = sbe[:]
    for i in range(len(sbe2)):
        if len(sbe2[i]) > 60 and 'FFN ' in sbe2[i][60:]:
            sbe2[i] = (sbe2[i].replace('FFN 0', 'N/A').
                       replace('FFN 1', 'N/A'))
    _comp_rpt(s, sbe2)

    results['kc_forces'].ext[:] = 0.
    results['kc_forces'].mn[:] = 0.
    results['kc_forces'].mx[:] = 0.
    with StringIO() as f:
        cla.rpttab1(results['kc_forces'], f, 'Title')
        s = f.getvalue().split('\n')
    sbe = [
        'Title',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Date:        22-Jan-2017',
        '',
        'Maximum Responses',
        '',
        ' Row     Description       FFN 0         FFN 1        Maximum         Case',
        '====== =============== ============= ============= ============= =============',
        '     1 Spring 1         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     2 Spring 2         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     3 Spring 3         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     4 Damper 1         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     5 Damper 2         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     6 Damper 3         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '',
        '',
        'Minimum Responses',
        '',
        ' Row     Description       FFN 0         FFN 1        Minimum         Case',
        '====== =============== ============= ============= ============= =============',
        '     1 Spring 1         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     2 Spring 2         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     3 Spring 3         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     4 Damper 1         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     5 Damper 2         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     6 Damper 3         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '',
        '',
        'Abs-Max Responses',
        '',
        ' Row     Description       FFN 0         FFN 1        Abs-Max         Case',
        '====== =============== ============= ============= ============= =============',
        '     1 Spring 1         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     2 Spring 2         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     3 Spring 3         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     4 Damper 1         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     5 Damper 2         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '     6 Damper 3         0.000000e+00  0.000000e+00  0.000000e+00 zero row',
        '',
        '',
        'Extrema Count',
        'Filter: 1e-06',
        '',
        '         Description       FFN 0         FFN 1',
        '       =============== ============= =============',
        '       Maxima Count                0             0',
        '       Minima Count                0             0',
        '       Max+Min Count               0             0',
        '       Abs-Max Count               0             0',
        '',
        '         Description       FFN 0         FFN 1',
        '       =============== ============= =============',
        '       Maxima Percent            0.0           0.0',
        '       Minima Percent            0.0           0.0',
        '       Max+Min Percent           0.0           0.0',
        '       Abs-Max Percent           0.0           0.0',
        '']
    _comp_rpt(s, sbe)

    lbls = results['kc_forces'].drminfo.labels[:]
    results['kc_forces'].drminfo.labels = lbls[:-1]
    with StringIO() as f:
        assert_raises(ValueError, cla.rpttab1,
                      results['kc_forces'], f, 'Title')

    results['kc_forces'].drminfo.labels = lbls
    assert_raises(ValueError, cla.rpttab1,
                  results['kc_forces'], 't.xlsx', 'Title')


def test_rptpct1():
    ext1 = [[120.0, -8.0],
            [8.0, -120.0]]
    ext2 = [[115.0, -5.0],
            [10.0, -125.0]]
    opts = {'domagpct': False,
            'dohistogram': False,
            'filterval': np.array([5.0, 1000.])}
    with StringIO() as f:
        dct = cla.rptpct1(ext1, ext2, f, **opts)
        s = f.getvalue().split('\n')
    sbe = [
        'PERCENT DIFFERENCE REPORT',
        '',
        'Description: No description provided',
        'Uncertainty: Not specified',
        'Units:       Not specified',
        'Filter:      <defined row-by-row>',
        'Notes:       % Diff = +/- abs(Self-Reference)/max(abs(Reference(max,min)))*100',
        '             Sign set such that positive % differences indicate exceedances',
        'Date:        22-Jan-2017',
        '',
        '                             Self        Reference                    Self    '
        '    Reference                    Self        Reference',
        '  Row    Description       Maximum        Maximum      % Diff       Minimum   '
        '     Minimum      % Diff       Abs-Max        Abs-Max      % Diff',
        '-------  -----------    -------------  -------------  -------    -------------'
        '  -------------  -------    -------------  -------------  -------',
        '      1  Row      1         120.00000      115.00000     4.35         -8.00000'
        '       -5.00000     2.61        120.00000      115.00000     4.35',
        '      2  Row      2           8.00000       10.00000  n/a           -120.00000'
        '     -125.00000  n/a            120.00000      125.00000  n/a    ',
        '',
        '',
        '',
        '    No description provided - Maximum Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         4.00          1    100.00',
        '',
        '    0.0% of values are within 1%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [4.00, 4.00, 4.0000, 0.0000]',
        '',
        '',
        '    No description provided - Minimum Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         3.00          1    100.00',
        '',
        '    0.0% of values are within 1%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [3.00, 3.00, 3.0000, 0.0000]',
        '',
        '',
        '    No description provided - Abs-Max Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         4.00          1    100.00',
        '',
        '    0.0% of values are within 1%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [4.00, 4.00, 4.0000, 0.0000]',
        '']
    _comp_rpt(s, sbe)


def test_rptpct1_2():
    (mass, damp, stiff,
     drms1, uf_reds, defaults, DR) = mass_spring_system()

    # define forces:
    h = 0.01
    t = np.arange(0, 1.0, h)
    f = {0: np.zeros((3, len(t))),
         1: np.zeros((3, len(t)))}
    f[0][0, 2:25] = 10.0
    f[1][0, 2:25] = 10.0
    f[1][0, 3:25:3] = 9.5

    # initialize results
    results = cla.DR_Results()
    # setup solver:
    ts = ode.SolveUnc(mass, damp, stiff, h, pre_eig=True)
    for i in range(2):
        sol = {uf_reds: ts.tsolve(f[i])}
        case = 'FFN {}'.format(i)
        # perform data recovery:
        results[case] = DR.prepare_results('Spring & Damper Forces', 'Steps')
        results[case].time_data_recovery(sol, None, case, DR, 1, 0)

    opts = {'domagpct': False,
            'dohistogram': False,
            'filterval': 0.3*np.ones(6)}
    drminfo = results['FFN 0']['kc_forces'].drminfo
    drminfo.labels = drminfo.labels[:]
    drminfo.labels[2] = 'SPRING 3'
    with StringIO() as f:
        dct = cla.rptpct1(results['FFN 0']['kc_forces'],
                          results['FFN 1']['kc_forces'],
                          f, **opts)
        s = f.getvalue().split('\n')
    sbe = [
        'PERCENT DIFFERENCE REPORT',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Filter:      <defined row-by-row>',
        'Notes:       % Diff = +/- abs(Self-Reference)/max(abs(Reference(max,min)))*100',
        '             Sign set such that positive % differences indicate exceedances',
        'Date:        23-Jan-2017',
        '',
        '                             Self        Reference                    Self        Reference                    Self        Reference',
        '  Row    Description       Maximum        Maximum      % Diff       Minimum        Minimum      % Diff       Abs-Max        Abs-Max      % Diff',
        '-------  -----------    -------------  -------------  -------    -------------  -------------  -------    -------------  -------------  -------',
        '      1  Spring 1            1.518830       1.509860     0.16        -5.753157      -5.603161     2.68         5.753157       5.603161     2.68',
        '      2  Spring 2            1.041112       1.031179     0.53        -1.931440      -1.887905     2.31         1.931440       1.887905     2.31',
        '      4  Damper 1            1.760988       1.714232     2.73        -1.760881      -1.697650     3.69         1.760988       1.714232     2.73',
        '      5  Damper 2            0.423522       0.415255     1.99        -0.411612      -0.399434     2.93         0.423522       0.415255     1.99',
        '      6  Damper 3            0.893351       0.873861     2.23        -0.890131      -0.861630     3.26         0.893351       0.873861     2.23',
        '',
        '',
        '',
        '    Spring & Damper Forces - Maximum Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         0.00          1     20.00',
        '         1.00          1     20.00',
        '         2.00          2     40.00',
        '         3.00          1     20.00',
        '',
        '    40.0% of values are within 1%',
        '    80.0% of values are within 2%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [0.00, 3.00, 1.6000, 1.1402]',
        '',
        '',
        '    Spring & Damper Forces - Minimum Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         2.00          1     20.00',
        '         3.00          3     60.00',
        '         4.00          1     20.00',
        '',
        '    0.0% of values are within 1%',
        '    20.0% of values are within 2%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.00, 4.00, 3.0000, 0.7071]',
        '',
        '',
        '    Spring & Damper Forces - Abs-Max Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         2.00          3     60.00',
        '         3.00          2     40.00',
        '',
        '    0.0% of values are within 1%',
        '    60.0% of values are within 2%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.00, 3.00, 2.4000, 0.5477]',
        '']
    _comp_rpt(s, sbe)

    opts = {'domagpct': False,
            'dohistogram': False,
            'filterval': 0.3*np.ones(1),
            'use_range': False}
    with StringIO() as f:
        dct = cla.rptpct1(results['FFN 0']['kc_forces'],
                          results['FFN 1']['kc_forces'],
                          f, **opts)
        s = f.getvalue().split('\n')
    sbe = [
        'PERCENT DIFFERENCE REPORT',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Filter:      0.3',
        'Notes:       % Diff = +/- abs((Self-Reference)/Reference)*100',
        '             Sign set such that positive % differences indicate exceedances',
        'Date:        30-Jan-2017',
        '',
        '                             Self        Reference                    Self    '
        '    Reference                    Self        Reference',
        '  Row    Description       Maximum        Maximum      % Diff       Minimum   '
        '     Minimum      % Diff       Abs-Max        Abs-Max      % Diff',
        '-------  -----------    -------------  -------------  -------    -------------'
        '  -------------  -------    -------------  -------------  -------',
        '      1  Spring 1            1.518830       1.509860     0.59        -5.753157'
        '      -5.603161     2.68         5.753157       5.603161     2.68',
        '      2  Spring 2            1.041112       1.031179     0.96        -1.931440'
        '      -1.887905     2.31         1.931440       1.887905     2.31',
        '      4  Damper 1            1.760988       1.714232     2.73        -1.760881'
        '      -1.697650     3.72         1.760988       1.714232     2.73',
        '      5  Damper 2            0.423522       0.415255     1.99        -0.411612'
        '      -0.399434     3.05         0.423522       0.415255     1.99',
        '      6  Damper 3            0.893351       0.873861     2.23        -0.890131'
        '      -0.861630     3.31         0.893351       0.873861     2.23',
        '',
        '',
        '',
        '    Spring & Damper Forces - Maximum Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         1.00          2     40.00',
        '         2.00          2     40.00',
        '         3.00          1     20.00',
        '',
        '    40.0% of values are within 1%',
        '    80.0% of values are within 2%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [1.00, 3.00, 1.8000, 0.8367]',
        '',
        '',
        '    Spring & Damper Forces - Minimum Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         2.00          1     20.00',
        '         3.00          3     60.00',
        '         4.00          1     20.00',
        '',
        '    0.0% of values are within 1%',
        '    20.0% of values are within 2%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.00, 4.00, 3.0000, 0.7071]',
        '',
        '',
        '    Spring & Damper Forces - Abs-Max Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         2.00          3     60.00',
        '         3.00          2     40.00',
        '',
        '    0.0% of values are within 1%',
        '    60.0% of values are within 2%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.00, 3.00, 2.4000, 0.5477]',
        '']
    _comp_rpt(s, sbe)

    opts = {'domagpct': False,
            'dohistogram': False,
            'prtbad': 2.5,
            'flagbad': 2.7,
            }
    with StringIO() as f:
        dct = cla.rptpct1(results['FFN 0']['kc_forces'],
                          results['FFN 1']['kc_forces'],
                          f, **opts)
        s = f.getvalue().split('\n')
    sbe = [
        'PERCENT DIFFERENCE REPORT',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Filter:      1e-06',
        'Notes:       % Diff = +/- abs(Self-Reference)/max(abs(Reference(max,min)))*100',
        '             Sign set such that positive % differences indicate exceedances',
        '             Printing rows where abs(% Diff) > 2.5%',
        '             Flagging (*) rows where abs(% Diff) > 2.7%',
        'Date:        30-Jan-2017',
        '',
        '                             Self        Reference                     Self        Reference                     Self        Reference',
        '  Row    Description       Maximum        Maximum      % Diff        Minimum        Minimum      % Diff        Abs-Max        Abs-Max      % Diff',
        '-------  -----------    -------------  -------------  --------    -------------  -------------  --------    -------------  -------------  --------',
        '      1  Spring 1            1.518830       1.509860     0.16         -5.753157      -5.603161     2.68          5.753157       5.603161     2.68 ',
        '      4  Damper 1            1.760988       1.714232     2.73*        -1.760881      -1.697650     3.69*         1.760988       1.714232     2.73*',
        '      5  Damper 2            0.423522       0.415255     1.99         -0.411612      -0.399434     2.93*         0.423522       0.415255     1.99 ',
        '      6  Damper 3            0.893351       0.873861     2.23         -0.890131      -0.861630     3.26*         0.893351       0.873861     2.23 ',
        '',
        '',
        '',
        '    Spring & Damper Forces - Maximum Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         0.00          1     20.00',
        '         1.00          1     20.00',
        '         2.00          2     40.00',
        '         3.00          1     20.00',
        '',
        '    40.0% of values are within 1%',
        '    80.0% of values are within 2%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [0.00, 3.00, 1.6000, 1.1402]',
        '',
        '',
        '    Spring & Damper Forces - Minimum Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         2.00          1     20.00',
        '         3.00          3     60.00',
        '         4.00          1     20.00',
        '',
        '    0.0% of values are within 1%',
        '    20.0% of values are within 2%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.00, 4.00, 3.0000, 0.7071]',
        '',
        '',
        '    Spring & Damper Forces - Abs-Max Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         2.00          3     60.00',
        '         3.00          2     40.00',
        '',
        '    0.0% of values are within 1%',
        '    60.0% of values are within 2%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.00, 3.00, 2.4000, 0.5477]',
        '']
    _comp_rpt(s, sbe)

    opts = {'domagpct': False,
            'dohistogram': False,
            'prtbadh': 2.5,
            'flagbadh': 2.7,
            }
    with StringIO() as f:
        dct = cla.rptpct1(results['FFN 0']['kc_forces'],
                          results['FFN 1']['kc_forces'],
                          f, **opts)
        s = f.getvalue().split('\n')
    sbe[8] = '             Printing rows where % Diff > 2.5%'
    sbe[9] = '             Flagging (*) rows where % Diff > 2.7%'
    _comp_rpt(s, sbe)

    opts = {'domagpct': False,
            'dohistogram': False,
            'prtbadl': 2.0,
            'flagbadl': 2.2,
            }
    with StringIO() as f:
        dct = cla.rptpct1(results['FFN 0']['kc_forces'],
                          results['FFN 1']['kc_forces'],
                          f, **opts)
        s = f.getvalue().split('\n')
    sbe[8] = '             Printing rows where % Diff < 2.0%'
    sbe[9] = '             Flagging (*) rows where % Diff < 2.2%'
    sbe[12:19] = [
        '                             Self        Reference                     Self        Reference                    Self        Reference',
        '  Row    Description       Maximum        Maximum      % Diff        Minimum        Minimum      % Diff       Abs-Max        Abs-Max      % Diff',
        '-------  -----------    -------------  -------------  --------    -------------  -------------  -------    -------------  -------------  --------',
        '      1  Spring 1            1.518830       1.509860     0.16*        -5.753157      -5.603161     2.68         5.753157       5.603161     2.68 ',
        '      2  Spring 2            1.041112       1.031179     0.53*        -1.931440      -1.887905     2.31         1.931440       1.887905     2.31 ',
        '      5  Damper 2            0.423522       0.415255     1.99*        -0.411612      -0.399434     2.93         0.423522       0.415255     1.99*',
        ]
    _comp_rpt(s, sbe)

    opts = {'domagpct': False,
            'dohistogram': False,
            'prtbadl': 2.0,
            'flagbadl': 2.2,
            'doabsmax': True,
            }
    with StringIO() as f:
        dct = cla.rptpct1(results['FFN 0']['kc_forces'],
                          results['FFN 1']['kc_forces'],
                          f, **opts)
        s = f.getvalue().split('\n')
    sbe = [
        'PERCENT DIFFERENCE REPORT',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Filter:      1e-06',
        'Notes:       % Diff = +/- abs((Self-Reference)/Reference)*100',
        '             Sign set such that positive % differences indicate exceedances',
        '             Printing rows where % Diff < 2.0%',
        '             Flagging (*) rows where % Diff < 2.2%',
        'Date:        30-Jan-2017',
        '',
        '                             Self           Self           Self        Reference',
        '  Row    Description       Maximum        Minimum        Abs-Max        Abs-Max      % Diff',
        '-------  -----------    -------------  -------------  -------------  -------------  --------',
        '      5  Damper 2            0.423522      -0.411612       0.423522       0.415255     1.99*',
        '',
        '',
        '',
        '    Spring & Damper Forces - Abs-Max Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         2.00          3     60.00',
        '         3.00          2     40.00',
        '',
        '    0.0% of values are within 1%',
        '    60.0% of values are within 2%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.00, 3.00, 2.4000, 0.5477]',
        '']
    _comp_rpt(s, sbe)

    opts = {'domagpct': False,
            'dohistogram': False,
            'prtbadl': 2.0,
            'flagbadl': 2.2,
            'shortabsmax': True,
            }
    with StringIO() as f:
        dct = cla.rptpct1(results['FFN 0']['kc_forces'],
                          results['FFN 1']['kc_forces'],
                          f, **opts)
        s = f.getvalue().split('\n')
    sbe = [
        'PERCENT DIFFERENCE REPORT',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Filter:      1e-06',
        'Notes:       % Diff = +/- abs((Self-Reference)/Reference)*100',
        '             Sign set such that positive % differences indicate exceedances',
        '             Printing rows where % Diff < 2.0%',
        '             Flagging (*) rows where % Diff < 2.2%',
        'Date:        30-Jan-2017',
        '',
        '                             Self        Reference',
        '  Row    Description       Abs-Max        Abs-Max      % Diff',
        '-------  -----------    -------------  -------------  --------',
        '      5  Damper 2            0.423522       0.415255     1.99*',
        '',
        '',
        '',
        '    Spring & Damper Forces - Abs-Max Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         2.00          3     60.00',
        '         3.00          2     40.00',
        '',
        '    0.0% of values are within 1%',
        '    60.0% of values are within 2%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.00, 3.00, 2.4000, 0.5477]',
        '']
    _comp_rpt(s, sbe)

    with StringIO() as f:
        # mxmn2 has different number of rows:
        assert_raises(
            ValueError, cla.rptpct1, results['FFN 0']['kc_forces'],
            results['FFN 1']['kc_forces'].ext[:4], f, **opts)

    drminfo0 = results['FFN 0']['kc_forces'].drminfo
    drminfo1 = results['FFN 1']['kc_forces'].drminfo
    drminfo0.labels = drminfo1.labels[:4]
    with StringIO() as f:
        # labels is wrong length
        assert_raises(
            ValueError, cla.rptpct1, results['FFN 0']['kc_forces'],
            results['FFN 1']['kc_forces'].ext, f, **opts)

    opts = {'domagpct': False,
            'dohistogram': False,
            'shortabsmax': True,
            'ignorepv' : np.array([False, False, True,
                                   False, True, True]),
            'roundvals': 3,
            'perpage': 3,
            }
    drminfo0.labels = drminfo1.labels[:]
    with StringIO() as f:
        cla.rptpct1(results['FFN 0']['kc_forces'],
                    results['FFN 1']['kc_forces'].ext, f, **opts)
        s = f.getvalue().split('\n')
    sbe = [
        'PERCENT DIFFERENCE REPORT',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Filter:      1e-06',
        'Notes:       % Diff = +/- abs((Self-Reference)/Reference)*100',
        '             Sign set such that positive % differences indicate exceedances',
        'Date:        30-Jan-2017',
        '',
        '                             Self        Reference',
        '  Row    Description       Abs-Max        Abs-Max      % Diff',
        '-------  -----------    -------------  -------------  -------',
        '      1  Spring 1            5.753000       5.603000     2.68',
        '      2  Spring 2            1.931000       1.888000     2.28',
        '      3  Spring 3            5.681000       5.572000  n/a    ',
        '',
        'PERCENT DIFFERENCE REPORT',
        '',
        'Description: Spring & Damper Forces',
        'Uncertainty: [Rigid, Elastic, Dynamic, Static] = [1, 1, 1, 1]',
        'Units:       N',
        'Filter:      1e-06',
        'Notes:       % Diff = +/- abs((Self-Reference)/Reference)*100',
        '             Sign set such that positive % differences indicate exceedances',
        'Date:        30-Jan-2017',
        '',
        '                             Self        Reference',
        '  Row    Description       Abs-Max        Abs-Max      % Diff',
        '-------  -----------    -------------  -------------  -------',
        '      4  Damper 1            1.761000       1.714000     2.74',
        '      5  Damper 2            0.424000       0.415000  n/a    ',
        '      6  Damper 3            0.893000       0.874000  n/a    ',
        '',
        '',
        '',
        '    Spring & Damper Forces - Abs-Max Comparison Histogram',
        '',
        '      % Diff      Count    Percent',
        '     --------   --------   -------',
        '         2.00          1     33.33',
        '         3.00          2     66.67',
        '',
        '    0.0% of values are within 1%',
        '    33.3% of values are within 2%',
        '    100.0% of values are within 5%',
        '',
        '    % Diff Statistics: [Min, Max, Mean, StdDev] = [2.00, 3.00, 2.6667, 0.5774]',
        '']
    _comp_rpt(s, sbe)


# pyyeti/cla.py 1803 45 98% 91, 98, 5007-5013, 5046-5047, 5049-5050,
# 5064, 5131-5138, 5146-5150, 5158, 5169-5173, 5182, 5188, 5211-5214,
# 5216-5219, 5222-5224, 5233-5235, 5280-5282
