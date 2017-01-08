import os
from subprocess import run

import sys
import itertools
import shutil
import inspect
import re
from types import SimpleNamespace
from glob import glob
import numpy as np
from scipy.io import matlab
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
                drm, ext_results[drm][event])

    # Done with setup; now we can use the standard cla tools:
    results.form_extreme(ext_name)
    return results


def test_rptpct1():
    results = cla.DR_Results()
    results.merge(
        (get_fake_cla_results('FLAC', _get_labels0, 0),
         get_fake_cla_results('VLC', _get_labels1, 1),
         get_fake_cla_results('PostVLC', _get_labels2, 2)),
        {'FLAC': 'FDLC',
         'PostVLC': 'VCL2'})

    results.form_extreme()
    assert repr(results).startswith('<pyyeti.cla.DR_Results object')
    assert str(results) == ('DR_Results with 4 categories: '
                            '[FDLC, VLC, VCL2, extreme]')
    try:
        results['extreme'].rpttab(direc='./temp_tab', excel='results')
        results['extreme'].rpttab(direc='./temp_tab')
        results['FDLC']['extreme'].rpttab(direc='./temp_fdlc_tab')
        # check results somehow
    finally:
        shutil.rmtree('./temp_tab', ignore_errors=True)
        shutil.rmtree('./temp_fdlc_tab', ignore_errors=True)


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

        # make some srs plots and tab files:
        # results.rptext()
        # results.rpttab()
        #results.rpttab(excel=event.lower())
        #results.srs_plots()
        results.srs_plots(Q=10, direc='srs_cases', showall=True,
                          plot=plt.semilogy)
        #results.resp_plots()

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
    with cd('summary'):
        pth = '../' + pth
        # Load both sets of results and report percent differences:
        results = cla.load('results.pgz')
        lvc = cla.load(pth+'summary/contractor_results_no_srs.pgz')
        results['extreme'].rptpct(lvc, names=('LSP', 'Contractor'))


def confirm(pth):
    cmp_files = glob('summary/compare/*.cmp')
    assert len(cmp_files) == 6
    for n in cmp_files:
        with open(n) as f:
            count = 0
            for line in f:
                if '% Diff Statistics:' in line:
                    count += 1
                    p = line.index(' = [')
                    stats = np.array([float(i)
                                      for i in line[p+4:-2].split(',')])
                    assert np.all(stats == 0.0)
        assert count == 3


def test_transfer_orbit_cla():
    try:
        if os.path.exists('temp_cla'):
            shutil.rmtree('./temp_cla', ignore_errors=True)
        os.mkdir('temp_cla')
        pth = '../pyyeti/tests/cla_test_data/'
        with cd('temp_cla'):
            prepare_4_cla(pth)
            toes(pth)
            owlab(pth)
            toeco(pth)
            summarize(pth)
            compare(pth)
            confirm(pth)
    finally:
        shutil.rmtree('./temp_cla', ignore_errors=True)


#    files = [
#        './temp_cla/prepare_4_cla.py',
#        './temp_cla/toes/toes.py',
#        './temp_cla/owlab/owlab.py',
#        './temp_cla/toeco/toeco.py',
#        './temp_cla/summary/summarize.py',
#        './temp_cla/summary/compare.py',
#        ]
#    try:
#        shutil.copytree('pyyeti/tests/cla_test_data', './temp_cla')
#        for fn in files:
#            direc, name = os.path.split(fn)
#            # print('Running {}'.format(name))
#            with cd(direc):
#                run(['python', name], check=True)
#    finally:
#        pass
#        # shutil.rmtree('./temp_cla', ignore_errors=True)


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

# pyyeti/cla.py 1801 168 91% 91, 98, 2162-2163, 2166,
# 2314, 2324, 2329, 2496, 2515, 2528, 2559, 2936-2937, 2941, 2946,
# 2981-2986, 2998, 3018, 3080-3083, 3186, 3254, 3268, 3611-3612,
# 3620-3622, 3634, 3713, 3725, 3738, 3745-3746, 3753, 3769, 3785,
# 3795, 3847, 3862, 3871, 4017, 4038, 4058, 4061-4063, 4374,
# 4392-4393, 4395, 4404, 4417-4425, 4430-4438, 4445-4451, 4465, 4489,
# 4491, 4521, 4562-4582, 4592, 4619, 4638, 4693, 4703, 4715, 4731,
# 4733, 4735-4736, 4764, 4794-4795, 4800, 4930, 4933, 4936, 4951,
# 4965-4967, 4976, 4984-4990, 5023-5024, 5026-5027, 5041, 5064,
# 5108-5115, 5123-5127, 5135, 5146-5150, 5155-5159, 5165, 5172-5174,
# 5178, 5190-5193, 5195-5198, 5201-5203, 5212-5214, 5259-5261
