import itertools
import shutil
import numpy as np
from nose.tools import *
from pyyeti import cla
from pyyeti.pp import PP


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
        # drfile is required, but can just use __name__ for
        # a dummy placeholder (will see a warning)
        # drfile = __name__
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
    try:
        results['extreme'].rpttab(direc='./temp_tab', excel='results')
        results['extreme'].rpttab(direc='./temp_tab')
        results['FDLC']['extreme'].rpttab(direc='./temp_fdlc_tab')
        # check results somehow
    finally:
        shutil.rmtree('./temp_tab', ignore_errors=True)
        shutil.rmtree('./temp_fdlc_tab', ignore_errors=True)
