# -*- coding: utf-8 -*-
import os
import copy
import warnings
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyyeti.nastran import n2p
from ._utilities import get_marker_cycle


def mk_plots(res, event=None, issrs=True, Q='auto', drms=None,
             inc0rb=True, fmt='pdf', onepdf=True, layout=(2, 3),
             figsize=(11, 8.5), showall=None, showboth=False,
             cases=None, direc='srs_plots', tight_layout_args=None,
             plot=plt.plot, show_figures=False):
    """
    Make SRS or response history plots

    Parameters
    ----------
    res : :class:`DR_Results` instance
        Subclass of dict containing categories with results (see
        :class:`DR_Results`). For example, ``results['SC_atm'].ext``.
    event : string or None; optional
        String for plot titles and file names (eg: 'Liftoff'). If
        None, `event` is determined from `res[drm].event`.
    issrs : bool; optional
        True if plotting SRS data; False otherwise.
    Q : scalar or iterable or 'auto'; optional
        The Q value(s) to plot. If 'auto', all the Q values for
        each category are plotted. Must be a scalar if `showall`
        is True (see below).
    drms : list of data recovery categories or None; optional
        Data recovery categories to plot. If None, plot all
        available. See also input `inc0rb`.
    inc0rb : bool; optional
        If True, the '_0rb' versions of each data recovery
        category are automatically included.
    fmt : string or None; optional
        If `fmt` == "pdf", all plots are written to one PDF file,
        unless `onepdf` is set to False.  If `fmt` is some other
        string, it is used as the `format` parameter in
        :func:`matplotlib.pyplot.savefig`. If None, no figures
        will be saved. Typical values for `fmt` are (from
        ``fig.canvas.get_supported_filetypes()``)::

            'eps': 'Encapsulated Postscript',
            'jpeg': 'Joint Photographic Experts Group',
            'jpg': 'Joint Photographic Experts Group',
            'pdf': 'Portable Document Format',
            'pgf': 'PGF code for LaTeX',
            'png': 'Portable Network Graphics',
            'ps': 'Postscript',
            'raw': 'Raw RGBA bitmap',
            'rgba': 'Raw RGBA bitmap',
            'svg': 'Scalable Vector Graphics',
            'svgz': 'Scalable Vector Graphics',
            'tif': 'Tagged Image File Format',
            'tiff': 'Tagged Image File Format'

        File naming conventions: if 'SC_atm' is a category, then
        example output filenames could be::

            'SC_atm_srs.pdf'
            'SC_atm_eqsine.pdf'
            'SC_atm_srs_0.png', 'SC_atm_srs_1.png', ...
            'SC_atm_eqsine_0.png', 'SC_atm_eqsine_1.png', ...

    onepdf : bool or string; optional
        If `onepdf` evaluates to True and `fmt` is 'pdf', all plots
        are written to one PDF file where the name is:

          ========   =======================================
          `onepdf`   PDF file name
          ========   =======================================
           string    All plots saved in: `onepdf` + ".pdf"
           True      If `issrs`: all plots saved in:
                     `event` + "_srs.pdf"
           True      If not `issrs`: all plots saved in:
                     `event` + "_??.pdf", where "??" is
                     either 'hist', 'psd', or 'frf' as
                     appropriate.
          ========   =======================================

        If False, each figure is saved to its own file named as
        described above (see `fmt`).
    layout : 2-element tuple/list; optional
        Subplot layout, eg: (2, 3) for 2 rows by 3 columns. See also
        `figsize`.
    figsize : 2-element tuple/list; optional
        Define page size in inches. See also `layout`.
    showall : bool or None; optional
        If True, show all SRS curves for all cases; otherwise just
        plot envelope. If None and `showboth` is True, `showall`
        is set to True.
    showboth : bool; optional
        If True, shows all SRS curves and the envelope; otherwise
        just plot which ever `showall` indicates.
    direc : string; optional
        Directory name to put all output plot files; will be
        created if it doesn't exist.
    cases : tuple/list of case names to plot or None; optional
        If None, all cases are plotted. This option is ignored if
        plotting SRS curves and `showall` is True.
    tight_layout_args : dict or None; optional
        Arguments for :func:`plt.tight_layout`. If None, defaults to::

                {'pad': 3.0,
                 'w_pad': 2.0,
                 'h_pad': 2.0,
                 'rect': (0.3 / figsize[0],
                          0.3 / figsize[1],
                          1.0 - 0.3 / figsize[0],
                          1.0 - 0.3 / figsize[1])}

    plot : function; optional
        The function that will draw each curve. Defaults to
        :func:`matplotlib.pyplot.plot`. As an example, for a plot with
        a linear X-axis but a log Y-axis, set `plot` to
        :func:`matplotlib.pyplot.semilogy`. You can also use a custom
        function of your own devising, but it is expected to accept
        the same arguments as :func:`matplotlib.pyplot.plot`.
    show_figures : bool; optional
        If True, plot figures will be displayed on the screen for
        interactive viewing. Warning: there may be many figures.

    Notes
    -----
    Set the `onepdf` parameter to a string to specify the name of the
    PDF file.

    Used by :func:`DR_Results.srs_plots` and
    :func:`DR_Results.resp_plots` for plot generation.
    """

    def _get_Qs(Q, res, name, showall):
        try:
            srsQs = res[name].srs.srs.keys()
        except AttributeError:
            return None
        srsQs = list(srsQs)
        if Q == 'auto':
            Qs = srsQs
        else:
            Q_in = n2p._ensure_iter(Q)
            Qs = []
            for q in Q_in:
                if q in srsQs:
                    Qs.append(q)
                else:
                    warnings.warn('no Q={} SRS data for {}'.
                                  format(q, name), RuntimeWarning)
            if len(Qs) == 0:
                return None
        Qs = n2p._ensure_iter(Qs)
        if len(Qs) > 1 and showall:
            raise ValueError('`Q` must be a scalar if `showall` '
                             'is true')
        return Qs

    def _set_vars(res, name, event, showall, showboth, cases):
        curres = res[name]
        res._check_labels_len(name, curres)
        if issrs:
            labels = curres.drminfo.srslabels
            rowpv = curres.drminfo.srspv
            lbl = srstype = curres.srs.type
            units = curres.drminfo.srsunits
            if showall:
                cases = curres.cases
                if showboth:
                    lbl = lbl + '_all_env'
                else:
                    lbl = lbl + '_all'
            else:
                cases = None
        else:
            labels = curres.drminfo.histlabels
            rowpv = curres.drminfo.histpv
            lbl = 'resp'
            srstype = None
            units = curres.drminfo.histunits
            if cases is None:
                cases = curres.cases
            else:
                for case in cases:
                    if case not in curres.cases:
                        raise ValueError('case {} not found for'
                                         ' {}'.format(case, name))

        if isinstance(rowpv, slice):
            rowpv = np.arange(len(curres.drminfo.labels))[rowpv]
        maxlen = len(max(labels, key=len))
        if event is not None:
            sname = event
        else:
            sname = curres.event

        return (labels, rowpv, maxlen, sname,
                srstype, lbl, units, cases)

    def _get_figname(nplots, perpage, fmt, onepdf,
                     name, lbl, sname):
        if nplots > perpage:
            if fmt == 'pdf' and onepdf:
                prefix = '{}_{}'.format(name, lbl)
                figname = '{} {}_{}'.format(sname, prefix,
                                            filenum)
            else:
                prefix = '{}_{}_{}'.format(name, lbl, filenum)
                figname = '{} {}'.format(sname, prefix)
        else:
            prefix = '{}_{}'.format(name, lbl)
            figname = '{} {}'.format(sname, prefix)
        return prefix, figname

    def _prep_subplot(rows, cols, sub, perpage, filenum, nplots,
                      fmt, name, lbl, sname, figsize, prefix):
        sub += 1
        if sub > perpage:
            sub = 1
            filenum += 1
            prefix, figname = _get_figname(
                nplots, perpage, fmt, onepdf, name, lbl, sname)
            if show_figures:
                plt.figure(figname, figsize=figsize)
                plt.clf()
            else:
                nonlocal cur_fig
                cur_fig = plt.figure(figsize=figsize)
        ax = plt.subplot(rows, cols, sub)
        ax.ticklabel_format(useOffset=False,
                            style='sci', scilimits=(-3, 4))
        txt = ax.get_yaxis().get_offset_text()
        txt.set_x(-0.22)
        txt.set_va('bottom')
        plt.grid(True)
        return sub, filenum, prefix

    def _add_title(name, label, maxlen, sname, row, cols,
                   q=None):
        def _add_q(ttl, q):
            if q is not None:
                ttl = '{}, Q={}'.format(ttl, q)
            return ttl

        if cols == 1:
            small = 'medium'
            big = 'large'
        elif cols == 2:
            small = 10
            big = 'large'
        else:
            small = 8
            big = 12

        if maxlen > 35:
            ttl = '{} {}\nRow {}'.format(name, sname, row)
            plt.annotate(label, xy=(0, 1),
                         xycoords='axes fraction',
                         fontsize=small,
                         xytext=(3, -3),
                         textcoords='offset points',
                         ha='left', va='top')
        else:
            ttl = '{} {}\n{}'.format(name, sname, label)
        ttl = _add_q(ttl, q)
        plt.title(ttl, fontsize=big)

    def _add_legend(leg_info):
        ax = plt.gca()
        fig = plt.gcf()
        handles, labels = ax.get_legend_handles_labels()
        if 'rect' in tight_layout_args:
            lx = tight_layout_args['rect'][2]
            ly = tight_layout_args['rect'][3]
        else:
            lx = 1.0 - 0.3 / figsize[0]
            ly = 1.0 - 0.3 / figsize[1]
        leg = fig.legend(
            handles,
            labels,
            loc='upper right',
            bbox_to_anchor=(lx, ly),
            fontsize='small',
            framealpha=0.5,
            # fancybox=True,
            # borderaxespad=0.,
            # labelspacing=legspace*.9,
        )
        legwidth = (leg.get_tightbbox(fig.canvas.get_renderer())
                    .inverse_transformed(fig.transFigure)
                    .width)
        leg_info[0] = leg
        leg_info[1] = legwidth

    def _legend_layout(leg_info):
        tla = tight_layout_args.copy()
        if leg_info[0]:
            if 'rect' in tla:
                tla['rect'] = (
                    tla['rect'][0],
                    tla['rect'][1],
                    tla['rect'][2] - leg_info[1],
                    tla['rect'][3])
            else:
                tla['rect'] = (0, 0, 1 - leg_info[1], 1)

        # if the legend belongs to an axes object, don't include it in
        # the tight_layout calculations:
        # 1: leg_in_layout = leg_info[0].get_in_layout()
        # 2: leg_info[0].set_in_layout(False)
        plt.tight_layout(**tla)
        # 3: leg_info[0].set_in_layout(leg_in_layout)
        leg_info[0] = None

    def _mark_srs(x, y, line, marker, label, **kwargs):
        me = signal.argrelextrema(y, np.greater)[0]
        return plot(x, y, line, marker=marker,
                    markevery=list(me), label=label, **kwargs)

    def _plot_all(curres, q, frq, hist, showboth, cases, sub,
                  cols, maxcol, name, label, maxlen,
                  sname, rowpv, j, leg_info):
        # legspace = matplotlib.rcParams['legend.labelspacing']
        if issrs:
            srsall = curres.srs.srs[q]
            srsext = curres.srs.ext[q]
            # srsall (cases x rows x freq)
            # srsext (each rows x freq)
            h = []
            marker = get_marker_cycle()
            for n, case in enumerate(cases):
                h += _mark_srs(
                    x, srsall[n, j], '-',
                    marker=next(marker), label=case)
            if showboth:
                # set zorder=-1 ?
                h.insert(0,
                         _mark_srs(
                             x, srsext[j], 'k-',
                             lw=2, alpha=0.5, marker=next(marker),
                             label=curres.event)[0])
        else:
            # hist (cases x rows x time | freq)
            h = []
            for n, case in enumerate(cases):
                h += plot(x, hist[n, j], linestyle='-', label=case)
        if sub == maxcol:
            _add_legend(leg_info)
        _add_title(name, label, maxlen, sname,
                   rowpv[j] + 1, cols, q)

    def _plot_ext(curres, q, frq, sub, cols, maxcol, name,
                  label, maxlen, sname, rowpv, j):
        srsext = curres.srs.ext[q]
        # srsext (each rows x freq)
        if sub == maxcol:
            plot(frq, srsext[j], label='Q={}'.format(q))
            plt.legend(loc='best', fontsize='small',
                       fancybox=True, framealpha=0.5)
        else:
            plot(frq, srsext[j])
        if q == Qs[0]:
            _add_title(name, label, maxlen, sname,
                       rowpv[j] + 1, cols)

    def _add_xy_labels(uj, xlab, ylab, srstype):
        if isinstance(units, str):
            u = units
        else:
            if uj > len(units):
                uj = 0
            u = units[uj]
            if len(units) < nplots:
                if sub == 1 or sub == 4:
                    uj += 1   # each label goes to 3 rows
            else:
                uj += 1
        if issrs:
            if srstype == 'eqsine':
                plt.ylabel('EQ-Sine ({})'.format(u))
            else:
                plt.ylabel('SRS ({})'.format(u))
        else:
            if ylab.startswith('PSD'):
                if len(u) == 1:
                    uu = ' ({}$^2$/Hz)'.format(u)
                else:
                    uu = ' ({})$^2$/Hz'.format(u)
                plt.ylabel(ylab + uu)
            else:
                plt.ylabel(ylab + ' ({})'.format(u))
        plt.xlabel(xlab)
        return uj

    # main routine
    if tight_layout_args is None:
        tight_layout_args = {
            'pad': 3.0,
            'w_pad': 2.0,
            'h_pad': 2.0,
            'rect': (0.3 / figsize[0],
                     0.3 / figsize[1],
                     1.0 - 0.3 / figsize[0],
                     1.0 - 0.3 / figsize[1])}

    if showboth and showall is None:
        showall = True

    if not os.path.exists(direc):
        os.mkdir(direc)

    rows = layout[0]
    cols = layout[1]
    perpage = rows * cols
    orientation = ('landscape' if figsize[0] > figsize[1]
                   else 'portrait')

    if drms is None:
        alldrms = sorted(res)
    else:
        alldrms = copy.copy(drms)
        if inc0rb:
            for name in drms:
                if name + '_0rb' in res:
                    alldrms.append(name + '_0rb')

    pdffile = None
    imode = plt.isinteractive()
    plt.interactive(show_figures)
    cur_fig = None

    try:
        for name in alldrms:
            if name not in res:
                raise ValueError('category {} does not exist.'
                                 .format(name))
            if issrs:
                if 'srs' not in res[name].__dict__:
                    if drms and name in drms:
                        warnings.warn(
                            'no SRS data for {}'.format(name),
                            RuntimeWarning)
                    continue
                Qs = _get_Qs(Q, res, name, showall)
                if Qs is None:
                    continue
                x = res[name].srs.frq
                y = None
                xlab = 'Frequency (Hz)'
                ylab = None
                ptype = 'srs'
            else:
                if 'hist' in res[name].__dict__:
                    x = res[name].time
                    y = res[name].hist
                    xlab = 'Time (s)'
                    ylab = 'Response'
                    ptype = 'hist'
                elif 'psd' in res[name].__dict__:
                    x = res[name].freq
                    y = res[name].psd
                    xlab = 'Frequency (Hz)'
                    ylab = 'PSD'
                    ptype = 'psd'
                elif 'frf' in res[name].__dict__:
                    x = res[name].freq
                    y = abs(res[name].frf)
                    xlab = 'Frequency (Hz)'
                    ylab = 'FRF'
                    ptype = 'frf'
                else:
                    if drms and name in drms:
                        warnings.warn('no response data for {}'.
                                      format(name), RuntimeWarning)
                    continue

            (labels, rowpv, maxlen,
             sname, srstype, lbl,
             units, _cases) = _set_vars(res, name, event, showall,
                                        showboth, cases)

            if fmt == 'pdf' and onepdf and pdffile is None:
                if isinstance(onepdf, str):
                    fname = os.path.join(direc, onepdf)
                    if not fname.endswith('.pdf'):
                        fname = fname + '.pdf'
                else:
                    fname = os.path.join(
                        direc, '{}_{}.pdf'.format(sname, ptype))
                pdffile = PdfPages(fname)

            filenum = 0
            uj = 0   # units index
            nplots = len(rowpv)
            maxcol = cols if nplots > cols else nplots
            sub = perpage
            prefix = None
            leg_info = [None, 0.0]
            for j in range(nplots):
                sub, filenum, prefix = _prep_subplot(
                    rows, cols, sub, perpage, filenum, nplots,
                    fmt, name, lbl, sname, figsize, prefix)
                label = ' '.join(labels[j].split())
                if issrs:
                    for q in Qs:
                        if showall:
                            _plot_all(
                                res[name], q, x, y,
                                showboth, _cases, sub, cols,
                                maxcol, name, label,
                                maxlen, sname, rowpv, j,
                                leg_info)
                        else:
                            _plot_ext(
                                res[name], q, x, sub, cols,
                                maxcol, name, label,
                                maxlen, sname, rowpv, j)
                else:
                    _plot_all(
                        res[name], None, x, y, showboth,
                        _cases, sub, cols, maxcol,
                        name, label, maxlen, sname,
                        rowpv, j, leg_info)
                _add_xy_labels(uj, xlab, ylab, srstype)

                if j + 1 == nplots or (j + 1) % perpage == 0:
                    _legend_layout(leg_info)

                    if fmt == 'pdf' and onepdf:
                        pdffile.savefig()
                        # orientation=orientation,
                        # papertype='letter')
                    elif fmt:
                        fname = os.path.join(direc,
                                             prefix + '.' + fmt)
                        if fmt != 'pdf':
                            kwargs = dict(
                                orientation=orientation,
                                dpi=200, bbox_inches='tight')
                        else:
                            kwargs = {}
                        plt.savefig(fname, format=fmt, **kwargs)
                    if not show_figures:
                        plt.close(cur_fig)
                        cur_fig = None
    finally:
        plt.interactive(imode)
        if pdffile:
            pdffile.close()
        if cur_fig:
            plt.close(cur_fig)
