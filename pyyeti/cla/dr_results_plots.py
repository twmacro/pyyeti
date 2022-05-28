# -*- coding: utf-8 -*-
import os
import copy
import warnings
import numpy as np
import scipy.signal as signal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from matplotlib.backends.backend_pdf import PdfPages
from pyyeti.nastran import n2p
from ._utilities import get_marker_cycle


def _get_Qs(Q, res, name, showall):
    try:
        srsQs = res[name].srs.srs.keys()
    except AttributeError:
        return None
    srsQs = list(srsQs)
    if Q == "auto":
        Qs = srsQs
    else:
        Q_in = n2p._ensure_iter(Q)
        Qs = []
        for q in Q_in:
            if q in srsQs:
                Qs.append(q)
            else:
                warnings.warn(f"no Q={q} SRS data for {name}", RuntimeWarning)
        if len(Qs) == 0:
            return None
    Qs = n2p._ensure_iter(Qs)
    if len(Qs) > 1 and showall:
        raise ValueError("`Q` must be a scalar if `showall` is true")
    return Qs


def _set_vars(issrs, res, name, event, showall, showboth, cases):
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
                lbl = lbl + "_all_env"
            else:
                lbl = lbl + "_all"
        else:
            cases = None
    else:
        labels = curres.drminfo.histlabels
        rowpv = curres.drminfo.histpv
        lbl = "resp"
        srstype = None
        units = curres.drminfo.histunits
        if cases is None:
            cases = curres.cases
        else:
            for case in cases:
                if case not in curres.cases:
                    raise ValueError(f"case {case} not found for {name}")

    if isinstance(rowpv, slice):
        rowpv = np.arange(len(curres.drminfo.labels))[rowpv]
    maxlen = len(max(labels, key=len))
    if event is not None:
        sname = event
    else:
        sname = curres.event

    return (labels, rowpv, maxlen, sname, srstype, lbl, units, cases)


def _get_figname(nplots, perpage, fmt, onepdf, name, lbl, sname, filenum):
    if nplots > perpage:
        if fmt == "pdf" and onepdf:
            prefix = f"{name}_{lbl}"
            figname = f"{sname} {prefix}_{filenum}"
        else:
            prefix = f"{name}_{lbl}_{filenum}"
            figname = f"{sname} {prefix}"
    else:
        prefix = f"{name}_{lbl}"
        figname = f"{sname} {prefix}"
    return prefix, figname


def _prep_subplot(
    rows,
    cols,
    sub,
    perpage,
    filenum,
    nplots,
    fmt,
    name,
    lbl,
    sname,
    figsize,
    prefix,
    onepdf,
    show_figures,
    cur_fig,
):
    sub += 1
    if sub > perpage:
        sub = 1
        prefix, figname = _get_figname(
            nplots, perpage, fmt, onepdf, name, lbl, sname, filenum
        )
        filenum += 1
        if show_figures:
            cur_fig = plt.figure(figname, figsize=figsize, clear=True)
        else:
            cur_fig = Figure(figsize=figsize)
            FigureCanvasAgg(cur_fig)

    ax = cur_fig.add_subplot(rows, cols, sub)
    ax.ticklabel_format(useOffset=False, style="sci", scilimits=(-3, 4))
    txt = ax.get_yaxis().get_offset_text()
    txt.set_x(-0.22)
    txt.set_va("bottom")
    ax.grid(True)
    return sub, filenum, prefix, cur_fig, ax


def _add_title(ax, name, label, maxlen, sname, row, cols, q=None):
    def _add_q(ttl, q):
        if q is not None:
            ttl = f"{ttl}, Q={q}"
        return ttl

    if cols == 1:
        small = "medium"
        big = "large"
    elif cols == 2:
        small = 10
        big = "large"
    else:
        small = 8
        big = 12

    if maxlen > 35:
        ttl = f"{name} {sname}\nRow {row}"
        ax.annotate(
            label,
            xy=(0, 1),
            xycoords="axes fraction",
            fontsize=small,
            xytext=(3, -3),
            textcoords="offset points",
            ha="left",
            va="top",
        )
    else:
        ttl = f"{name} {sname}\n{label}"
    ttl = _add_q(ttl, q)
    ax.set_title(ttl, fontsize=big)


def _get_legopts(legopts, legend_args):
    if legend_args:
        legopts.update(legend_args)
    return legopts


def _add_legend(ax, leg_info, figsize, tight_layout_args, legend_args):
    fig = ax.get_figure()
    handles, labels = ax.get_legend_handles_labels()
    if "rect" in tight_layout_args:
        lx = tight_layout_args["rect"][2]
        ly = tight_layout_args["rect"][3]
    else:
        lx = 1.0 - 0.3 / figsize[0]
        ly = 1.0 - 0.3 / figsize[1]

    legopts = {
        "loc": "upper right",
        "bbox_to_anchor": (lx, ly),
        "fontsize": "small",
        "framealpha": 0.5,
        # "fancybox": True,
        # "borderaxespad": 0.,
        # "labelspacing": legspace * 0.9,
    }
    leg = fig.legend(handles, labels, **_get_legopts(legopts, legend_args))
    legwidth = (
        leg.get_tightbbox(fig.canvas.get_renderer())
        .transformed(fig.transFigure.inverted())
        .width
    )

    leg_info[0] = leg
    leg_info[1] = legwidth


def _legend_layout(fig, leg_info, tight_layout_args):
    tla = tight_layout_args.copy()
    if leg_info[0]:
        if "rect" in tla:
            tla["rect"] = (
                tla["rect"][0],
                tla["rect"][1],
                tla["rect"][2] - leg_info[1],
                tla["rect"][3],
            )
        else:
            tla["rect"] = (0, 0, 1 - leg_info[1], 1)

    # if the legend belongs to an axes object, don't include it in
    # the tight_layout calculations:
    # 1: leg_in_layout = leg_info[0].get_in_layout()
    # 2: leg_info[0].set_in_layout(False)
    fig.tight_layout(**tla)
    # 3: leg_info[0].set_in_layout(leg_in_layout)
    leg_info[0] = None


def _mark_srs(plotfunc, x, y, line, marker, label, **kwargs):
    if len(y) > 1:
        me = list(signal.argrelextrema(y, np.greater)[0])
        # to capture end points too:
        if y[0] > y[1]:
            me.insert(0, 0)
        if y[-1] > y[-2]:
            me.append(len(y) - 1)
    else:
        me = [0]
    return plotfunc(x, y, line, marker=marker, markevery=me, label=label, **kwargs)


def _plot_all(
    ax,
    issrs,
    plotfunc,
    curres,
    q,
    x,
    hist,
    showboth,
    cases,
    sub,
    cols,
    maxcol,
    name,
    label,
    maxlen,
    sname,
    rowpv,
    j,
    leg_info,
    figsize,
    tight_layout_args,
    legend_args,
):
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
                plotfunc, x, srsall[n, j], "-", marker=next(marker), label=case
            )
        if showboth:
            # set zorder=-1 ?
            h.insert(
                0,
                _mark_srs(
                    plotfunc,
                    x,
                    srsext[j],
                    "k-",
                    lw=2,
                    alpha=0.5,
                    marker=next(marker),
                    label=curres.event,
                )[0],
            )
    else:
        # hist (cases x rows x time | freq)
        h = []
        for n, case in enumerate(cases):
            h += plotfunc(x, hist[n, j], linestyle="-", label=case)

    if sub == maxcol:
        _add_legend(ax, leg_info, figsize, tight_layout_args, legend_args)
    _add_title(ax, name, label, maxlen, sname, rowpv[j] + 1, cols, q)


def _plot_ext(
    ax,
    plotfunc,
    curres,
    q,
    Qs,
    frq,
    sub,
    cols,
    maxcol,
    name,
    label,
    maxlen,
    sname,
    rowpv,
    j,
    legend_args,
):
    srsext = curres.srs.ext[q]
    # srsext (each rows x freq)
    if sub == maxcol:
        plotfunc(frq, srsext[j], label=f"Q={q}")
        legopts = {
            "loc": "best",
            "fontsize": "small",
            "framealpha": 0.5,
            "fancybox": True,
        }
        ax.legend(**_get_legopts(legopts, legend_args))
    else:
        plotfunc(frq, srsext[j])
    if q == Qs[0]:
        _add_title(ax, name, label, maxlen, sname, rowpv[j] + 1, cols)


def _add_xy_labels(ax, issrs, units, uj, xlab, ylab, nplots, sub, srstype):
    if isinstance(units, str):
        u = units
    else:
        if uj > len(units):
            uj = 0
        u = units[uj]
        if len(units) < nplots:
            if sub == 1 or sub == 4:
                uj += 1  # each label goes to 3 rows
        else:
            uj += 1
    if issrs:
        if srstype == "eqsine":
            ax.set_ylabel(f"EQ-Sine ({u})")
        else:
            ax.set_ylabel(f"SRS ({u})")
    else:
        if ylab.startswith("PSD"):
            if len(u) == 1:
                uu = f" ({u}$^2$/Hz)"
            else:
                uu = f" ({u})$^2$/Hz"
            ax.set_ylabel(ylab + uu)
        else:
            ax.set_ylabel(ylab + f" ({u})")
    ax.set_xlabel(xlab)
    return uj


def mk_plots(
    res,
    *,
    event=None,
    issrs=True,
    Q="auto",
    drms=None,
    inc0rb=True,
    fmt="pdf",
    onepdf=True,
    layout=(2, 3),
    figsize=(11, 8.5),
    showall=None,
    showboth=False,
    cases=None,
    direc="srs_plots",
    tight_layout_args=None,
    legend_args=None,
    plot="plot",
    show_figures=False,
):
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
        :meth:`matplotlib.figure.Figure.savefig`. If None, no figures
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
        Arguments for :meth:`matplotlib.figure.Figure.tight_layout`.
        If None, defaults to::

            {
               "pad": 3.0,
               "w_pad": 2.0,
               "h_pad": 2.0,
               "rect": (0.3 / figsize[0],
                        0.3 / figsize[1],
                        1.0 - 0.3 / figsize[0],
                        1.0 - 0.3 / figsize[1]),
            }

    legend_args : dict or None; optional
        Arguments for :meth:`matplotlib.figure.Figure.legend` or
        :meth:`matplotlib.axes.Axes.legend`. The internally set
        arguments for the legend call depend on `showall` and
        `issrs`. If `issrs` is False and/or `showall` is True,
        `legend_args` defaults to::

            {
               "loc": "upper right",
               "bbox_to_anchor": (lx, ly),
               "fontsize": "small",
               "framealpha": 0.5,
            }

        If `issrs` is True and `showall` is False (only plotting SRS
        envelopes)::

            {
               "loc": "best",
               "fontsize": "small",
               "framealpha": 0.5,
               "fancybox": True,
            }

        If `legend_args` is a dictionary, it will update those default
        settings.
    plot : string; optional
        The name of a function in :class:`matplotlib.axes.Axes` that
        will draw each curve. Defaults to "plot". Common options:

            +------------+
            | `plot`     |
            +============+
            | "plot"     |
            +------------+
            | "loglog"   |
            +------------+
            | "semilogx" |
            +------------+
            | "semilogy" |
            +------------+

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

    if not isinstance(plot, str):
        raise TypeError(f"`plot` must be a string, not a {type(plot)}")

    if tight_layout_args is None:
        tight_layout_args = {
            "pad": 3.0,
            "w_pad": 2.0,
            "h_pad": 2.0,
            "rect": (
                0.3 / figsize[0],
                0.3 / figsize[1],
                1.0 - 0.3 / figsize[0],
                1.0 - 0.3 / figsize[1],
            ),
        }

    if showboth and showall is None:
        showall = True

    if not os.path.exists(direc):
        os.mkdir(direc)

    rows = layout[0]
    cols = layout[1]
    perpage = rows * cols
    # The following removed in v0.95.5:
    # orientation = "landscape" if figsize[0] > figsize[1] else "portrait"

    if drms is None:
        alldrms = sorted(res)
    else:
        alldrms = copy.copy(drms)
        if inc0rb:
            for name in drms:
                if name + "_0rb" in res:
                    alldrms.append(name + "_0rb")

    pdffile = None
    cur_fig = None

    try:
        for name in alldrms:
            if name not in res:
                raise ValueError(f"category {name} does not exist.")
            if issrs:
                if "srs" not in res[name].__dict__:
                    if drms and name in drms:
                        warnings.warn(f"no SRS data for {name}", RuntimeWarning)
                    continue
                Qs = _get_Qs(Q, res, name, showall)
                if Qs is None:
                    continue
                x = res[name].srs.frq
                y = None
                xlab = "Frequency (Hz)"
                ylab = None
                ptype = "srs"
            else:
                if "hist" in res[name].__dict__:
                    x = res[name].time
                    y = res[name].hist
                    xlab = "Time (s)"
                    ylab = "Response"
                    ptype = "hist"
                elif "psd" in res[name].__dict__:
                    x = res[name].freq
                    y = res[name].psd
                    xlab = "Frequency (Hz)"
                    ylab = "PSD"
                    ptype = "psd"
                elif "frf" in res[name].__dict__:
                    x = res[name].freq
                    y = abs(res[name].frf)
                    xlab = "Frequency (Hz)"
                    ylab = "FRF"
                    ptype = "frf"
                else:
                    if drms and name in drms:
                        warnings.warn(f"no response data for {name}", RuntimeWarning)
                    continue

            (labels, rowpv, maxlen, sname, srstype, lbl, units, _cases) = _set_vars(
                issrs, res, name, event, showall, showboth, cases
            )

            if fmt == "pdf" and onepdf and pdffile is None:
                if isinstance(onepdf, str):
                    fname = os.path.join(direc, onepdf)
                    if not fname.endswith(".pdf"):
                        fname = fname + ".pdf"
                else:
                    fname = os.path.join(direc, f"{sname}_{ptype}.pdf")
                pdffile = PdfPages(fname)

            filenum = 0
            uj = 0  # units index
            nplots = len(rowpv)
            maxcol = cols if nplots > cols else nplots
            sub = perpage
            prefix = None
            leg_info = [None, 0.0]
            for j in range(nplots):
                sub, filenum, prefix, cur_fig, ax = _prep_subplot(
                    rows,
                    cols,
                    sub,
                    perpage,
                    filenum,
                    nplots,
                    fmt,
                    name,
                    lbl,
                    sname,
                    figsize,
                    prefix,
                    onepdf,
                    show_figures,
                    cur_fig,
                )
                plotfunc = getattr(ax, plot)
                label = " ".join(labels[j].split())
                if issrs:
                    for q in Qs:
                        if showall:
                            _plot_all(
                                ax,
                                issrs,
                                plotfunc,
                                res[name],
                                q,
                                x,
                                y,
                                showboth,
                                _cases,
                                sub,
                                cols,
                                maxcol,
                                name,
                                label,
                                maxlen,
                                sname,
                                rowpv,
                                j,
                                leg_info,
                                figsize,
                                tight_layout_args,
                                legend_args,
                            )
                        else:
                            _plot_ext(
                                ax,
                                plotfunc,
                                res[name],
                                q,
                                Qs,
                                x,
                                sub,
                                cols,
                                maxcol,
                                name,
                                label,
                                maxlen,
                                sname,
                                rowpv,
                                j,
                                legend_args,
                            )
                else:
                    _plot_all(
                        ax,
                        issrs,
                        plotfunc,
                        res[name],
                        None,
                        x,
                        y,
                        showboth,
                        _cases,
                        sub,
                        cols,
                        maxcol,
                        name,
                        label,
                        maxlen,
                        sname,
                        rowpv,
                        j,
                        leg_info,
                        figsize,
                        tight_layout_args,
                        legend_args,
                    )

                _add_xy_labels(ax, issrs, units, uj, xlab, ylab, nplots, sub, srstype)

                if j + 1 == nplots or (j + 1) % perpage == 0:
                    _legend_layout(cur_fig, leg_info, tight_layout_args)

                    if fmt == "pdf" and onepdf:
                        pdffile.savefig(cur_fig)
                    elif fmt:
                        fname = os.path.join(direc, prefix + "." + fmt)
                        cur_fig.savefig(fname, format=fmt)
                    if not show_figures:
                        cur_fig = None
    finally:
        if pdffile:
            pdffile.close()
