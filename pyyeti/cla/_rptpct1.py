# -*- coding: utf-8 -*-
"""
Low level tool for writing percent difference reports. Typically, this
is called via: :func:`cla.DR_Results.rptpct`.
"""
from io import StringIO
from types import SimpleNamespace
import warnings
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyyeti import ytools, locate, writer, guitools
from ._utilities import _get_rpt_headers, _get_numform, _proc_filterval
from ._magpct import magpct


__all__ = ["rptpct1"]


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def _apply_pv(value, pv, oldlen):
    # if value has a len that's > 1, try to partition it down;
    # otherwise, return it as is:
    try:
        n = len(value)
    except TypeError:
        return value
    else:
        if n == 1:
            return value

    # `value` is a vector with len > 1 ... ensure it is a true numpy
    # array:
    value = np.atleast_1d(value)

    # oldlen is either 0 (for `value` vectors that are expected to be
    # full size ... currently, only the `filterval` and
    # `magpct_filterval` vectors), or it is the length of the
    # dimension that the `value` index type of partition vector
    # (currently, only the `ignorepv` vector) was originally defined
    # to partition.
    if oldlen == 0:
        # `value` is `filterval` or `magpct_filterval` ... these just
        # need to be partitioned down:
        newvalue = value[pv]
    else:
        # `value` is `ignorepv` ... it needs to be redefined to
        # correspond to reduced size:
        truefalse = locate.index2bool(value, oldlen)
        newvalue = truefalse[pv].nonzero()[0]
    return newvalue


def _align_mxmn(mxmn1, mxmn2, labels2, row_number, infodct):
    if infodct["labels"] and infodct["labels"] != labels2:
        n = len(infodct["labels"])
        pv1, pv2 = locate.list_intersect(infodct["labels"], labels2)
        mxmn1 = mxmn1[pv1]
        mxmn2 = mxmn2[pv2]
        infodct["labels"] = [infodct["labels"][i] for i in pv1]
        row_number = row_number[pv1]
        infodct["filterval"] = _apply_pv(infodct["filterval"], pv1, 0)
        infodct["magpct_filterval"] = _apply_pv(infodct["magpct_filterval"], pv1, 0)
        infodct["ignorepv"] = _apply_pv(infodct["ignorepv"], pv1, n)
    return mxmn1, mxmn2, row_number


def _get_filtline(filterval):
    if len(filterval) > 1:
        filtline = "Filter:      <defined row-by-row>\n"
    else:
        filtline = f"Filter:      {filterval[0]}\n"
    return filtline


def _get_noteline(use_range, names, prtbads, flagbads):
    noteline = "Notes:       "
    tab = "             "
    if not use_range:
        noteline += "% Diff = +/- abs(({0}-{1})/{1})*100\n".format(*names)
    else:
        noteline += "% Diff = +/- abs({0}-{1})/max(abs({1}(max,min)))*100\n".format(
            *names
        )

    noteline += tab + "Sign set such that positive % differences indicate exceedances\n"
    prtbad, prtbadh, prtbadl = prtbads
    flagbad, flagbadh, flagbadl = flagbads
    if prtbad is not None or prtbadh is not None or prtbadl is not None:
        if prtbad is not None:
            prtbad = abs(prtbad)
            noteline += tab + f"Printing rows where abs(% Diff) > {prtbad}%\n"
        elif prtbadh is not None:
            noteline += tab + f"Printing rows where % Diff > {prtbadh}%\n"
        else:
            noteline += tab + f"Printing rows where % Diff < {prtbadl}%\n"

    if flagbad is not None or flagbadh is not None or flagbadl is not None:
        if flagbad is not None:
            flagbad = abs(flagbad)
            noteline += tab + f"Flagging (*) rows where abs(% Diff) > {flagbad}%\n"
        elif flagbadh is not None:
            noteline += tab + f"Flagging (*) rows where % Diff > {flagbadh}%\n"
        else:
            noteline += tab + f"Flagging (*) rows where % Diff < {flagbadl}%\n"
    return noteline


def _get_badpv(pct, pv, bad, badh, badl, defaultpv=False):
    if bad is not None or badh is not None or badl is not None:
        badpv = pv.copy()
        if bad is not None:
            badpv &= abs(pct) > bad
        elif badh is not None:
            badpv &= pct > badh
        else:
            badpv &= pct < badl
    else:
        badpv = np.empty(len(pct), bool)
        badpv[:] = defaultpv
    return badpv


def _get_pct_diff(a, b, filt, pv, nastring, mxmn_b=None, ismax=True, flagbads=None):
    # either can pass filter to be kept:
    pv &= (abs(a) > filt) | (abs(b) > filt)

    if mxmn_b is not None:
        denom = np.nanmax(abs(mxmn_b), axis=1)
    else:
        denom = abs(b)

    # put 1's in for filtered values ... this is temporary
    a = a.copy()
    b = b.copy()
    a[~pv] = 1.0
    b[~pv] = 1.0

    z = denom == 0.0
    denom[z] = 1.0
    pct = 100 * abs(a - b) / denom
    pct[z] = 100.0  # np.inf

    # make less extreme values negative
    neg = a < b if ismax else a > b
    pct[neg] *= -1.0

    # put nan's in for the filtered or n/a rows:
    pct[~pv] = np.nan

    # make 7 char version:
    spct = [f"{p:7.2f}" for p in pct]
    badpv = _get_badpv(pct, pv, *flagbads, False)
    for j in badpv.nonzero()[0]:
        spct[j] += "*"
    for j in (~pv).nonzero()[0]:
        spct[j] = nastring

    return pct, spct


def _get_histogram_str(desc, hdr, pctinfo):
    pctcount = pctinfo["hsto"]
    s = [
        (f"\n\n    {desc} - {hdr} Comparison Histogram\n\n"),
        ("      % Diff      Count    Percent\n     --------   --------   -------\n"),
    ]
    with StringIO() as f:
        writer.vecwrite(f, "     {:8.2f}   {:8.0f}   {:7.2f}\n", pctcount)
        s.append(f.getvalue())
        s.append("\n")

    # total_percent_10 will either be 0 or 1000:
    #  - 0 if all % diffs are "n/a"
    #  - 1000 otherwise
    total_percent_10 = np.round(pctcount[:, 2].sum() * 10)
    last = -1.0
    for pdiff in [1, 2, 5, 10, 15, 20, 25, 50, 100, 500]:
        pvdiff = abs(pctcount[:, 0]) <= pdiff
        num = pctcount[pvdiff, 2].sum()
        if num > last:
            s.append(f"    {num:.1f}% of values are within {pdiff}%\n")
        if np.round(num * 10) == total_percent_10:
            break
        last = num

    pct = pctinfo["pct"]
    n = len(pct)
    if n == 0:
        s.append(
            "\n    % Diff Statistics: [Min, Max, Mean, StdDev]"
            " = [n/a, n/a, n/a, n/a]\n"
        )
    else:
        stddev = 0.0 if n <= 1 else pct.std(ddof=1)
        s.append(
            "\n    % Diff Statistics: [Min, Max, Mean, StdDev]"
            f" = [{pct.min():.2f}, {pct.max():.2f}, {pct.mean():.4f}, {stddev:.4f}]\n"
        )
    return "".join(s)


def _proc_pct(
    ext1,
    ext2,
    filterval,
    magpct_filterval,
    *,
    names,
    mxmn1,
    comppv,
    mxmn_b,
    ismax,
    histogram_inc,
    prtbads,
    flagbads,
    numform,
    valhdr,
    maxhdr,
    minhdr,
    absmhdr,
    pdhdr,
    nastring,
    doabsmax,
    shortabsmax,
    print_info,
):
    # handle magpct stuff here:
    mag = ext1[comppv], ext2[comppv]
    if magpct_filterval is not None and len(magpct_filterval) > 1:
        magfilt = magpct_filterval[comppv]
    else:
        magfilt = magpct_filterval

    pv = comppv.copy()
    pct, spct = _get_pct_diff(
        ext1,
        ext2,
        filterval,
        pv,
        nastring,
        mxmn_b=mxmn_b,
        ismax=ismax,
        flagbads=flagbads,
    )
    pct_ret = pct[pv]
    hsto = ytools.histogram(pct_ret, histogram_inc)

    # for trimming down if prtbad set:
    prtpv = _get_badpv(pct, pv, *prtbads, True)
    pctlen = max(len(pdhdr), len(max(spct, key=len)))
    sformatpd = f"{{:{pctlen}}}"

    # for writer.formheader:
    numlen = max(13, len(max(names, key=len)), len(numform.format(np.pi)))
    if not doabsmax:
        print_info.headers1.extend([*names, ""])
        print_info.headers2.extend([valhdr, valhdr, pdhdr])
        print_info.formats.extend([numform, numform, sformatpd])
        print_info.printargs.extend([ext1, ext2, spct])
        print_info.widths.extend([numlen, numlen, pctlen])
        print_info.seps.extend([4, 2, 2])
        print_info.justs.extend(["c", "c", "c"])
    elif shortabsmax:
        print_info.headers1.extend([*names, ""])
        print_info.headers2.extend([absmhdr, absmhdr, pdhdr])
        print_info.formats.extend([numform, numform, sformatpd])
        print_info.printargs.extend([ext1, ext2, spct])
        print_info.widths.extend([numlen, numlen, pctlen])
        print_info.seps.extend([4, 2, 2])
        print_info.justs.extend(["c", "c", "c"])
    else:
        print_info.headers1.extend([names[0], names[0], names[0], names[1], ""])
        print_info.headers2.extend([maxhdr, minhdr, absmhdr, absmhdr, pdhdr])
        print_info.formats.extend([numform, numform, numform, numform, sformatpd])
        print_info.printargs.extend([mxmn1[:, 0], mxmn1[:, 1], ext1, ext2, spct])
        print_info.widths.extend([numlen, numlen, numlen, numlen, pctlen])
        print_info.seps.extend([4, 2, 2, 2, 2])
        print_info.justs.extend(["c", "c", "c", "c", "c"])
    return dict(
        pct=pct_ret, spct=spct, hsto=hsto, prtpv=prtpv, mag=mag, magfilt=magfilt
    )


def _figure_on(name, doabsmax, show_figures):
    figsize = [8.5, 11.0]
    if doabsmax:
        figsize[1] /= 3.0
    if show_figures:
        fig = plt.figure(name, figsize=figsize, clear=True)
    else:
        fig = Figure(figsize=figsize)
        FigureCanvasAgg(fig)
    return fig


def _figure_off(show_figures):
    pass
    # if not show_figures:
    #     plt.close()


def _prep_subplot(pctinfo, fig, sp):
    if "mx" in pctinfo:
        # if not just doing absmax
        if sp > 311:
            prev_axes = fig.gca()
            return fig.add_subplot(sp, sharex=prev_axes)
        return fig.add_subplot(sp)
    return fig.add_subplot()


def _plot_magpct(
    pctinfo,
    names,
    desc,
    doabsmax,
    filename,
    magpct_options,
    use_range,
    maxhdr,
    minhdr,
    absmhdr,
    show_figures,
    tight_layout_args,
):
    ptitle = f"{desc} - {{}} Comparison vs Magnitude"
    xl = f"{names[1]} Magnitude"
    yl = f"% Diff of {names[0]} vs {names[1]}"
    fig = _figure_on("Magpct - " + desc, doabsmax, show_figures)
    try:
        for lbl, hdr, sp, ismax in (
            ("mx", maxhdr, 311, True),
            ("mn", minhdr, 312, False),
            ("amx", absmhdr, 313, True),
        ):
            axes = _prep_subplot(pctinfo, fig, sp)
            if lbl in pctinfo:
                if use_range:
                    ref = pctinfo["amx"]["mag"][1]
                else:
                    ref = None
                magpct(
                    pctinfo[lbl]["mag"][0],
                    pctinfo[lbl]["mag"][1],
                    Ref=ref,
                    ismax=ismax,
                    filterval=pctinfo[lbl]["magfilt"],
                    ax=axes,
                    **magpct_options,
                )
                axes.set_title(ptitle.format(hdr))
                axes.set_xlabel(xl)
                axes.set_ylabel(yl)
            axes.grid(True)
        fig.tight_layout(**tight_layout_args)
        if isinstance(filename, str):
            fig.savefig(filename + ".magpct.png")
    finally:
        _figure_off(show_figures)


def _plot_histogram(
    pctinfo,
    names,
    desc,
    doabsmax,
    filename,
    histogram_inc,
    maxhdr,
    minhdr,
    absmhdr,
    show_figures,
    tight_layout_args,
):
    ptitle = f"{desc} - {{}} Comparison Histogram"
    xl = f"% Diff of {names[0]} vs {names[1]}"
    yl = "Percent Occurrence (%)"
    fig = _figure_on("Histogram - " + desc, doabsmax, show_figures)
    try:
        for lbl, hdr, sp in (
            ("mx", maxhdr, 311),
            ("mn", minhdr, 312),
            ("amx", absmhdr, 313),
        ):
            axes = _prep_subplot(pctinfo, fig, sp)
            if lbl in pctinfo:
                width = histogram_inc
                x = pctinfo[lbl]["hsto"][:, 0]
                y = pctinfo[lbl]["hsto"][:, 2]
                colors = ["b"] * len(x)
                ax = abs(x)
                pv1 = ((ax > 5) & (ax <= 10)).nonzero()[0]
                pv2 = (ax > 10).nonzero()[0]
                for pv, c in ((pv1, "m"), (pv2, "r")):
                    for i in pv:
                        colors[i] = c
                axes.bar(x, y, width=width, color=colors, align="center")
                axes.set_title(ptitle.format(hdr))
                axes.set_xlabel(xl)
                axes.set_ylabel(yl)
                x = abs(max(axes.get_xlim(), key=abs))
                if x < 5:
                    axes.set_xlim(-5, 5)
            axes.grid(True)
        fig.tight_layout(**tight_layout_args)
        if isinstance(filename, str):
            fig.savefig(filename + ".histogram.png")
    finally:
        _figure_off(show_figures)


def rptpct1(
    mxmn1,
    mxmn2,
    filename,
    *,
    title="PERCENT DIFFERENCE REPORT",
    names=("Self", "Reference"),
    desc=None,
    filterval=None,
    labels=None,
    units=None,
    ignorepv=None,
    uf_reds=None,
    use_range=True,
    numform=None,
    prtbad=None,
    prtbadh=None,
    prtbadl=None,
    flagbad=None,
    flagbadh=None,
    flagbadl=None,
    dohistogram=True,
    histogram_inc=1.0,
    domagpct=True,
    magpct_options=None,
    doabsmax=False,
    shortabsmax=False,
    roundvals=-1,
    rowhdr="Row",
    deschdr="Description",
    maxhdr="Maximum",
    minhdr="Minimum",
    absmhdr="Abs-Max",
    perpage=-1,
    tight_layout_args=None,
    show_figures=False,
    align_by_label=True,
):
    """
    Write a percent difference report between 2 sets of max/min data

    Parameters
    ----------
    mxmn1 : 2d array_like or SimpleNamespace
        The max/min data to compare to the `mxmn2` set. If 2-column
        array_like, its columns are: [max, min]. If SimpleNamespace,
        it must be as defined in :class:`DR_Results` and have these
        members:

        .. code-block:: none

            .ext = [max, min]
            .drminfo = SimpleNamespace which has (at least):
               .desc      = one line description of category
               .filterval = the filter value; (see `filterval`
                            description below)
               .labels    = a list of descriptions; one per row
               .ignorepv  = these rows will get 'n/a' for % diff
               .units     = string with units
               .uf_reds   = uncertainty factors

        Note that the inputs `desc`, `labels`, etc, override the
        values above.
    mxmn2 : 2d array_like or SimpleNamespace
        The reference set of max/min data. Format is the same as
        `mxmn1`.

        .. note::
            If both `mxmn1` and `mxmn2` are SimpleNamespaces and have
            the ``.drminfo.labels`` attribute, this routine will, by
            default, use the labels to align the data sets for
            comparison. To prevent this, set the `align_by_label`
            parameter to False.

    filename : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    title : string; must be named; optional
        Title for the report
    names : list/tuple; must be named; optional
        Two (short) strings identifying the two sets of data
    desc : string or None; must be named; optional
        A one line description of the table. Overrides
        `mxmn1.drminfo.desc`. If neither are input,
        'No description provided' is used.
    filterval : scalar, 1d array_like or None; must be named; optional
        Numbers with absolute value <= than `filterval` will get a
        'n/a' % diff. If vector, length must match number of rows in
        `mxmn1` and `mxmn2` data. Overrides `mxmn1.drminfo.filterval`.
        If neither are input, `filterval` is set to 1.e-6.
    labels : list or None; must be named; optional
        A list of strings briefly describing each row. Overrides
        `mxmn1.drminfo.labels`. If neither are input,
        ``['Row 1','Row 2',...]`` is used.
    units : string or None; must be named; optional
        Specifies the units. Overrides `mxmn1.drminfo.units`. If
        neither are input, 'Not specified' is used.
    ignorepv : 1d array or None; must be named; optional
        0-offset index vector specifying which rows of `mxmn1` to
        ignore (they get the 'n/a' % diff). Overrides
        `mxmn1.drminfo.ignorepv`. If neither are input, no rows are
        ignored (though `filterval` is still used).

        .. note::
            `ignorepv` applies *before* any alignment by labels is
            done (when `align_by_label` is True, which is the
            default).

    uf_reds : 1d array or None; must be named; optional
        Uncertainty factors: [rigid, elastic, dynamic, static].
        Overrides `mxmn1.drminfo.uf_reds`. If neither is input,
        'Not specified' is used.
    use_range : bool; must be named, optional
        If True, the denominator of the % diff calc for both the max
        & min for each row is the absolute maximum of the reference
        max & min for that row. If False, the denominator is the
        applicable reference max or min. A quick example shows why
        ``use_range=True`` might be useful:

        .. code-block:: none

            If [max1, min1] = [12345, -10] and
               [max2, min2] = [12300,  50]
            Then:
               % diff = [0.37%,   0.49%]  if use_range is True
               % diff = [0.37%, 120.00%]  if use_range is False

        Note that the sign of the % diff is defined such that a
        positive % diff means an exceedance: where ``max1 > max2`` or
        ``min1 < min2``.

        `use_range` is ignored if `doabsmax` is True.

    numform : string or None; must be named; optional
        Format of the max & min numbers. If None, it is set internally
        to be 13 chars wide and depends on the range of numbers to
        print:

           - if range is "small", numform='{:13.xf}' where "x" ranges
             from 0 to 7
           - if range is "large", numform='{:13.6e}'

    prtbad : scalar or None; must be named; optional
        Only print rows where ``abs(%diff) > prtbad``. For example, to
        print rows off by more than 5%, use ``prtbad=5``. `prtbad`
        takes precedence over `prtbadh` and `prtbadl`.
    prtbadh : scalar or None; must be named; optional
        Only print rows where ``%diff > prtbadh``. Handy for showing
        just the exceedances. `prtbadh` takes precedence over
        `prtbadl`.
    prtbadl : scalar or None; must be named; optional
        Only print rows where ``%diff < prtbadl``. Handy for showing
        where reference rows are higher.
    flagbad : scalar or None; must be named; optional
        Flag % diffs where ``abs(%diff) > flagbad``. Works similar to
        `prtbad`. The flag is an asterisk (*).
    flagbadh : scalar or None; must be named; optional
        Flag % diffs where ``%diff > flagbadh``. Works similar to
        `prtbadh`. Handy for flagging exceedances. `flagbadh` takes
        precedence over `flagbadl`.
    flagbadl : scalar or None; must be named; optional
        Flag % diffs where ``%diff < flagbadl``. Works similar to
        `prtbadl`.
    dohistogram : bool; must be named; optional
        If True, plot the histograms. Plots will be written to
        "`filename`.histogram.png".
    histogram_inc : scalar; must be named; optional
        The histogram increment; defaults to 1.0 (for 1%).
    domagpct : bool; must be named; optional
        If True, plot the percent differences versus magnitude via
        :func:`magpct`. Plots will be written to
        "`filename`.magpct.png". Filtering for the "magpct" plot is
        controlled by the ``magpct_options['filterval']`` and
        ``magpct_options['symlogy']`` options. By default, all percent
        differences are shown, but the larger values (according to the
        `filterval` filter) are emphasized by using a mixed linear/log
        y-axis. The percent differences for the `ignorepv` rows are
        not plotted.
    magpct_options : None or dict; must be named; optional
        If None, it is internally reset to::

            magpct_options = {'filterval': 'same'}

        Use this parameter to provide any options to :func:`magpct`
        but note that the `filterval` option for :func:`magpct` is
        treated specially. Here, in addition to any of the values that
        :func:`magpct` accepts, it can also be set to the string
        "same" as in the default case shown above. If set to "same",
        ``magpct_options['filterval']`` gets internally reset to the
        final value of `filterval` so that the comparison table, the
        histogram, and the magpct plot all use the same filter value.
        For backward compatibility, the string "filterval" is accepted
        as well and works like "same".

        .. note::
            The call to :func:`magpct` is *after* applying `ignorepv`
            and doing any data aligning by labels.

        .. note::
           Unless the :func:`magpct` option `plot_all` is set to
           False, all values (even those smaller than `filterval`) are
           compared and shown on the :func:`magpct` plot in the shaded
           region.

    doabsmax : bool; must be named; optional
        If True, compare only absolute maximums.
    shortabsmax : bool; must be named; optional
        If True, set ``doabsmax=True`` and do not print the max1 and
        min1 columns.
    roundvals : integer; must be named; optional
        Round max & min numbers at specified decimal. If negative, no
        rounding.
    rowhdr : string; must be named; optional
        Header for row number column
    deschdr : string; must be named; optional
        Header for description column
    maxhdr : string; must be named; optional
        Header for the column 1 data
    minhdr : string; must be named; optional
        Header for the column 2 data
    absmhdr : string; must be named; optional
        Header for abs-max column
    perpage : integer; must be named; optional
        The number of lines to write perpage. If < 1, there is no
        limit (one page).
    tight_layout_args : dict or None; must be named; optional
        Arguments for :func:`matplotlib.pyplot.tight_layout`. If None,
        defaults to ``{'pad': 3.0}``.
    show_figures : bool; must be named; optional
        If True, plot figures will be displayed on the screen for
        interactive viewing. Warning: there may be many figures.
    align_by_label : bool; must be named; optional
        If True, use labels to align the two sets of data for
        comparison. See note above under the `mxmn2` option.

    Returns
    -------
    pdiff_info : dict
        Dictionary with 'amx' (abs-max), 'mx' (max), and 'mn' keys:

        .. code-block:: none

            <class 'dict'>[n=3]
                'amx': <class 'dict'>[n=5]
                    'hsto' : float64 ndarray 33 elems: (11, 3)
                    'mag'  : [n=2]: (float64 ndarray: (100,), ...
                    'pct'  : float64 ndarray 100 elems: (100,)
                    'prtpv': bool ndarray 100 elems: (100,)
                    'spct' : [n=100]: ['  -2.46', '  -1.50', ...
                'mn' : <class 'dict'>[n=5]
                    'hsto' : float64 ndarray 33 elems: (11, 3)
                    'mag'  : [n=2]: (float64 ndarray: (100,), ...
                    'pct'  : float64 ndarray 100 elems: (100,)
                    'prtpv': bool ndarray 100 elems: (100,)
                    'spct' : [n=100]: ['   1.55', '   1.53', ...
                'mx' : <class 'dict'>[n=5]
                    'hsto' : float64 ndarray 27 elems: (9, 3)
                    'mag'  : [n=2]: (float64 ndarray: (100,), ...
                    'pct'  : float64 ndarray 100 elems: (100,)
                    'prtpv': bool ndarray 100 elems: (100,)
                    'spct' : [n=100]: ['  -2.46', '  -1.50', ...

        Where:

        .. code-block:: none

            'hsto'  : output of :func:`histogram`: [center, count, %]
            'mag'   : inputs to :func:`magpct`
            'pct'   : percent differences
            'prtpv' : rows to print partition vector
            'spct'  : string version of 'pct'

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import cla
    >>> ext1 = [[120.0, -8.0],
    ...         [8.0, -120.0]]
    >>> ext2 = [[115.0, -5.0],
    ...         [10.0, -125.0]]

    Run :func:`rptpct1` multiple times to get a more complete picture
    of all the output (the table is very wide). Also, the plots will
    be turned off for this example.

    First, the header:

    >>> opts = {'domagpct': False, 'dohistogram': False}
    >>> dct = cla.rptpct1(ext1, ext2, 1, **opts)  # doctest: +ELLIPSIS
    PERCENT DIFFERENCE REPORT
    <BLANKLINE>
    Description: No description provided
    Uncertainty: Not specified
    Units:       Not specified
    Filter:      1e-06
    Notes:       % Diff = +/- abs(Self-Reference)/max(abs(Reference...
                 Sign set such that positive % differences indicate...
    Date:        ...
    ...

    Then, the max/min/absmax percent difference table in 3 calls:

    >>> dct = cla.rptpct1(ext1, ext2, 1, **opts)  # doctest: +ELLIPSIS
    PERCENT DIFFERENCE REPORT
    ...
                                 Self        Reference             ...
      Row    Description       Maximum        Maximum      % Diff  ...
    -------  -----------    -------------  -------------  -------  ...
          1  Row      1         120.00000      115.00000     4.35  ...
          2  Row      2           8.00000       10.00000    -1.60  ...
    ...
    >>> dct = cla.rptpct1(ext1, ext2, 1, **opts)  # doctest: +ELLIPSIS
    PERCENT DIFFERENCE REPORT
    ...
                         ...     Self        Reference             ...
      Row    Description ...   Minimum        Minimum      % Diff  ...
    -------  ----------- ...-------------  -------------  -------  ...
          1  Row      1  ...     -8.00000       -5.00000     2.61  ...
          2  Row      2  ...   -120.00000     -125.00000    -4.00  ...
    ...
    >>> dct = cla.rptpct1(ext1, ext2, 1, **opts)  # doctest: +ELLIPSIS
    PERCENT DIFFERENCE REPORT
    ...
                         ...     Self        Reference
      Row    Description ...   Abs-Max        Abs-Max      % Diff
    -------  ----------- ...-------------  -------------  -------
          1  Row      1  ...    120.00000      115.00000     4.35
          2  Row      2  ...    120.00000      125.00000    -4.00
    ...

    Finally, the histogram summaries:

    >>> dct = cla.rptpct1(ext1, ext2, 1, **opts)  # doctest: +ELLIPSIS
    PERCENT DIFFERENCE REPORT
    ...
        No description provided - Maximum Comparison Histogram
    <BLANKLINE>
          % Diff      Count    Percent
         --------   --------   -------
            -2.00          1     50.00
             4.00          1     50.00
    <BLANKLINE>
        0.0% of values are within 1%
        50.0% of values are within 2%
        100.0% of values are within 5%
    <BLANKLINE>
        % Diff Statistics: [Min, Max, Mean, StdDev] = [-1.60, 4.35,...
    <BLANKLINE>
    <BLANKLINE>
        No description provided - Minimum Comparison Histogram
    <BLANKLINE>
          % Diff      Count    Percent
         --------   --------   -------
            -4.00          1     50.00
             3.00          1     50.00
    <BLANKLINE>
        0.0% of values are within 1%
        100.0% of values are within 5%
    <BLANKLINE>
        % Diff Statistics: [Min, Max, Mean, StdDev] = [-4.00, 2.61,...
    <BLANKLINE>
    <BLANKLINE>
        No description provided - Abs-Max Comparison Histogram
    <BLANKLINE>
          % Diff      Count    Percent
         --------   --------   -------
            -4.00          1     50.00
             4.00          1     50.00
    <BLANKLINE>
        0.0% of values are within 1%
        100.0% of values are within 5%
    <BLANKLINE>
        % Diff Statistics: [Min, Max, Mean, StdDev] = [-4.00, 4.35,...
    """
    if tight_layout_args is None:
        tight_layout_args = {"pad": 3.0}

    if magpct_options is None:
        magpct_options = {"filterval": "same"}
    else:
        magpct_options = magpct_options.copy()

    # magpct_options['filterval'] gets special treatment:
    magpct_filterval = magpct_options["filterval"]
    del magpct_options["filterval"]

    reset_magpct_filterval = False
    if isinstance(magpct_filterval, str):
        if magpct_filterval in ("same", "filterval"):
            reset_magpct_filterval = True
            magpct_filterval = None
        else:
            raise ValueError(
                "``magpct_options['filterval']`` is an invalid "
                f"string: {magpct_filterval!r} (can only "
                "be 'filterval' if a string)"
            )

    infovars = (
        "desc",
        "filterval",
        "magpct_filterval",
        "labels",
        "units",
        "ignorepv",
        "uf_reds",
    )
    dct = locals()
    infodct = {n: dct[n] for n in infovars}
    del dct

    # check mxmn1:
    if isinstance(mxmn1, SimpleNamespace):
        sns = mxmn1.drminfo
        for key, value in infodct.items():
            if value is None:
                infodct[key] = getattr(sns, key, None)
        del sns
        mxmn1 = mxmn1.ext
    else:
        mxmn1 = np.atleast_2d(mxmn1)
    row_number = np.arange(1, mxmn1.shape[0] + 1)

    # check mxmn2:
    if isinstance(mxmn2, SimpleNamespace) and getattr(mxmn2, "drminfo", None):
        labels2 = mxmn2.drminfo.labels
        mxmn2 = mxmn2.ext
        if align_by_label:
            # use labels and labels2 to align data; this is in case
            # the two sets of results recover some of the same items,
            # but not all
            mxmn1, mxmn2, row_number = _align_mxmn(
                mxmn1, mxmn2, labels2, row_number, infodct
            )
    else:
        mxmn2 = np.atleast_2d(mxmn2)

    desc = infodct["desc"]
    if desc is None:
        desc = "No description provided"

    R = mxmn1.shape[0]
    if R != mxmn2.shape[0]:
        raise ValueError(
            f"`mxmn1` and `mxmn2` have a different number of rows: "
            f"{R} vs {mxmn2.shape[0]} for category with `desc` = {desc}"
        )

    filterval = infodct["filterval"]
    magpct_filterval = infodct["magpct_filterval"]
    labels = infodct["labels"]
    units = infodct["units"]
    ignorepv = infodct["ignorepv"]
    uf_reds = infodct["uf_reds"]
    del infodct

    if filterval is None:
        filterval = 1.0e-6
    filterval = _proc_filterval(filterval, R, "filterval")
    if reset_magpct_filterval:
        magpct_filterval = filterval
    else:
        magpct_filterval = _proc_filterval(
            magpct_filterval, R, "magpct_options['filterval']"
        )

    if labels is None:
        labels = [f"Row {i + 1:6d}" for i in range(R)]
    elif len(labels) != R:
        raise ValueError(
            "length of `labels` does not match number"
            f" of rows in `mxmn1`: {len(labels)} vs {R} for "
            f"category with `desc` = {desc}"
        )
    if units is None:
        units = "Not specified"
    if numform is None:
        numform = _get_numform(mxmn1)

    pdhdr = "% Diff"
    nastring = "n/a "
    comppv = np.ones(R, bool)
    if ignorepv is not None:
        comppv[ignorepv] = False

    # for row labels:
    w = max(11, len(max(labels, key=len)))
    frm = f"{{:{w}}}"

    # start preparing for writer.formheader:
    print_info = SimpleNamespace(
        headers1=["", ""],
        headers2=[rowhdr, deschdr],
        formats=["{:7d}", frm],
        printargs=[row_number, labels],
        widths=[7, w],
        seps=[0, 2],
        justs=["c", "l"],
    )

    if shortabsmax:
        doabsmax = True
    if doabsmax:
        use_range = False
    if roundvals > -1:
        mxmn1 = np.round(mxmn1, roundvals)
        mxmn2 = np.round(mxmn2, roundvals)

    prtbads = (prtbad, prtbadh, prtbadl)
    flagbads = (flagbad, flagbadh, flagbadl)

    # compute percent differences
    pctinfo = {}
    kwargs = dict(
        names=names,
        mxmn1=mxmn1,
        comppv=comppv,
        histogram_inc=histogram_inc,
        numform=numform,
        prtbads=prtbads,
        flagbads=flagbads,
        maxhdr=maxhdr,
        minhdr=minhdr,
        absmhdr=absmhdr,
        pdhdr=pdhdr,
        nastring=nastring,
        doabsmax=doabsmax,
        shortabsmax=shortabsmax,
        print_info=print_info,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        mx1 = np.nanmax(abs(mxmn1), axis=1)
        mx2 = np.nanmax(abs(mxmn2), axis=1)
    if not doabsmax:
        max1, min1 = mxmn1[:, 0], mxmn1[:, 1]
        max2, min2 = mxmn2[:, 0], mxmn2[:, 1]
        mxmn_b = mxmn2 if use_range else None
        prtpv = np.zeros(R, bool)
        for i in zip(
            ("mx", "mn", "amx"),
            (max1, min1, mx1),
            (max2, min2, mx2),
            (True, False, True),
            (maxhdr, minhdr, absmhdr),
        ):
            lbl, ext1, ext2, ismax, valhdr = i
            pctinfo[lbl] = _proc_pct(
                ext1,
                ext2,
                filterval,
                magpct_filterval,
                mxmn_b=mxmn_b,
                ismax=ismax,
                valhdr=valhdr,
                **kwargs,
            )
            prtpv |= pctinfo[lbl]["prtpv"]
        prtpv &= comppv
    else:
        pctinfo["amx"] = _proc_pct(
            mx1,
            mx2,
            filterval,
            magpct_filterval,
            mxmn_b=None,
            ismax=True,
            valhdr=absmhdr,
            **kwargs,
        )
        prtpv = pctinfo["amx"]["prtpv"]
    hu, frm = writer.formheader(
        [print_info.headers1, print_info.headers2],
        print_info.widths,
        print_info.formats,
        sep=print_info.seps,
        just=print_info.justs,
    )

    # format page header:
    misc = _get_filtline(filterval) + _get_noteline(use_range, names, prtbads, flagbads)
    hdrs = _get_rpt_headers(desc=desc, uf_reds=uf_reds, units=units, misc=misc)
    header = title + "\n\n" + hdrs + "\n"

    imode = plt.isinteractive()
    plt.interactive(show_figures)
    try:
        if domagpct:
            _plot_magpct(
                pctinfo,
                names,
                desc,
                doabsmax,
                filename,
                magpct_options,
                use_range,
                maxhdr,
                minhdr,
                absmhdr,
                show_figures,
                tight_layout_args,
            )
        if dohistogram:
            _plot_histogram(
                pctinfo,
                names,
                desc,
                doabsmax,
                filename,
                histogram_inc,
                maxhdr,
                minhdr,
                absmhdr,
                show_figures,
                tight_layout_args,
            )
    finally:
        plt.interactive(imode)

    # write results
    @guitools.write_text_file
    def _wtcmp(f, header, hu, frm, printargs, perpage, prtpv, pctinfo, desc):
        prtpv = prtpv.nonzero()[0]
        if perpage < 1:
            # one additional in case size is zero
            perpage = prtpv.size + 1
        pages = (prtpv.size + perpage - 1) // perpage
        if prtpv.size < len(printargs[0]):
            for i, item in enumerate(printargs):
                printargs[i] = [item[j] for j in prtpv]
        tabhead = header + hu
        pager = "\n"  # + chr(12)
        for p in range(pages):
            if p > 0:
                f.write(pager)
            f.write(tabhead)
            b = p * perpage
            e = b + perpage
            writer.vecwrite(f, frm, *printargs, so=slice(b, e))
        f.write(pager)
        for lbl, hdr in zip(("mx", "mn", "amx"), (maxhdr, minhdr, absmhdr)):
            if lbl in pctinfo:
                f.write(_get_histogram_str(desc, hdr, pctinfo[lbl]))

    _wtcmp(
        filename, header, hu, frm, print_info.printargs, perpage, prtpv, pctinfo, desc
    )
    return pctinfo
