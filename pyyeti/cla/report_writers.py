# -*- coding: utf-8 -*-
"""
Low level tools for writing reports.
"""
import os
import copy
import datetime
from io import StringIO
from types import SimpleNamespace
import warnings
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import xlsxwriter
from pyyeti import ytools, locate, writer
from pyyeti.nastran import n2p
from ._utilities import nan_absmax, _proc_filterval, get_marker_cycle
from ._magpct import magpct

__all__ = [
    '_get_rpt_headers',
    'rptext1',
    '_get_numform',
    'rpttab1',
    'rptpct1',
]


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass


def _get_rpt_headers(res=None, desc=None, uf_reds=None,
                     units=None, misc=''):
    if res is not None:
        desc = res.drminfo.desc
        uf_reds = res.drminfo.uf_reds
        units = res.drminfo.units
    descline = 'Description: {}\n'.format(desc)
    if uf_reds is None:
        unceline = 'Uncertainty: Not specified\n'
    else:
        unceline = ('Uncertainty: [Rigid, Elastic, Dynamic, Static] '
                    '= [{}, {}, {}, {}]\n'
                    .format(*uf_reds))
    unitline = 'Units:       {}\n'.format(units)
    currdate = datetime.date.today().strftime('%d-%b-%Y')
    dateline = 'Date:        {}\n'.format(currdate)
    return descline + unceline + unitline + misc + dateline


def rptext1(res, filename,
            title='M A X / M I N  S U M M A R Y',
            doabsmax=False, numform='{:13.5e}', perpage=-1):
    """
    Writes .ext file for max/min results.

    Parameters
    ----------
    res : SimpleNamespace
        Results data structure with attributes `.ext`, `.cases`, etc
        (see example in :class:`DR_Results`)
    filename : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :func:`StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    title : string; optional
        Title for report
    doabsmax : bool; optional
        If True, report only absolute maximums. Note that signs are
        retained.
    numform : string; optional
        Format of the max & min numbers.
    perpage : integer; optional
        The number of lines to write perpage. If < 0, there is no
        limit (one page).

    Notes
    -----
    The output files contain the maximums, minimums and cases as
    applicable.
    """
    headers = []
    formats = []
    printargs = []
    widths = []
    seps = []
    justs = []

    def _add_column(hdr, frm, arg, width, sep, just):
        headers.append(hdr)
        formats.append(frm)
        printargs.append(arg)
        widths.append(width)
        seps.append(sep)
        justs.append(just)

    # row
    nrows = res.ext.shape[0]
    _add_column('Row', '{:7d}', np.arange(1, nrows + 1), 7, 0, 'c')

    # labels
    labels = getattr(res.drminfo, 'labels', None)
    if labels is not None:
        if len(res.drminfo.labels) != res.ext.shape[0]:
            raise ValueError('length of `labels` does not match '
                             'number of rows in `res.ext` (`desc`'
                             ' = {})'.format(res.drminfo.desc))
        w = max(11, len(max(labels, key=len)))
        frm = '{{:{:d}}}'.format(w)
        _add_column('Description', frm, labels, w, 2, 'l')

    # max, mx_x, case, min, mn_x, case
    domain = getattr(res, 'domain', None)
    if domain:
        domain = domain.capitalize()
    else:
        domain = 'X-Value'
    if res.maxcase is not None:
        casewidth = max(4,
                        len(max(res.maxcase, key=len)),
                        len(max(res.mincase, key=len)))
        casefrm = '{{:{:d}}}'.format(casewidth)

    one_col = res.ext.ndim == 1 or res.ext.shape[1] == 1
    for col, hdr, case in zip((0, 1),
                              ('Maximum', 'Minimum'),
                              (res.maxcase, res.mincase)):
        # maximum or minimum
        w = len(numform.format(np.pi))
        if doabsmax and not one_col:
            mx = nan_absmax(res.ext[:, 0], res.ext[:, 1])[0]
        else:
            mx = res.ext if one_col else res.ext[:, col]
        _add_column(hdr, numform, mx, w, 4, 'c')

        # x
        if res.ext_x is not None:
            t = res.ext_x if one_col else res.ext_x[:, col]
            _add_column(domain, '{:8.3f}', t, 8, 2, 'c')

        # case
        if case is not None:
            _add_column('Case', casefrm, case, casewidth, 2, 'l')

        if doabsmax or one_col:
            break

    hu, frm = writer.formheader(headers, widths, formats,
                                seps, justs)

    # format page header:
    header = title + '\n\n' + _get_rpt_headers(res) + '\n' + hu

    # write results
    @ytools.write_text_file
    def _wtext(f, header, frm, printargs, perpage):
        if perpage < 1:
            # one additional in case nrows is zero
            perpage = nrows + 1
        pages = (nrows + perpage - 1) // perpage
        for p in range(pages):
            if p > 0:
                f.write(chr(12))
            f.write(header)
            b = p * perpage
            e = b + perpage
            writer.vecwrite(f, frm, *printargs, so=slice(b, e))

    _wtext(filename, header, frm, printargs, perpage)


def _get_numform(mxmn1, excel=False):
    # excel logic is different than text:
    # - it avoids scientific notation since the actual values are
    #   there ... the user can just change the format
    pv = (mxmn1 != 0.0) & np.isfinite(mxmn1)
    if not np.any(pv):
        return '{:13.0f}' if not excel else '#,##0.'
    pmx = int(np.floor(np.log10(abs(mxmn1[pv]).max())))
    if excel:
        numform = '#,##0.' + '0' * (5 - pmx)
    else:
        pmn = int(np.floor(np.log10(abs(mxmn1[pv]).min())))
        if pmx - pmn < 6 and pmn > -3:
            if pmn < 5:
                numform = '{{:13.{}f}}'.format(5 - pmn)
            else:
                numform = '{:13.0f}'
        else:
            numform = '{:13.6e}'
    return numform


def rpttab1(res, filename, title, count_filter=1e-6, name=None):
    """
    Write results tables with bin count information.

    Parameters
    ----------
    res : SimpleNamespace
        Results data structure with attributes `.ext`, `.cases`,
        `.drminfo`, etc (see example in :class:`DR_Results`)
    filename : string or file_like or 1 or None
        If a string that ends with '.xlsx', a Microsoft Excel file is
        written.

        Otherwise, `filename` is either a name of a file, or is a
        file_like object as returned by :func:`open` or
        :func:`StringIO`. Input as integer 1 to write to stdout. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.
    title : string
        Title for report
    count_filter : scalar; optional
        Filter to use for the bin count; only numbers larger (in
        the absolute value sense) than the filter are counted
    name : string or None; optional
        For '.xlsx' files, this string is used for sheet naming. If
        None and writing an '.xlsx' file, a ValueError is raised.

    Notes
    -----
    The output files contain the maximums, minimums, abs-max
    tables. The extrema value is also included along with the case
    that produced it.

    After those three tables, a table of bin counts is printed
    showing the number of extrema values produced by each event.
    """
    def _event_count(mat, j, filter_=1e-6):
        n = mat.shape[1]
        count = np.zeros(n, np.int64)
        peaks = mat[np.arange(mat.shape[0]), j]
        pv = (abs(peaks) > filter_).nonzero()[0]
        if pv.size > 0:
            j = j[pv]
            for k in range(n):
                count[k] = np.sum(j == k)
            return count, count / len(pv) * 100
        return count, count

    def _add_zero_case(mat, cases):
        pv = (mat == 0).all(axis=1).nonzero()[0]
        if cases is None:
            new_cases = ['N/A'] * mat.shape[0]
        else:
            new_cases = copy.copy(cases)
        for i in pv:
            new_cases[i] = 'zero row'
        return new_cases

    def _get_absmax(res):
        amx, pv = nan_absmax(res.mx, res.mn)
        aext, pv = nan_absmax(res.ext[:, 0], res.ext[:, 1])
        if res.maxcase is not None:
            amxcase = res.maxcase[:]
            for j in pv.nonzero()[0]:
                amxcase[j] = res.mincase[j]
        else:
            amxcase = None
        return amx, amxcase, aext

    def _add_max_plus_min(ec):
        count = ec['Maximum'][0] + ec['Minimum'][0]
        sumcount = count.sum()
        if sumcount > 0:
            countperc = 100 * count / sumcount
        else:
            countperc = count
        ec['Max+Min'] = {0: count, 1: countperc}

    def _rowlbls_table(lbl, ec, j):
        rowlabels = ['Maxima ' + lbl,
                     'Minima ' + lbl,
                     'Max+Min ' + lbl,
                     'Abs-Max ' + lbl]
        table = np.vstack((ec['Maximum'][j],
                           ec['Minimum'][j],
                           ec['Max+Min'][j],
                           ec['Abs-Max'][j]))
        return rowlabels, table

    def _wttab_eventcount(f, res, ec, count_filter):
        # extrema count
        f.write('Extrema Count\nFilter: {}\n\n'
                .format(count_filter))
        widths = [desclen, *([caselen] * n)]
        headers = ['Description', *res.cases]
        for j, frm, lbl in zip((0, 1),
                               ('{{:{}d}}'.format(caselen),
                                '{{:{}.1f}}'.format(caselen)),
                               ('Count', 'Percent')):
            formats = [descfrm, *([frm] * n)]
            hu_, frm_ = writer.formheader(
                headers, widths, formats, sep=[7, 1],
                just='c', ulchar='=')
            f.write(hu_)
            rowlabels, table = _rowlbls_table(lbl, ec, j)
            writer.vecwrite(f, frm_, rowlabels, table)
            if j == 0:
                f.write('\n')

    def _wtxlsx_eventcount(workbook, header, bold, hform, res, ec,
                           name, count_filter):
        # extrema count
        sheet = '{} Count'.format(name)
        worksheet = workbook.add_worksheet(sheet)

        title = header.split('\n')
        for i, ln in enumerate(title):
            worksheet.write(i, 0, ln, bold)

        n = len(title)
        worksheet.write(n, 0, 'Extrema Count', bold)
        worksheet.write(
            n + 1, 0, 'Filter: {}'.format(count_filter), bold)
        worksheet.write(n + 1, 1, count_filter)
        n += 2
        ncases = len(res.cases)
        headers = ['Description', *res.cases]
        chart_positions = ((25, 0), (25, 8), (55, 0), (55, 8))
        data_labels = {'category': True,
                       'percentage': True,
                       'separator': '\n'}
        chart_opts = {'x_offset': 25, 'y_offset': 10}
        chart_size = {'width': 600, 'height': 500}

        for j, frm, lbl in zip((0, 1),
                               ('#,##', '#,##0.0'),
                               ('Count', 'Percent')):
            number = workbook.add_format({'num_format': frm})
            worksheet.write_row(n, 0, headers, hform)
            rowlabels, table = _rowlbls_table(lbl, ec, j)

            # write table and pie charts:
            n += 1
            for i, (rowlbl, chpos) in enumerate(
                    zip(rowlabels, chart_positions)):
                worksheet.write(n + i, 0, rowlbl)
                worksheet.write_row(n + i, 1, table[i], number)
                if j == 1:
                    chart = workbook.add_chart({'type': 'pie'})
                    chart.add_series(
                        {'name': rowlbl,
                         'categories': [sheet, n - 1, 1, n - 1,
                                        ncases],
                         'values': [sheet, n + i, 1, n + i, ncases],
                         'data_labels': data_labels})
                    chart.set_title({'name': rowlbl})
                    chart.set_size(chart_size)
                    worksheet.insert_chart(*chpos, chart, chart_opts)
            n += len(rowlabels) + 1

        # adjust column widths
        worksheet.set_column(0, 0, 20)
        worksheet.set_column(1, len(headers) - 1, 14)

    @ytools.write_text_file
    def _wttab(f, header, hu, frm, res, loop_vars):
        f.write(header)
        ec = {}   # event counter
        for lbl, vals, case_pv, ext, extcases in loop_vars:
            f.write('{} Responses\n\n'.format(lbl))
            hu_ = hu.replace('Maximum', lbl)
            f.write(hu_)
            ec[lbl] = _event_count(vals, case_pv, count_filter)
            extcases = _add_zero_case(vals, extcases)
            writer.vecwrite(f, frm, rows, res.drminfo.labels,
                            vals, ext, extcases)
            f.write('\n\n')
        _add_max_plus_min(ec)
        _wttab_eventcount(f, res, ec, count_filter)

    def _wtxlsx(workbook, header, headers, res, loop_vars, name):
        bold = workbook.add_format({'bold': True})
        hform = workbook.add_format({'bold': True,
                                     'align': 'center'})
        frm = _get_numform(res.ext, excel=True)
        number = workbook.add_format({'num_format': frm})
        ec = {}
        for lbl, vals, case_pv, ext, extcases in loop_vars:
            worksheet = workbook.add_worksheet(
                '{} {}'.format(name, lbl))
            title = header.split('\n')
            for i, ln in enumerate(title):
                worksheet.write(i, 0, ln, bold)
            h = [i.replace('Maximum', lbl) for i in headers]
            worksheet.write_row(len(title), 0, h, hform)

            ec[lbl] = _event_count(vals, case_pv, count_filter)
            extcases = _add_zero_case(vals, extcases)

            # write table:
            n = len(title) + 1
            labels = res.drminfo.labels
            ncases = vals.shape[1]
            for i, rw in enumerate(rows):
                worksheet.write(n + i, 0, rw)
                worksheet.write(n + i, 1, labels[i])
                worksheet.write_row(
                    n + i, 2, vals[i], number)
                worksheet.write(
                    n + i, 2 + ncases, ext[i], number)
                worksheet.write(
                    n + i, 3 + ncases, extcases[i])

            # adjust column widths and freeze row and col panes
            worksheet.set_column(1, 1, 20)  # description
            worksheet.set_column(2, 3 + ncases, 14)
            worksheet.freeze_panes(n, 2)

        _add_max_plus_min(ec)
        _wtxlsx_eventcount(workbook, header, bold, hform, res, ec,
                           name, count_filter)

    if len(res.drminfo.labels) != res.mx.shape[0]:
        raise ValueError('length of `labels` does not match number of'
                         ' rows in `res.mx` (`desc` = {})'
                         .format(res.drminfo.desc))

    rows = np.arange(res.mx.shape[0]) + 1
    headers = ['Row', 'Description', *res.cases,
               'Maximum', 'Case']
    header = title + '\n\n' + _get_rpt_headers(res) + '\n'

    amx, amxcase, aext = _get_absmax(res)
    # order of tables: max, min, abs-max with sign:
    loop_vars = (('Maximum', res.mx, np.nanargmax(res.mx, axis=1),
                  res.ext[:, 0], res.maxcase),
                 ('Minimum', res.mn, np.nanargmin(res.mn, axis=1),
                  res.ext[:, 1], res.mincase),
                 ('Abs-Max', amx, np.nanargmax(abs(amx), axis=1),
                  aext, amxcase))

    if isinstance(filename, xlsxwriter.workbook.Workbook):
        excel = 'old'
    elif isinstance(filename, str) and filename.endswith('.xlsx'):
        excel = 'new'
    else:
        excel = ''

    if not excel:
        desclen = max(15, len(max(res.drminfo.labels, key=len)))
        caselen = max(13, len(max(res.cases, key=len)))
        n = len(res.cases)
        widths = [6, desclen, *([caselen] * n), caselen, caselen]
        descfrm = '{{:{:d}}}'.format(desclen)
        numform = '{{:{}.6e}}'.format(caselen)
        formats = ['{:6d}', descfrm, *([numform] * n),
                   numform, '{}']
        hu, frm = writer.formheader(
            headers, widths, formats, sep=[0, 1],
            just='c', ulchar='=')
        _wttab(filename, header, hu, frm, res, loop_vars)
    else:
        if not name:
            raise ValueError('`name` must be input when writing'
                             ' ".xlsx" files')
        if excel == 'new':
            opts = {'nan_inf_to_errors': True}
            with xlsxwriter.Workbook(filename, opts) as workbook:
                _wtxlsx(workbook, header, headers, res,
                        loop_vars, name)
        else:
            _wtxlsx(filename, header, headers, res, loop_vars, name)


def rptpct1(mxmn1, mxmn2, filename, *,
            title='PERCENT DIFFERENCE REPORT',
            names=('Self', 'Reference'),
            desc=None, filterval=None, labels=None,
            units=None, ignorepv=None, uf_reds=None,
            use_range=True, numform=None,
            prtbad=None, prtbadh=None, prtbadl=None,
            flagbad=None, flagbadh=None, flagbadl=None,
            dohistogram=True, histogram_inc=1.0,
            domagpct=True, magpct_filterval='filterval',
            magpct_symlog=True, doabsmax=False, shortabsmax=False,
            roundvals=-1, rowhdr='Row', deschdr='Description',
            maxhdr='Maximum', minhdr='Minimum', absmhdr='Abs-Max',
            perpage=-1, tight_layout_args=None, show_figures=False):
    """
    Write a percent difference report between 2 sets of max/min data.

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
    filename : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :func:`StringIO`. Input as integer 1 to
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
        `mxmn1` and `mxmn2` data. Overrides
        `mxmn1.drminfo.filterval`. If neither are input, `filterval`
        is set to 1.e-6.
    labels : list or None; must be named; optional
        A list of strings briefly describing each row. Overrides
        `mxmn1.drminfo.labels`. If neither are input,
        ``['Row 1','Row 2',...]`` is used.
    units : string or None; must be named; optional
        Specifies the units. Overrides `mxmn1.drminfo.units`. If
        neither are input, 'Not specified' is used.
    ignorepv : 1d array or None; must be named; optional
        0-offset index vector specifying which rows to ignore (they
        get the 'n/a' % diff). Overrides `mxmn1.drminfo.units`. If
        neither are input, no rows are ignored (though `filterval` is
        still used).
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
        controlled by the `magpct_filterval` and `magpct_symlog`
        options. By default, all percent differences are shown, but
        the larger values (according to the `filterval` filter) are
        emphasized by using a mixed linear/log y-axis. The percent
        differences for the `ignorepv` rows are not plotted.
    magpct_filterval : None, string, scalar, or 1d ndarray; optional
        Unless `magpct_filterval` is a string, this is directly used
        as the ``filterval`` input to :func:`magpct`. If it is
        'filterval', `magpct_filterval` is first reset to the final
        value of `filterval` (described above). In any case, if
        `magpct_filterval` is not None, see also the `magpct_symlog`
        option, which specifies how the filter is to be used in
        :func:`magpct`.

        .. note::

           The two filter value options (`filterval` and
           `magpct_filterval`) have different defaults: None and
           'filterval`, respectively. They also differ on how the
           ``None`` setting is used: for `filterval`, None is replaced
           by 1.e-6 while for `magpct_filterval`, None means that the
           "magpct" plot will not have any filters applied at all.

    magpct_symlog : bool; must be named; optional
        Directly used as the ``symlog`` input to :func:`magpct`; see
        that routine for a description.
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
    tight_layout_args : dict or None; optional
        Arguments for :func:`plt.tight_layout`. If None, defaults to
        ``{'pad': 3.0}``.
    show_figures : bool; optional
        If True, plot figures will be displayed on the screen for
        interactive viewing. Warning: there may be many figures.

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
    def _apply_pv(value, pv):
        try:
            newvalue = value[pv]
        except (TypeError, IndexError):
            pass
        else:
            value = newvalue
        return value

    def _align_mxmn(mxmn1, mxmn2, labels2, row_number, infodct):
        if infodct['labels'] and infodct['labels'] != labels2:
            pv1, pv2 = locate.list_intersect(
                infodct['labels'], labels2)
            mxmn1 = mxmn1[pv1]
            mxmn2 = mxmn2[pv2]
            infodct['labels'] = [infodct['labels'][i] for i in pv1]
            row_number = row_number[pv1]
            infodct['filterval'] = _apply_pv(
                infodct['filterval'], pv1)
            infodct['ignorepv'] = _apply_pv(infodct['ignorepv'], pv1)
        return mxmn1, mxmn2, row_number

    def _get_filtline(filterval):
        if len(filterval) > 1:
            filtline = 'Filter:      <defined row-by-row>\n'
        else:
            filtline = 'Filter:      {}\n'.format(filterval[0])
        return filtline

    def _get_noteline(use_range, names, prtbads, flagbads):
        noteline = 'Notes:       '
        tab = '             '
        if not use_range:
            noteline += ('% Diff = +/- abs(({0}-{1})/'
                         '{1})*100\n'.format(*names))
        else:
            noteline += ('% Diff = +/- abs({0}-{1})/'
                         'max(abs({1}(max,min)))*100\n'
                         .format(*names))

        noteline += (tab + 'Sign set such that positive % '
                     'differences indicate exceedances\n')
        prtbad, prtbadh, prtbadl = prtbads
        flagbad, flagbadh, flagbadl = flagbads
        if (prtbad is not None or prtbadh is not None or
                prtbadl is not None):
            if prtbad is not None:
                prtbad = abs(prtbad)
                noteline += (tab + 'Printing rows where abs(% Diff)'
                             ' > {}%\n'.format(prtbad))
            elif prtbadh is not None:
                noteline += (tab + 'Printing rows where % Diff > '
                             '{}%\n'.format(prtbadh))
            else:
                noteline += (tab + 'Printing rows where % Diff < '
                             '{}%\n'.format(prtbadl))

        if (flagbad is not None or flagbadh is not None or
                flagbadl is not None):
            if flagbad is not None:
                flagbad = abs(flagbad)
                noteline += (tab + 'Flagging (*) rows where '
                             'abs(% Diff) > {}%\n'.format(flagbad))
            elif flagbadh is not None:
                noteline += (tab + 'Flagging (*) rows where '
                             '% Diff > {}%\n'.format(flagbadh))
            else:
                noteline += (tab + 'Flagging (*) rows where '
                             '% Diff < {}%\n'.format(flagbadl))
        return noteline

    def _get_badpv(pct, pv, bad, badh, badl, defaultpv=False):
        if (bad is not None or badh is not None or
                badl is not None):
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

    def _get_pct_diff(a, b, filt, pv, mxmn_b=None, ismax=True,
                      flagbads=None):
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
        pct[z] = 100.0    # np.inf

        # make less extreme values negative
        neg = a < b if ismax else a > b
        pct[neg] *= -1.0

        # put nan's in for the filtered or n/a rows:
        pct[~pv] = np.nan

        # make 7 char version:
        spct = ['{:7.2f}'.format(p) for p in pct]
        badpv = _get_badpv(pct, pv, *flagbads, False)
        for j in badpv.nonzero()[0]:
            spct[j] += '*'
        for j in (~pv).nonzero()[0]:
            spct[j] = nastring

        return pct, spct

    def _get_histogram_str(desc, hdr, pctinfo):
        pctcount = pctinfo['hsto']
        s = [('\n\n    {} - {} Comparison Histogram\n\n'
              .format(desc, hdr)),
             ('      % Diff      Count    Percent\n'
              '     --------   --------   -------\n')]
        with StringIO() as f:
            writer.vecwrite(f, '     {:8.2f}   {:8.0f}   {:7.2f}\n',
                            pctcount)
            s.append(f.getvalue())
            s.append('\n')

        last = -1.0
        for pdiff in [1, 2, 5, 10, 15, 20, 25, 50, 100, 500]:
            pvdiff = abs(pctcount[:, 0]) <= pdiff
            num = pctcount[pvdiff, 2].sum()
            if num > last:
                s.append('    {:.1f}% of values are within {:d}%\n'
                         .format(num, pdiff))
            if np.round(num * 10) == 1000:
                break
            last = num

        pct = pctinfo['pct']
        n = len(pct)
        stddev = 0.0 if n <= 1 else pct.std(ddof=1)
        s.append('\n    % Diff Statistics: [Min, Max, Mean, StdDev]'
                 ' = [{:.2f}, {:.2f}, {:.4f}, {:.4f}]\n'
                 .format(pct.min(), pct.max(), pct.mean(), stddev))
        return ''.join(s)

    def _proc_pct(ext1, ext2, filterval, *, names, mxmn1,
                  comppv, mxmn_b, ismax, histogram_inc,
                  prtbads, flagbads, numform, valhdr, maxhdr,
                  minhdr, absmhdr, pdhdr):
        pv = comppv.copy()
        mag = ext1[comppv], ext2[comppv]  # good here?
        pct, spct = _get_pct_diff(ext1, ext2, filterval, pv,
                                  mxmn_b=mxmn_b, ismax=ismax,
                                  flagbads=flagbads)
        pct_ret = pct[pv]
        hsto = ytools.histogram(pct_ret, histogram_inc)

        # for trimming down if prtbad set:
        prtpv = _get_badpv(pct, pv, *prtbads, True)
        pctlen = max(len(pdhdr),
                     len(max(spct, key=len)))
        sformatpd = '{{:{}}}'.format(pctlen)

        # for writer.formheader:
        numlen = max(13,
                     len(max(names, key=len)),
                     len(numform.format(np.pi)))
        if not doabsmax:
            headers1.extend([*names, ''])
            headers2.extend([valhdr, valhdr, pdhdr])
            formats.extend([numform, numform, sformatpd])
            printargs.extend([ext1, ext2, spct])
            widths.extend([numlen, numlen, pctlen])
            seps.extend([4, 2, 2])
            justs.extend(['c', 'c', 'c'])
        elif shortabsmax:
            headers1.extend([*names, ''])
            headers2.extend([absmhdr, absmhdr, pdhdr])
            formats.extend([numform, numform, sformatpd])
            printargs.extend([mx1, mx2, spct])
            widths.extend([numlen, numlen, pctlen])
            seps.extend([4, 2, 2])
            justs.extend(['c', 'c', 'c'])
        else:
            headers1.extend([names[0], names[0],
                             names[0], names[1], ''])
            headers2.extend([maxhdr, minhdr, absmhdr,
                             absmhdr, pdhdr])
            formats.extend([numform, numform, numform,
                            numform, sformatpd])
            printargs.extend([mxmn1[:, 0], mxmn1[:, 1],
                              mx1, mx2, spct])
            widths.extend([numlen, numlen, numlen,
                           numlen, pctlen])
            seps.extend([4, 2, 2, 2, 2])
            justs.extend(['c', 'c', 'c', 'c', 'c'])
        return dict(pct=pct_ret, spct=spct, hsto=hsto,
                    prtpv=prtpv, mag=mag)

    def _figure_on(name, doabsmax):
        figsize = [8.5, 11.0]
        if doabsmax:
            figsize[1] /= 3.0
        if show_figures:
            plt.figure(name, figsize=figsize)
            plt.clf()
        else:
            plt.figure(figsize=figsize)

    def _figure_off():
        if not show_figures:
            plt.close()

    def _prep_subplot(pctinfo, sp):
        if 'mx' in pctinfo:
            # if not just doing absmax
            if sp > 311:
                plt.subplot(sp, sharex=plt.gca())
            else:
                plt.subplot(sp)

    def _plot_magpct(pctinfo, names, desc, doabsmax, filename,
                     magpct_filterval, magpct_symlog):
        ptitle = '{} - {{}} Comparison vs Magnitude'.format(desc)
        xl = '{} Magnitude'.format(names[1])
        yl = '% Diff of {} vs {}'.format(*names)
        _figure_on('Magpct - ' + desc, doabsmax)
        try:
            for lbl, hdr, sp, ismax in (('mx', maxhdr, 311, True),
                                        ('mn', minhdr, 312, False),
                                        ('amx', absmhdr, 313, True)):
                _prep_subplot(pctinfo, sp)
                if lbl in pctinfo:
                    if use_range:
                        ref = pctinfo['amx']['mag'][1]
                    else:
                        ref = None
                    magpct(pctinfo[lbl]['mag'][0],
                           pctinfo[lbl]['mag'][1],
                           Ref=ref,
                           ismax=ismax,
                           filterval=magpct_filterval,
                           symlog=magpct_symlog)
                    plt.title(ptitle.format(hdr))
                    plt.xlabel(xl)
                    plt.ylabel(yl)
                plt.grid(True)
            plt.tight_layout(**tight_layout_args)
            if isinstance(filename, str):
                plt.savefig(filename + '.magpct.png')
        finally:
            _figure_off()

    def _plot_histogram(pctinfo, names, desc, doabsmax, filename):
        ptitle = '{} - {{}} Comparison Histogram'.format(desc)
        xl = '% Diff of {} vs {}'.format(*names)
        yl = 'Percent Occurrence (%)'
        _figure_on('Histogram - ' + desc, doabsmax)
        try:
            for lbl, hdr, sp in (('mx', maxhdr, 311),
                                 ('mn', minhdr, 312),
                                 ('amx', absmhdr, 313)):
                _prep_subplot(pctinfo, sp)
                if lbl in pctinfo:
                    width = histogram_inc
                    x = pctinfo[lbl]['hsto'][:, 0]
                    y = pctinfo[lbl]['hsto'][:, 2]
                    colors = ['b'] * len(x)
                    ax = abs(x)
                    pv1 = ((ax > 5) & (ax <= 10)).nonzero()[0]
                    pv2 = (ax > 10).nonzero()[0]
                    for pv, c in ((pv1, 'm'),
                                  (pv2, 'r')):
                        for i in pv:
                            colors[i] = c
                    plt.bar(x, y, width=width, color=colors,
                            align='center')
                    plt.title(ptitle.format(hdr))
                    plt.xlabel(xl)
                    plt.ylabel(yl)
                    x = abs(max(plt.xlim(), key=abs))
                    if x < 5:
                        plt.xlim(-5, 5)
                plt.grid(True)
            plt.tight_layout(**tight_layout_args)
            if isinstance(filename, str):
                plt.savefig(filename + '.histogram.png')
        finally:
            _figure_off()

    # main routine
    if tight_layout_args is None:
        tight_layout_args = {'pad': 3.0}
    infovars = ('desc', 'filterval', 'labels',
                'units', 'ignorepv', 'uf_reds')
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
    if (isinstance(mxmn2, SimpleNamespace) and
            getattr(mxmn2, 'drminfo', None)):
        labels2 = mxmn2.drminfo.labels
        mxmn2 = mxmn2.ext
        # use labels and labels2 to align data; this is in case
        # the two sets of results recover some of the same items,
        # but not all
        mxmn1, mxmn2, row_number = _align_mxmn(
            mxmn1, mxmn2, labels2, row_number, infodct)
    else:
        mxmn2 = np.atleast_2d(mxmn2)

    desc = infodct['desc']
    filterval = infodct['filterval']
    labels = infodct['labels']
    units = infodct['units']
    ignorepv = infodct['ignorepv']
    uf_reds = infodct['uf_reds']

    R = mxmn1.shape[0]
    if R != mxmn2.shape[0]:
        raise ValueError('`mxmn1` and `mxmn2` have a different'
                         ' number of rows (`desc` = {})'.format(desc))
    if desc is None:
        desc = 'No description provided'

    if filterval is None:
        filterval = 1.e-6
    filterval = _proc_filterval(filterval, R)

    if labels is None:
        labels = ['Row {:6d}'.format(i + 1)
                  for i in range(R)]
    elif len(labels) != R:
        raise ValueError('length of `labels` does not match number'
                         ' of rows in `mxmn1` (`desc` = {})'
                         .format(desc))
    if units is None:
        units = 'Not specified'
    if numform is None:
        numform = _get_numform(mxmn1)

    pdhdr = '% Diff'
    nastring = 'n/a '
    comppv = np.ones(R, bool)
    if ignorepv is not None:
        comppv[ignorepv] = False

    # for row labels:
    w = max(11, len(max(labels, key=len)))
    frm = '{{:{:d}}}'.format(w)

    # start preparing for writer.formheader:
    headers1 = ['', '']
    headers2 = [rowhdr, deschdr]
    formats = ['{:7d}', frm]
    printargs = [row_number, labels]
    widths = [7, w]
    seps = [0, 2]
    justs = ['c', 'l']

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
    kwargs = dict(names=names, mxmn1=mxmn1, comppv=comppv,
                  histogram_inc=histogram_inc, numform=numform,
                  prtbads=prtbads, flagbads=flagbads,
                  maxhdr=maxhdr, minhdr=minhdr, absmhdr=absmhdr,
                  pdhdr=pdhdr)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', r'All-NaN (slice|axis) encountered')
        mx1 = np.nanmax(abs(mxmn1), axis=1)
        mx2 = np.nanmax(abs(mxmn2), axis=1)
    if not doabsmax:
        max1, min1 = mxmn1[:, 0], mxmn1[:, 1]
        max2, min2 = mxmn2[:, 0], mxmn2[:, 1]
        mxmn_b = mxmn2 if use_range else None
        prtpv = np.zeros(R, bool)
        for i in zip(('mx', 'mn', 'amx'),
                     (max1, min1, mx1),
                     (max2, min2, mx2),
                     (True, False, True),
                     (maxhdr, minhdr, absmhdr)):
            lbl, ext1, ext2, ismax, valhdr = i
            pctinfo[lbl] = _proc_pct(ext1, ext2, filterval,
                                     mxmn_b=mxmn_b, ismax=ismax,
                                     valhdr=valhdr, **kwargs)
            prtpv |= pctinfo[lbl]['prtpv']
        prtpv &= comppv
    else:
        pctinfo['amx'] = _proc_pct(mx1, mx2, filterval,
                                   mxmn_b=None, ismax=True,
                                   valhdr=absmhdr, **kwargs)
        prtpv = pctinfo['amx']['prtpv']
    hu, frm = writer.formheader([headers1, headers2], widths,
                                formats, sep=seps, just=justs)

    # format page header:
    misc = (_get_filtline(filterval) +
            _get_noteline(use_range, names, prtbads, flagbads))
    header = (title + '\n\n' +
              _get_rpt_headers(desc=desc, uf_reds=uf_reds,
                                 units=units, misc=misc) +
              '\n')

    imode = plt.isinteractive()
    plt.interactive(show_figures)
    try:
        if domagpct:
            if isinstance(magpct_filterval, str):
                if magpct_filterval != 'filterval':
                    raise ValueError(
                        '`magpct_filterval` is an invalid string: '
                        '"{}" (can only be "filterval")'
                        .format(magpct_filterval))
                magpct_filterval = filterval
            _plot_magpct(pctinfo, names, desc, doabsmax, filename,
                         magpct_filterval, magpct_symlog)
        if dohistogram:
            _plot_histogram(pctinfo, names, desc, doabsmax, filename)
    finally:
        plt.interactive(imode)

    # write results
    @ytools.write_text_file
    def _wtcmp(f, header, hu, frm, printargs, perpage,
               prtpv, pctinfo, desc):
        prtpv = prtpv.nonzero()[0]
        if perpage < 1:
            # one additional in case size is zero
            perpage = prtpv.size + 1
        pages = (prtpv.size + perpage - 1) // perpage
        if prtpv.size < len(printargs[0]):
            for i, item in enumerate(printargs):
                printargs[i] = [item[j] for j in prtpv]
        tabhead = header + hu
        pager = '\n'  # + chr(12)
        for p in range(pages):
            if p > 0:
                f.write(pager)
            f.write(tabhead)
            b = p * perpage
            e = b + perpage
            writer.vecwrite(f, frm, *printargs, so=slice(b, e))
        f.write(pager)
        for lbl, hdr in zip(('mx', 'mn', 'amx'),
                            (maxhdr, minhdr, absmhdr)):
            if lbl in pctinfo:
                f.write(_get_histogram_str(
                    desc, hdr, pctinfo[lbl]))

    _wtcmp(filename, header, hu, frm, printargs, perpage, prtpv,
           pctinfo, desc)
    return pctinfo


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
