# -*- coding: utf-8 -*-
"""
Low level tool for tables of results. Typically, this is called
via: :func:`cla.DR_Results.rpttab`.
"""
import copy
import numpy as np
import xlsxwriter
from pyyeti import guitools, writer
from ._utilities import nan_absmax, _get_rpt_headers, _get_numform


__all__ = ["rpttab1"]


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


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
        new_cases = ["N/A"] * mat.shape[0]
    else:
        new_cases = copy.copy(cases)
    for i in pv:
        new_cases[i] = "zero row"
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
    count = ec["Maximum"][0] + ec["Minimum"][0]
    sumcount = count.sum()
    if sumcount > 0:
        countperc = 100 * count / sumcount
    else:
        countperc = count
    ec["Max+Min"] = {0: count, 1: countperc}


def _rowlbls_table(lbl, ec, j):
    rowlabels = ["Maxima " + lbl, "Minima " + lbl, "Max+Min " + lbl, "Abs-Max " + lbl]
    table = np.vstack(
        (ec["Maximum"][j], ec["Minimum"][j], ec["Max+Min"][j], ec["Abs-Max"][j])
    )
    return rowlabels, table


def _wttab_eventcount(f, res, ec, count_filter, desclen, descfrm, caselen):
    # extrema count
    n = len(res.cases)
    f.write(f"Extrema Count\nFilter: {count_filter}\n\n")
    widths = [desclen, *([caselen] * n)]
    headers = ["Description", *res.cases]
    for j, frm, lbl in zip(
        (0, 1), (f"{{:{caselen}d}}", f"{{:{caselen}.1f}}"), ("Count", "Percent")
    ):
        formats = [descfrm, *([frm] * n)]
        hu_, frm_ = writer.formheader(
            headers, widths, formats, sep=[7, 1], just="c", ulchar="="
        )
        f.write(hu_)
        rowlabels, table = _rowlbls_table(lbl, ec, j)
        writer.vecwrite(f, frm_, rowlabels, table)
        if j == 0:
            f.write("\n")


def _wtxlsx_eventcount(workbook, header, bold, hform, res, ec, name, count_filter):
    # extrema count
    sheet = f"{name} Count"
    worksheet = workbook.add_worksheet(sheet)

    title = header.split("\n")
    for i, ln in enumerate(title):
        worksheet.write(i, 0, ln, bold)

    n = len(title)
    worksheet.write(n, 0, "Extrema Count", bold)
    worksheet.write(n + 1, 0, f"Filter: {count_filter}", bold)
    worksheet.write(n + 1, 1, count_filter)
    n += 2
    ncases = len(res.cases)
    headers = ["Description", *res.cases]
    chart_positions = ((25, 0), (25, 8), (55, 0), (55, 8))
    data_labels = {"category": True, "percentage": True, "separator": "\n"}
    chart_opts = {"x_offset": 25, "y_offset": 10}
    chart_size = {"width": 600, "height": 500}

    for j, frm, lbl in zip((0, 1), ("#,##", "#,##0.0"), ("Count", "Percent")):
        number = workbook.add_format({"num_format": frm})
        worksheet.write_row(n, 0, headers, hform)
        rowlabels, table = _rowlbls_table(lbl, ec, j)

        # write table and pie charts:
        n += 1
        for i, (rowlbl, chpos) in enumerate(zip(rowlabels, chart_positions)):
            worksheet.write(n + i, 0, rowlbl)
            worksheet.write_row(n + i, 1, table[i], number)
            if j == 1:
                chart = workbook.add_chart({"type": "pie"})
                chart.add_series(
                    {
                        "name": rowlbl,
                        "categories": [sheet, n - 1, 1, n - 1, ncases],
                        "values": [sheet, n + i, 1, n + i, ncases],
                        "data_labels": data_labels,
                    }
                )
                chart.set_title({"name": rowlbl})
                chart.set_size(chart_size)
                worksheet.insert_chart(*chpos, chart, chart_opts)
        n += len(rowlabels) + 1

    # adjust column widths
    worksheet.set_column(0, 0, 20)
    worksheet.set_column(1, len(headers) - 1, 14)


@guitools.write_text_file
def _wttab(
    f, header, hu, frm, res, loop_vars, rows, count_filter, desclen, descfrm, caselen
):
    f.write(header)
    ec = {}  # event counter
    for lbl, vals, case_pv, ext, extcases in loop_vars:
        f.write(f"{lbl} Responses\n\n")
        hu_ = hu.replace("Maximum", lbl)
        f.write(hu_)
        ec[lbl] = _event_count(vals, case_pv, count_filter)
        extcases = _add_zero_case(vals, extcases)
        writer.vecwrite(f, frm, rows, res.drminfo.labels, vals, ext, extcases)
        f.write("\n\n")
    _add_max_plus_min(ec)
    _wttab_eventcount(f, res, ec, count_filter, desclen, descfrm, caselen)


def _wtxlsx(workbook, header, headers, res, loop_vars, name, rows, count_filter):
    bold = workbook.add_format({"bold": True})
    hform = workbook.add_format({"bold": True, "align": "center"})
    frm = _get_numform(res.ext, excel=True)
    number = workbook.add_format({"num_format": frm})
    ec = {}
    for lbl, vals, case_pv, ext, extcases in loop_vars:
        worksheet = workbook.add_worksheet(f"{name} {lbl}")
        title = header.split("\n")
        for i, ln in enumerate(title):
            worksheet.write(i, 0, ln, bold)
        h = [i.replace("Maximum", lbl) for i in headers]
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
            worksheet.write_row(n + i, 2, vals[i], number)
            worksheet.write(n + i, 2 + ncases, ext[i], number)
            worksheet.write(n + i, 3 + ncases, extcases[i])

        # adjust column widths and freeze row and col panes
        worksheet.set_column(1, 1, 20)  # description
        worksheet.set_column(2, 3 + ncases, 14)
        worksheet.freeze_panes(n, 2)

    _add_max_plus_min(ec)
    _wtxlsx_eventcount(workbook, header, bold, hform, res, ec, name, count_filter)


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
        :class:`io.StringIO`. Input as integer 1 to write to
        stdout. Can also be the name of a directory or None; in these
        cases, a GUI is opened for file selection.
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

    if len(res.drminfo.labels) != res.mx.shape[0]:
        raise ValueError(
            "length of `labels` does not match number of"
            f" rows in `res.mx` (`desc` = {res.drminfo.desc})"
        )

    rows = np.arange(res.mx.shape[0]) + 1
    headers = ["Row", "Description", *res.cases, "Maximum", "Case"]
    header = title + "\n\n" + _get_rpt_headers(res) + "\n"

    amx, amxcase, aext = _get_absmax(res)
    # order of tables: max, min, abs-max with sign:
    loop_vars = (
        ("Maximum", res.mx, np.nanargmax(res.mx, axis=1), res.ext[:, 0], res.maxcase),
        ("Minimum", res.mn, np.nanargmin(res.mn, axis=1), res.ext[:, 1], res.mincase),
        ("Abs-Max", amx, np.nanargmax(abs(amx), axis=1), aext, amxcase),
    )

    if isinstance(filename, xlsxwriter.workbook.Workbook):
        excel = "old"
    elif isinstance(filename, str) and filename.endswith(".xlsx"):
        excel = "new"
    else:
        excel = ""

    if not excel:
        desclen = max(15, len(max(res.drminfo.labels, key=len)))
        caselen = max(13, len(max(res.cases, key=len)))
        n = len(res.cases)
        widths = [6, desclen, *([caselen] * n), caselen, caselen]
        descfrm = f"{{:{desclen:d}}}"
        numform = f"{{:{caselen}.6e}}"
        formats = ["{:6d}", descfrm, *([numform] * n), numform, "{}"]
        hu, frm = writer.formheader(
            headers, widths, formats, sep=[0, 1], just="c", ulchar="="
        )
        _wttab(
            filename,
            header,
            hu,
            frm,
            res,
            loop_vars,
            rows,
            count_filter,
            desclen,
            descfrm,
            caselen,
        )
    else:
        if not name:
            raise ValueError('`name` must be input when writing ".xlsx" files')
        if excel == "new":
            opts = {"nan_inf_to_errors": True}
            with xlsxwriter.Workbook(filename, opts) as workbook:
                _wtxlsx(
                    workbook, header, headers, res, loop_vars, name, rows, count_filter
                )
        else:
            _wtxlsx(filename, header, headers, res, loop_vars, name, rows, count_filter)
