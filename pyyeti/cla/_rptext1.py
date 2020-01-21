# -*- coding: utf-8 -*-
"""
Low level tool for writing extrema reports. Typically, this is called
via: :func:`cla.DR_Results.rptext`.
"""
from types import SimpleNamespace
import numpy as np
from pyyeti import ytools, writer
from ._utilities import nan_absmax, _get_rpt_headers


__all__ = ["rptext1"]


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def _add_column(hdr, frm, arg, width, sep, just, print_info):
    print_info.headers.append(hdr)
    print_info.formats.append(frm)
    print_info.printargs.append(arg)
    print_info.widths.append(width)
    print_info.seps.append(sep)
    print_info.justs.append(just)


# write results
@ytools.write_text_file
def _wtext(f, header, frm, printargs, perpage, nrows):
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


def rptext1(
    res,
    filename,
    title="M A X / M I N  S U M M A R Y",
    doabsmax=False,
    numform="{:13.5e}",
    perpage=-1,
):
    """
    Writes .ext file for max/min results.

    Parameters
    ----------
    res : SimpleNamespace
        Results data structure with attributes `.ext`, `.cases`, etc
        (see example in :class:`DR_Results`)
    filename : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
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
    print_info = SimpleNamespace(
        headers=[], formats=[], printargs=[], widths=[], seps=[], justs=[]
    )

    # row
    nrows = res.ext.shape[0]
    _add_column("Row", "{:7d}", np.arange(1, nrows + 1), 7, 0, "c", print_info)

    # labels
    labels = getattr(res.drminfo, "labels", None)
    if labels is not None:
        if len(res.drminfo.labels) != res.ext.shape[0]:
            raise ValueError(
                "length of `labels` does not match number of rows in `res.ext` "
                f"(`desc` = {res.drminfo.desc})"
            )
        w = max(11, len(max(labels, key=len)))
        frm = f"{{:{w}}}"
        _add_column("Description", frm, labels, w, 2, "l", print_info)

    # max, mx_x, case, min, mn_x, case
    domain = getattr(res, "domain", None)
    if domain:
        domain = domain.capitalize()
    else:
        domain = "X-Value"
    if res.maxcase is not None:
        casewidth = max(
            4, len(max(res.maxcase, key=len)), len(max(res.mincase, key=len))
        )
        casefrm = f"{{:{casewidth}}}"

    one_col = res.ext.ndim == 1 or res.ext.shape[1] == 1
    for col, hdr, case in zip(
        (0, 1), ("Maximum", "Minimum"), (res.maxcase, res.mincase)
    ):
        # maximum or minimum
        w = len(numform.format(np.pi))
        if doabsmax and not one_col:
            mx = nan_absmax(res.ext[:, 0], res.ext[:, 1])[0]
        else:
            mx = res.ext if one_col else res.ext[:, col]
        _add_column(hdr, numform, mx, w, 4, "c", print_info)

        # x
        if res.ext_x is not None:
            t = res.ext_x if one_col else res.ext_x[:, col]
            _add_column(domain, "{:8.3f}", t, 8, 2, "c", print_info)

        # case
        if case is not None:
            _add_column("Case", casefrm, case, casewidth, 2, "l", print_info)

        if doabsmax or one_col:
            break

    hu, frm = writer.formheader(
        print_info.headers,
        print_info.widths,
        print_info.formats,
        print_info.seps,
        print_info.justs,
    )

    # format page header:
    header = title + "\n\n" + _get_rpt_headers(res) + "\n" + hu

    _wtext(filename, header, frm, print_info.printargs, perpage, nrows)
