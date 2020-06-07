# -*- coding: utf-8 -*-
"""
Collection of tools for writing formatted text to files.
"""

import numpy as np
from pyyeti import ytools


def getith(i, args, fncs):
    """
    Return list with i'th value from each input, typically called by
    :func:`vecwrite`.

    Parameters
    ----------
    i : integer
        Specifies which value to extract from each input; starts at 0.
    args : list of variables
        Variable to extract the i'th value from. Must be compatibly
        sized (scalars or vectors of equal length). Strings are
        considered scalars.
    fncs : list of functions
        Same length as args; the function is used to extract the
        i'th item. Call signature:  ith_element_of_a = func(a, i).
        The function must return an iterable of items (eg, list).

    Returns
    -------
    lst : list
        List of the i'th items extracted from each variable in `args`.

    Examples
    --------
    >>> from pyyeti import writer
    >>> import numpy as np
    >>> r = np.array([1.2, 45.])
    >>> s = 'test string'
    >>> i = 5
    >>> v = ['One', 'Two']
    >>> def f(a, i): return [a]
    >>> def f2(a, i): return [a[i]]
    >>> args = [r, s, i, v]
    >>> fncs = [f2, f, f, f2]
    >>> writer.getith(0, args, fncs)
    [1.2, 'test string', 5, 'One']
    >>> writer.getith(1, args, fncs)
    [45.0, 'test string', 5, 'Two']
    """
    lst = []
    for arg, fnc in zip(args, fncs):
        lst.extend(fnc(arg, i))
    return lst


@ytools.write_text_file
def _vecwrite(fout, string, length, args, fncs, postfunc, pfargs, so):
    """Utility routine for :func:`vecwrite`."""
    v = range(length)
    if so is not None:
        v = v[so]
    if postfunc:
        if pfargs is None:
            pfargs = []
        for i in v:
            curargs = getith(i, args, fncs)
            s = postfunc(string.format(*curargs), *pfargs)
            fout.write(s)
    else:
        for i in v:
            curargs = getith(i, args, fncs)
            fout.write(string.format(*curargs))


def vecwrite(f, string, *args, postfunc=None, pfargs=None, so=None):
    """
    Vectorized write.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    string : string
        The formatting string for the write, Python 3 format as in:
        `string`.format(a,b)
    *args : list of variables
        Variables to write. Must be compatibly sized (scalars or
        vectors or numpy arrays of compatible sizes). numpy arrays of
        length 1 are considered scalars. For 2-d numpy arrays, each
        row is written on one line and each element of the row must
        have a conversion specifier. 1-d numpy arrays are treated
        like a column 2-d numpy array. Strings are considered
        scalars.
    postfunc : function or None
        If a function, it is called with the final string (for each
        line) as the argument and it must return a string. The return
        string is what gets output. This can be handy for final string
        substitutions, for example. This input must be named and must
        be after the arguments to be printed; see example.
    pfargs : iterable or None
        If an iterable, contains extra arguments to pass to `postfunc`
        after the string argument. Must be named and after the
        arguments to be printed.
    so : slice object or None
        Allows selection of limited range and custom increment; eg:
        ``slice(0, 10, 2)``. Scalars are not sliced. Must be named and
        after the arguments to be printed.

    Returns
    -------
    None.

    Notes
    -----
    The expected vector length is determined from the first non-scalar
    input. Note that scalar values are repeated automatically as
    necessary.

    Raises
    ------
    ValueError
        When the lengths of print arguments do not match (for
        lengths > 1). Note that the slice object `so` can make
        otherwise incompatible arguments compatible; for example,
        arguments of length 10 and length 100 would be compatible if
        ``so = slice(10)`` (or similar).

    Examples
    --------
    >>> from pyyeti import writer
    >>> import sys
    >>> import numpy as np
    >>> r = np.array([1.2, 45.8])
    >>> s = 'test string'
    >>> i = 5
    >>> v = ['short string', 'a bit longer string']
    >>> frm = '{:3}, {:5.1f}, {:<25}, {}' + chr(10)
    >>> writer.vecwrite(sys.stdout, frm, i, r, v, s)
      5,   1.2, short string             , test string
      5,  45.8, a bit longer string      , test string

    >>> r = np.array([[1.1, 1.2, 1.3], [10.1, 10.2, 10.3]])
    >>> frm = '{:2}, {:=^25} : ' + '  {:6.2f}'*3 + chr(10)
    >>> writer.vecwrite(sys.stdout, frm, i, v, r)
     5, ======short string======= :     1.10    1.20    1.30
     5, ===a bit longer string=== :    10.10   10.20   10.30

    >>> def pf(s):
    ...     return s.replace('0 ', '  ')
    >>> writer.vecwrite(sys.stdout, frm, i, v, r, postfunc=pf)
     5, ======short string======= :     1.1     1.2     1.30
     5, ===a bit longer string=== :    10.1    10.2    10.30

    >>> def pf(s, s_old, s_new):
    ...     return s.replace(s_old, s_new)
    >>> writer.vecwrite(1, frm, i, v, r, postfunc=pf,
    ...                 pfargs=['0 ', '  '])
     5, ======short string======= :     1.1     1.2     1.30
     5, ===a bit longer string=== :    10.1    10.2    10.30
    """

    def _get_scalar(a, i):
        return [a]

    def _get_scalar1(a, i):
        return [a[0]]

    def _get_itemi(a, i):
        return [a[i]]

    def _get_matrow(a, i):
        return a[i]

    length = 1
    fncs = []
    for i, arg in enumerate(args):
        if not isinstance(arg, str) and hasattr(arg, "__len__"):
            if np.ndim(arg) == 2:
                fncs.append(_get_matrow)
                curlen = np.size(arg, 0)
            elif len(arg) == 1:
                fncs.append(_get_scalar1)
                curlen = 1
            else:
                fncs.append(_get_itemi)
                curlen = len(arg)
            if curlen > 1:
                if length > 1:
                    if so is not None:
                        if range(curlen)[so] != range(length)[so]:
                            msg = (
                                "length mismatch with slice object:"
                                f" arg # {i + 1} is incompatible with "
                                "previous args"
                            )
                            raise ValueError(msg)
                    elif curlen != length:
                        msg = (
                            f"length mismatch: arg # {i + 1} has "
                            f"length {curlen}; expected {length} or 1."
                        )
                        raise ValueError(msg)
                length = curlen
        else:
            fncs.append(_get_scalar)
    _vecwrite(f, string, length, args, fncs, postfunc, pfargs, so)


def formheader(headers, widths, formats, sep=(0, 2), just=-1, ulchar="-"):
    """
    Form a nice table header for formatted output via f.write().

    Parameters
    ----------
    headers : list or tuple
        List or tuple of column header strings, eg:
        ['Desc', 'Maximum', 'Time']. Can also be a list of lists (or
        tuples) to support multiple header lines, eg:
        [['Maximum', 'Minimum', 'Time'], ['(lbs)', '(lbs)', '(sec)']]
    widths : iterable
        Iterable of field widths, eg: (25, 10, 8) or [25, 10, 8]. If
        an element in `widths` is < length of corresponding word in a
        header-line, the length of the word is used for that field.
        Note that if this doesn't match with `formats`, the columns
        will not line up nicely.
    formats : list or tuple
        List or tuple of format specifiers for the values in the table,
        eg: ['{:25s}', '{:10f}', '{:8.3f}']
    sep : string, list, tuple, or integer
        Defines 'spacer' in front of each word:

            - if a string, that string is used in front of all headers
            - use a list or tuple of strings for complete control
            - if an integer, that many spaces are used in front of all
              headers
            - use a vector of integers to specify a variable number of
              spaces
            - if len(sep) < len(headers), the last element is used for
              all remaining elements

    just : string or integer or list
        Justification flag or flags for each header string:

            - 'l', 'c', 'r' (or -1, 0, 1) to left, center, or right
              justify headers in their fields
            - can be a list or tuple of len(headers) for complete
              control

    ulchar : string
        Character to use for underlining of headers.

    Returns
    -------
    hu : string
        Contains formatted header string(s) and the underline string.
    f : string
        Final formatting string.

    Examples
    --------
    >>> import numpy as np
    >>> import sys
    >>> from pyyeti import writer
    >>> descs = ['Item 1', 'A different item']
    >>> mx = np.array([[1.2, 2.3], [3.4, 4.5]]) * 1000
    >>> time = np.array([[1.234], [2.345]])
    >>> headers = [['The']*3, ['Descriptions', 'Maximum', 'Time']]
    >>> formats = ['{:<25s}', '{:10.2f}', '{:8.3f}']
    >>> widths  = [25, 10, 8]
    >>> hu, f = writer.formheader(headers, widths, formats,
    ...                           sep=[4, 5, 2], just=0)
    >>> fout = sys.stdout
    >>> if 1:   # just so all output is together
    ...     b = fout.write(hu)
    ...     writer.vecwrite(fout, f, descs, mx, time)
                   The                   The        The
               Descriptions            Maximum      Time
        -------------------------     ----------  --------
        Item 1                           1200.00  2300.000
        A different item                 3400.00  4500.000
    """
    if not isinstance(headers, (list, tuple)):
        raise ValueError("input 'headers' must be a list or tuple")
    if isinstance(headers[0], (list, tuple)):
        length = len(headers[0])
        nheaders = len(headers)
        mxlengths = np.array([len(s) for s in headers[0]])
        for j in range(1, nheaders):
            if len(headers[j]) != length:
                raise ValueError(
                    f"headers[{len(headers[j])}] != length of previous headers"
                )
            for k in range(length):
                mxlengths[k] = max(mxlengths[k], len(headers[j][k]))
    else:
        nheaders = 0
        mxlengths = np.array([len(s) for s in headers])
        length = len(headers)

    if not length == len(formats) == len(widths):
        s = ""
        if isinstance(headers[0], (list, tuple)):
            s = "[*]"
        raise ValueError(
            f"this check failed: ``len(headers{s}) == len(formats) == len(widths)``"
        )

    def strexp(string, width, just):
        if just == -1 or just == "l":
            return string.ljust(width)
        if just == 0 or just == "c":
            return string.center(width)
        return string.rjust(width)

    if isinstance(just, (str, int)):
        just = [just]
    if isinstance(sep, int):
        sep = " " * sep
    if isinstance(sep, str):
        sep = [sep]

    if nheaders > 0:
        h = [""] * nheaders
    else:
        h = ""
    u, f = "", ""
    for j in range(length):
        if j >= len(just):
            cj = just[-1]
        else:
            cj = just[j]
        if j >= len(sep):
            csep = sep[-1]
        else:
            csep = sep[j]
        if isinstance(csep, int):
            csep = " " * csep
        w = max(widths[j], mxlengths[j])
        if nheaders > 0:
            for k in range(nheaders):
                h[k] += csep + strexp(headers[k][j], w, just=cj)
        else:
            h += csep + strexp(headers[j], w, just=cj)
        u += csep + ulchar * w
        f += csep + formats[j]

    if nheaders > 0:
        h = [hj.rstrip() + "\n" for hj in h]
    else:
        h = h.rstrip() + "\n"
    u = u.rstrip() + "\n"
    f = f.rstrip() + "\n"
    return "".join(h) + u, f
