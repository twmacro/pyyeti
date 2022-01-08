# -*- coding: utf-8 -*-
"""
Collection of tools for reading/writing Nastran bulk data.

The functions provided by this module can be accessed by just
importing the "nastran" package. For example, you can access the
:func:`rdgrids` function in these two ways:

>>> from pyyeti import nastran
>>> from pyyeti.nastran import bulk
>>> bulk.rdgrids is nastran.rdgrids
True
"""

import os
import re
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyyeti import locate, writer, ytools, guitools
from pyyeti.nastran import n2p, op4, op2


__all__ = [
    "nas_sscanf",
    "fsearch",
    "rdgpwg",
    "rdcards",
    "rddmig",
    "mkcomment",
    "wtdmig",
    "rdgrids",
    "rdcord2cards",
    "wtgrids",
    "rdtabled1",
    "wttabled1",
    "bulk2uset",
    "uset2bulk",
    "rdspoints",
    "rdseconct",
    "asm2uset",
    "rdwtbulk",
    "rdeigen",
    "wtnasints",
    "rdcsupers",
    "rdextrn",
    "wtcsuper",
    "wtspc1",
    "wtxset1",
    "wtqcset",
    "wtrbe2",
    "wtrbe3",
    "wtseset",
    "wtset",
    "wtrspline",
    "wtrspline_rings",
    "wtcoordcards",
    "wtextrn",
    "wtextseout",
    "mknast",
    "rddtipch",
]


def nas_sscanf(s, keep_string=False):
    """
    Read a single integer or Nastran-formatted floating point number

    Parameters
    ----------
    s : string
        May be formatted in the NASTRAN shortcut way; eg, '1.7-4'
        instead of '1.7e-4'. May also use 'd' instead of 'e'.
    keep_string : bool; optional
        Only used if `s` is a string that cannot be interpreted as a
        number. In that case, if `keep_string` is True, ``s.strip()``
        is returned; otherwise, if `keep_string` is False, None is
        returned.

    Returns
    -------
    v : int or float or string or None
        The scalar value that the string represents. If string is
        empty, returns None. If `keep_string` is True, `v` will be
        ``s.strip()`` when `s` cannot be converted to a number.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> nastran.nas_sscanf(' 10')
    10
    >>> nastran.nas_sscanf('1.7e-4')
    0.00017
    >>> nastran.nas_sscanf('1.7-4')
    0.00017
    >>> nastran.nas_sscanf('1.7d4')
    17000.0
    >>> print(nastran.nas_sscanf(' '))
    None
    >>> print(nastran.nas_sscanf('string'))
    None
    >>> print(nastran.nas_sscanf('string', keep_string=True))
    string
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            S = s = s.strip()
            if len(s) == 0:
                return None
            s = s.lower().replace("d", "e")
            try:
                return float(s)
            except ValueError:
                s = s[0] + s[1:].replace("+", "e+").replace("-", "e-")
                try:
                    return float(s)
                except ValueError:
                    if keep_string:
                        return S
                    return None


def fsearch(f, s):
    """
    Search for a line in a file.

    Parameters
    ----------
    f : file_like
        File_like object to search in.
    s : string
        String to search for.

    Returns
    -------
    line : string or None
        The first line that contains the string. None if not found.
    p : integer or None
        The position where `s` begins in `line`. None if not found.

    Notes
    -----
    The search begins at the current file position.
    """
    for line in f:
        p = line.find(s)
        if p > -1:
            return line, p
    return None, None


@guitools.read_text_file
def rdgpwg(f, search_strings=None):
    """
    Read a GPWG table from a Nastran F06 file.

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.
    search_strings : None, string, or list_like of strings
        If a string, this routine will scan the file until the string
        is found before reading the GPWG table. If multiple strings in
        a list or tuple, the routine will scan for each string, in the
        order provided, before reading the GPWG table. If None, the
        first GPWG table is read.

    Returns
    -------
    mass : 2d ndarray
        The 6x6 mass (or weight) matrix.
    cg : 2d ndarray
        The 3x4 center-of-gravity table::

            [ mass(x, y, z) x-cg y-cg z-cg ]

    refpt : integer
        The reference point listed just before the mass matrix in the
        F06 file.
    Is : 2d ndarray
        The I(S) matrix. This is the 3 x 3 mass moment of inertia
        partition with respect to the center of gravity referred to
        the principal mass axes (the S system). This is not
        necessarily a diagonal matrix because the determination of the
        S system does not involve second moments. The values of
        inertias at the center of gravity are found from the values at
        the reference point employing the parallel axis rule.

    Notes
    -----
    The routine will load the first GPWG table it finds after
    searching for each string in `search_strings`, if provided. After
    searching for these strings, the routine will search for the next
    "W E I G H T" string.

    All outputs will be set to None if no GPWG table is found or if a
    search string is not found.

    Example usage::

        from pyyeti import nastran
        str1 = '           SUPERELEMENT 100'
        str2 = '^^^F-SET MASS TABLE'
        mass, cg, ref, I = nastran.rdgpwg('modes.f06', (str1, str2))
    """
    f.seek(0, 0)
    default = None, None, None, None
    if search_strings is not None:
        if isinstance(search_strings, str):
            search_strings = (search_strings,)
        for s in search_strings:
            for line in f:
                if line.find(s) > -1:
                    break
            else:
                return default

    line, p = fsearch(f, "W E I G H T")
    if line is None:
        return default
    line, p = fsearch(f, "REFERENCE POINT =")
    refpt = int(line[p + 17 :])

    f.readline()
    mass = []
    for _ in range(6):
        line = f.readline().strip()
        mass.append([float(item) for item in line[1:-1].split()])
    mass = np.array(mass)
    line, p = fsearch(f, "MASS AXIS SYSTEM")

    cg = []
    for _ in range(3):
        line = f.readline().strip()
        cg.append([float(item) for item in line[1:].split()])
    cg = np.array(cg)

    f.readline()
    Is = []
    for _ in range(3):
        line = f.readline().strip()
        Is.append([float(item) for item in line[1:-1].split()])
    Is = np.array(Is)
    return mass, cg, refpt, Is


# def _readline_keep_comments(it, Vals):
#     try:
#         s = next(it)
#         while len(s) > 0 and s[0] == "$":
#             Vals.append(s)
#             s = next(it)
#         return s
#     except StopIteration:
#         return None
#
#
# def _readline_skip_comments(it, Vals):
#     try:
#         s = next(it)
#         while len(s) > 0 and s[0] == "$":
#             s = next(it)
#         return s
#     except StopIteration:
#         return None
#
#
# def _get_line(fiter, _readline, Vals, s=None, trim=True):
#     if s is None:
#         s = _readline(fiter, Vals)
#         if s is None:
#             return None
#     s = s.expandtabs()
#     if trim:
#         s = s[:72]
#     p = s.find("$")
#     if p > -1:
#         s = s[:p]
#     s = s.rstrip()
#     return s


def _proc_line(s):
    p = s.find("$")
    if p > -1:
        s = s[:p]
    return s.rstrip()


def _handle_comments(comment_list, Vals):
    if comment_list:
        # if there were comments read in, store them
        # and clear cache:
        Vals.extend(comment_list)
        comment_list.clear()


def _next_line(f, name, regex, keep_comments, Vals):
    if regex:
        prog = re.compile(name, re.IGNORECASE)

        def ismatch(line, prog):
            return prog.match(line)

    else:
        prog = name.lower()

        def ismatch(line, prog):
            return line.lower().find(prog) == 0

    do_match = True
    if not keep_comments:
        for line in f:
            line = line.expandtabs()
            if do_match:
                if ismatch(line, prog):
                    # found the start of a matching card, yield line
                    do_match = yield line
                    # if card continues on multiple lines, do_match
                    # will be set to False
            else:
                # do_match is False (meaning reading continuation
                # lines), so get next line no matter what it is:
                do_match = yield line
                # if do_match is now True, the last line read is the
                # next line that needs to be examined:
                if do_match:
                    if ismatch(line, prog):
                        do_match = yield line

        do_match = yield None  # mark the end of input
    else:
        comment_list = []
        for line in f:
            if line.startswith("$"):
                comment_list.append(line)
                continue
            line = line.expandtabs()

            # this is very much like above, so see those
            # comments. added comments pertain to handling comments
            if do_match:
                if ismatch(line, prog):
                    _handle_comments(comment_list, Vals)
                    do_match = yield line
            else:
                do_match = yield line
                if do_match:
                    if ismatch(line, prog):
                        _handle_comments(comment_list, Vals)
                        do_match = yield line

        do_match = yield None  # mark the end of input
        _handle_comments(comment_list, Vals)


def _rdfixed(fiter, s, n, conchar, blank, tolist, keep_name):
    """
    Read fixed field cards:
    fiter : file iterator
    s : string, current line from file (card start)
    n : field width, 8 or 16
    conchar : string, set of chars that make up continuation
              - either ' +' or '*'
    blank : value to use for blank fields and string-valued fields
    tolist : if True, return list, else return np.array; strings are
             kept in this case (instead of being turned into `blank`)
    keep_name : if True and tolist is True, keep the card name in the
                output
    """
    vals = []
    length = len(s)
    i = -1
    nfields = -1
    if n > 8:
        inc = 4  # number of values per line
    else:
        inc = 8

    s = _proc_line(s[:72])

    if tolist and keep_name:
        # read card name outside of loop:
        v = nas_sscanf(s[:8], tolist)
        vals.append(v)

    maxstart = 72 - n
    while 1:
        for i in range(i, nfields):
            vals.append(blank)
        i = nfields
        j = 8
        while j <= maxstart and length > j:
            i += 1
            v = nas_sscanf(s[j : j + n], tolist)
            if v is not None:
                vals.append(v)
            else:
                vals.append(blank)
            j += n
        s = fiter.send(False)
        if s is None or len(s) == 0 or conchar.find(s[0]) < 0:
            break
        s = _proc_line(s[:72])
        length = len(s)
        nfields += inc
    return vals


def _rdcomma(fiter, s, conchar, blank, tolist, keep_name):
    """
    Read comma delimited cards:
    fiter : file iterator
    s : string, current line from file (card start)
    n : field width, 8 or 16
    conchar : string, set of chars that make up continuation
              - ' +, '
    blank : value to use for blank fields and string-valued fields
    tolist : if True, return list, else return np.array; strings are
             kept in this case (instead of being turned into `blank`)
    keep_name : if True and tolist is True, keep the card name in the
                output
    """
    vals = []
    i = -1
    nfields = -1
    inc = 8
    start_field = 0 if (tolist and keep_name) else 1
    s = _proc_line(s)
    while 1:
        for i in range(i, nfields):
            vals.append(blank)
        i = nfields
        tok = s.split(",")
        lentok = min(len(tok), 9)
        for j in range(start_field, lentok):
            i += 1
            v = nas_sscanf(tok[j], tolist)
            if v is not None:
                vals.append(v)
            else:
                vals.append(blank)
        # first field for continuation cards will never be retained:
        start_field = 1
        s = fiter.send(False)
        if s is None or len(s) == 0 or conchar.find(s[0]) < 0:
            break
        s = _proc_line(s)
        nfields += inc
    return vals


@guitools.read_text_file
def rdcards(
    f,
    name,
    blank=None,
    return_var="array",
    dtype=float,
    no_data_return=None,
    regex=False,
    keep_name=False,
    keep_comments=False,
):
    r"""
    Read Nastran cards (lines) into a matrix, dictionary, or list.

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.
    name : string
        Usually the card name, but is really just the initial part of
        the string to look for. This means that `name` can span more
        than one field, in case that is handy. It can also be a
        regular expression if the `regex` option is set to True. This
        routine is case insensitive: if `regex` is False, `name` is
        converted to lower case as are the lines that are read in from
        the file; if `regex` is True, the ``re.IGNORECASE`` option is
        used.
    blank : None, scalar or string; optional
        If reading into an array or dictionary (see `return_var`),
        `blank` is expected to be a numeric value to use for blank
        fields and string-valued fields. If reading into a list, it
        can also be a string (like ``""``). If None (the default), it
        is internally reset to ``""`` if `return_var` is ``"list"``
        and to ``0`` otherwise.
    return_var : string; optional
        Specifies which data structure to use for the data:

        ============   ============================================
        `return_var`   Routine returns
        ============   ============================================
          'array'      a numpy ndarray of type `dtype`; non-numeric
                       fields are set to `blank`
          'list'       a list of lists; this is the "truest" reader
                       since string fields are retained and only
                       blank fields are set to `blank`
          'dict'       a dictionary; each card becomes a dictionary
                       entry: the value is a numpy ndarray of type
                       `dtype` and the key is the first element of
                       that ndarray (but before converting to
                       `dtype`)
        ============   ============================================

    dtype : data-type; optional
        The desired array data-type for the 'array' or 'dict' output
        options.
    no_data_return : any variable; optional
        If no data is found, this routine returns `no_data_return`.
    regex : bool; optional
        If set to True, Use regular expression matching instead of
        literal string matching.
    keep_name : bool; optional
        If reading into a list, `keep_name` can be set to True to
        retain the card name in the output. This option is ignored if
        not reading into a list.
    keep_comments : bool; optional
        If reading into a list, `keep_comments` can be set to True to
        retain all comment cards in the output. This option is ignored
        if not reading into a list. Note that only the comments that
        start at the beginning of the line are retained.

    Returns
    -------
    cards : ndarray or list or dictionary or no_data_return

        - If an ndarray is returned, each row is a card, padded with
          blanks as necessary.
        - If a list (of lists) is returned, each item is a card and is
          a list of values from the card, the length of which is the
          number of items on the card.
        - If a dictionary is returned, the first number from each card
          is used as the key and the value are all the numbers from
          the card in a row vector (including the first value).
        - `no_data_return` is returned if no cards of requested name
          were found.

    Notes
    -----
    This routine can read fixed field (8 or 16 chars wide) or
    comma-delimited and can handle any number of continuations. Note
    that the characters in the continuation fields are ignored. It
    also uses :func:`nas_sscanf` to read the numbers, so numbers like
    1.-3 (which means 1.e-3) are okay. Blank fields and string fields
    are set to `blank`, except when `return_var` is 'list'; in that
    case, string fields are kept as is and only blank fields are set
    to `blank`.

    Note: this routine is has no knowledge of any card, which means
    that it will not append trailing blanks to a card. For example, if
    a GRID card is: 'GRID, 1', then this routine would return ``[1]``,
    not ``[1, 0, 0, 0, 0, 0, 0, 0]``. The :func:`rdgrids` routine
    would return ``[1, 0, 0, 0, 0, 0, 0, 0]`` since it knows the
    number of fields a GRID card has.

    Examples
    --------
    Create some bulk data to read (not necessarily valid):

    >>> from io import StringIO
    >>> from pyyeti.nastran import bulk
    >>> fs = StringIO('''
    ... DTI     SELOAD         1       2
    ... dti     seload         3       4
    ... $ a comment for testing
    ... dti,seload,5,6
    ... DTI, SELOAD, , 8.0, 'a'
    ... DTI,SETREE,100,0
    ... ''')

    Read all the DTI cards to a list:

    >>> lst = bulk.rdcards(fs, 'dti', return_var='list')
    >>> for item in lst: print(item)
    ['SELOAD', 1, 2]
    ['seload', 3, 4]
    ['seload', 5, 6]
    ['SELOAD', '', 8.0, "'a'"]
    ['SETREE', 100, 0]

    Read some of the DTI,SELOAD cards (it's case-insensitive):

    >>> bulk.rdcards(fs, 'dti     seload')
    array([[ 0.,  1.,  2.],
           [ 0.,  3.,  4.]])

    Use a regular expression for more power:

    >>> bulk.rdcards(fs, r'DTI(,\s*|\s+)SELOAD', regex=True)
    array([[ 0.,  1.,  2.,  0.],
           [ 0.,  3.,  4.,  0.],
           [ 0.,  5.,  6.,  0.],
           [ 0.,  0.,  8.,  0.]])

    Same, but read into a list:

    >>> lst = bulk.rdcards(fs, r'DTI(,\s*|\s+)SELOAD', regex=True,
    ...                    return_var='list')
    >>> for item in lst: print(item)
    ['SELOAD', 1, 2]
    ['seload', 3, 4]
    ['seload', 5, 6]
    ['SELOAD', '', 8.0, "'a'"]

    Include the card name as well:

    >>> lst = bulk.rdcards(fs, r'DTI(,\s*|\s+)SELOAD', regex=True,
    ...                    return_var='list', keep_name=True)
    >>> for item in lst: print(item)
    ['DTI', 'SELOAD', 1, 2]
    ['dti', 'seload', 3, 4]
    ['dti', 'seload', 5, 6]
    ['DTI', 'SELOAD', '', 8.0, "'a'"]

    For the most accurate read of the data, include comments:

    >>> lst = bulk.rdcards(fs, r'DTI(,\s*|\s+)SELOAD', regex=True,
    ...                    return_var='list', keep_name=True,
    ...                    keep_comments=True)
    >>> for item in lst: print(f"{item!r}")
    ['DTI', 'SELOAD', 1, 2]
    ['dti', 'seload', 3, 4]
    '$ a comment for testing\n'
    ['dti', 'seload', 5, 6]
    ['DTI', 'SELOAD', '', 8.0, "'a'"]
    """

    if return_var not in ("array", "list", "dict"):
        raise ValueError(
            'invalid `return_var` setting; must be one of: ("array", "list", "dict")'
        )

    if return_var == "dict":
        Vals = {}
        todict = True
        tolist = False
    else:
        todict = False
        tolist = return_var == "list"
        Vals = []

    if blank is None:
        blank = "" if tolist else 0

    mxlen = 0
    f.seek(0, 0)

    fiter = _next_line(f, name, regex, tolist and keep_comments, Vals)
    s = next(fiter)
    while s is not None:
        # if here, have matching line
        if s.find(",") > -1:
            vals = _rdcomma(fiter, s, " +,", blank, tolist, keep_name)
        else:
            s = s[:72].rstrip()
            p = s[:8].find("*")
            field, continuation = (16, "*") if p > -1 else (8, " +")
            vals = _rdfixed(fiter, s, field, continuation, blank, tolist, keep_name)
        if tolist:
            Vals.append(vals)
        else:
            cur = len(vals)
            mxlen = max(mxlen, cur)
            key = vals[0]  # before it gets turned into dtype
            vals = np.array(vals).astype(dtype)
            if todict:
                Vals[key] = vals
            else:
                Vals.append(vals)
        try:
            s = fiter.send(True)
        except StopIteration:
            break

    # flush out iterator if needed (get any last comments)
    try:
        s = fiter.send(True)
    except StopIteration:
        pass
    del fiter

    if len(Vals) > 0:
        if not (todict or tolist):
            npVals = np.empty((len(Vals), mxlen), dtype=dtype)
            npVals[:] = blank
            for i, vals in enumerate(Vals):
                npVals[i, : len(vals)] = vals
            Vals = npVals
        return Vals
    return no_data_return


def rddmig(f, dmig_names=None, *, expanded=False, square=False):
    """
    Read DMIG entries from a Nastran punch (bulk) or output2 file.

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a punch or output2 file, or is a file_like
        object as returned by :func:`open`. If handle, punch format is
        assumed and file is rewound first. Can also be the name of a
        directory or None; in these cases, a GUI is opened for file
        selection.
    dmig_names : None, string, or list_like of strings
        If not None, it is the name or names of DMIG entry(s) to
        read. If None, all DMIG entries will be returned.
    expanded : bool; optional; must be named
        If True, row and column indices for "GRID" IDs will be
        expanded to include all 6 DOF, even if some of those DOF are
        not used. Otherwise, if `expanded` is False, only the row and
        column DOF actually referenced on the DMIG will be
        included.
    square : bool; optional; must be named
        Only used for "square" matrices (``form=1``). If True, ensures
        that the row and column indices are the same by filling in
        zeros as necessary.

    Returns
    -------
    dct : dictionary
        Dictionary of pandas DataFrames containing the DMIG
        entries. The column and row indices are pandas MultiIndex
        objects with node ID in level 0 and DOF in level 1 (the names
        are "id" and "dof"). The exception is for form "9" matrices:
        the column index in that case is just the column number (as
        specified by Nastran).

    Notes
    -----
    For the punch format, this routine first reads all DMIG entries
    via :func:`rdcards` using the 'list' output option. It then builds
    a dictionary of DataFrames for only those entries specified in
    `dmig_names`. It is assumed that DMIG entries are in order in the
    file.

    For the output2 format, this routine scans through the file
    looking for matching DMIG data-blocks and only reads those in.

    .. note::
        This routine is more lenient than Nastran. Nastran will issue
        a FATAL message if a symmetric matrix has entries for element
        (i, j) and (j, i). This routine does not check for that (the
        last value for (i, j) or (j, i) will be used for both).

    Examples
    --------
    Manually create a punch format DMIG string, treat it as a file,
    and read it in with a couple different options:

    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> from io import StringIO
    >>> dmig = (
    ...  'DMIG    MAT            0       1       2       0'
    ...  '                       2\\n'
    ...  'DMIG*   MAT                            1               1\\n'
    ...  '*                      1               3 1.200000000D+01\\n'
    ...  '*                     10               0 1.000000000D+02\\n'
    ... )
    >>> with StringIO(dmig) as f:
    ...     dct1 = nastran.rddmig(f)
    ...     dct2 = nastran.rddmig(f, expanded=True)
    ...     dct3 = nastran.rddmig(f, square=True)
    ...     dct4 = nastran.rddmig(f, expanded=True, square=True)
    >>> dct1['mat']    # doctest: +ELLIPSIS
    id          1
    dof         1
    id dof...
    1  3     12.0
    10 0    100.0
    >>> dct2['mat']    # doctest: +ELLIPSIS
    id          1...
    dof         1    2    3    4    5    6
    id dof...
    1  1      0.0  0.0  0.0  0.0  0.0  0.0
       2      0.0  0.0  0.0  0.0  0.0  0.0
       3     12.0  0.0  0.0  0.0  0.0  0.0
       4      0.0  0.0  0.0  0.0  0.0  0.0
       5      0.0  0.0  0.0  0.0  0.0  0.0
       6      0.0  0.0  0.0  0.0  0.0  0.0
    10 0    100.0  0.0  0.0  0.0  0.0  0.0
    >>> dct3['mat']    # doctest: +ELLIPSIS
    id         1         10
    dof         1    3    0
    id dof...
    1  1      0.0  0.0  0.0
       3     12.0  0.0  0.0
    10 0    100.0  0.0  0.0
    >>> dct4['mat']    # doctest: +ELLIPSIS
    id         1                             10
    dof         1    2    3    4    5    6    0
    id dof...
    1  1      0.0  0.0  0.0  0.0  0.0  0.0  0.0
       2      0.0  0.0  0.0  0.0  0.0  0.0  0.0
       3     12.0  0.0  0.0  0.0  0.0  0.0  0.0
       4      0.0  0.0  0.0  0.0  0.0  0.0  0.0
       5      0.0  0.0  0.0  0.0  0.0  0.0  0.0
       6      0.0  0.0  0.0  0.0  0.0  0.0  0.0
    10 0    100.0  0.0  0.0  0.0  0.0  0.0  0.0
    """
    # Notes on DMIG format:
    # - rows are always g-set or p-set size
    # - square:
    #   - format 1 is general square
    #   - format 6 is symmetric and only one of each (i, j) or (j, i)
    #     pair may be provided (Nastran will FATAL otherwise)
    # - rectangular:
    #   - format 2
    #   - format 9 uses 'ncol' and doesn't have actual grid/dof pairs
    #     for columns; but the pairs provided do determine sort order

    def _add_iddof_expanded(ids, iddof, nid, dof):
        # iddof is a list
        if nid not in ids:
            ids.add(nid)
            if dof > 0:
                iddof.extend([(nid, k) for k in range(1, 7)])
            else:
                iddof.append((nid, 0))

    def _add_iddof_minimal(ids, iddof, nid, dof):
        # iddof is a set
        iddof.add((nid, dof))

    def _mk_index(iddof):
        iddof = sorted(iddof, key=lambda x: 10 * x[0] + x[1])
        return pd.MultiIndex.from_tuples(iddof, names=["id", "dof"])

    def _prep_dataframe(mtype, form, row_ids, row_iddof, col_iddof, ncol):
        # create dataframe:
        dtype = float if mtype < 3 else complex
        if form != 9:
            if form == 6 or (form == 1 and square):
                for nid, dof in col_iddof:
                    add_iddof(row_ids, row_iddof, nid, dof)
                rowindex = _mk_index(row_iddof)
                colindex = rowindex
            else:
                rowindex = _mk_index(row_iddof)
                colindex = _mk_index(col_iddof)
        else:
            rowindex = _mk_index(row_iddof)
            if expanded:
                colindex = np.arange(1, ncol + 1)
            else:
                colindex = sorted(nid for nid, dof in col_iddof)
        data = np.zeros((len(rowindex), len(colindex)), dtype)
        return data, rowindex, colindex

    def _get_1d_index(index):
        ind = index.to_frame().values
        return ind[:, 0] * 10 + ind[:, 1]

    def _cards_to_df(c, dmig_names):
        # punch file
        def _next_i(c, i):
            while i < len(c) and c[i][0].lower() not in dmig_names:
                i += 1
            return i

        dct = {}
        i = 0
        while i < len(c):
            if dmig_names is not None:
                i = _next_i(c, i)
                if i == len(c):
                    break
            name = c[i][0].lower()

            # form: 1 square, 2 rect, 6 symmetric, 9 rect
            form = c[i][2]
            # mtype: 1 real, 2 double, 3 complex, 4 complex double
            mtype = c[i][3]
            ncol = c[i][7] if form == 9 else None

            # Count DOF for DMIG name
            j = i + 1
            row_ids = set()
            col_ids = set()
            col_iddof = iddof_initializer()
            row_iddof = iddof_initializer()

            # look across columns (each DMIG entry is a column)
            while j < len(c) and c[j][0].lower() == name:
                nid, dof = c[j][1:3]
                add_iddof(col_ids, col_iddof, nid, dof)

                # for each column, look across rows:
                rowids = c[j][4::4]
                rowdofs = c[j][5::4]
                for nid, dof in zip(rowids, rowdofs):
                    add_iddof(row_ids, row_iddof, nid, dof)
                j += 1

            # create dataframe:
            mat, rowindex, colindex = _prep_dataframe(
                mtype, form, row_ids, row_iddof, col_iddof, ncol
            )

            j = i + 1
            # put data into dataframe:
            r_id_dof = _get_1d_index(rowindex)
            if form != 9:
                c_id_dof = _get_1d_index(colindex)
            else:
                c_id_dof = colindex
            while j < len(c) and c[j][0].lower() == name:
                nid, dof = c[j][1:3]
                if form != 9:
                    nid = nid * 10 + dof
                ci = np.searchsorted(c_id_dof, nid)

                # for each column, look across rows:
                rowids = c[j][4::4]
                rowdofs = c[j][5::4]
                reals = c[j][6::4]
                if mtype < 3:
                    for nid, dof, real in zip(rowids, rowdofs, reals):
                        ri = np.searchsorted(r_id_dof, nid * 10 + dof)
                        mat[ri, ci] = real
                        if form == 6:
                            mat[ci, ri] = real
                else:
                    imags = c[j][7::4]
                    for nid, dof, real, imag in zip(rowids, rowdofs, reals, imags):
                        val = real + 1j * imag
                        ri = np.searchsorted(r_id_dof, nid * 10 + dof)
                        mat[ri, ci] = val
                        if form == 6:
                            mat[ci, ri] = val
                j += 1
            i = j
            dct[name] = pd.DataFrame(mat, index=rowindex, columns=colindex)
        return dct

    def _read_op2_dmig(o2, dmig_names):
        o2._fileh.seek(o2._postheaderpos)
        dct = {}
        name, trailer, dbtype = o2.rdop2nt()
        while name is not None:
            name = name.lower()
            # print(f'op2: found {name}')
            if dmig_names is not None and name.lower() not in dmig_names:
                o2.skipop2table()
            else:
                rec = o2.rdop2record()
                if np.all(rec[:3] == (114, 1, 120)):
                    dct[name] = rec
                o2.go_to_next_db()
            name, trailer, dbtype = o2.rdop2nt()
        return dct

    def _recs_to_df(recs):
        dct = {}
        for name, rec in recs.items():
            form = rec[6]
            # mtype: 1 real, 2 double, 3 complex, 4 complex double
            mtype = rec[7]
            ncol = rec[11] if form == 9 else None

            if mtype > 2:
                ints_per_number = 2 * (mtype - 2)
                dtype = np.complex128 if mtype == 4 else np.complex64
            else:
                ints_per_number = mtype
                dtype = np.float64 if mtype == 2 else np.float32
            step = 2 + ints_per_number  # row id, dof, <number>

            # Count DOF for DMIG
            row_ids = set()
            col_ids = set()
            col_iddof = iddof_initializer()
            row_iddof = iddof_initializer()

            # Look across columns for id, dof pairs. Within each
            # column, look across rows for id, dof pairs.

            # - each DMIG entry is a column
            # - end of each DMIG entry is marked with row:
            #     [id, dof] = [-1, -1]

            # 12 & 13 4-byte integers are 1st col id, dof pair
            j = 12
            while j < len(rec) - 2:
                nid, dof = rec[j : j + 2]
                add_iddof(col_ids, col_iddof, nid, dof)

                j += 2
                rowids = rec[j::step]
                rowdofs = rec[j + 1 :: step]
                for i, (nid, dof) in enumerate(zip(rowids, rowdofs)):
                    if nid == -1:
                        j += step * i + 2
                        break
                    add_iddof(row_ids, row_iddof, nid, dof)

            # create dataframe:
            mat, rowindex, colindex = _prep_dataframe(
                mtype, form, row_ids, row_iddof, col_iddof, ncol
            )

            # put data into dataframe:
            # df = dct[name]
            r_id_dof = _get_1d_index(rowindex)
            if form != 9:
                c_id_dof = _get_1d_index(colindex)
            else:
                c_id_dof = colindex
            j = 12
            while j < len(rec) - 2:
                nid, dof = rec[j : j + 2]
                if form != 9:
                    nid = nid * 10 + dof
                ci = np.searchsorted(c_id_dof, nid)

                j += 2
                nid = rec[j]
                while nid != -1:
                    dof = rec[j + 1]
                    j += 2
                    val = rec[j : j + ints_per_number]
                    val.dtype = dtype
                    j += ints_per_number
                    ri = np.searchsorted(r_id_dof, nid * 10 + dof)
                    mat[ri, ci] = val
                    if form == 6:
                        mat[ci, ri] = val
                    nid = rec[j]
                j += 2
            dct[name] = pd.DataFrame(mat, index=rowindex, columns=colindex)
        return dct

    # MAIN routine
    if expanded:
        add_iddof = _add_iddof_expanded
        iddof_initializer = list
    else:
        add_iddof = _add_iddof_minimal
        iddof_initializer = set

    if dmig_names is not None:
        if isinstance(dmig_names, str):
            dmig_names = (dmig_names,)
        dmig_names = [i.lower() for i in dmig_names]

    if f is not None and not isinstance(f, str):
        # assume file handle, assume punch
        cards = rdcards(f, name="dmig", return_var="list", blank="")
        return _cards_to_df(cards, dmig_names)

    # read op2 or punch ... try op2, if that fails, assume punch:
    dmigfile = guitools.get_file_name(f, read=True)
    try:
        o2 = op2.OP2(dmigfile)
    except ValueError:
        cards = rdcards(dmigfile, name="dmig", return_var="list", blank="")
        dct = _cards_to_df(cards, dmig_names)
    else:
        o2dct = _read_op2_dmig(o2, dmig_names)
        del o2
        dct = _recs_to_df(o2dct)
    return dct


def mkcomment(comment, width=72, start="$ ", surround=True):
    r"""
    Formats a string into a Nastran comment with wrapping.

    Parameters
    ----------
    comment : string
        The string to write out, wrapping to `width` characters.
    width : integer; optional
        Specify maximum line length.
    start : string; optional
        String (i.e. "$ ", "# ") to start each comment line with.
    surround : bool; optional
        If True, a leading and trailing blank comment line will be
        included.

    Returns
    -------
    string
        The formatted comment.

    Notes
    -----
    Uses the Python function :func:`textwrap.fill` to format the
    string.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> s = ('This is a long comment string to '
    ...      'demonstrate the wrapping feature of '
    ...      'this algorithm. It wraps to 72 '
    ...      'characters by default, but that can '
    ...      'changed by the user via the `width` '
    ...      'option. The default start for each '
    ...      'line is "$ ", but that can be changed '
    ...      'as well.')
    >>> print(nastran.mkcomment(s, width=55), end='')
    $
    $ This is a long comment string to demonstrate the
    $ wrapping feature of this algorithm. It wraps to 72
    $ characters by default, but that can changed by the
    $ user via the `width` option. The default start for
    $ each line is "$ ", but that can be changed as well.
    $
    """
    s = (
        textwrap.fill(
            comment, width=width, initial_indent=start, subsequent_indent=start
        )
        + "\n"
    )
    if surround:
        s = "$\n" + s + "$\n"
    return s


@guitools.write_text_file
def wtdmig(f, dct):
    """
    Writes Nastran DMIG cards to a file.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    dct : dictionary
        Dictionary of pandas DataFrames. The keys will be used as the
        names of the DMIG entries. With one exception, the column and
        row indices of each DataFrame are pandas MultiIndex objects
        with node ID in level 0 and DOF in level 1 (the names are "id"
        and "dof"). The exception is for form "9" matrices: the column
        index in that case is just the column number (starting at one
        for Nastran). Use an OrderedDict from the standard Python
        "collections" module to write the entries in a specific order.

    Returns
    -------
    None

    Notes
    -----
    If a DataFrame has a single level 'columns' attribute, the form
    for that matrix is 9. In that case, the 'NCOL' DMIG header card
    value is determined by maximum value in the 'columns' attribute.

    Otherwise, if a DataFrame has a 2-level 'columns' attribute, the
    form is determined by checking the matrix shape and, if square,
    whether the matrix is symmetric or not. 'NCOL' is set to the
    number of columns in the matrix (but is ignored by Nastran).

    The type is determined by the numpy 'dtype' attribute.

    Currently, both 'POLAR' and 'TOUT' header card values are set to
    0. That means that complex numbers are written in real/imaginary
    parts and the resulting precision in Nastran is automatically
    set.

    Raises
    ------
    ValueError
        When a DataFrame row index is not 2 levels, or when a column
        index is not 1 or 2 levels.

    Examples
    --------
    Create a symmetric matrix and write DMIG to screen:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pyyeti import nastran
    >>> a = np.array([[3.5, -1.2, -2.4],
    ...               [-1.2, 8.8, 6.5],
    ...               [-2.4, 6.5, 9.9]])
    >>> ind = pd.MultiIndex.from_product([[100], [1, 2, 6]])
    >>> k = pd.DataFrame(a, index=ind, columns=ind)
    >>> k              # doctest: +ELLIPSIS
           100...
             1    2    6
    100 1  3.5 -1.2 -2.4
        2 -1.2  8.8  6.5
        6 -2.4  6.5  9.9
    >>> nastran.wtdmig(1, {'k': k})
    DMIG    K              0       6       2       0       0               3
    DMIG*   K                            100               1
    *                    100               1 3.500000000D+00
    *                    100               2-1.200000000D+00
    *                    100               6-2.400000000D+00
    DMIG*   K                            100               2
    *                    100               2 8.800000000D+00
    *                    100               6 6.500000000D+00
    DMIG*   K                            100               6
    *                    100               6 9.900000000D+00

    As another demo, write to a string and use :func:`rddmig` to read
    back in:

    >>> import io
    >>> with io.StringIO() as sio:
    ...     nastran.wtdmig(sio, {'k': k})
    ...     k2 = nastran.rddmig(sio)['k']
    >>> k2              # doctest: +ELLIPSIS
    id       100...
    dof        1    2    6
    id  dof...
    100 1    3.5 -1.2 -2.4
        2   -1.2  8.8  6.5
        6   -2.4  6.5  9.9
    >>> np.all((k2 == k).values)
    True
    """
    for name, value in dct.items():
        name = name.upper()
        rowids = value.index
        colids = value.columns
        # row index must be 2-level MultiIndex:
        if rowids.nlevels != 2:
            raise ValueError(
                f'"{name}" must have a 2-level row index but has '
                f"{rowids.nlevels} levels"
            )

        if colids.nlevels > 2:
            raise ValueError(
                f'"{name}" must have a 1 or 2-level column index but has '
                f"{colids.nlevels} levels"
            )

        m = value.values
        ncol = value.shape[1]

        # determine form of matrix:
        if colids.nlevels == 1:
            form = 9
            ncol = colids.max()
        elif value.shape[0] != value.shape[1]:
            form = 2
        else:
            if np.allclose(m.transpose(), m):
                form = 6
            else:
                form = 1

        # determine type of matrix:
        if np.iscomplexobj(m):
            mtype = 4 if m.dtype.itemsize > 8 else 3
        else:
            mtype = 2 if m.dtype.itemsize > 4 else 1

        # write header card:
        polar = 0
        tout = 0

        #        DMIG  NAME  0    IFO  TIN  TOUT POLAR     NCOL
        f.write(
            f"{'DMIG':<8s}{name:<8s}{0:8d}{form:8d}{mtype:8d}{tout:8d}"
            f"{polar:8d}{'':8s}{ncol:8d}\n"
        )

        # write a column at a time:
        for col in range(m.shape[1]):
            if m[:, col].any():
                start_row = col if form == 6 else 0
                if colids.nlevels == 2:
                    gj, cj = colids[col]
                else:
                    gj = colids[col]
                    cj = 0
                f.write(f"{'DMIG*':<8s}{name:<16s}{gj:16d}{cj:16d}\n")
                for row in range(start_row, m.shape[0]):
                    num = m[row, col]
                    if num != 0.0:
                        gi, ci = rowids[row]
                        if mtype < 3:  # real
                            num_str = f"{num:16.9E}"
                        else:  # complex
                            num_str = f"{num.real:16.9E}{num.imag:16.9E}"
                        if mtype & 1 == 0:  # if even
                            num_str = num_str.replace("E", "D")
                        f.write(f"{'*':<8s}{gi:16d}{ci:16d}{num_str:s}\n")


def rdgrids(f):
    """
    Read Nastran GRID cards from a Nastran bulk file.

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.

    Returns
    -------
    grids : ndarray or None
        8-column ndarray: [ id, cs, x, y, z, cs, spc, seid ]. Returns
        None if no grid cards found

    Notes
    -----
    This routine uses :func:`rdcards` to load the data (can read 8 or
    16 fixed field or comma-delimited).

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> from io import StringIO
    >>> from pandas import DataFrame
    >>> xyz = np.array([[.1, .2, .3], [1.1, 1.2, 1.3]])
    >>> with StringIO() as f:
    ...     nastran.wtgrids(f, [100, 200], xyz=xyz, cd=10)
    ...     g = nastran.rdgrids(f)
    >>> df = DataFrame(g)
    >>> intcols = [0, 1, 5, 6, 7]
    >>> df[intcols] = df[intcols].astype(int)
    >>> df
         0  1    2    3    4   5  6  7
    0  100  0  0.1  0.2  0.3  10  0  0
    1  200  0  1.1  1.2  1.3  10  0  0
    """
    v = rdcards(f, "grid")
    if v is not None:
        c = np.size(v, 1)
        if c < 8:
            v = np.hstack((v, np.zeros((np.size(v, 0), 8 - c))))
        return v


def _convert_card(card):
    if len(card) != 12:
        raise ValueError(
            f"expected 12 fields but got {len(card)}, check: {card[0]}, {card[1]}"
        )
    char = card[0][5]
    if char in "rR":
        ctype = 1
    elif char in "cC":
        ctype = 2
    else:
        ctype = 3

    # for error message, in case it's needed:
    name, ident = card[0], card[1]

    card[0], card[1] = card[1], ctype
    try:
        card = np.array(card).astype(np.float)
    except ValueError as e:
        msg = f"{e.args[0]}, check: {name}, {ident}"
        e.args = (msg,)
        raise

    return card


def rdcord2cards(f):
    """
    Read CORD2* cards from a Nastran bulk file

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.

    Returns
    -------
    dictionary
        Dictionary with the keys being the coordinate system id and
        the values being the 5x3 matrix::

            [id  type 0]  # output coord. sys. id and type
            [xo  yo  zo]  # origin of coord. system
            [    T     ]  # 3x3 transformation to basic
            Note that T is for the coordinate system, not a grid
            (unless type = 1 which means rectangular)

    See also
    --------
    :func:`bulk2uset`, :func:`rdcards`.
    """
    cards = rdcards(
        f, r"(cord2[rcs])\b", return_var="list", regex=True, keep_name=True, blank=0
    )

    if cards is None:
        return {}

    return n2p.build_coords(np.array([_convert_card(card) for card in cards]))


def wtgrids(
    f,
    grids,
    cp=0,
    xyz=np.array([[0.0, 0.0, 0.0]]),
    cd=0,
    ps="",
    seid="",
    form="{:16.8f}",
):
    """
    Writes Nastran GRID cards to a file.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    grids : 1d array_like; length N
        Vector of grid ids. The length of `grids` will be referenced
        as ``N`` in explanations below.
    cp : integer scalar or vector
        Id of coordinate system(s) grids are defined in; either scalar
        or vector with N rows.
    xyz : 2d array_like
        3-column matrix of grid locations; 1 row or N rows.
    cd : integer scalar or vector
        Id of displacement coordinate system for grids; either scalar
        or vector with N rows.
    ps : integer scalar or vector
        Permanent constraints for grids, eg: 123456; either scalar or
        vector with N rows.
    seid : integer scalar or vector
        Superelement id; either scalar or vector with N rows.
    format : string
        String specifying format of XYZ values; must produce 8 or 16
        character strings.

    Returns
    -------
    None

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> nastran.wtgrids(1, np.arange(1, 4))
    GRID*                  1               0      0.00000000      0.00000000
    *             0.00000000               0
    GRID*                  2               0      0.00000000      0.00000000
    *             0.00000000               0
    GRID*                  3               0      0.00000000      0.00000000
    *             0.00000000               0
    >>> xyz = np.array([[.1, .2, .3], [1.1, 1.2, 1.3]])
    >>> nastran.wtgrids(1, [100, 200], xyz=xyz, cd=10,
    ...                 form='{:8.2f}')
    GRID         100       0    0.10    0.20    0.30      10
    GRID         200       0    1.10    1.20    1.30      10
    """
    grids = np.atleast_1d(grids).ravel()
    xyz = np.atleast_2d(xyz)
    teststr = form.format(1.0)
    length = len(teststr)
    if length != 8 and length != 16:
        raise ValueError(
            f"`form` produces a {length} length string. It must be 8 or 16.\n"
        )
    if ps == seid == "":
        if len(teststr) > 8:
            string = (
                "GRID*   {:16d}{:16d}" + form * 2 + "\n*       " + form + "{:16d}\n"
            )
        else:
            string = "GRID    {:8d}{:8d}" + form * 3 + "{:8d}\n"
        writer.vecwrite(f, string, grids, cp, xyz[:, 0], xyz[:, 1], xyz[:, 2], cd)
    else:
        if len(teststr) > 8:
            string = (
                "GRID*   {:16d}{:16d}"
                + form * 2
                + "\n*       "
                + form
                + "{:16d}{:>16}{:>16}\n"
            )
        else:
            string = "GRID    {:8d}{:8d}" + form * 3 + "{:8d}{:>8}{:>8}\n"
        writer.vecwrite(
            f, string, grids, cp, xyz[:, 0], xyz[:, 1], xyz[:, 2], cd, ps, seid
        )


def rdtabled1(f, name="tabled1"):
    """
    Read Nastran TABLED1 or other identically formatted cards from a
    Nastran bulk file.

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.
    name : string; optional
        Name of cards to read.

    Returns
    -------
    dct : dictionary
        Dictionary of TABLED1 (or similar) cards:

        .. code-block:: none

            {ID1: [time, data],
             ID2: [time, data], ...}

    Notes
    -----
    This routine uses :func:`rdcards` to load the data (can read 8
    or 16 fixed field or comma-delimited).

    See also
    --------
    :func:`wttabled1`

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> from io import StringIO
    >>> t = np.arange(0, 1, .05)
    >>> d = np.sin(2*np.pi*3*t)
    >>> with StringIO() as f:
    ...     nastran.wttabled1(f, 4000, t, d, '3 Hz Sine Wave',
    ...                       form='{:8.2f}{:8.5f}')
    ...     dct = nastran.rdtabled1(f)
    >>> np.allclose(t, dct[4000][:, 0])
    True
    >>> np.allclose(d, dct[4000][:, 1])
    True
    """
    d = rdcards(f, name, return_var="dict")
    for tid in d:
        vec = d[tid]
        d[tid] = np.vstack([vec[8:-1:2], vec[9:-1:2]]).T
    return d


@guitools.write_text_file
def wttabled1(f, tid, t, d, title=None, form="{:16.9E}{:16.9E}", tablestr="TABLED1"):
    """
    Writes a Nastran TABLED1 (or similar) card to a file.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    tid : integer
        ID for TABLED1 card
    t : 1d array_like
        time vector
    d : 1d array_like
        data vector
    title : string or None; optional
        If a string, it is written as a comment before table. If None,
        no string is written.
    form : string; optional
        String specifying the format of a single [time, data] pair.
        Expected to result in a string either 16 or 32 characters
        long.
    tablestr : string; optional
        Name of card to write; must be same format as TABLED1.

    Returns
    -------
    None

    Notes
    -----
    In the format string, include a # sign to force a decimal point to
    appear without trailing digits. For example, '{:8.2f}{:#8.0f}'
    would print like this: ' 12.34 123456.'.

    See also
    --------
    :func:`rdtabled1`

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> t = np.arange(0, .91, .05)
    >>> d = np.sin(2*np.pi*3*t)
    >>> nastran.wttabled1(1, 4000, t, d, '3 Hz Sine Wave',
    ...                   form='{:8.2f}{:8.5f}')
    $ 3 Hz Sine Wave
    TABLED1     4000
                0.00 0.00000    0.05 0.80902    0.10 0.95106    0.15 0.30902
                0.20-0.58779    0.25-1.00000    0.30-0.58779    0.35 0.30902
                0.40 0.95106    0.45 0.80902    0.50 0.00000    0.55-0.80902
                0.60-0.95106    0.65-0.30902    0.70 0.58779    0.75 1.00000
                0.80 0.58779    0.85-0.30902    0.90-0.95106ENDT
    >>> nastran.wttabled1(1, 4000, [1, 2, 3], [1, 2, 3],
    ...                   form='{:16.2f}{:16.5f}')
    TABLED1*            4000
    *
    *                   1.00         1.00000            2.00         2.00000
    *                   3.00         3.00000ENDT
    """
    t, d = np.atleast_1d(t, d)
    t = t.ravel()
    d = d.ravel()
    npts = len(t)
    if len(d) != npts:
        raise ValueError(f"len(d) is {len(d)} but len(t) is {npts}")

    # determine if using single or double field:
    n = len(form.format(1, 1))
    if n != 16 and n != 32:
        raise ValueError(f"`form` produces a {n} length string. It must be 16 or 32.")
    if title:
        f.write(f"$ {title:s}\n")
    if n == 32:
        tablestr = tablestr + "*"
        f.write(f"{tablestr:<8s}{tid:16d}\n*\n")
        rows = npts // 2
        r = rows * 2
        writer.vecwrite(
            f, "*       " + form * 2 + "\n", t[:r:2], d[:r:2], t[1:r:2], d[1:r:2]
        )
        f.write("*       ")
        for j in range(r, npts):
            f.write(form.format(t[j], d[j]))
    else:
        f.write(f"{tablestr:<8s}{tid:8d}\n")
        rows = npts // 4
        r = rows * 4
        writer.vecwrite(
            f,
            "        " + form * 4 + "\n",
            t[:r:4],
            d[:r:4],
            t[1:r:4],
            d[1:r:4],
            t[2:r:4],
            d[2:r:4],
            t[3:r:4],
            d[3:r:4],
        )
        f.write("        ")
        for j in range(r, npts):
            f.write(form.format(t[j], d[j]))
    f.write("ENDT\n")


def bulk2uset(*args):
    """
    Read CORD2* and GRID cards from file(s) to make a USET table

    Parameters
    ----------
    *args
        File names or file handles as returned by :func:`open`. Files
        referred to by handle are rewound first. A GUI is open for
        file selection if no arguments are provided or if an argument
        is either a directory name or None.

    Returns
    -------
    uset : pandas DataFrame
        A DataFrame as output by
        :func:`pyyeti.nastran.op2.OP2.rdn2cop2`
    coordref : dictionary
        Dictionary with the keys being the coordinate system id and
        the values being the 5x3 matrix::

            [id  type 0]  # output coord. sys. id and type
            [xo  yo  zo]  # origin of coord. system
            [    T     ]  # 3x3 transformation to basic
            Note that T is for the coordinate system, not a grid
            (unless type = 1 which means rectangular)

    Notes
    -----
    This is the reverse of :func:`uset2bulk`.

    Note that :func:`asm2uset` is similar to this routine but written
    specifically for Nastran .asm files. That routine will return a
    USET table ordered to match the model (according to the SECONCT
    card); it also returns a b-set boolean partition vector.

    All grids are put in the 'b' set. This routine uses
    :func:`pyyeti.nastran.n2p.addgrid` to build the output.

    See also
    --------
    :func:`asm2uset`, :func:`uset2bulk`, :func:`rdcards`,
    :func:`rdgrids`, :func:`pyyeti.nastran.op2.OP2.rdn2cop2`,
    :mod:`pyyeti.nastran.n2p`.
    """
    grids = np.zeros((0, 8))
    coords = {}

    if len(args) == 0:
        args = [None]

    for f in args:
        f = guitools.get_file_name(f, read=True)
        coords.update(rdcord2cards(f))
        g = rdgrids(f)
        if g is not None:
            grids = np.vstack((grids, g))

    i = np.argsort(grids[:, 0])
    grids = grids[i, :]
    uset = n2p.addgrid(
        None,
        grids[:, 0].astype(np.int64),
        "b",
        grids[:, 1],
        grids[:, 2:5],
        grids[:, 5],
        coords,
    )
    return uset, coords


@guitools.write_text_file
def uset2bulk(f, uset):
    """
    Write CORD2* and GRID cards from a USET table

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    uset : pandas DataFrame
        A DataFrame as output by
        :func:`pyyeti.nastran.op2.OP2.rdn2cop2`

    Returns
    -------
    None

    Notes
    -----
    This is the reverse of :func:`bulk2uset`.

    See also
    --------
    :func:`bulk2uset`, :func:`pyyeti.nastran.n2p.mkcordcardinfo`,
    :func:`wtcoordcards`, :func:`wtgrids`.
    """

    # Get some data from the uset table:
    ci = n2p.mkcordcardinfo(uset)
    grids = uset.index.get_level_values("id")
    dof = uset.index.get_level_values("dof")
    pv = dof == 1
    grids = grids[pv]
    xyz = uset.loc[pv, "x":"z"].values
    pv = dof == 2
    cd = uset.loc[pv, "x"].values.astype(int)

    # Write coordinate system cards if needed:
    if ci:
        f.write("$\n$ COORDINATE SYSTEM DATA\n$\n")
        wtcoordcards(f, ci)

    # Write Grid data:
    f.write("$\n$ GRID DATA\n$\n")
    wtgrids(f, grids, 0, xyz, cd)


def rdspoints(f):
    r"""
    Read Nastran SPOINT cards from a Nastran bulk file.

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.

    Returns
    -------
    spoints : 1d ndarray
        Array of SPOINT ids. Will be empty if no SPOINT cards found.

    Notes
    -----
    This routine uses :func:`rdcards` to load the data.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> from io import StringIO
    >>> bulkdata = (
    ...     "SPOINT,1,2,100\n"
    ...     "SPOINT,200,THRU,202\n"
    ...     "SPOINT,5\n"
    ... )
    >>> with StringIO(bulkdata) as f:
    ...     spoints = nastran.rdspoints(f)
    >>> spoints                      # doctest: +ELLIPSIS
    array([  1,   2, 100, 200, 201, 202,   5]...)
    >>> with StringIO("no data") as f:
    ...     spoints = nastran.rdspoints(f)
    >>> spoints                      # doctest: +ELLIPSIS
    array([]...)

    """
    spoint_data = rdcards(f, "spoint", return_var="list")
    spoints = []
    if spoint_data is not None:
        for card in spoint_data:
            if len(card) > 2 and isinstance(card[1], str) and card[1].lower() == "thru":
                spoints.extend([i for i in range(card[0], card[2] + 1)])
            else:
                spoints.extend(card)
    return np.array(spoints, dtype=np.int64)


def rdseconct(f):
    r"""
    Read Nastran SECONCT cards from a Nastran bulk file.

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.

    Returns
    -------
    a_ids, b_ids : 1d ndarrays
        Array of GRID and SPOINT ids for superelement A and
        superelement B, respectively. Both outputs will be empty if no
        SECONCT cards found.

    Notes
    -----
    This routine uses :func:`rdcards` to load the data.

    Examples
    --------
    >>> from io import StringIO
    >>> from pyyeti import nastran
    >>> bulkdata = (
    ...     "SECONCT,101,0,,NO\n"
    ...     ",3,30,11,110,19,190,27,270\n"
    ...     "$\n"
    ...     "SECONCT,101,0,,NO\n"
    ...     ",1101,THRU,1104,2101,THRU,2104\n"
    ... )
    >>> with StringIO(bulkdata) as f:
    ...     a_ids, b_ids = nastran.rdseconct(f)
    >>> a_ids                      # doctest: +ELLIPSIS
    array([   3,   11,   19,   27, 1101, 1102, 1103, 1104]...)
    >>> b_ids                      # doctest: +ELLIPSIS
    array([  30,  110,  190,  270, 2101, 2102, 2103, 2104]...)
    """

    def _get_pairs(card):
        # truncate and filter out blanks:
        card = [i for i in card[8:] if i]
        it = iter(card)
        for a, b in zip(it, it):
            yield a, b

    seconct_data = rdcards(f, "seconct", return_var="list")
    a_ids = []
    b_ids = []
    if seconct_data is not None:
        for card in seconct_data:
            it = iter(_get_pairs(card))
            for a, b in it:
                if isinstance(b, str):  # it must be "THRU"
                    # BTA says I'm a jerk if I don't check for "THRU" :)
                    a2, b1 = next(it)
                    thru, b2 = next(it)
                    a_ids.extend([i for i in range(a, a2 + 1)])
                    b_ids.extend([i for i in range(b1, b2 + 1)])
                else:
                    a_ids.append(a)
                    b_ids.append(b)
    return np.array(a_ids, dtype=np.int64), np.array(b_ids, dtype=np.int64)


@guitools.read_text_file
def asm2uset(f):
    r"""
    Read CORD2* and GRID cards from a ".asm" file to make a USET table

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.

    Returns
    -------
    uset : pandas DataFrame
        A DataFrame as output by
        :func:`pyyeti.nastran.op2.OP2.rdn2cop2`. Contains all GRID and
        SPOINT DOF in the .asm file in the order specified on the
        SECONCT card. This is compatible with the model matrices (eg,
        in the .op4 file).
    coordref : dictionary
        Dictionary with the keys being the coordinate system id and
        the values being the 5x3 matrix::

            [id  type 0]  # output coord. sys. id and type
            [xo  yo  zo]  # origin of coord. system
            [    T     ]  # 3x3 transformation to basic
            Note that T is for the coordinate system, not a grid
            (unless type = 1 which means rectangular)

    bset_bool : 1d ndarray
        A boolean partition vector with True for the b-set. This is
        created for convenience by::

            from pyyeti import nastran
            bset_bool = nastran.mksetpv(uset, 'a', 'b')

    Examples
    --------
    >>> import numpy as np
    >>> from io import StringIO
    >>> from pyyeti import nastran
    >>> asm_bulk = (
    ...    "$ SE101 ASSEMBLY FILE\n"
    ...    "SEBULK,101,EXTOP4,,MANUAL,,,101\n"
    ...    "SECONCT,101,0,,NO\n"
    ...    ",3,3,110,110,19,19,27,27\n"
    ...    "$ Coordinate 10:\n"
    ...    "CORD2R,10,0,0.0,0.0,0.0,1.0,0.0,0.0\n"
    ...    ",0.0,1.0,0.0\n"
    ...    "GRID,3,0,600.,0.,300.,0\n"
    ...    "GRID,19,0,600.,300.,0.,0\n"
    ...    "GRID,27,0,600.,0.,0.\n"
    ...    "SPOINT,110\n"
    ... )
    >>> with StringIO(asm_bulk) as f:
    ...     u, c, b = nastran.asm2uset(f)
    >>> u            # doctest: +ELLIPSIS
              nasset      x      y      z
    id  dof...
    3   1    2097154  600.0    0.0  300.0
        2    2097154    0.0    1.0    0.0
        3    2097154    0.0    0.0    0.0
        4    2097154    1.0    0.0    0.0
        5    2097154    0.0    1.0    0.0
        6    2097154    0.0    0.0    1.0
    110 0    4194304    0.0    0.0    0.0
    19  1    2097154  600.0  300.0    0.0
        2    2097154    0.0    1.0    0.0
        3    2097154    0.0    0.0    0.0
        4    2097154    1.0    0.0    0.0
        5    2097154    0.0    1.0    0.0
        6    2097154    0.0    0.0    1.0
    27  1    2097154  600.0    0.0    0.0
        2    2097154    0.0    1.0    0.0
        3    2097154    0.0    0.0    0.0
        4    2097154    1.0    0.0    0.0
        5    2097154    0.0    1.0    0.0
        6    2097154    0.0    0.0    1.0
    >>> c            # doctest: +SKIP
    {0: array([[ 0.,  1.,  0.],
            [ 0.,  0.,  0.],
            [ 1.,  0.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  0.,  1.]]),
     10: array([[ 10.,   1.,   0.],
            [  0.,   0.,   0.],
            [  0.,   0.,   1.],
            [  1.,   0.,   0.],
            [  0.,   1.,   0.]])}
    >>> np.set_printoptions(linewidth=55)
    >>> b
    array([ True,  True,  True,  True,  True,  True, False,
            True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True], dtype=bool)
    """
    uset, coords = bulk2uset(f)

    # add spoints to uset table:
    spoints = rdspoints(f)
    if spoints is not None:
        n = len(spoints)
        dof = np.zeros((n, 2), np.int64)
        dof[:, 0] = spoints
        uset_spoints = n2p.make_uset(dof, n2p.mkusetmask("q"), np.zeros((n, 3)))
        uset = pd.concat([uset, uset_spoints], axis=0)

    a_ids, b_ids = rdseconct(f)
    uset_ordered = uset.loc[a_ids]
    if uset_ordered.shape[0] != uset.shape[0]:
        raise RuntimeError(
            "Number of SECONCT superelement 'A' DOF do not match GRID and"
            " SPOINT cards in file:\n"
            f"    # DOF on SECONCT = {uset_ordered.shape[0]}\n"
            f"    # DOF on GRIDS/SPOINTS = {uset.shape[0]}"
        )

    return uset_ordered, coords, n2p.mksetpv(uset_ordered, "a", "b")


def rdwtbulk(fin, fout):
    """
    Get bulk data from a sorted Nastran output file.

    Parameters
    ----------
    fin : string or file_like or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.
    fout : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.

    Returns
    -------
    None

    Notes
    -----
    Reads the bulk data from input file and writes it to the output
    file. All continuation codes are stripped off.

    Has not been tested on unsorted bulk.

    Example usage::

        from pyyeti import nastran
        nastran.rdwtbulk('meco1.out', 'meco1.blk')
    """

    @guitools.read_text_file
    def _rdbulk(f):
        fsearch(f, "COUNT        .   1  ..   2  ..   3")
        s = []
        prog = re.compile(r"[ ]{13}[ 0-9]{8}-[ ]{8}(.{72})")
        for line in f:
            if line.startswith("                              ENDDATA"):
                break
            m = prog.match(line)
            if m:
                match = m.group(1).strip()
                if match[0] == "+" or match[0] == "*":
                    if len(match) > 8:
                        match = match[0] + "       " + match[8:]
                    else:
                        match = match[0]
                s.append(match)
        return "\n".join(s) + "\n"

    @guitools.write_text_file
    def _wtbulk(f, blk):
        f.write(blk)

    _wtbulk(fout, _rdbulk(fin))


@guitools.read_text_file
def rdeigen(f, use_pandas=True):
    """
    Read eigenvalue tables from a Nastran f06 file.

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.
    use_pandas : bool; optional
        If True, the values returned in the dictionary will be pandas
        DataFrames

    Returns
    -------
    dictionary
        The keys are the superelement IDs and the values is the 7
        column eigenvalue table: [mode number, extraction order,
        eigenvalue, radians, cycles, generalized mass, generalized
        stiffness]

    Notes
    -----
    The last table read for a superelement replaces tables previously
    read in.

    If the pandas output is chosen (the default) the mode number is
    used as the `index` and `columns` is::

        c = ['Mode #', 'ext #', 'eigenvalue', 'radians',
             'cycles', 'genmass', 'genstif']
    """

    def _find_eigen(f):
        SE = "SUPERELEMENT "
        EIG = "R E A L   E I G E N V A L U E S"
        se = 0
        for line in f:
            if len(line) > 116 and line[104:].startswith(SE):
                se = int(line[116:])
            if len(line) > 76 and line[46:].startswith(EIG):
                return se
        return None

    def _rd_eigen_table(f):
        """Eigenvalue table found, read it."""
        for line in f:
            if line.startswith("    NO."):
                break
        table = []
        continued = True
        while continued:
            for line in f:
                try:
                    row = [float(i) for i in line.split()]
                except ValueError:
                    break
                if len(row) == 7:
                    table.append(row)
            for _ in range(8):
                line = f.readline()
                if line.startswith("1 "):
                    continued = False
                    break
                if line.startswith("    NO."):
                    break
            else:
                break
        return np.array(table)

    dct = {}
    f.seek(0, 0)
    while True:
        se = _find_eigen(f)
        if se is None:
            return dct
        table = _rd_eigen_table(f)
        if use_pandas:
            i = table[:, 0].astype(int)
            c = [
                "mode #",
                "ext #",
                "eigenvalue",
                "radians",
                "cycles",
                "genmass",
                "genstif",
            ]
            table = pd.DataFrame(table, index=i, columns=c)
        dct[se] = table


@guitools.write_text_file
def wtnasints(f, start, ints):
    """
    Utility routine for the nastran 'wt*' routines.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    start : integer
        Beginning field for the integers; card name is in field 1, so
        start should be >= 2.
    ints : 1d array_like
        Vector of integers to write

    Returns
    -------
    None

    Notes
    -----

    This routine is typically not called directly, but could be useful
    to write cards that are not already accounted for in this
    module. As an example usage, here is the code inside
    :func:`wtcsuper`::

        f.write(f"CSUPER  {superid:8d}{0:8d}")
        wtnasints(f, 4, grids)

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> nastran.wtnasints(1, 2, np.arange(28))
           0       1       2       3       4       5       6       7
                   8       9      10      11      12      13      14      15
                  16      17      18      19      20      21      22      23
                  24      25      26      27
    """
    n = len(ints)
    firstline = 10 - start
    if n >= firstline:
        i = firstline
        f.write(("{:8d}" * i + "\n").format(*ints[:i]))
        while n >= i + 8:
            f.write(("{:8s}" + "{:8d}" * 8 + "\n").format("", *ints[i : i + 8]))
            i += 8
        if n > i:
            n -= i
            f.write(("{:8s}" + "{:8d}" * n + "\n").format("", *ints[i:]))
    else:
        f.write(("{:8d}" * n + "\n").format(*ints))


def rdcsupers(f):
    r"""
    Read CSUPER entries

    Parameters
    ----------
    f : string or file_like or None
        Either a filename, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.

    Returns
    -------
    dictionary
        The dictionary keys are the CSUPER IDs. The dictionary values
        are all the ids from the CSUPER card in the form: [csuper_id, 0,
        node_id_1, node_id_2, … node_id_n]

    Notes
    -----
    Any "THRU" fields will be set to -1. This routine simply calls the
    more general routine :func:`rdcards`::

        rdcards(f, 'csuper', return_var='dict', dtype=np.int64, blank=-1)

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> from io import StringIO
    >>> np.set_printoptions(linewidth=60)
    >>> f = StringIO("CSUPER,101,0,3,11,19,27,1995001,1995002\n"
    ...              ",1995003,thru,1995010  $ comment")
    >>> nastran.rdcsupers(f)             # doctest: +ELLIPSIS
    {101: array([    101,       0,       3,      11,      19,      27,
           1995001, 1995002, 1995003,      -1, 1995010]...)}
    """
    return rdcards(f, "csuper", return_var="dict", dtype=np.int64, blank=-1)


def rdextrn(f, expand=True):
    r"""
    Read EXTRN entry from .pch file created by Nastran

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.
    expand : bool; optional
        If True, expand rows like this::

            [100, 123456]

        into 6 separate rows like this::

            [100, 1],
            [100, 2],
            [100, 3],
            [100, 4],
            [100, 5],
            [100, 6],

    Returns
    -------
    2d ndarray
        Two column array: [ID, DOF]

    Notes
    -----
    The expansion is done by :func:`pyyeti.nastran.n2p.expanddof`.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> from io import StringIO
    >>> f = StringIO('EXTRN,3,123456,11,123456,19,123456,27,123456\n'
    ...              ',2995001,0,2995002,0,2995003,0,2995004,0\n'
    ...              ',2995005,0,2995006,0,2995007,0,2995008,0\n')
    >>> nastran.rdextrn(f, expand=False)   # doctest: +ELLIPSIS
    array([[      3,  123456],
           [     11,  123456],
           [     19,  123456],
           [     27,  123456],
           [2995001,       0],
           [2995002,       0],
           [2995003,       0],
           [2995004,       0],
           [2995005,       0],
           [2995006,       0],
           [2995007,       0],
           [2995008,       0]]...)
    >>> f = StringIO('EXTRN,3,123456,2995001,0')
    >>> nastran.rdextrn(f)             # doctest: +ELLIPSIS
    array([[      3,       1],
           [      3,       2],
           [      3,       3],
           [      3,       4],
           [      3,       5],
           [      3,       6],
           [2995001,       0]]...)
    """
    extrn = rdcards(f, "extrn", dtype=np.int64).reshape(-1, 2)
    if expand:
        extrn = n2p.expanddof(extrn)
    return extrn


@guitools.write_text_file
def wtcsuper(f, superid, grids):
    """
    Writes a Nastran CSUPER card to a file.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    superid : integer
        Superelement ID
    grids : 1d array_like
        Vector of grid ids.

    Returns
    -------
    None

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> nastran.wtcsuper(1, 100, np.arange(1, 10))
    CSUPER       100       0       1       2       3       4       5       6
                   7       8       9
    """
    f.write(f"CSUPER  {superid:8d}{0:8d}")
    wtnasints(f, 4, grids)


@guitools.write_text_file
def wtspc1(f, eid, dof, grids, name="SPC1"):
    """
    Writes a Nastran SPC1 (or similar) card to a file.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    eid : integer
        Element ID
    dof : integer
        An integer concatenation of the DOF (ex: 123456)
    grids : 1d array_like
        Vector of grid ids.
    name : string; optional
        Name of NASTRAN card to write; card must be in same format as
        the SPC1 card.

    Returns
    -------
    None

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> nastran.wtspc1(1, 1, 123456, np.arange(1, 10))
    SPC1           1  123456       1       2       3       4       5       6
                   7       8       9
    >>> nastran.wtspc1(1, 200, 123456, np.arange(2001, 2031),
    ...                'SEQSET1')
    SEQSET1      200  123456    2001    2002    2003    2004    2005    2006
                2007    2008    2009    2010    2011    2012    2013    2014
                2015    2016    2017    2018    2019    2020    2021    2022
                2023    2024    2025    2026    2027    2028    2029    2030
    """
    f.write(f"{name:<8s}{eid:8d}{dof:8d}")
    wtnasints(f, 4, grids)


@guitools.write_text_file
def wtxset1(f, dof, grids, name="BSET1"):
    """
    Writes a Nastran BSET1, QSET1 (or similar) card to a file.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    dof : integer
        An integer concatenation of the DOF (ex: 123456)
    grids : 1d array_like
        Vector of grid ids.
    name : string; optional
        Name of NASTRAN card to write; card must be in same format as
        the SPC1 card.

    Returns
    -------
    None

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> nastran.wtxset1(1, 123456, np.arange(1, 11))
    BSET1     123456       1       2       3       4       5       6       7
                   8       9      10
    >>> nastran.wtxset1(1, 0, np.arange(2001, 2013), 'QSET1')
    QSET1          0    2001    2002    2003    2004    2005    2006    2007
                2008    2009    2010    2011    2012
    """
    f.write(f"{name:<8s}{dof:8d}")
    wtnasints(f, 3, grids)


@guitools.write_text_file
def wtqcset(f, startgrid, nq):
    """
    Writes Nastran QSET1 and CSET1 cards for GRID modal DOF for use in
    the DMAP "xtmatrix".

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    startgrid : integer
        The start ID for the modal grids that need to be assigned to
        the Q-set and C-set. Any extra DOF are assigned to the C-set.
        Created grid IDs will be sequential.
    nq : integer
        Number of modal DOF

    Returns
    -------
    None

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> nastran.wtqcset(1, 990001, 14)
    QSET1     123456  990001 THRU     990002
    QSET1         12  990003
    CSET1       3456  990003
    """
    ngrids = (nq + 5) // 6
    endgrid = startgrid + ngrids - 1
    xdof = nq - 6 * (ngrids - 1)
    xdofs = "123456"[:xdof]
    cdofs = "123456"[xdof:]
    # write qset and cset cards:
    if xdof == 6:
        if ngrids > 1:
            f.write(
                f"{'QSET1':<8s}{123456:8d}{startgrid:8d}{' THRU ':<8s}{endgrid:8d}\n"
            )
        else:
            f.write(f"{'QSET1':<8s}{123456:8d}{startgrid:8d}\n")
    else:
        if ngrids > 2:
            f.write(
                f"{'QSET1':<8s}{123456:8d}{startgrid:8d}{' THRU ':<8s}"
                f"{endgrid - 1:8d}\n"
            )
        elif ngrids == 2:
            f.write(f"{'QSET1':<8s}{123456:8d}{startgrid:8d}\n")
        f.write(f"{'QSET1':<8s}{xdofs:>8s}{endgrid:8d}\n")
        f.write(f"{'CSET1':<8s}{cdofs:>8s}{endgrid:8d}\n")


@guitools.write_text_file
def wtrbe2(f, eid, indep, dof, dep):
    """
    Writes a Nastran RBE2 card to a file.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    eid : integer
        Element ID
    indep : integer
        Independent grid ID. All 6 DOF are automatically independent
        and are not specified.
    dof : integer
        An integer concatenation of the DOF (ex: 123456) that applies
        for all the dependent DOF.
    dep : 1d array_like
        Vector of dependent grid IDs

    Returns
    -------
    None

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> nastran.wtrbe2(1, 1, 100, 123456, np.arange(101, 111))
    RBE2           1     100  123456     101     102     103     104     105
                 106     107     108     109     110
    """
    f.write(f"RBE2    {eid:8d}{indep:8d}{dof:8d}")
    wtnasints(f, 5, dep)


@guitools.write_text_file
def wtrbe3(f, eid, GRID_dep, DOF_dep, Ind_List, UM_List=None, alpha=None):
    """
    Writes a Nastran RBE3 card to a file.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    eid : integer
        Element ID
    GRID_dep : integer
        Dependent grid ID
    DOF_dep : integer
        An integer concatenation of the DOF (ex: 123456)
    Ind_List : list
        [ DOF_Ind1, GRIDS_Ind1, DOF_Ind2, GRIDS_Ind2, ... ], where::

            DOF_Ind1   : 1 or 2 element array_like containing the
                         component DOF (ie, 123456) of the nodes in
                         GRIDS_Ind1 and, optionally, the weighting
                         factor for these DOF. If not input, the
                         weighting factor defaults to 1.0.
            GRIDS_Ind1 : array_like of node ids corresponding to
                         DOF_Ind1
            ...
            eg:  [ [123, 1.2], [95, 195, 1000], 123456, 95]

        `Ind_List` must be even length.

    UM_List : None or list; optional
        [ GRID_MSET1, DOF_MSET1, GRID_MSET2, DOF_MSET2, ... ] where::

              GRID_MSET1 : first grid in the M-set
              DOF_MSET1  : DOF of first grid in M-set (integer subset
                           of 123456). No weighting factors are
                           allowed here.
              GRID_MSET2 : second grid in the M-set
              DOF_MSET2  : DOF of second grid in M-set
              ...

    alpha : None or string; optional
        Thermal expansion coefficient in a 8-char string (or less).

    Returns
    -------
    None

    Examples
    --------
    >>> from pyyeti import nastran
    >>> nastran.wtrbe3(1, 100, 9900, 123456,
    ...                [123, [9901, 9902, 9903, 9904],
    ...                 123456, [450001, 200]])
    RBE3         100            9900  123456   1.000     123    9901    9902
                9903    9904   1.000  123456  450001     200
    >>> nastran.wtrbe3(1, 100, 9900, 123456,
    ...                [123, [9901, 9902, 9903, 9904],
    ...                 [123456, 1.2], [450001, 200]],
    ...                UM_List=[9901, 12, 9902, 3, 9903, 12, 9904, 3],
    ...                alpha='6.5e-6')
    RBE3         100            9900  123456   1.000     123    9901    9902
                9903    9904   1.200  123456  450001     200
            UM          9901      12    9902       3    9903      12
                        9904       3
            ALPHA     6.5e-6
    """
    if len(Ind_List) & 1:
        raise ValueError(f"`Ind_List` must have even length (it is {len(Ind_List)})")

    f.write(f"RBE3    {eid:8d}        {GRID_dep:8d}{DOF_dep:8d}")
    field = 5

    def _Inc_Field(f, field):
        field += 1
        if field == 10:
            f.write("\n        ")
            field = 2
        return field

    # loop over independents
    for j in range(0, len(Ind_List), 2):
        DOF_ind = np.atleast_1d(Ind_List[j])
        GRIDS_ind = np.atleast_1d(Ind_List[j + 1])
        dof = int(DOF_ind[0])
        if len(DOF_ind) == 2:
            wt = float(DOF_ind[1])
        else:
            wt = 1.0
        field = _Inc_Field(f, field)
        f.write(f"{wt:8.3f}")
        field = _Inc_Field(f, field)
        f.write(f"{dof:8d}")
        for g in GRIDS_ind:
            field = _Inc_Field(f, field)
            f.write(f"{g:8d}")
    f.write("\n")

    def _Inc_UM_Field(f, field):
        field += 1
        if field == 9:
            f.write("\n                ")
            field = 3
        return field

    if UM_List is not None:
        f.write("        UM      ")
        field = 2
        for j in UM_List:
            field = _Inc_UM_Field(f, field)
            f.write("{:8d}".format(j))
        f.write("\n")

    if alpha is not None:
        f.write(f"        ALPHA   {alpha:>8s}\n")


@guitools.write_text_file
def wtseset(f, superid, grids):
    """
    Writes a Nastran SESET card to a file.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    superid: integer
        Superelement ID
    grids : 1d array_like
        Vector of grid ids.

    Returns
    -------
    None

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> nastran.wtseset(1, 100, np.arange(1, 26))
    SESET        100       1       2       3       4       5       6       7
    SESET        100       8       9      10      11      12      13      14
    SESET        100      15      16      17      18      19      20      21
    SESET        100      22      23      24      25
    """
    n = len(grids)
    # 7 grids per SESET:
    frm = "SESET   {:8d}" + "{:8d}" * 7 + "\n"
    i = 0
    while i + 7 <= n:
        f.write(frm.format(superid, *grids[i : i + 7]))
        i += 7
    if i < n:
        frm = "SESET   {:8d}" + "{:8d}" * (n - i) + "\n"
        f.write(frm.format(superid, *grids[i:]))


@guitools.write_text_file
def wtset(f, setid, ids):
    """
    Writes a Nastran case-control SET card to a file.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    setid: integer
        Set ID
    ids : 1d array_like
        Vector of IDs

    Returns
    -------
    None

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> nastran.wtset(1, 100, np.arange(1, 26))
    SET 100 = 1, 2, 3, 4, 5, 6, 7,
     8, 9, 10, 11, 12, 13, 14,
     15, 16, 17, 18, 19, 20, 21,
     22, 23, 24, 25
    """
    f.write(f"SET {setid} =")
    n = len(ids)
    # 7 ids per line:
    frm = " {:d}," * 7 + "\n"
    i = 0
    while i + 7 < n:
        f.write(frm.format(*ids[i : i + 7]))
        i += 7
    frm = " {:d}," * (n - i - 1) + " {:d}\n"
    f.write(frm.format(*ids[i:]))


@guitools.write_text_file
def wtrspline(f, rid, ids, DoL="0.1"):
    """
    Write Nastran RSPLINE card(s).

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    rid : integer
        ID for 1st RSPLINE; gets incremented by 1 for each RSPLINE
    ids : 2d array_like
        2 column matrix: [grid_ids, independent_flag]; has grids for
        RSPLINE in desired order and a 0 or 1 flag for each where 1
        means the grid is independent. Note: the 1st and last grids
        must be independent.
    DoL : string or real scalar
        Specifies ratio of diameter of elastic tube to the sum of the
        lengths of all segments. Written with: ``f'{DoL:<8}'``

    Returns
    -------
    None

    Notes
    -----
    The RSPLINEs will follow this pattern::

        RSPLINE1  independent1 - dependents1 - independent2
        RSPLINE2  independent2 - dependents2 - independent3
        ...

    Note that there can be multiple dependents sandwiched between the
    two outside independent nodes per RSPLINE.

    The spline element assumes a linear interpolation for displacement
    and torsion along the axis of the spline, a quadratic
    interpolation for rotations normal to the axis of the spline, and
    a cubic interpolation for displacements normal to the axis of the
    spline.

    See also
    --------
    :func:`wtrspline_rings`

    Raises
    ------
    ValueError

        If there are less than 3 grids for RSPLINE.

        If either the first or last grids is dependent.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> ids = np.array([[100, 1],
    ...                 [101, 0],
    ...                 [102, 0],
    ...                 [103, 1],
    ...                 [104, 0],
    ...                 [105, 1]])
    >>> nastran.wtrspline(1, 10, ids)
    RSPLINE       10     0.1     100     101  123456     102  123456     103
    RSPLINE       11     0.1     103     104  123456     105
    """
    ids = np.atleast_2d(ids).astype(np.int64)
    N = ids.shape[0]
    if N < 3:
        raise ValueError("not enough grids for RSPLINE")

    ids, ind = ids.T
    if ind.all():
        raise ValueError("there are no dependent DOF for RSPLINE")

    if not ind[0] == ind[-1] == 1:
        raise ValueError("the first and last grids must be independent")

    # find indices of all dependents:
    deps = (ind == 0).nonzero()[0]

    # find gaps in indices of dependents list:
    gaps = (np.diff(deps) != 1).nonzero()[0]

    # starting indices of dependents:
    firsts = deps[np.hstack((0, gaps + 1))]

    # ending indices of dependents:
    lasts = deps[np.hstack((gaps, len(deps) - 1))]

    # write one RSPLINE per series of dependents:
    for d0, d1 in zip(firsts, lasts):
        f.write(f"RSPLINE {rid:8}{DoL:>8}{ids[d0 - 1]:8}")
        rid += 1
        field = 5  # next nastran field (1 to 10)

        # write all but last grid of current segment:
        for j in range(d0, d1 + 1):
            f.write(f"{ids[j]:8}")
            field += 1
            if field == 10:
                f.write("\n        ")
                field = 2
            f.write("  123456")
            field += 1

        # write closing independent:
        f.write(f"{ids[d1 + 1]:8}\n")


def _intersect(circA, circB, xyA, draw=False):
    """
    Find where a line that passes through the center of circA and
    point xyA intersects circB.

    Parameters
    ----------
    circA : 3-element array_like
        [xc, yc, R] for circle A
    circB : 3-element array_like
        [xc, yc, R] for circle B
    xyA : 2-element array_like
        [x, y] point (probably on circle A)
    draw : bool; optional
        If True, plot circles and points for visual inspection in
        figure 'intersect'

    Returns
    -------
    pt : 1d ndarray
        [x, y] location of intersection point closest to xyA

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from pyyeti.nastran.bulk import _intersect
        >>> circA = (0., 0., 10.)
        >>> circB = (0., 0., 15.)
        >>> xyA = (0., 5.)
        >>> _intersect(circA, circB, [0., 1.], draw=1)
        array([  0.,  15.])
        >>> _intersect(circA, circB, [0., -1.])
        array([  0., -15.])
        >>> _intersect(circA, circB, [1., 0.])
        array([ 15.,   0.])
        >>> _intersect(circA, circB, [-1., 0.])
        array([-15.,   0.])
    """
    (x1, y1, R1) = circA
    (x2, y2) = xyA
    (x3, y3, R3) = circB
    if abs(x2 - x1) > 100 * np.finfo(float).eps:
        m = (y2 - y1) / (x2 - x1)
        b = (x2 * y1 - x1 * y2) / (x2 - x1)
        # (mx+b-y3)**2 + (x-x3)**2 = R3**2
        a2 = m * m + 1
        a1 = 2 * (m * (b - y3) - x3) / a2
        a0 = ((b - y3) ** 2 + x3 ** 2 - R3 ** 2) / a2
        x1 = -a1 / 2 + np.sqrt((a1 / 2) ** 2 - a0)
        x2 = -a1 / 2 - np.sqrt((a1 / 2) ** 2 - a0)
        y1 = m * x1 + b
        y2 = m * x2 + b
    else:
        # line is vertical (x1 & x2 are unchanged)
        #    (y-y3)**2 + (x1-x3)**2 = R3**2
        #    y-y3 = +- np.sqrt(R3**2 - (x1-x3)**2)
        y1 = y3 + np.sqrt(R3 ** 2 - (x1 - x3) ** 2)
        y2 = y3 - np.sqrt(R3 ** 2 - (x1 - x3) ** 2)
    err1 = abs(xyA[0] - x1) + abs(xyA[1] - y1)
    err2 = abs(xyA[0] - x2) + abs(xyA[1] - y2)
    if err1 < err2:
        pt = np.array([x1, y1])
    else:
        pt = np.array([x2, y2])
    if draw:
        plt.figure("intersect")
        plt.clf()
        th = np.arange(0, 361) * np.pi / 180.0
        (x1, y1, R1) = circA
        plt.plot(x1 + R1 * np.cos(th), y1 + R1 * np.sin(th), label="CircA")
        plt.plot(x3 + R3 * np.cos(th), y3 + R3 * np.sin(th), label="CircB")
        plt.scatter(*xyA, c="g", marker="o", s=60, label="xyA")
        plt.scatter(*pt, c="b", marker="v", s=60, label="intersection on CircB")
        plt.axis("equal")
        plt.legend(loc="best", scatterpoints=1)
    return pt


def _check_z_alignment(circ_parms, tol):
    # Have two different transforms to a local system ... have to use
    # same one to get proper node aligning to occur for RSPLINE. But,
    # the z-axis for both better match up ... compute cosine of angle
    # between the two z-axes ... should be very close to 1.0:
    z1 = circ_parms[0].basic2local[2]
    z2 = circ_parms[1].basic2local[2]
    # these are unit vectors; no need to divide by magnitudes:
    cosine = z1.dot(z2)
    if abs(cosine) < tol:
        raise ValueError(
            "perpendicular directions of `r1grids` and "
            f"`r2grids` do not match: {z1} vs. {z2}; the "
            f"cosine of the angle between them is {cosine}"
        )


def _wt_circle1_coord(f, cord_id, center, basic2local, node0, node_id0, names):
    ref_cord_id = 0
    dist = max(1.0, np.linalg.norm(center))
    p = 10 ** np.floor(np.log10(dist))
    dist = round(dist / p) * p
    local_cord = {
        cord_id: [
            "CORD2R",
            np.vstack(
                (
                    [cord_id, 1, ref_cord_id],
                    center,
                    center + dist * basic2local[2],
                    center + dist * basic2local[0],
                )
            ),
        ]
    }
    comment = (
        f"Origin of local CORD2R {cord_id} is at center of {names[0]} ring;"
        " z is perpendicular to plane of circle, and x is aligned with "
        f"node {node0} (and new node {node_id0})."
    )
    f.write(mkcomment(comment))
    wtcoordcards(f, local_cord)


def _wtgrids_rbe2s(
    f, circ_parms, center, basic2local, cord_id, node_id0, rbe2_id0, ring1_ids, names
):
    n1 = circ_parms[0].local.shape[1]
    newpts = np.zeros((n1, 3))

    # define z location of newpts:
    newpts[:, 2] = circ_parms[1].local[2].mean()

    # compute center of circle 2 in local 1 coordinates:
    center2 = basic2local @ (circ_parms[1].center - center)
    radius = circ_parms[0].radius
    for j in range(n1):
        newpts[j, :2] = _intersect(
            [0.0, 0.0, radius],
            [center2[0], center2[1], circ_parms[1].radius],
            circ_parms[0].local[:2, j],
        )

    # write new grids
    comment = (
        f"Grids to RBE2 to the {names[0]} grids. These grids line "
        f"up with the {names[1]} circle so that the RSPLINE (which "
        f"ties together these new grids and the {names[1]} grids) will "
        "be smooth."
    )
    f.write(mkcomment(comment))
    vec = np.arange(n1)
    newids = node_id0 + vec
    rbe2ids = rbe2_id0 + vec
    wtgrids(f, newids, xyz=newpts, cp=cord_id)

    comment = (
        f"RBE2 the {names[0]} nodes to new nodes created above "
        "(the new nodes are independent):"
    )
    f.write(mkcomment(comment))
    writer.vecwrite(f, "RBE2,{},{},123456,{}\n", rbe2ids, newids, ring1_ids)
    return newpts, newids


def _sort_n_write(
    f, independent, circ_parms, newpts, newids, ring2_ids, rspline_id0, DoL, names
):
    n1 = circ_parms[0].local.shape[1]
    n2 = circ_parms[1].local.shape[1]

    # - to do this in order, we need to compute their angles
    th_1 = np.arctan2(newpts[:, 1], newpts[:, 0])
    r2_local = circ_parms[1].local.T
    th_2 = np.arctan2(r2_local[:, 1], r2_local[:, 0])
    th = np.hstack((th_1, th_2))
    th[th < -1e-8] += 2 * np.pi

    ids_1 = np.column_stack((newids, np.zeros(n1, np.int64)))
    ids_2 = np.column_stack((ring2_ids, np.zeros(n2, np.int64)))

    if independent == "ring1":
        comment = (
            f"RSPLINE the {names[1]} nodes to the new nodes created "
            "above, with the new nodes being independent."
        )
        ids_1[:, 1] = 1
    else:
        comment = (
            f"RSPLINE the new nodes created above to the {names[1]} "
            f"nodes, with the {names[1]} nodes being independent."
        )
        ids_2[:, 1] = 1

    f.write(mkcomment(comment))
    ids = np.vstack((ids_1, ids_2))

    # sort by angular location:
    i = np.argsort(th)
    ids = ids[i]
    i = 0
    while ids[i, 1] == 0:
        i += 1
    # stack ids, ensuring starting and ending node is independent:
    ids = np.vstack((ids[i:], ids[:i], ids[i]))
    wtrspline(f, rspline_id0, ids, DoL=DoL)
    return ids


def _plot_rspline(
    ax,
    circ_parms,
    xyz,
    newpts,
    newids,
    basic2local,
    center,
    rspline_nodes,
    ring2_ids,
    names,
):
    for item in "xyz":
        get_func = getattr(ax, f"get_{item}label")
        if not get_func():
            set_func = getattr(ax, f"set_{item}label")
            set_func(item.upper())

    # draw the fit for circles:
    th = np.deg2rad(np.arange(0.0, 361))
    for num, (parms, line) in enumerate(zip(circ_parms, ("+", "x"))):
        x = parms.radius * np.cos(th)
        y = parms.radius * np.sin(th)
        z = 0 * x
        # transform to basic coordinates and plot:
        circle_basic = (
            parms.center + (np.column_stack((x, y, z)) @ parms.basic2local)
        ).T
        h = ax.plot(
            *xyz[num].T,
            line,
            markersize=8.0,
            markeredgewidth=2.0,
            label=f"{names[num]} nodes",
        )[0]
        ax.plot(*circle_basic, h.get_color(), label=f"{names[num]} best-fit circle")

    # get basic coordinates of newpts:
    newpts_basic = newpts @ basic2local + center

    segments = np.empty((3 * newpts.shape[0], 3))
    segments[::3] = newpts_basic
    segments[1::3] = xyz[0]
    segments[2::3] = np.nan

    h = ax.plot(
        *newpts_basic.T,
        "o",
        markersize=5.0,
        markeredgewidth=2.0,
        label=("New {} nodes\n - should be on {} circle").format(*names),
    )[0]
    ax.plot(
        *segments.T,
        "-",
        color=h.get_color(),
        label=f"RBE2s - should be {names[0]} radial",
    )

    # plot the rspline:
    unsorted_rspline_nodes = np.hstack((newids, ring2_ids))
    pv = locate.mat_intersect(unsorted_rspline_nodes, rspline_nodes[:, 0], 2)

    rspline_xyz = np.vstack((newpts_basic, xyz[1]))[pv[0]]
    ax.plot(*rspline_xyz.T, "--", alpha=0.7, linewidth=3.0, label="Final RSPLINE")

    ytools.axis_equal_3d(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.get_figure().tight_layout()


def wtrspline_rings(
    f,
    r1grids,
    r2grids,
    node_id0,
    rspline_id0,
    rbe2_id0=None,
    cord_id=None,
    makeplot="new",
    DoL="0.1",
    independent="ring1",
):
    """
    Creates a smooth RSPLINE to connect two rings of grids

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    r1grids : 2d array_like or DataFrame or tuple
        Ignoring the tuple input option for now, `r1grids` contains
        the ids and locations of the ring 1 grids in basic
        coordinates. If 2d array_like, it has 4 columns describing the
        ring 1 grids::

             [id, x, y, z] < -- basic coordinates

        If DataFrame, it is assumed to be the USET DataFrame
        containing just the ring 1 grids. The format of this
        DataFrame is described in
        :func:`pyyeti.nastran.op2.OP2.rdn2cop2`.

        The default name in the output comments for ring 1 is 'Ring
        1'. By using the tuple input option, you can provide a
        name. For example, the default is equivalent to::

            ('Ring 1', r1grids)

        .. note::
             Note that when using a USET table, the grids can be
             defined in any local coordinate system(s). However, both
             `r1grids` and `r2grids` must use the same basic
             coordinate system.

    r2grids : 2d array_like or DataFrame or tuple
        Contains the ids and locations (and optionally, the name) of
        the ring 2 grids in basic coordinates. See `r1grids` for
        description of format. The default name for ring 2 is 'Ring
        2'.
    node_id0 : integer
        1st id of new nodes created to 'move' ring 1 nodes
    rspline_id0 : integer
        1st id of RSPLINEs
    rbe2_id0 : integer or None; optional
        1st id of RBE2 elements that will connect old ring 1 nodes to
        new ones. If None, ``rbe2_id0 = node_id0``.
    cord_id : integer or None; optional
        ID for the local coordinate system for the `r1grids`. If None,
        it is set to ``node_id0 * 10``.
    makeplot : string or axes object; optional
        Specifies if and how to plot data showing the RSPLINE.

        ===========   ===============================
        `makeplot`    Description
        ===========   ===============================
            'no'      do not plot
         'clear'      plot after clearing figure
           'add'      plot without clearing figure
           'new'      plot in new figure
        axes object   plot in given axes (like 'add')
        ===========   ===============================

        Note that if `makeplot` is 'add' or an axes object, it must be
        3d; otherwise a ValueError exception is raised.

    DoL : string or real scalar; optional
        Specifies ratio of diameter of elastic tybe to the sum of the
        lengths of all segments. Written with: ``f'{DoL:<8}'``
    independent : str; optional
        Either 'ring1' or 'ring2' or the assigned name if one was
        provided in the `r1grids` or `r2grids` input. This input is
        case-insensitive and all spaces are ignored, so 'Ring 1' is
        the same as 'ring1'. Specifies which ring will be independent
        on the RSPLINEs. Note that is different than just switching
        the order of `r1grids` and `r2grids`. This option modifies
        step 4 below while switching `r1grids` and `r2grids` modifies
        all the steps.

    Returns
    -------
    None

    Notes
    -----
    This routine writes GRID, RBE2 and RSPLINE lines to the output
    file.

    The approach is as follows (N = number of ring 1 grids):

      1. Fit a circle through both the ring 1 nodes and the ring 2
         nodes. A new "ring 1" local coordinate system is defined (see
         :func:`pyyeti.ytools.fit_circle_3d`) to simplify the creation
         of the new RSPLINE. In the local system, z is perpendicular
         to the plane of the circle and x is aligned with node 1 of
         ring 1.
      2. Create N new ring 1 grids at station and radius of ring 2
         grids, but at the same angular location as original N.
      3. RBE2 these new grids to the N original grids. The new grids
         are independent.
      4. Write RSPLINE cards using :func:`wtrspline`. The first
         RSPLINE starts at the independent grid (on ring 1 or ring 2
         according to `independent`) with the lowest angular
         location. The angular locations range from 0 to 360 degrees,
         counter-clockwise. The RSPLINEs proceed counter-clockwise
         around the circle.

    See also
    --------
    :func:`wtrspline`

    Raises
    ------
    ValueError

        When the perpendicular directions of `r1grids` and `r2grids`
        do not match within tolerance: absolute value of the cosine of
        the angle between the two perpendicular directions must be >
        0.99.

        When a `r1grids` or `r2grids` DataFrame does not have a
        multiple of 6 rows.

        When the `makeplot` option tries to add to an axes object that
        is not using a 3d projection.

    Examples
    --------
    Define two rings of grids:

    1. Ring 1 will be at station 0.0 with 5 nodes on ring of radius
       50. IDs will be 1 to 5.

    2. Ring 2 will be at station 1.0 with 7 nodes on ring of radius
       45. IDs will be 101 to 107.

    For demonstration, ring 1 will be named 'Spacecraft' while the
    default will be accepted for ring 2 (which is 'Ring 2').

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyyeti import nastran
        >>> theta1 = np.arange(0, 359, 360/5)*np.pi/180
        >>> rad1 = 50.
        >>> sta1 = 0.
        >>> n1 = len(theta1)
        >>> ring1 = np.vstack((np.arange(1, n1+1),      # ID
        ...                    sta1*np.ones(n1),        # x
        ...                    rad1*np.cos(theta1),     # y
        ...                    rad1*np.sin(theta1))).T  # z
        >>> # Provide a name for ring 1:
        >>> ring1 = ('Spacecraft', ring1)
        >>> theta2 = np.arange(10, 359, 360/7)*np.pi/180
        >>> rad2 = 45.
        >>> sta2 = 1.
        >>> n2 = len(theta2)
        >>> ring2 = np.vstack((np.arange(1, n2+1)+100,  # ID
        ...                    sta2*np.ones(n2),        # x
        ...                    rad2*np.cos(theta2),     # y
        ...                    rad2*np.sin(theta2))).T  # z
        >>> fig = plt.figure('Example', figsize=(8, 6))
        >>> fig.clf()
        >>> ax = fig.add_subplot(1, 1, 1, projection='3d')
        >>> nastran.wtrspline_rings(
        ...     1, ring1, ring2, 1001, 2001,
        ...     makeplot=ax)   # doctest: +SKIP
        $
        $ Fit of Spacecraft grids:
        $     Center: [0.000, 0.000, 0.000] (in basic)
        $     Radius: 50.0
        $
        $ Fit of Ring 2 grids:
        $     Center: [1.000, -1.776e-15, -3.553e-15] (in basic)
        $     Radius: 45.0
        $
        $ Origin of local CORD2R 10010 is at center of Spacecraft; z is
        $ perpendicular to plane of circle, and x is aligned with node 1 (and
        $ new node 1001).
        $
        $ Coordinate 10010:
        CORD2R*            10010               0  0.00000000e+00  0.00000000e+00*
        *         0.00000000e+00  1.00000000e+00  0.00000000e+00  0.00000000e+00*
        *         0.00000000e+00  1.00000000e+00  0.00000000e+00
        $
        $ Grids to RBE2 to the Spacecraft grids. These grids line up with the
        $ Ring 2 circle so that the RSPLINE (which ties together these new grids
        $ and the Ring 2 grids) will be smooth.
        $
        GRID*               1001           10010     45.00000000     -0.00000000
        *             1.00000000               0
        GRID*               1002           10010     13.90576475     42.79754323
        *             1.00000000               0
        GRID*               1003           10010    -36.40576475     26.45033635
        *             1.00000000               0
        GRID*               1004           10010    -36.40576475    -26.45033635
        *             1.00000000               0
        GRID*               1005           10010     13.90576475    -42.79754323
        *             1.00000000               0
        $
        $ RBE2 the Spacecraft nodes to new nodes created above (the new nodes
        $ are independent):
        $
        RBE2,1001,1001,123456,1
        RBE2,1002,1002,123456,2
        RBE2,1003,1003,123456,3
        RBE2,1004,1004,123456,4
        RBE2,1005,1005,123456,5
        $
        $ RSPLINE the Ring 2 nodes to the new nodes created above, with the new
        $ nodes being independent.
        $
        RSPLINE     2001     0.1    1001     101  123456     102  123456    1002
        RSPLINE     2002     0.1    1002     103  123456    1003
        RSPLINE     2003     0.1    1003     104  123456     105  123456    1004
        RSPLINE     2004     0.1    1004     106  123456    1005
        RSPLINE     2005     0.1    1005     107  123456    1001
    """

    def _check_grids(grids, name):
        if isinstance(grids, tuple) and len(grids) == 2:
            return grids
        return name, grids

    def _despace(s):
        return ("".join(s.split())).lower()

    r1name, r1grids = _check_grids(r1grids, "Ring 1")
    r2name, r2grids = _check_grids(r2grids, "Ring 2")

    # ensure that independent is either 'ring1' or 'ring2':
    independent = _despace(independent)
    if independent == _despace(r1name):
        independent = "ring1"
    elif independent == _despace(r2name):
        independent = "ring2"
    if independent not in ("ring1", "ring2"):
        raise ValueError("invalid `independent` option")

    if rbe2_id0 is None:
        rbe2_id0 = node_id0
    if cord_id is None:
        cord_id = node_id0 * 10

    names = (r1name, r2name)
    IDs = []
    xyz = []
    for ring, name in ((r1grids, "r1grids"), (r2grids, "r2grids")):
        if isinstance(ring, pd.DataFrame):
            r = ring.shape[0]
            if (r // 6) * 6 != r:
                raise ValueError(
                    f"number of rows `{name}` is not " "multiple of 6 for USET input"
                )
            IDs.append(ring.index.get_level_values("id")[::6])
            xyz.append(ring.iloc[::6, 1:].values)
        else:
            ring = np.atleast_2d(ring)
            IDs.append(ring[:, 0].astype(np.int64))
            xyz.append(ring[:, 1:])

    # fit both circles:
    circ_parms = [ytools.fit_circle_3d(xyz[i].T) for i in (0, 1)]

    # z axes better be aligned:
    _check_z_alignment(circ_parms, 0.99)

    # use 1st basic2local transform on second set of data to get all
    # coordinates in same local system:
    basic2local = circ_parms[0].basic2local
    center = circ_parms[0].center
    circ_parms[1].local = basic2local @ (xyz[1] - center).T

    # the center point will often have near-zeros but be numerically
    # off ... check for this and adjust if that's the case:
    radius = circ_parms[0].radius
    for i, value in enumerate(center):
        if abs(value) < 1e-7 * radius:
            center[i] = 0.0

    @guitools.write_text_file
    def _write_file(f):
        # write circle info for reference:
        for i in range(2):
            ctr = circ_parms[i].center
            ctr_string = f"[{ctr[0]:#.4g}, {ctr[1]:#.4g}, {ctr[2]:#.4g}]"
            comment = (
                "$\n"
                f"$ Fit of {names[i]} grids:\n"
                f"$     Center: {ctr_string} (in basic)\n"
                f"$     Radius: {circ_parms[i].radius}\n"
            )
            f.write(comment)

        # write local coordinate system to file:
        _wt_circle1_coord(f, cord_id, center, basic2local, IDs[0][0], node_id0, names)

        # create new nodes that will be RBE2'd to ring 1 nodes, but
        # located on ring 2 circle
        # - these nodes will be defined in the local system but output
        #   in basic
        newpts, newids = _wtgrids_rbe2s(
            f,
            circ_parms,
            center,
            basic2local,
            cord_id,
            node_id0,
            rbe2_id0,
            IDs[0],
            names,
        )

        # rspline will tie the 'newpts' and the ring 2 nodes together:
        rspline_nodes = _sort_n_write(
            f, independent, circ_parms, newpts, newids, IDs[1], rspline_id0, DoL, names
        )

        return newpts, newids, rspline_nodes

    # write the rspline:
    newpts, newids, rspline_nodes = _write_file(f)

    ax = ytools._check_makeplot(makeplot, figsize=[8, 6], need3d=True)
    if ax:
        _plot_rspline(
            ax,
            circ_parms,
            xyz,
            newpts,
            newids,
            basic2local,
            center,
            rspline_nodes,
            IDs[1],
            names,
        )


def wtcoordcards(f, ci):
    """
    Write Nastran CORD2* cards to a file

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    ci : dictionary or None
        Dictionary of coordinate card info as returned by
        :func:`pyyeti.nastran.n2p.mkcordcardinfo`. If None or if dict
        is empty, this routine quietly does nothing.

    Returns
    -------
    None

    Notes
    -----
    Typically called by :func:`wtextseout`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> ci = {10: ['CORD2R', np.array([[10, 1, 0],
    ...                                [100., 0., 0.],
    ...                                [100., 0., 100.],
    ...                                [200., 0., 0.]])]}
    >>> nastran.wtcoordcards(1, ci)
    $ Coordinate 10:
    CORD2R*               10               0  1.00000000e+02  0.00000000e+00*
    *         0.00000000e+00  1.00000000e+02  0.00000000e+00  1.00000000e+02*
    *         2.00000000e+02  0.00000000e+00  0.00000000e+00
    """
    if ci is None or len(ci) == 0:
        return

    @guitools.write_text_file
    def _wtcoords(f, ci):
        for k in ci:
            data = ci[k]  # [name, [[id, type, ref]; A; B; C]]
            coord = data[1]
            abc = +coord[1:]
            abc[abs(abc) < abs(abc).max() * 1e-15] = 0.0
            f.write(f"$ Coordinate {k}:\n")
            f.write(
                "{:<8s}{:16d}{:16d}{:16.8e}{:16.8e}*\n".format(
                    data[0] + "*", k, int(coord[0, 2]), *abc[0, :2]
                )
            )
            f.write(("{:<8s}" + "{:16.8e}" * 4 + "*\n").format("*", abc[0, 2], *abc[1]))
            f.write(("{:<8s}" + "{:16.8e}" * 3 + "\n").format("*", *abc[2]))

    _wtcoords(f, ci)


@guitools.write_text_file
def wtextrn(f, ids, dof):
    """
    Writes a Nastran EXTRN card to a file.

    Parameters
    ----------
    f : string or file_like or 1 or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open` or :class:`io.StringIO`. Input as integer 1 to
        write to stdout. Can also be the name of a directory or None;
        in these cases, a GUI is opened for file selection.
    ids : 1d array_like
        Vector of node ids
    dof : 1d array_like
        Vector of DOF

    Returns
    -------
    None

    Notes
    -----
    Typically called by :func:`wtextseout`.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> nastran.wtextrn(1, [999, 10001, 10002, 10003, 10004, 10005],
    ...                    [123456, 0, 0, 0, 0, 0])
    EXTRN        999  123456   10001       0   10002       0   10003       0
               10004       0   10005       0
    """
    f.write("EXTRN   ")
    ints = np.zeros(len(ids) * 2, dtype=int)
    ints[::2] = ids
    ints[1::2] = dof
    wtnasints(f, 2, ints)


def wtextseout(name, *, se, maa, baa, kaa, bset, uset, spoint1, sedn=0, **kwargs):
    """
    Write .op4, .asm, .pch and possibly the damping DMIG file for an
    external SE.

    Note that all inputs except `name` must be named and can be input
    in any order.

    Parameters
    ----------
    name : string
        Basename for files; eg: 'spacecraft'. Files with the
        extensions '.op4', '.asm', and '.pch' will be created. If `bh`
        is True, a '.baa_dmig' file will also be created.
    se : integer
        Superelement id; also used as the Fortran unit number on the
        SEBULK entry.
    maa, baa, kaa : 2d array_like
        These are the Craig-Bampton mass, damping and stiffness
        matrices respectively.
    bset : 1d array_like
        Index partition vector for the bset
    uset : pandas DataFrame
        A DataFrame as output by
        :func:`pyyeti.nastran.op2.OP2.rdn2cop2`.

        .. warning::
            Unlike :func:`pyyeti.cb.mk_net_drms`, this USET table
            defines the b-set nodes relative to the basic coordinate
            system of superelement 0. This is so the external
            superelement is positioned properly. Note however that the
            "displacement" coordinate system(s) (specified after the
            coordinates on the "GRID" card) must match whatever the
            displacement coordinate system(s) are for the
            Craig-Bampton component.

    spoint1 : integer
        Starting value for the SPOINTs (for modal DOF)
    sedn : integer; optional
        Downstream superelement id
    **kwargs : optional
        Allows user to input other matrices to be written to the op4
        file. Name must be one of the following to be included (order
        is as written to op4)::

            'k4xx', 'pa', 'gpxx', 'gdxx', 'va', 'mug1', 'mug1o',
            'mes1', 'mes1o', 'mee1', 'mee1o', 'mgpf', 'mgpfo', 'mef1',
            'mef1o', 'mqg1', 'mqg1o', 'mqmg1', 'mqmg1o'

        By default, 1x1 zero matrices are written for all these names
        except for two: 'pa' is a vector of zeros and 'va' is a vector
        of ones.

    Returns
    -------
    None
    """
    maa, baa, kaa = np.atleast_2d(maa, baa, kaa)
    bset = np.atleast_1d(bset)
    n = maa.shape[0]
    nq = n - len(bset)

    # prepare standard Nastran op4 file:
    k4xx = 0.0
    pa = np.zeros((n, 1))
    gpxx = 0.0
    gdxx = 0.0
    va = np.ones((n, 1))
    mug1 = 0.0
    mug1o = 0.0
    mes1 = 0.0
    mes1o = 0.0
    mee1 = 0.0
    mee1o = 0.0
    mgpf = 0.0
    mgpfo = 0.0
    mef1 = 0.0
    mef1o = 0.0
    mqg1 = 0.0
    mqg1o = 0.0
    mqmg1 = 0.0
    mqmg1o = 0.0
    namelist = [
        "kaa",
        "maa",
        "baa",
        "k4xx",
        "pa",
        "gpxx",
        "gdxx",
        "va",
        "mug1",
        "mug1o",
        "mes1",
        "mes1o",
        "mee1",
        "mee1o",
        "mgpf",
        "mgpfo",
        "mef1",
        "mef1o",
        "mqg1",
        "mqg1o",
        "mqmg1",
        "mqmg1o",
    ]
    dct = {**locals(), **kwargs}
    varlist = [dct[i] for i in namelist]
    op4.write(name + ".op4", namelist, varlist)

    # Get some data from the uset table:
    ci = n2p.mkcordcardinfo(uset)
    grids = uset.index.get_level_values("id")
    dof = uset.index.get_level_values("dof")
    pv = dof == 1
    grids = grids[pv]
    xyz = uset.loc[pv, "x":"z"].values
    pv = dof == 2
    cd = uset.loc[pv, "x"].values.astype(int)

    # Write out ASM file
    unit = se
    spointn = spoint1 + nq - 1
    with open(name + ".asm", "w") as f:
        f.write(
            f"$ {name.upper()} ASSEMBLY FILE FOR RESIDUAL RUN...INCLUDE IN BULK DATA\n"
        )
        f.write("$\n")
        f.write(f"SEBULK  {se:8d}  EXTOP4          MANUAL                {unit:8d}\n")
        f.write(f"SECONCT {se:8d}{sedn:8d}              NO\n")
        f.write("        ")
        gids = np.vstack((grids, grids)).T
        wtnasints(f, 2, gids.ravel())

        # Write coordinate system cards if needed:
        f.write("$\n$ COORDINATE SYSTEM DATA\n$\n")
        wtcoordcards(f, ci)

        # Write Grid data:
        f.write("$\n$ BOUNDARY GRID DATA\n$\n")
        wtgrids(f, grids, 0, xyz, cd)
        if spointn >= spoint1:
            f.write("$\n")
            f.write(f"SECONCT {se:8d}{sedn:8d}              NO\n")
            f.write(
                f"        {spoint1:8d}    THRU{spointn:8d}"
                f"{spoint1:8d}    THRU{spointn:8d}\n"
            )
            f.write("$\n")
            f.write(f"SPOINT  {spoint1:8d}    THRU{spointn:8d}\n")

    # Write out PCH file
    with open(name + ".pch", "w") as f:
        f.write(f"$ {name.upper()} PUNCH FILE FOR RESIDUAL RUN...INCLUDE AT END\n")
        f.write("$\n")
        f.write(f"BEGIN SUPER{se:8d}\n$\n")
        ids = np.hstack((grids, spoint1 + np.arange(nq)))
        dof = np.zeros_like(ids, dtype=int)
        dof[: len(grids)] = 123456
        wtextrn(f, ids, dof)
        f.write("$\n")
        f.write("$ COORDINATE SYSTEM DATA\n$\n")
        wtcoordcards(f, ci)
        f.write("$\n$ BOUNDARY GRID DATA\n$\n")
        wtgrids(f, grids, 0, xyz, cd)
        f.write("$\n")

        f.write("$ BSET\n$\n")
        f.write(f"ASET1   {123456:8d}")
        wtnasints(f, 3, grids)
        if spointn >= spoint1:
            f.write("$\n")
            f.write("$ QSET\n$\n")
            f.write(f"QSET1   {0:8d}{spoint1:8d}    THRU{spointn:8d}\n")
            f.write("$\n")
            f.write(f"SPOINT  {spoint1:8d}    THRU{spointn:8d}\n")


def mknast(
    script=None,
    *,
    nascom="nast9p1",
    nasopt="batch=no",
    ext="out",
    stoponfatal=False,
    shell="/bin/sh",
    files=None,
    before="",
    after="",
    top="",
    bottom="",
):
    """
    Creates a shell script for running a chain of Nastran (or other)
    runs.

    Note that all inputs except `script` must be named and can be
    input in any order.

    Parameters
    ----------
    script : string or None; optional
        Name of shell script to create; if None, user is prompted for
        name.
    nascom : string; optional
        Nastran command; note: this can actually specify any program,
        but `nasopt` would likely need adjusting too.
    nasopt : string; optional
        Options for nastran command
    ext : string; optional
        The extension of the Nastran output (f06) file; usually 'f06'
        or 'out'
    stoponfatal : bool; optional
        Set to True if you want the script to exit on first FATAL
        instead of continuing on
    shell : string; optional
        Shell program to write at top of script, eg: #!`shell`
    files : string or None; optional
        List of filenames to run; if None, user is prompted for names
    before : string; optional
        String to put in shell script before each nastran command
    after : string; optional
        String to put in shell script after each nastran command
    top : string; optional
        String to put in shell script at the top
    bottom : string; optional
        String to put in shell script at the bottom

    Returns
    -------
    None

    Notes
    -----

    If you're in the 'bash' shell, the resulting script can be run in
    the background with, assuming the script name is 'doruns.sh'::

       exec ./doruns.sh > doruns.log 2>&1 &

    Example usage::

        from pyyeti import nastran
        nastran.mknast('doruns.sh', files=['file1.dat', 'file2.dat'])
    """
    # Name of script to create:
    if script is None:
        script = input("Name of shell script to create (doruns.sh): ")
        if not script:
            script = "doruns.sh"

    # initialize shell script
    GREP = "grep -l '[*^][*^][*^].*FATAL'"
    with open(script, "w") as f:
        f.write(f"#!{shell}\n")
        curdir = os.getcwd()
        f.write(f"cd {curdir}\n\n")
        if top:
            f.write(f"{top}\n")

        i = -1
        while 1:  # loop over file names
            i += 1
            if files is not None:
                if i >= len(files):
                    break
                else:
                    nasfile = files[i]
            else:
                p = f"File #{i + 1:2d} (blank to quit): "
                nasfile = input(p)
                if not nasfile:
                    break

            if not os.path.exists(nasfile):
                print(f"Warning:  file '{nasfile}' not found.\n")

            f.write(f"\n# ******** File {nasfile} ********\n")
            p = nasfile.rfind("/")
            if p > -1:
                filepath = nasfile[:p]
                filename = nasfile[p + 1 :]
                f.write(f"  cd {filepath}\n")
                docd = 1
            else:
                filename = nasfile
                docd = 0

            if before:
                f.write(f"{before}\n")
            f.write(f"  {nascom} '{nasopt}' '{filename}'\n")

            if stoponfatal:
                p = filename.rfind(".")
                if p > -1:
                    f06file = filename[: p + 1] + ext
                else:
                    f06file = filename + "." + ext

                f.write(f"  if [ X != X`{GREP} {f06file}` ] ; then\n")
                f.write("    exit\n")
                f.write("  fi\n")
            if after:
                f.write(f"{after}\n")

            if docd:
                f.write(f"  cd {curdir}\n")

        if bottom:
            f.write(f"{bottom}\n")
    os.system(f"chmod a+rx '{script}'")


def rddtipch(f, name="TUG1"):
    """
    Read the 2nd record of specific DTIs from a .pch file.

    Parameters
    ----------
    f : string or file_like or None
        Either a name of a file, or is a file_like object as returned
        by :func:`open`. If file_like object, it is rewound first. Can
        also be the name of a directory or None; in these cases, a GUI
        is opened for file selection.
    name : string
        Name of DTI table to read from the .pch file

    Returns
    -------
    id_dof : ndarray
        2-column matrix of the form: ``[id, dof]``. The number of rows
        in `id_dof` corresponds to the matching matrix in .op4 file
        (see Notes).

    Notes
    -----
    This routine is useful for reading data from .pch files written by
    the EXTSEOUT command. The 2nd record of "TUG1" (for example)
    contains the DOF that correspond to the rows of the "MUG1" matrix
    in the .op4 file. That matrix can be read by the
    :mod:`pyyeti.nastran.op4` module.

    Example usage::

        # read mug1 and tug1 (created from EXTSEOUT):
        from pyyeti import nastran
        from pyyeti.nastran import op4
        mug1 = op4.read('data.op4', 'mug1')['mug1']
        tug1 = nastran.rddtipch('data.pch')

        # form DRM to recovery grid 100, dof 4:
        from pyyeti import locate
        row = locate.find_rows(tug1, [100, 4])
        drm = mug1[row, :]
    """
    string = f"DTI     {name:<8s}2"
    c = rdcards(f, string)
    c = c[0, 16:-1].reshape(-1, 4).astype(np.int64)
    m = c[:, 1:]

    # array([[      3,       6,       1],
    #        [     11,       6,       7],
    #        [     19,       6,      13],
    #        [     27,       6,      19],
    #        [     45,       6,      25],
    #        [     60,       6,      31],
    #        [1995002,       6,      37]])

    if np.all(m[:, 1] == m[0, 1]):
        # all ids have same number of components, so take shortcut:
        c = m[0, 1]
        ids = np.dot(m[:, 0:1], np.ones((1, c), np.int64))
        ids = ids.reshape((-1, 1))
        dof = np.arange(1, c + 1, dtype=np.int64).reshape(1, -1)
        dofs = np.dot(np.ones((np.size(m, 0), 1), np.int64), dof)
        dofs = dofs.reshape((-1, 1))
        iddof = np.hstack((ids, dofs))
    else:
        nrows = m[-1, 2] + m[-1, 1] - 1
        iddof = np.zeros((nrows, 2), np.int64)
        j = 0
        for J in range(np.size(m, 0)):
            pv = np.arange(0, m[J, 1], dtype=np.int64)
            iddof[pv + j, 0] = m[J, 0]
            iddof[pv + j, 1] = pv + 1
            j += m[J, 1]
    return iddof
