# -*- coding: utf-8 -*-
"""
Collection of tools for reading/writing Nastran data.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from pyyeti import n2p, locate, writer, ytools, op4, op2, guitools


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
            s = s.lower().replace('d', 'e')
            try:
                return float(s)
            except ValueError:
                s = s[0]+s[1:].replace('+', 'e+').replace('-', 'e-')
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
        File handle to search in.
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


def _rdgpwg(f, search_strings):
    """
    Routine used by :func:`rdgpwg`. See documentation for
    :func:`rdgpwg`.
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

    line, p = fsearch(f, 'W E I G H T')
    if line is None:
        return default
    line, p = fsearch(f, 'REFERENCE POINT =')
    refpt = int(line[p+17:])

    f.readline()
    mass = []
    for _ in range(6):
        line = f.readline().strip()
        mass.append([float(item) for item in line[1:-1].split()])
    mass = np.array(mass)
    line, p = fsearch(f, 'MASS AXIS SYSTEM')

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


def rdgpwg(f, search_strings=None):
    """
    Read a GPWG table from a Nastran F06 file.

    Parameters
    ----------
    f : string or file handle or None
        Either a name of a file or a file handle as returned by
        :func:`open`. If handle, file is rewound first. Can also be
        the name of a directory or None; in these cases, a GUI is
        opened for file selection.
    search_strings : None, string, or list-like of strings
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
    return ytools.rdfile(f, _rdgpwg, search_strings)


def _get_line(fiter, s=None, trim=True):
    if s is None:
        try:
            s = next(fiter)  # f.readline()
        except StopIteration:
            return None
    s = s.expandtabs()
    if trim:
        s = s[:72]
    p = s.find('$')
    if p > -1:
        s = s[:p]
    s = s.rstrip()
    return s


def _rdfixed(fiter, s, n, conchar, blank, tolist):
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
    """
    vals = []
    length = len(s)
    i = -1
    I = -1
    if n > 8:
        inc = 4  # number of values per line
    else:
        inc = 8
    maxstart = 72-n
    while 1:
        for i in range(i, I):
            vals.append(blank)
        i = I
        j = 8
        while j <= maxstart and length > j:
            i += 1
            v = nas_sscanf(s[j:j+n], tolist)
            if v is not None:
                vals.append(v)
            else:
                vals.append(blank)
            j += n
        s = _get_line(fiter)
        if s is None or len(s) == 0:
            break
        if conchar.find(s[0]) < 0:
            break
        length = len(s)
        I += inc
    return vals, s


def _rdcomma(fiter, s, conchar, blank, tolist):
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
    """
    vals = []
    i = -1
    I = -1
    inc = 8
    while 1:
        for i in range(i, I):
            vals.append(blank)
        i = I
        tok = s.split(',')
        lentok = min(len(tok), 9)
        for j in range(1, lentok):
            i += 1
            v = nas_sscanf(tok[j], tolist)
            if v is not None:
                vals.append(v)
            else:
                vals.append(blank)
        s = _get_line(fiter, trim=0)
        if s is None or len(s) == 0:
            break
        if conchar.find(s[0]) < 0:
            break
        I += inc
    return vals, s


def _rdcards(fin, name, blank, todict, tolist, dtype, no_data_return):
    """
    Routine used by :func:`rdcards`. See documentation for
    :func:`rdcards`.
    """
    name = name.lower()
    if todict:
        Vals = {}
        tolist = False
    else:
        Vals = []
    mxlen = 0
    fin.seek(0, 0)

    def _readline(it):
        try:
            s = next(it)
        except StopIteration:
            s = None
        return s

    it = iter(fin)
    s = _readline(it)
    while True:
        while s is not None and s.lower().find(name) != 0:
            s = _readline(it)
        if s is None:
            break
        s = _get_line(it, s, trim=0)
        p = s.find(',')
        if p > -1:
            vals, s = _rdcomma(it, s, ' +,', blank, tolist)
        else:
            s = s[:72].rstrip()
            p = s[:8].find('*')
            if p > -1:
                vals, s = _rdfixed(it, s, 16, '*', blank, tolist)
            else:
                vals, s = _rdfixed(it, s, 8, ' +', blank, tolist)
        if tolist:
            Vals.append(vals)
        else:
            cur = len(vals)
            mxlen = max(mxlen, cur)
            vals = np.array(vals).astype(dtype)
            if todict:
                Vals[vals[0]] = vals
            else:
                Vals.append(vals)
    if len(Vals) > 0:
        if not (todict or tolist):
            npVals = np.empty((len(Vals), mxlen), dtype=dtype)
            npVals[:] = blank
            for i, vals in enumerate(Vals):
                npVals[i, :len(vals)] = vals
            Vals = npVals
        return Vals
    return no_data_return


def rdcards(f, name, blank=0, todict=False, tolist=False, dtype=float,
            no_data_return=None):
    """
    Read Nastran cards (lines) into a matrix, dictionary, or list.

    Parameters
    ----------
    f : string or file handle or None
        Either a name of a file or a file handle as returned by
        :func:`open`. If handle, file is rewound first. Can also be
        the name of a directory or None; in these cases, a GUI is
        opened for file selection.
    name : string
        Usually the card name, but is really just the initial part of
        the string to look for. This means that `name` can span more
        then one field, in case that is handy.
    blank : scalar; optional
        Numeric value to use for blank fields and string-valued
        fields.
    todict : bool; optional
        If True, return a dictionary. Otherwise, return a matrix
        unless `tolist` is True.
    tolist : bool; optional
        If True (and `todict` is False), return a list of
        lists. Otherwise, return a matrix or dictionary according to
        `todict`. Returning a list is the only way to preserve strings
        and is therefore the "truest" read of the cards.
    dtype : data-type, optional
        The desired data-type for the output.
    no_data_return : any variable; optional
        If no data is found, this routine returns `no_data_return`.

    Returns
    -------
    cards : ndarray or dictionary or list or no_data_return

        - If a matrix is returned, each row is a card, padded with
          blanks as necessary.
        - If a dictionary is returned, the first number from each card
          is used as the key and the value are all the numbers from the
          card in a row vector (including the first value).
        - If a list (of lists) is returned, each item is a card and is
          a list of values from the card, the length of which is the
          number of items on the card.
        - `no_data_return` is returned if no cards of requested name
          were found.

    Notes
    -----
    This routine can read fixed field (8 or 16 chars wide) or
    comma-delimited and can handle any number of continuations. Note
    that the characters in the continuation fields are ignored. It
    also uses nas_sscanf to read the numbers, so numbers like 1.-3
    (which means 1.e-3) are okay. Blank fields and string fields are
    set to `blank`, except when `tolist` is True; in that case, string
    fields are kept as is and only blank fields are set to `blank`.

    Note:  this routine is has no knowledge of any card, which means
    that it will not append trailing blanks to a card. For example,
    if a GRID card is: 'GRID, 1', then this routine would return 1, not
    [1, 0, 0, 0, 0, 0, 0, 0]. The :func:`rdgrids` routine would return
    [1, 0, 0, 0, 0, 0, 0, 0] since it knows the number of fields a GRID
    card has.
    """
    return ytools.rdfile(f, _rdcards, name, blank, todict, tolist,
                         dtype, no_data_return)


def rddmig(f, dmig_names=None, *, expanded=False, square=False):
    """
    Read DMIG entries from a Nastran punch (bulk) or output2 file.

    Parameters
    ----------
    f : string or file handle or None
        Either a name of a punch or output2 file or a file handle as
        returned by :func:`open`. If handle, punch format is assumed
        and file is rewound first. Can also be the name of a directory
        or None; in these cases, a GUI is opened for file selection.
    dmig_names : None, string, or list-like of strings
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
        objects with node ID in level 0 and DOF in level 1. The
        exception is for type "9" matrices: the column index in that
        case is just the column number (as specified by Nastran).

    Notes
    -----
    For the punch format, this routine first reads all DMIG entries
    via :func:`rdcards` using the `tolist` option. It then builds a
    dictionary of DataFrames for only those entries specified in
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
    >>> dct1['mat']
    ID          1
    DOF         1
    ID DOF       
    1  3     12.0
    10 0    100.0
    >>> dct2['mat']
    ID          1                         
    DOF         1    2    3    4    5    6
    ID DOF                                
    1  1      0.0  0.0  0.0  0.0  0.0  0.0
       2      0.0  0.0  0.0  0.0  0.0  0.0
       3     12.0  0.0  0.0  0.0  0.0  0.0
       4      0.0  0.0  0.0  0.0  0.0  0.0
       5      0.0  0.0  0.0  0.0  0.0  0.0
       6      0.0  0.0  0.0  0.0  0.0  0.0
    10 0    100.0  0.0  0.0  0.0  0.0  0.0
    >>> dct3['mat']
    ID         1         10
    DOF         1    3    0
    ID DOF                 
    1  1      0.0  0.0  0.0
       3     12.0  0.0  0.0
    10 0    100.0  0.0  0.0
    >>> dct4['mat']
    ID         1                             10
    DOF         1    2    3    4    5    6    0
    ID DOF                                     
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
        iddof = sorted(iddof, key=lambda x: 10*x[0]+x[1])
        return pd.MultiIndex.from_tuples(iddof, names=['ID', 'DOF'])

    def _mk_dataframe(mtype, form, row_ids, row_iddof, col_iddof):
        # create dataframe:
        dtype = float if mtype < 3 else complex
        if form != 9:
            if form == 6 or (form ==1 and square):
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
                colindex = np.arange(1, ncol+1)
            else:
                colindex = sorted(nid for nid, dof in col_iddof)

        data = np.zeros((len(rowindex), len(colindex)), dtype)
        return pd.DataFrame(data, index=rowindex, columns=colindex)

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
            dct[name] = _mk_dataframe(mtype, form, row_ids,
                                      row_iddof, col_iddof)

            j = i + 1
            # put data into dataframe:
            df = dct[name]
            while j < len(c) and c[j][0].lower() == name:
                nid, dof = c[j][1:3]
                if form != 9:
                    col = (nid, dof)
                else:
                    col = nid

                # for each column, look across rows:
                rowids = c[j][4::4]
                rowdofs = c[j][5::4]
                reals = c[j][6::4]
                if mtype < 3:
                    for nid, dof, real in zip(rowids, rowdofs, reals):
                        df.loc[(nid, dof), col] = real
                        if form == 6:
                            df.loc[col, (nid, dof)] = real
                else:
                    imags = c[j][7::4]
                    for nid, dof, real, imag in zip(rowids, rowdofs,
                                                    reals, imags):
                        val = real + 1j*imag
                        df.loc[(nid, dof), col] = val
                        if form == 6:
                            df.loc[col, (nid, dof)] = val
                j += 1
            i = j
        return dct

    def _read_op2_dmig(o2, dmig_names):
        o2._fileh.seek(o2._postheaderpos)
        dct = {}
        name, trailer, dbtype = o2._rdop2nt()
        while name is not None:
            # print(f'op2: found {name}')
            if (dmig_names is not None and
                    name.lower() not in dmig_names):
                o2.skipop2table()
            else:
                rec = o2.rdop2record()
                if np.all(rec[:3] == (114, 1, 120)):
                    dct[name] = rec
                o2.go_to_next_db()
            name, trailer, dbtype = o2._rdop2nt()
        return dct

    def _recs_to_df(recs):
        dct = {}
        for name, rec in recs.items():
            form = rec[6]
            mtype = rec[7]  # 1 real, 2 double, 3 complex, 4 complex double
            ncol = rec[11] if form == 9 else None

            ints_per_number = mtype
            if mtype > 2:
                ints_per_number = 2*(mtype-2)
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

            # Look across columns for id, dof pairs. Within each column,
            # look across rows for id, dof pairs.

            # - each DMIG entry is a column
            # - end of each DMIG entry is marked with row:
            #     [ID, DOF] = [-1, -1]

            # 12 & 13 4-byte integers are 1st col id, dof pair
            j = 12
            while j < len(rec)-2:
                nid, dof = rec[j:j+2]
                add_iddof(col_ids, col_iddof, nid, dof)

                j += 2
                rowids = rec[j::step]
                rowdofs = rec[j+1::step]
                for i, (nid, dof) in enumerate(zip(rowids, rowdofs)):
                    if nid == -1:
                        j += step*i + 2
                        break
                    add_iddof(row_ids, row_iddof, nid, dof)

            # create dataframe:
            dct[name] = _mk_dataframe(mtype, form, row_ids,
                                      row_iddof, col_iddof)

            # put data into dataframe:
            df = dct[name]
            j = 12
            while j < len(rec)-2:
                nid, dof = rec[j:j+2]
                if form != 9:
                    col = (nid, dof)
                else:
                    col = nid

                j += 2
                nid = rec[j]
                while nid != -1:
                    dof = rec[j+1]
                    j += 2
                    val = rec[j:j+ints_per_number]
                    val.dtype = dtype
                    j += ints_per_number
                    df.loc[(nid, dof), col] = val
                    if form == 6:
                        df.loc[col, (nid, dof)] = val
                    nid = rec[j]
                j += 2
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
        cards = rdcards(f, name='dmig', tolist=True, blank='')
        return _cards_to_df(cards, dmig_names)

    # read op2 or punch ... try op2, if that fails, assume punch:
    dmigfile = guitools.get_file_name(f, read=True)
    try:
        o2 = op2.OP2(dmigfile)
    except ValueError:
        cards = rdcards(dmigfile, name='dmig', tolist=True, blank='')
        dct = _cards_to_df(cards, dmig_names)
    else:
        o2dct = _read_op2_dmig(o2, dmig_names)
        del o2
        dct = _recs_to_df(o2dct)
    return dct


def rdgrids(f):
    """
    Read Nastran GRID cards from a Nastran bulk file.

    Parameters
    ----------
    f : string or file handle or None
        Either a name of a file or a file handle as returned by
        :func:`open`. If handle, file is rewound first. Can also be
        the name of a directory or None; in these cases, a GUI is
        opened for file selection.

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
    v = rdcards(f, 'grid')
    if v is not None:
        c = np.size(v, 1)
        if c < 8:
            v = np.hstack((v, np.zeros((np.size(v, 0), 8-c))))
        return v


def wtgrids(f, grids, cp=0, xyz=np.array([[0., 0., 0.]]),
            cd=0, ps="", seid="", form="{:16.8f}"):
    """
    Writes Nastran GRID cards to a file.

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
    grids : 1d array_like
        Vector of grid ids; N = len(grids).
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
    if not isinstance(ps, str):
        ps = '{}'.format(ps)
    if not isinstance(seid, str):
        seid = '{}'.format(seid)

    xyz = np.atleast_2d(xyz)
    teststr = form.format(1.)
    length = len(teststr)
    if length != 8 and length != 16:
        raise ValueError("`form` produces a {} length string. It"
                         " must be 8 or 16.\n", length)
    if ps == seid == "":
        if len(teststr) > 8:
            string = ("GRID*   {:16d}{:16d}" + form*2 +
                      "\n*       " + form + "{:16d}\n")
        else:
            string = ("GRID    {:8d}{:8d}" + form*3 + "{:8d}\n")
        writer.vecwrite(f, string, grids, cp, xyz[:, 0], xyz[:, 1],
                        xyz[:, 2], cd)
    else:
        if len(teststr) > 8:
            string = ("GRID*   {:16d}{:16d}" + form*2 +
                      "\n*       " + form +
                      "{:16d}{:>16s}{:>16s}\n")
        else:
            string = ("GRID    {:8d}{:8d}" + form*3 +
                      "{:8d}{:>8s}{:>8s}\n")
        writer.vecwrite(f, string, grids, cp, xyz[:, 0], xyz[:, 1],
                        xyz[:, 2], cd, ps, seid)


def rdtabled1(f, name='tabled1'):
    """
    Read Nastran TABLED1 or other identically formatted cards from a
    Nastran bulk file.

    Parameters
    ----------
    f : string or file handle or None
        Either a name of a file or a file handle as returned by
        :func:`open`. If handle, file is rewound first. Can also be
        the name of a directory or None; in these cases, a GUI is
        opened for file selection.
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
    d = rdcards(f, name, todict=1)
    for tid in d:
        vec = d[tid]
        d[tid] = np.vstack([vec[8:-1:2], vec[9:-1:2]]).T
    return d


def _wttabled1(f, tid, t, d, title, form, tablestr):
    """
    Routine used by :func:`wttabled1`. See documentation for
    :func:`wttabled1`.
    """
    t, d = np.atleast_1d(t, d)
    t = t.ravel()
    d = d.ravel()
    npts = len(t)
    if len(d) != npts:
        raise ValueError('len(d) is {} but len(t) is {}'.
                         format(len(d), npts))

    # determine if using single or double field:
    n = len(form.format(1, 1))
    if n != 16 and n != 32:
        raise ValueError('`form` produces a {} length string. It '
                         'must be 16 or 32.'.format(n))
    if title:
        f.write('$ {:s}\n'.format(title))
    if n == 32:
        tablestr = tablestr + '*'
        f.write('{:<8s}{:16d}\n*\n'.format(tablestr, tid))
        rows = npts // 2
        r = rows*2
        writer.vecwrite(f, '*       '+form*2+'\n',
                        t[:r:2], d[:r:2], t[1:r:2], d[1:r:2])
        f.write('*       ')
        for j in range(r, npts):
            f.write(form.format(t[j], d[j]))
    else:
        f.write('{:<8s}{:8d}\n'.format(tablestr, tid))
        rows = npts // 4
        r = rows*4
        writer.vecwrite(f, '        '+form*4+'\n',
                        t[:r:4], d[:r:4], t[1:r:4], d[1:r:4],
                        t[2:r:4], d[2:r:4], t[3:r:4], d[3:r:4])
        f.write('        ')
        for j in range(r, npts):
            f.write(form.format(t[j], d[j]))
    f.write('ENDT\n')


def wttabled1(f, tid, t, d, title=None, form="{:16.9E}{:16.9E}",
              tablestr="TABLED1"):
    """
    Writes a Nastran TABLED1 (or similar) card to a file.

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
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
    return ytools.wtfile(f, _wttabled1, tid, t, d,
                         title, form, tablestr)


def bulk2uset(*args):
    """
    Read CORD2* and GRID cards from file(s) to make a USET table and
    coordinate system dictionary via :func:`pyyeti.n2p.addgrid`.

    Parameters
    ----------
    *args
        File names (or handles as returned by :func:`open`). Files
        referred to by handle are rewound first. If an argument is
        None or a directory name, a GUI is open for file selection.

    Returns
    -------
    uset : ndarray
        A 6-column matrix as output by
        :func:`pyyeti.op2.OP2.rdn2cop2`.
    coordref : dictionary
        Dictionary with the keys being the coordinate system id and
        the values being the 5x3 matrix::

            [id  type 0]  # output coord. sys. id and type
            [xo  yo  zo]  # origin of coord. system
            [    T     ]  # 3x3 transformation to basic
            Note that T is for the coordinate system, not a grid
            (unless type = 0 which means rectangular)

    Notes
    -----
    All grids are put in the 'b' set.

    See also
    --------
    :func:`rdcards`, :func:`rdgrids`, :func:`pyyeti.op2.OP2.rdn2cop2`,
    :mod:`pyyeti.n2p`.
    """
    grids = np.zeros((0, 8))
    no_data = np.zeros((0, 11))
    cord2r = cord2c = cord2s = no_data
    if len(args) == 0:
        args = [None]
    for f in args:
        f = guitools.get_file_name(f, read=True)
        cord2r = np.vstack((cord2r, rdcards(f, 'cord2r',
                                            no_data_return=no_data)))
        cord2c = np.vstack((cord2c, rdcards(f, 'cord2c',
                                            no_data_return=no_data)))
        cord2s = np.vstack((cord2s, rdcards(f, 'cord2s',
                                            no_data_return=no_data)))
        g = rdgrids(f)
        if g is not None:
            grids = np.vstack((grids, g))

    def _expand_cords(cord, v):
        r = np.size(cord, 0)
        if r > 0:
            O = np.ones((r, 1))*v
            cord = np.hstack((cord[:, :1], O, cord[:, 1:]))
            return cord
        return np.zeros((0, 12))

    cord2r = _expand_cords(cord2r, 1.)
    cord2c = _expand_cords(cord2c, 2.)
    cord2s = _expand_cords(cord2s, 3.)
    # each cord matrix is 12 columns:
    #   [id, type, ref, a1, a2, a3, b1, b2, b3, c1, c2, c3]
    cords = np.vstack((cord2r, cord2c, cord2s))
    coordref = n2p.build_coords(cords)
    i = np.argsort(grids[:, 0])
    grids = grids[i, :]
    n = np.size(grids, 0)
    uset = np.zeros((n*6, 6))
    for j in range(n):
        grid = grids[j, 0]
        csin = grids[j, 1]
        csout = grids[j, 5]
        xyz = grids[j, 2:5]
        r = j*6
        uset[r:r+6, :] = n2p.addgrid(None, grid, 'b', csin, xyz, csout,
                                     coordref)
    return uset, coordref


def rdwtbulk(fin, fout):
    """
    Get bulk data from a sorted Nastran output file.

    Parameters
    ----------
    fin : string or file handle
        Either a name of a file or a file handle as returned by
        :func:`open`. If handle, file is rewound first. Can also be
        the name of a directory or None; in these cases, a GUI is
        opened for file selection.
    fout : string or file handle or 1
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.

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
    def _rdbulk(f):
        fsearch(f, "COUNT        .   1  ..   2  ..   3")
        s = []
        prog = re.compile(r'[ ]{13}[ 0-9]{8}-[ ]{8}(.{72})')
        for line in f:
            if line.startswith('                              ENDDATA'):
                break
            m = prog.match(line)
            if m:
                match = m.group(1).strip()
                if match[0] == '+' or match[0] == '*':
                    if len(match) > 8:
                        match = match[0]+'       '+match[8:]
                    else:
                        match = match[0]
                s.append(match)
        return '\n'.join(s)+'\n'

    def _wtbulk(f, blk):
        f.write(blk)

    blk = ytools.rdfile(fin, _rdbulk)
    return ytools.wtfile(fout, _wtbulk, blk)


def rdeigen(f, use_pandas=True):
    """
    Read eigenvalue tables from a Nastran f06 file.

    Parameters
    ----------
    f : string or file handle or None
        Either a name of a file or a file handle as returned by
        :func:`open`. If handle, file is rewound first. Can also be
        the name of a directory or None; in these cases, a GUI is
        opened for file selection.
    use_pandas : bool; optional
        If True, the values with be pandas objects

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
        SE = 'SUPERELEMENT '
        EIG = 'R E A L   E I G E N V A L U E S'
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
            if line.startswith('    NO.'):
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
                if line.startswith('1 '):
                    continued = False
                    break
                if line.startswith('    NO.'):
                    break
            else:
                break
        return np.array(table)

    def _rdeigen(f, use_pandas):
        dct = {}
        f.seek(0, 0)
        while True:
            se = _find_eigen(f)
            if se is None:
                return dct
            table = _rd_eigen_table(f)
            if use_pandas:
                i = table[:, 0].astype(int)
                c = ['Mode #', 'ext #', 'eigenvalue', 'radians',
                     'cycles', 'genmass', 'genstif']
                table = pd.DataFrame(table, index=i, columns=c)
            dct[se] = table

        while True:
            se = _find_eigen(f)
            if se is None:
                break
            dct[se] = _rd_eigen_table(f)
        if use_pandas:
            for se in dct:
                table = dct[se]
                i = table[:, 0].astype(int)
                c = ['Mode #', 'ext #', 'eigenvalue', 'radians',
                     'cycles', 'genmass', 'genstif']
                dct[se] = pd.DataFrame(table, index=i, columns=c)
        return dct

    return ytools.rdfile(f, _rdeigen, use_pandas)


def wtnasints(f, start, ints):
    """
    Utility routine for the nastran 'wt*' routines.

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
    start : integer
        Beginning field for the integers; card name is in field 1, so
        start should be >= 2.
    ints : 1d array_like
        Vector of integers to write

    Returns
    -------
    None

    Examples
    --------
    >>> from pyyeti import nastran
    >>> wtnasints(1, 2, np.arange(20))
           0       1       2       3       4       5       6       7
                   8       9      10      11      12      13      14      15
                  16      17      18      19
    """
    def _wtints(f, start, ints):
        n = len(ints)
        firstline = 10-start
        if n >= firstline:
            i = firstline
            f.write(('{:8d}'*i+'\n').format(*ints[:i]))
            while n >= i+8:
                f.write(('{:8s}'+'{:8d}'*8+'\n').
                        format('', *ints[i:i+8]))
                i += 8
            if n > i:
                n -= i
                f.write(('{:8s}'+'{:8d}'*n+'\n').
                        format('', *ints[i:]))
        else:
            f.write(('{:8d}'*n+'\n').format(*ints))

    return ytools.wtfile(f, _wtints, start, ints)


def rdcsupers(f):
    r"""
    Read CSUPER entries

    Parameters
    ----------
    f : string or file handle or None
        Either a name of a file or a file handle as returned by
        :func:`open`. If handle, file is rewound first. Can also be
        the name of a directory or None; in these cases, a GUI is
        opened for file selection.

    Returns
    -------
    dictionary
        The keys are the CSUPER IDs and the values are the ids from
        the card: [csuper_id, 0, node_ids]

    Notes
    -----
    Any "THRU" fields will be set to -1. This routine simply calls the
    more general routine :func:`rdcards`::

        rdcards(f, 'csuper', todict=True, dtype=np.int64, blank=-1)

    Examples
    --------
    >>> from pyyeti import nastran
    >>> from io import StringIO
    >>> f = StringIO("CSUPER,101,0,3,11,19,27,1995001,1995002\n"
    ...              ",1995003,thru,1995010  $ comment")
    >>> nastran.rdcsupers(f)             # doctest: +ELLIPSIS
    {101: array([    101,       0,       3,      11,      19,      27, 1995001,
           1995002, 1995003,      -1, 1995010]...)}
    """
    return rdcards(f, 'csuper', todict=True, dtype=np.int64, blank=-1)


def rdextrn(f, expand=True):
    r"""
    Read EXTRN entry from .pch file created by Nastran

    Parameters
    ----------
    f : string or file handle or None
        Either a name of a file or a file handle as returned by
        :func:`open`. If handle, file is rewound first. Can also be
        the name of a directory or None; in these cases, a GUI is
        opened for file selection.
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
    The expansion is done by :func:`pyyeti.n2p.expanddof`.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> from io import StringIO
    >>> f = StringIO('EXTRN,3,123456,11,123456,19,123456,27,123456\n'
    ...              ',2995001,0,2995002,0,2995003,0,2995004,0\n'
    ...              ',2995005,0,2995006,0,2995007,0,2995008,0\n')
    >>> nastran.rdextrn(f, expand=False)             # doctest: +ELLIPSIS
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
    extrn = rdcards(f, 'extrn', dtype=np.int64).reshape(-1, 2)
    if expand:
        extrn = n2p.expanddof(extrn)
    return extrn


def wtcsuper(f, superid, grids):
    """
    Writes a Nastran CSUPER card to a file.

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
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
    def _wtcsuper(f, superid, grids):
        f.write('CSUPER  {:8d}{:8d}'.format(superid, 0))
        wtnasints(f, 4, grids)
    return ytools.wtfile(f, _wtcsuper, superid, grids)


def wtspc1(f, eid, dof, grids, name='SPC1'):
    """
    Writes a Nastran SPC1 (or similar) card to a file.

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
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
    def _wtspc1(f, eid, dof, grids, name):
        f.write('{:<8s}{:8d}{:8d}'.format(name, eid, dof))
        wtnasints(f, 4, grids)
    return ytools.wtfile(f, _wtspc1, eid, dof, grids, name)


def wtxset1(f, dof, grids, name="BSET1"):
    """
    Writes a Nastran BSET1, QSET1 (or similar) card to a file.

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
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
    def _wtxset1(f, dof, grids, name):
        f.write('{:<8s}{:8d}'.format(name, dof))
        wtnasints(f, 3, grids)
    return ytools.wtfile(f, _wtxset1, dof, grids, name)


def wtqcset(f, startgrid, nq):
    """
    Writes Nastran QSET1 and CSET1 cards for GRID modal DOF for use in
    the DMAP "xtmatrix".

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
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
    def _wtqcset(f, startgrid, nq):
        ngrids = (nq + 5) // 6
        endgrid = startgrid + ngrids - 1
        xdof = nq - 6*(ngrids-1)
        xdofs = '123456'[:xdof]
        cdofs = '123456'[xdof:]
        # write qset and cset cards:
        if xdof == 6:
            if ngrids > 1:
                f.write('{:<8s}{:8d}{:8d}{:<8s}{:8d}\n'.
                        format('QSET1', 123456,
                               startgrid, ' THRU ', endgrid))
            else:
                f.write('{:<8s}{:8d}{:8d}\n'.
                        format('QSET1', 123456, startgrid))
        else:
            if ngrids > 2:
                f.write('{:<8s}{:8d}{:8d}{:<8s}{:8d}\n'.
                        format('QSET1', 123456,
                               startgrid, ' THRU ', endgrid-1))
            elif ngrids == 2:
                f.write('{:<8s}{:8d}{:8d}\n'.
                        format('QSET1', 123456, startgrid))
            f.write('{:<8s}{:>8s}{:8d}\n'.
                    format('QSET1', xdofs, endgrid))
            f.write('{:<8s}{:>8s}{:8d}\n'.
                    format('CSET1', cdofs, endgrid))

    return ytools.wtfile(f, _wtqcset, startgrid, nq)


def wtrbe2(f, eid, indep, dof, dep):
    """
    Writes a Nastran RBE2 card to a file.

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
    eid : integer
        Element ID
    indep : integer
        Independent grid ID
    dof : integer
        An integer concatenation of the DOF (ex: 123456)
    dep : 1d array_like
        Vector of dependend grid IDs

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
    def _wtrbe2(f, eid, indep, dof, dep):
        f.write('RBE2    {:8d}{:8d}{:8d}'.format(eid, indep, dof))
        wtnasints(f, 5, dep)
    return ytools.wtfile(f, _wtrbe2, eid, indep, dof, dep)


def wtrbe3(f, eid, GRID_dep, DOF_dep, Ind_List,
           UM_List=None, alpha=None):
    """
    Writes a Nastran RBE3 card to a file.

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
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
    def _wtrbe3(f, eid, GRID_dep, DOF_dep, Ind_List, UM_List, alpha):
        f.write('RBE3    {:8d}        {:8d}{:8d}'.
                format(eid, GRID_dep, DOF_dep))
        field = 5

        def _Inc_Field(f, field):
            field += 1
            if field == 10:
                f.write('\n        ')
                field = 2
            return field

        # loop over independents
        for j in range(0, len(Ind_List), 2):
            DOF_ind = np.atleast_1d(Ind_List[j])
            GRIDS_ind = np.atleast_1d(Ind_List[j+1])
            dof = int(DOF_ind[0])
            if len(DOF_ind) == 2:
                wt = float(DOF_ind[1])
            else:
                wt = 1.0
            field = _Inc_Field(f, field)
            f.write('{:8.3f}'.format(wt))
            field = _Inc_Field(f, field)
            f.write('{:8d}'.format(dof))
            for g in GRIDS_ind:
                field = _Inc_Field(f, field)
                f.write('{:8d}'.format(g))
        f.write('\n')

        def _Inc_UM_Field(f, field):
            field += 1
            if field == 9:
                f.write('\n                ')
                field = 3
            return field

        if UM_List is not None:
            f.write('        UM      ')
            field = 2
            for j in UM_List:
                field = _Inc_UM_Field(f, field)
                f.write('{:8d}'.format(j))
            f.write('\n')

        if alpha is not None:
            f.write('        ALPHA   {:>8s}\n'.format(alpha))

    if len(Ind_List) & 1:
        raise ValueError('`Ind_List` must have even length '
                         '(it is {})'.format(len(Ind_List)))
    return ytools.wtfile(f, _wtrbe3, eid, GRID_dep, DOF_dep,
                         Ind_List, UM_List, alpha)


def wtseset(f, superid, grids):
    """
    Writes a Nastran SESET card to a file.

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
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
    def _wtseset(f, superid, grids):
        n = len(grids)
        # 7 grids per SESET:
        frm = 'SESET   {:8d}' + '{:8d}'*7 + '\n'
        i = 0
        while i+7 <= n:
            f.write(frm.format(superid, *grids[i:i+7]))
            i += 7
        if i < n:
            frm = 'SESET   {:8d}' + '{:8d}'*(n-i) + '\n'
            f.write(frm.format(superid, *grids[i:]))
    return ytools.wtfile(f, _wtseset, superid, grids)


def wtset(f, setid, ids):
    """
    Writes a Nastran case-control SET card to a file.

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
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
    def _wtset(f, setid, ids):
        f.write('SET {:d} ='.format(setid))
        n = len(ids)
        # 7 ids per line:
        frm = ' {:d},'*7 + '\n'
        i = 0
        while i+7 < n:
            f.write(frm.format(*ids[i:i+7]))
            i += 7
        frm = ' {:d},'*(n-i-1) + ' {:d}\n'
        f.write(frm.format(*ids[i:]))
    return ytools.wtfile(f, _wtset, setid, ids)


def _wtrspline(f, rid, ids, nper, DoL):
    """
    Routine used by :func:`wtrspline`. See documentation for
    :func:`wtrspline`.
    """
    ids = np.atleast_2d(ids)
    N = ids.shape[0]
    if N < 3:
        raise ValueError('not enough grids for RSPLINE')
    dof = ['123456', '']
    ind = ids[:, 1].astype(np.int64)
    ids = ids[:, 0]
    ind[0] = ind[-1] = 1
    j = 0
    while j < N-1:
        f.write('RSPLINE {:8}{:>8}{:8}'.format(rid, DoL, ids[j]))
        rid += 1
        j += 1
        pos = 5
        count = 1
        done = 0
        # write all but last grid of current segment:
        while not done and j < N-1:
            f.write('{:8}'.format(ids[j]))
            pos += 1
            count += 1
            if pos == 10:
                f.write('\n        ')
                pos = 2
            f.write('{:>8}'.format(dof[ind[j]]))
            j += 1
            pos += 1
            if ind[j] and count > nper:
                done = 1
        # write last grid for this rspline
        f.write('{:8}\n'.format(ids[j]))


def wtrspline(f, rid, ids, nper=1, DoL='0.1'):
    """
    Write Nastran RSPLINE card(s).

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
    rid : integer
        ID for 1st RSPLINE; gets incremented by 1 for each RSPLINE
    ids : 2d array_like
        2 column matrix: [grid_ids, independent_flag]; has grids for
        RSPLINE in desired order and a 0 or 1 flag for each where 1
        means the grid is independent. Note: the flag is ignored for
        the 1st and last grids; these are forced to be independent.
    nper : integer
        Number of grids to write per RSPLINE before starting to look
        for next independent grid which will end the RSPLINE. Routine
        will actually write a minimum of 3 nodes::

                  independent - dependent - independent

        ``nper = 1`` ensures the smallest rsplines are written.
    DoL : string or real scalar
        Specifies ratio of diameter of elastic tybe to the sum of the
        lengths of all segments. Written with: ``'{:<8}'.format(DoL)``

    Returns
    -------
    None

    Notes
    -----
    If ``nper = 1``, the RSPLINEs will follow this pattern::

        RSPLINE1  independent1 - dependent1 - independent2
        RSPLINE2  independent2 - dependent2 - independent3
        ...

    Note that there can be multiple dependents per RSPLINE.

    See also
    --------
    :func:`wtrspline_rings`

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> ids = np.array([[100, 1], [101, 0], [102, 0],
    ...                 [103, 1], [104, 0], [105, 1]])
    >>> nastran.wtrspline(1, 10, ids)
    RSPLINE       10     0.1     100     101  123456     102  123456     103
    RSPLINE       11     0.1     103     104  123456     105
    >>> nastran.wtrspline(1, 10, ids, nper=15)
    RSPLINE       10     0.1     100     101  123456     102  123456     103
                         104  123456     105
    """
    return ytools.wtfile(f, _wtrspline, rid, ids, nper, DoL)


def findcenter(x, y, doplot=False):
    """
    Find radius and center point of x-y data points

    Parameters
    ----------
    x, y : 1d array_like
        Vectors x, y data points (in cartesian coordinates) that are
        on a circle: [x, y]
    doplot : bool; optional
        If True, a figure named 'findcenter' will be opened to show
        the fit.

    Returns
    -------
    p : 1d ndarray
        Vector: [xc, yc, R]

    Notes
    -----
    Uses :func:`scipy.optimize.leastsq` to find optimum circle
    parameters.

    Examples
    --------
    For a test, provide precise x, y coordinates, but only for a 1/4
    circle:

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from pyyeti.nastran import findcenter
        >>> xc, yc, R = 1., 15., 35.
        >>> th = np.linspace(0., np.pi/2, 10)
        >>> x = xc + R*np.cos(th)
        >>> y = yc + R*np.sin(th)
        >>> findcenter(x, y, doplot=1)
        array([  1.,  15.,  35.])
    """
    x, y = np.atleast_1d(x, y)
    clx, cly = x.mean(), y.mean()
    R0 = (x.max()-clx + y.max()-cly)/2

    # The optimization routine leastsq needs a function that returns
    # the residuals:
    #       y - func(p, x)
    # where "func" is the fit you're trying to match
    def circle(p, d):
        # p is [xc, yc, R]
        # d is [x;y] coordinates
        xc, yc, R = p
        n = len(d) // 2
        theta = np.arctan2(d[n:]-yc, d[:n]-xc)
        return d - np.hstack((xc + R*np.cos(theta),
                              yc + R*np.sin(theta)))

    p0 = (clx, cly, R0)
    d = np.hstack((x, y))
    res = leastsq(circle, p0, args=(d,), full_output=1)
    sol = res[0]
    if res[-1] not in (1, 2, 3, 4):
        raise ValueError(':func:`scipy.optimization.leastsq` failed: '
                         '{}'.res[-2])
    ssq = np.sum(res[2]['fvec']**2)
    if ssq > .01:
        msg = ('data points do not appear to form a good circle, sum '
               'square of residuals = {}'.format(ssq))
        warnings.warn(msg, RuntimeWarning)

    if doplot:
        plt.figure('findcenter')
        plt.clf()
        plt.scatter(x, y, c='r', marker='o', s=60,
                    label='Input Points')
        th = np.arange(0, 361)*np.pi/180.
        (x, y, R) = sol
        plt.plot(x+R*np.cos(th), y+R*np.sin(th), label='Fit')
        plt.axis('equal')
        plt.legend(loc='best', scatterpoints=1)

    return sol


def intersect(circA, circB, xyA, draw=False):
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
        If true, plot circles and points for visual inspection in
        figure 'intersect'

    Returns
    -------
    pt : 1d ndarray
        [x, y] location of intersection point closest to xyA

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from pyyeti.nastran import intersect
        >>> circA = (0., 0., 10.)
        >>> circB = (0., 0., 15.)
        >>> xyA = (0., 5.)
        >>> intersect(circA, circB, [0., 1.], draw=1)
        array([  0.,  15.])
        >>> intersect(circA, circB, [0., -1.])
        array([  0., -15.])
        >>> intersect(circA, circB, [1., 0.])
        array([ 15.,   0.])
        >>> intersect(circA, circB, [-1., 0.])
        array([-15.,   0.])
    """
    (x1, y1, R1) = circA
    (x2, y2) = xyA
    (x3, y3, R3) = circB
    if abs(x2-x1) > 100*np.finfo(float).eps:
        m = (y2 - y1) / (x2 - x1)
        b = (x2*y1 - x1*y2) / (x2 - x1)
        # (mx+b-y3)**2 + (x-x3)**2 = R3**2
        a2 = (m*m + 1)
        a1 = 2*(m*(b-y3)-x3)/a2
        a0 = ((b-y3)**2 + x3**2 - R3**2)/a2
        x1 = -a1/2 + np.sqrt((a1/2)**2 - a0)
        x2 = -a1/2 - np.sqrt((a1/2)**2 - a0)
        y1 = m*x1 + b
        y2 = m*x2 + b
    else:
        # line is vertical (x1 & x2 are unchanged)
        #    (y-y3)**2 + (x1-x3)**2 = R3**2
        #    y-y3 = +- np.sqrt(R3**2 - (x1-x3)**2)
        y1 = y3 + np.sqrt(R3**2 - (x1-x3)**2)
        y2 = y3 - np.sqrt(R3**2 - (x1-x3)**2)
    err1 = abs(xyA[0]-x1) + abs(xyA[1]-y1)
    err2 = abs(xyA[0]-x2) + abs(xyA[1]-y2)
    if err1 < err2:
        pt = np.array([x1, y1])
    else:
        pt = np.array([x2, y2])
    if draw:
        plt.figure('intersect')
        plt.clf()
        th = np.arange(0, 361)*np.pi/180.
        (x1, y1, R1) = circA
        plt.plot(x1+R1*np.cos(th), y1+R1*np.sin(th), label='CircA')
        plt.plot(x3+R3*np.cos(th), y3+R3*np.sin(th), label='CircB')
        plt.scatter(*xyA, c='g', marker='o', s=60,
                    label='xyA')
        plt.scatter(*pt, c='b', marker='v', s=60,
                    label='intersection on CircB')
        plt.axis('equal')
        plt.legend(loc='best', scatterpoints=1)
    return pt


def _wtrspline_rings(f, r1grids, r2grids, node_id0, rspline_id0,
                     rbe2_id0, doplot, nper, DoL, independent):
    """
    Routine used by :func:`wtrspline`. See documentation for
    :func:`wtrspline`.
    """
    rgrids = np.atleast_2d(r1grids, r2grids)
    IDs = []
    xyz = []
    for ring, name in zip(rgrids, ('r1grids', 'r2grids')):
        if ring.shape[1] == 6:
            r = ring.shape[0]
            if (r // 6)*6 != r:
                raise ValueError('number of rows `{}` is not '
                                 'multiple of 6 for USET input'.
                                 format(name))
            IDs.append(ring[::6, 0].astype(np.int64))
            xyz.append(ring[::6, 3:])
        else:
            IDs.append(ring[:, 0].astype(np.int64))
            xyz.append(ring[:, 1:])

    n1 = len(IDs[0])
    n2 = len(IDs[1])
    ax = np.argmin(np.max(abs(np.diff(xyz[0], axis=0)), axis=0))
    ax2 = np.argmin(np.max(abs(np.diff(xyz[1], axis=0)), axis=0))
    if ax != ax2:
        raise ValueError('perpendicular directions of `r1grids` and '
                         '`r2grids` do not match: {} vs. {}'.
                         format(ax, ax2))

    lat = np.delete(np.arange(0, 3), ax)
    center = [findcenter(*xyz[0][:, lat].T),
              findcenter(*xyz[1][:, lat].T)]

    # the center point will often be at (0.0, 0.0) but be numerically
    # off ... check for this and adjust if that's the case:
    for circ in center:
        for i in (0, 1):
            if abs(circ[i]) < 1e-9 * circ[2]:
                circ[i] = 0.0

    if doplot:
        fig = plt.figure('rspline check 1', figsize=(8, 8))
        plt.clf()
        plt.scatter(xyz[0][:, lat[0]], xyz[0][:, lat[1]],
                    c='b', marker='+', s=60, label='R1 nodes')
        plt.scatter(xyz[1][:, lat[0]], xyz[1][:, lat[1]],
                    c='g', marker='x', s=60, label='R2 nodes')
        plt.axis('equal')
        plt.title('Old ring nodes and new ring 1 nodes -- should '
                  'see circles')
        plt.xlabel('XYZ'[lat[0]])
        plt.ylabel('XYZ'[lat[1]])
        fig.subplots_adjust(bottom=.3)

    newpts = np.zeros((n1, 3))
    newpts[:, ax] = xyz[1][0, ax]
    for j in range(n1):
        newpts[j, lat] = intersect(center[0], center[1],
                                   xyz[0][j, lat])

    if doplot:
        th = np.arange(0, np.pi*2+.005, .01)
        x = center[1][2]*np.cos(th) + center[1][0]
        y = center[1][2]*np.sin(th) + center[1][1]
        plt.plot(x, y, 'g-', label='R2 best-fit circle')
        plt.scatter(newpts[:, lat[0]], newpts[:, lat[1]], c='r',
                    marker='o', s=40,
                    label='New R1 nodes - should be on R2 circle')
        segments_x = np.empty(3*newpts.shape[0])
        segments_y = np.empty_like(segments_x)
        segments_x[::3] = newpts[:, lat[0]]
        segments_y[::3] = newpts[:, lat[1]]
        segments_x[1::3] = xyz[0][:, lat[0]]
        segments_y[1::3] = xyz[0][:, lat[1]]
        segments_x[2::3] = np.nan
        segments_y[2::3] = np.nan
        plt.plot(segments_x, segments_y, 'r-',
                 label='RBE2s - should be R1 radial')
        plt.legend(loc='upper center', bbox_to_anchor=(.5, -.1))

    # write new grids
    f.write('$\n$ Grids to RBE2 to Ring 1 grids. These grids line '
            'up with Ring 2 circle.\n')
    f.write('$ These will be used in an RSPLINE (which will be '
            'smooth)\n$\n')
    vec = np.arange(n1)
    newids = node_id0 + vec
    rbe2ids = rbe2_id0 + vec
    wtgrids(f, newids, xyz=newpts)
    f.write('$\n$ RBE2 old Ring 1 nodes to new nodes created above '
            '(new nodes are\n$ independent):\n$\n')
    writer.vecwrite(f, 'RBE2,{},{},123456,{}\n',
                    rbe2ids, newids, IDs[0])

    th_1 = np.arctan2(newpts[:, lat[1]], newpts[:, lat[0]])
    th_2 = np.arctan2(xyz[1][:, lat[1]], xyz[1][:, lat[0]])
    th = np.hstack((th_1, th_2))
    th[th < 0] += 2*np.pi

    ids_1 = np.column_stack((newids, np.zeros(n1, np.int64)))
    ids_2 = np.column_stack((IDs[1], np.zeros(n2, np.int64)))

    if independent == 'ring1':
        f.write('$\n$ RSPLINE Ring 2 nodes to new nodes created above, '
                'with the new nodes\n$ being independent.\n$\n')
        ids_1[:, 1] = 1
    else:
        f.write('$\n$ RSPLINE Ring 2 nodes to new nodes created above, '
                'with the Ring 2 nodes\n$ being independent.\n$\n')
        ids_2[:, 1] = 1

    ids = np.vstack((ids_1, ids_2))

    # sort by angular location:
    i = np.argsort(th)
    ids = ids[i]
    i = 0
    while ids[i, 1] == 0:
        i += 1
    ids = np.vstack((ids[i:], ids[:i], ids[i]))
    wtrspline(f, rspline_id0, ids, nper=nper, DoL=DoL)

    if doplot:
        # plot the rspline:
        plt.figure('rspline check 2')
        plt.clf()
        ids_with_coords = np.hstack((newids, IDs[1]))
        pv = locate.mat_intersect(ids_with_coords, ids[:, 0], 2)
        xy = np.vstack((newpts[:, lat], xyz[1][:, lat]))[pv[0]]
        plt.plot(xy[:, 0], xy[:, 1])
        plt.axis('equal')
        plt.title('Final RSPLINE - should be no sharp direction '
                  'changes')
        plt.xlabel('XYZ'[lat[0]])
        plt.ylabel('XYZ'[lat[1]])


def wtrspline_rings(f, r1grids, r2grids, node_id0, rspline_id0,
                    rbe2_id0=None, doplot=1, nper=1, DoL='0.1',
                    independent='ring1'):
    """
    Creates a smooth RSPLINE to connect two rings of grids.

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
    r1grids : 2d array_like
        4 or 6 column matrix of info on ring 1 grids:

          - If 4 columns: ``[id, x, y, z]``  <-- basic coordinates
          - If 6 columns: input is assumed to be USET table of ring 1
            grids (see :func:`pyyeti.op2.OP2.rdn2cop2` for
            description).

    r2grids : 2d array_like
        4 or 6 column matrix of info on ring 2 grids (same format as
        `r1grids`)
    node_id0 : integer
        1st id of new nodes created to 'move' ring 1 nodes
    rspline_id0 : integer
        1st id of RSPLINEs
    rbe2_id0 : integer or None; optional
        1st id of RBE2 elements that will connect old ring 1 nodes to
        new ones. If None, ``rbe2_id0 = node_id0``.
    doplot : bool; optional
        If True, 2 figures will be created for visual error checking:

          - figure 'rspline check 1' plots node locations and RBE2s
            for inspection; read title & legend for a couple hints of
            what to look for
          - figure 'rspline check 2' plots the final RSPLINE -- should
            be smooth

    nper : integer; optional
        Number of grids to write per RSPLINE before starting to look
        for next independent grid which will end the RSPLINE. Routine
        will actually write a minimum of 3 nodes::

                  independent - dependent - independent

        ``nper = 1`` ensures the smallest RSPLINEs are written.
    DoL : string or real scalar; optional
        Specifies ratio of diameter of elastic tybe to the sum of the
        lengths of all segments. Written with: ``'{:<8}'.format(DoL)``
    independent : str; optional
        Either 'ring1' or 'ring2'; specifies which ring will be
        independent on the RSPLINEs.

    Returns
    -------
    None

    Notes
    -----
    This routine writes GRID, RBE2 and RSPLINE lines to the output
    file.

    The approach is as follows (N = number of ring 1 grids):

      1. Create N new ring 1 grids at station and radius of ring 2
         grids, but at the same angular location as original N.
      2. RBE2 these new grids to the N original grids ... new grids
         are independent.
      3. Write RSPLINE cards using :func:`wtrspline`. The first
         RSPLINE starts at the independent grid (on ring 1 or ring 2
         according to `independent`) with the lowest angular
         location. The angular locations range from 0 to 360 degrees,
         counter-clockwise. The RSPLINEs proceed counter-clockwise and
         the last grid is also the first grid to complete the circle.

    Examples
    --------
    Define two rings of grids:

    1. Ring 1 will be at station 0.0 with 5 nodes on ring of radius
       50. IDs will be 1 to 5.

    2. Ring 2 will be at station 1.0 with 7 nodes on ring of radius
       45. IDs will be 101 to 107.

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from pyyeti import nastran
        >>> theta1 = np.arange(0, 359, 360/5)*np.pi/180
        >>> rad1 = 50.
        >>> sta1 = 0.
        >>> n1 = len(theta1)
        >>> ring1 = np.vstack((np.arange(1, n1+1),      # ID
        ...                    sta1*np.ones(n1),        # x
        ...                    rad1*np.cos(theta1),     # y
        ...                    rad1*np.sin(theta1))).T  # z
        >>> theta2 = np.arange(10, 359, 360/7)*np.pi/180
        >>> rad2 = 45.
        >>> sta2 = 1.
        >>> n2 = len(theta2)
        >>> ring2 = np.vstack((np.arange(1, n2+1)+100,  # ID
        ...                    sta2*np.ones(n2),        # x
        ...                    rad2*np.cos(theta2),     # y
        ...                    rad2*np.sin(theta2))).T  # z
        >>> nastran.wtrspline_rings(1, ring1, ring2, 1001, 2001)
        $
        $ Grids to RBE2 to Ring 1 grids. These grids line up with Ring 2 circle.
        $ These will be used in an RSPLINE (which will be smooth)
        $
        GRID*               1001               0      1.00000000     45.00000000
        *             0.00000000               0
        GRID*               1002               0      1.00000000     13.90576475
        *            42.79754323               0
        GRID*               1003               0      1.00000000    -36.40576475
        *            26.45033635               0
        GRID*               1004               0      1.00000000    -36.40576475
        *           -26.45033635               0
        GRID*               1005               0      1.00000000     13.90576475
        *           -42.79754323               0
        $
        $ RBE2 old Ring 1 nodes to new nodes created above (new nodes are
        $ independent):
        $
        RBE2,1001,1001,123456,1
        RBE2,1002,1002,123456,2
        RBE2,1003,1003,123456,3
        RBE2,1004,1004,123456,4
        RBE2,1005,1005,123456,5
        $
        $ RSPLINE Ring 2 nodes to new nodes created above, with the new nodes
        $ being independent.
        $
        RSPLINE     2001     0.1    1001     101  123456     102  123456    1002
        RSPLINE     2002     0.1    1002     103  123456    1003
        RSPLINE     2003     0.1    1003     104  123456     105  123456    1004
        RSPLINE     2004     0.1    1004     106  123456    1005
        RSPLINE     2005     0.1    1005     107  123456    1001
    """
    if independent not in ('ring1', 'ring2'):
        raise ValueError('invalid `independent` option')
    if rbe2_id0 is None:
        rbe2_id0 = node_id0
    return ytools.wtfile(f, _wtrspline_rings, r1grids, r2grids,
                         node_id0, rspline_id0, rbe2_id0, doplot,
                         nper, DoL, independent)


def wtvcomp(f, baa, kaa, bset, spoint1):
    """
    Write the VCOMP DMIG bulk data for P. Blelloch's BH DMAP

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
    baa : 2d array_like
        Craig-Bampton damping matrix
    kaa : 2d array_like
        Craig-Bampton stiffness matrix
    bset : 1d array_like
        Index partition vector for the bset
    spoint1 : integer
        Starting value for the SPOINTs (for modal DOF)

    Returns
    -------
    None

    Notes
    -----
    Typically called by :func:`wt_extseout`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> kaad = np.hstack((np.zeros(6), np.arange(100, 1000, 100.)))
    >>> zeta = .02
    >>> baad = 2*zeta*np.sqrt(kaad)
    >>> kaa = np.diag(kaad)
    >>> baa = np.diag(baad)
    >>> b = np.arange(6)
    >>> nastran.wtvcomp(1, baa, kaa, b, 1001)
    $ Critical damping ratios:
    DMIG    VCOMP          0       9       1       1                       1
    DMIG*   VCOMP                          1               0
    *                   1001               0            0.02
    *                   1002               0            0.02
    *                   1003               0            0.02
    *                   1004               0            0.02
    *                   1005               0            0.02
    *                   1006               0            0.02
    *                   1007               0            0.02
    *                   1008               0            0.02
    *                   1009               0            0.02
    """
    baa, kaa = np.atleast_2d(baa, kaa)
    qset = locate.flippv(bset, kaa.shape[0])
    qq = np.ix_(qset, qset)
    kd = np.diag(kaa[qq])
    bd = np.diag(baa[qq])
    zeta = bd / (2*np.sqrt(kd))

    def _wtvcomp(f, zeta, spoint1):
        f.write('$ Critical damping ratios:\n')
        f.write('DMIG    VCOMP          0       9       1'
                '       1                       1\n')
        f.write('DMIG*   VCOMP                          1'
                '               0\n')
        writer.vecwrite(f, '*       {:16d}{:16d}{:16.10g}\n',
                        spoint1+np.arange(len(zeta)), 0, zeta)

    return ytools.wtfile(f, _wtvcomp, zeta, spoint1)


def wtcoordcards(f, ci):
    """
    Write Nastran CORD2* cards to a file

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
    ci : dictionary or None
        Dictionary of coordinate card info as returned by
        :func:`pyyeti.n2p.coordcardinfo`. If None or if dict is empty,
        this routine quietly does nothing.

    Returns
    -------
    None

    Notes
    -----
    Typically called by :func:`wt_extseout`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> ci = {10: ['CORD2R', np.array([[10, 0, 0],
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

    def _wtcoords(f, ci):
        for k in ci:
            data = ci[k]  # [name, [[id, type, ref]; A; B; C]]
            coord = data[1]
            f.write('$ Coordinate {:d}:\n'.format(k))
            f.write('{:<8s}{:16d}{:16d}{:16.8e}{:16.8e}*\n'.
                    format(data[0]+'*', k, int(coord[0, 2]),
                           *coord[1, :2]))
            f.write(('{:<8s}'+'{:16.8e}'*4+'*\n').
                    format('*', coord[1, 2], *coord[2]))
            f.write(('{:<8s}'+'{:16.8e}'*3+'\n').
                    format('*', *coord[3]))

    return ytools.wtfile(f, _wtcoords, ci)


def wtextrn(f, ids, dof):
    """
    Writes a Nastran EXTRN card to a file.

    Parameters
    ----------
    f : string or file handle or 1 or None
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to stdout. Can also be the name
        of a directory or None; in these cases, a GUI is
        opened for file selection.
    ids : 1d array_like
        Vector of node ids
    dof : 1d array_like
        Vector of DOF

    Returns
    -------
    None

    Notes
    -----
    Typically called by :func:`wt_extseout`.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> nastran.wtextrn(1, [999, 10001, 10002, 10003, 10004, 10005],
    ...                    [123456, 0, 0, 0, 0, 0])
    EXTRN        999  123456   10001       0   10002       0   10003       0
               10004       0   10005       0
    """
    def _wtextrn(f, ids, dof):
        f.write('EXTRN   ')
        ints = np.zeros(len(ids)*2, dtype=int)
        ints[::2] = ids
        ints[1::2] = dof
        wtnasints(f, 2, ints)

    return ytools.wtfile(f, _wtextrn, ids, dof)


def wt_extseout(name, *, se, maa, baa, kaa, bset, uset, spoint1,
                sedn=0, bh=False):
    """
    Write .op4, .asm, .pch and possibly the damping DMIG file for an
    external SE.

    Note that all inputs except `f` must be named and can be input in
    any order.

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
    uset : ndarray
        A 6-column matrix as output by
        :func:`pyyeti.op2.OP2.rdn2cop2`.
    spoint1 : integer
        Starting value for the SPOINTs (for modal DOF)
    sedn : integer; optional
        Downstream superelement id
    bh : bool; optional
        If True, Benfield-Hruda damping is being used and a '.baa_dmig'
        file will be created with VCOMP DMIG. In this case, the BAA
        matrix written to the .op4 file is zeroed out to avoid double
        application of damping.

    Returns
    -------
    None
    """
    maa, baa, kaa, uset = np.atleast_2d(maa, baa, kaa, uset)
    bset = np.atleast_1d(bset)
    n = maa.shape[0]
    nq = n - len(bset)

    if bh:
        # write damping to DMIG VCOMP card for the BH method:
        wtvcomp(name+'.baa_dmig', baa, kaa, bset, spoint1)
        baa = np.zeros_like(baa)

    # prepare standard Nastran op4 file:
    k4xx = 0.
    pa = np.zeros((n, 1))
    gpxx = 0.
    gdxx = 0.
    va = np.ones((n, 1))
    mug1 = 0.
    mug1o = 0.
    mes1 = 0.
    mes1o = 0.
    mee1 = 0.
    mee1o = 0.
    mgpf = 0.
    mgpfo = 0.
    mef1 = 0.
    mef1o = 0.
    mqg1 = 0.
    mqg1o = 0.
    mqmg1 = 0.
    mqmg1o = 0.
    namelist = ['kaa', 'maa', 'baa', 'k4xx', 'pa', 'gpxx', 'gdxx',
                'va', 'mug1', 'mug1o', 'mes1', 'mes1o', 'mee1',
                'mee1o', 'mgpf', 'mgpfo', 'mef1', 'mef1o', 'mqg1',
                'mqg1o', 'mqmg1', 'mqmg1o']
    dct = locals()
    varlist = [dct[i] for i in namelist]
    op4.write(name+'.op4', namelist, varlist)

    # Get some data from the uset table:
    ci = n2p.coordcardinfo(uset)
    pv = np.nonzero(uset[:, 1] == 1)[0]
    grids = uset[pv, 0].astype(int)
    xyz = uset[pv, 3:]
    cd = uset[pv+1, 3].astype(int)

    # Write out ASM file
    unit = se
    spointn = spoint1 + nq-1
    with open(name+'.asm', 'w') as f:
        f.write(('$ {:s} ASSEMBLY FILE FOR RESIDUAL RUN...INCLUDE '
                 'IN BULK DATA\n').format(name.upper()))
        f.write('$\n')
        f.write(('SEBULK  {:8d}  EXTOP4          MANUAL'
                 '                {:8d}\n').format(se, unit))
        f.write('SECONCT {:8d}{:8d}              NO\n'.
                format(se, sedn))
        f.write('        ')
        gids = np.vstack((grids, grids)).T
        wtnasints(f, 2, gids.ravel())

        # Write coordinate system cards if needed:
        f.write('$\n')
        f.write('$ COORDINATE SYSTEM DATA\n$\n')
        wtcoordcards(f, ci)

        # Write Grid data:
        f.write('$\n')
        f.write('$ BOUNDARY GRID DATA\n')
        f.write('$\n')
        wtgrids(f, grids, 0, xyz, cd)
        if spointn >= spoint1:
            f.write('$\n')
            f.write('SECONCT {:8d}{:8d}              NO\n'.
                    format(se, sedn))
            f.write('        {:8d}    THRU{:8d}{:8d}    THRU{:8d}\n'.
                    format(spoint1, spointn, spoint1, spointn))
            f.write('$\n')
            f.write('SPOINT  {:8d}    THRU{:8d}\n'.
                    format(spoint1, spointn))

    # Write out PCH file
    with open(name+'.pch', 'w') as f:
        f.write(('$ {:s} PUNCH FILE FOR RESIDUAL RUN...INCLUDE '
                 'AT END\n').format(name.upper()))
        f.write('$\n')
        f.write('BEGIN SUPER{:8d}\n$\n'.format(se))
        ids = np.hstack((grids, spoint1+np.arange(nq)))
        dof = np.zeros_like(ids, dtype=int)
        dof[:len(grids)] = 123456
        wtextrn(f, ids, dof)
        f.write('$\n')
        f.write('$ COORDINATE SYSTEM DATA\n$\n')
        wtcoordcards(f, ci)
        f.write('$\n$ BOUNDARY GRID DATA\n$\n')
        wtgrids(f, grids, 0, xyz, cd)
        f.write('$\n')

        f.write('$ BSET\n$\n')
        f.write('ASET1   {:8d}'.format(123456))
        wtnasints(f, 3, grids)
        if spointn >= spoint1:
            f.write('$\n')
            f.write('$ QSET\n$\n')
            f.write('QSET1   {:8d}{:8d}    THRU{:8d}\n'.
                    format(0, spoint1, spointn))
            f.write('$\n')
            f.write('SPOINT  {:8d}    THRU{:8d}\n'.
                    format(spoint1, spointn))


def mknast(script=None, *, nascom='nast9p1', nasopt='batch=no',
           ext='out', stoponfatal='no', shell='/bin/sh', files=None,
           before='', after='', top='', bottom=''):
    """
    Create shell script to run chain of nastran (or other) runs.

    Note that all inputs except `script` must be named and can be input in
    any order.

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
        The extension on the f06 file; usually 'f06' or 'out'
    stoponfatal : string; optional
        'yes' or 'no'; 'yes' if you want script to exit on first fatal
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
        script = input('Name of shell script to create (doruns.sh): ')
        if not script:
            script = 'doruns.sh'

    # initialize shell script
    GREP = "grep -l '[*^][*^][*^].*FATAL'"
    with open(script, 'w') as f:
        f.write('#!{:s}\n'.format(shell))
        curdir = os.getcwd()
        f.write('cd {:s}\n\n'.format(curdir))
        if top:
            f.write('{:s}\n'.format(top))

        i = -1
        while 1:  # loop over file names
            i += 1
            if files is not None:
                if i >= len(files):
                    break
                else:
                    nasfile = files[i]
            else:
                p = 'File #{:2d} (blank to quit): '.format(i+1)
                nasfile = input(p)
                if not nasfile:
                    break

            if not os.path.exists(nasfile):
                print("Warning:  file '{:s}' not found.\n".
                      format(nasfile))

            f.write('\n# ******** File {:s} ********\n'.
                    format(nasfile))
            p = nasfile.rfind('/')
            if p > -1:
                filepath = nasfile[:p]
                filename = nasfile[p+1:]
                f.write('  cd {:s}\n'.format(filepath))
                docd = 1
            else:
                filename = nasfile
                docd = 0

            if before:
                f.write('{:s}\n'.format(before))
            f.write("  {:s} '{:s}' '{:s}'\n".
                    format(nascom, nasopt, filename))

            if stoponfatal == 'yes':
                p = filename.rfind('.')
                if p > -1:
                    f06file = filename[:p+1]+ext
                else:
                    f06file = filename+'.'+ext

                f.write("  if [ X != X`{:s} {:s}` ] ; then\n".
                        format(GREP, f06file))
                f.write("    exit\n")
                f.write("  fi\n")
            if after:
                f.write('{:s}\n'.format(after))

            if docd:
                f.write('  cd {:s}\n'.format(curdir))

        if bottom:
            f.write('{:s}\n'.format(bottom))
    os.system("chmod a+rx '{:s}'".format(script))


def rddtipch(f, name='TUG1'):
    """
    Read the 2nd record of specific DTIs from a .pch file.

    Parameters
    ----------
    f : string or file handle or None
        Either a name of a file or a file handle as returned by
        :func:`open`. If handle, file is rewound first. Can also be
        the name of a directory or None; in these cases, a GUI is
        opened for file selection.
    name : string
        Name of DTI table to read from the .pch file

    Returns
    -------
    id_dof : ndarray
        2-column matrix:  [id, dof]. Number of rows corresponds to
        matrix in .op4 file.

    Notes
    -----
    This routine is useful for .pch files written by the EXTSEOUT
    command. The 2nd record of TUG1 contains the DOF that correspond
    to the rows of the MUG1 matrix on the .op4 file. That matrix can
    be read by the op4 module.

    Example usage::

        # read mug1 and tug1 (created from EXTSEOUT):
        from pyyeti import op4
        from pyyeti import nastran
        mug1 = op4.read('data.op4', 'mug1')['mug1']
        tug1 = nastran.rddtipch('data.pch')

        # form DRM to recovery grid 100, dof 4:
        from pyyeti import locate
        row = locate.find_rows(tug1, [100, 4])
        drm = mug1[row, :]
    """
    string = 'DTI     {:<8s}2'.format(name)
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
        dof = np.arange(1, c+1, dtype=np.int64).reshape(1, -1)
        dofs = np.dot(np.ones((np.size(m, 0), 1), np.int64), dof)
        dofs = dofs.reshape((-1, 1))
        iddof = np.hstack((ids, dofs))
    else:
        nrows = m[-1, 2] + m[-1, 1] - 1
        iddof = np.zeros((nrows, 2), np.int64)
        j = 0
        for J in range(np.size(m, 0)):
            pv = np.arange(0, m[J, 1], dtype=np.int64)
            iddof[pv+j, 0] = m[J, 0]
            iddof[pv+j, 1] = pv + 1
            j += m[J, 1]
    return iddof
