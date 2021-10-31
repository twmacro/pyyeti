# -*- coding: utf-8 -*-
"""
Python tools for reading/writing Nastran .op4 files. Can read and
write all formats (as far as I know) with the restrictions that the
output files created by this class are always double precision, and
all matrices are read in as double precision. The binary files can be
in big or little endian format.

Notes on sparse matrices:

  1. By default, matrices read from .op4 files will be regular
     :class:`numpy.ndarray` matrices. However, :mod:`scipy.sparse`
     matrices can be created instead. See the `sparse` option in
     :func:`read` and :func:`load`.

  2. By default, matrices written to .op4 files will follow the Python
     type: :class:`numpy.ndarray` matrices will be written in dense
     format and :mod:`scipy.sparse` matrices will be written in
     "bigmat" sparse format. This can be overridden by specifying the
     `sparse` option in :func:`write`.

.. note::
    Some features of this module are demonstrated in the pyYeti
    :ref:`tutorial`: :doc:`/tutorials/op4`. There is also a link to
    the source Jupyter notebook at the top of the tutorial.
"""

import itertools as it
import struct
import sys
import warnings
import collections
import numpy as np
import scipy.sparse as sp
from pyyeti import guitools


def _ensure_dp(m):
    """
    Ensure double precision values
    """
    if np.iscomplexobj(m):
        if m.dtype != np.complex128:
            return m.astype(np.complex128)
    elif m.dtype != np.float64:
        return m.astype(np.float64)
    return m


# ensure double precision 2d arrays or tuple as returned by
# scipy.sparse.find:
def _ensure_2d_dp(m):
    """
    Ensures 2d double precision array or tuple as returned by
    scipy.sparse.find
    """
    if sp.issparse(m):
        # return m
        i, j, v = sp.find(m)
        return m, i, j, _ensure_dp(v)
    m = np.atleast_2d(m)
    if m.ndim > 2:
        raise ValueError("found array with greater than 2 dimensions.")
    return _ensure_dp(m)


def _check_write_names(names):
    """
    Ensure valid names
    """
    outnames = []
    for i, name in enumerate(names):
        if not name.isidentifier():
            oldname, name = name, f"m{i}"
            warnings.warn(
                f"Matrix for output4 write has name: {oldname!r}. "
                f"Changing to {name!r}.",
                RuntimeWarning,
            )
        elif len(name) > 8:
            oldname, name = name, name[:8]
            warnings.warn(
                f"Matrix for output4 write has name: {oldname!r}. "
                f"Truncating to {name!r}.",
                RuntimeWarning,
            )
        outnames.append(name)
    return outnames


class OP4:
    """
    Class for reading/writing Nastran output4 (.op4) files.

    See demo below and refer to the help on these functions for more
    information: :func:`write` (or :func:`save`), :func:`load` (or the
    lower level :func:`dctload`, :func:`listload`), and
    :func:`dir`. `save` is an alias for `write`.

    Examples
    --------
    Instantiate the class and create matrices for demo:

    >>> from pyyeti.nastran import op4
    >>> o4 = op4.OP4()
    >>> import numpy as np
    >>> r = np.random.randn(3, 5)
    >>> c = 1j*np.random.randn(3, 5) + r

    Write binary op4 file, with 'r' first:

    >>> o4.write('testbin.op4', ['r', 'c'], [r, c])

    Write ascii op4 file without caring about order:

    >>> o4.write('testascii.op4', dict(r=r, c=c), binary=False)

    To read an op4 file into a dictionary (indexed by the name in
    lower case):

    >>> dct = o4.load('testbin.op4', into='dct')

    Note: to preserve order, the dictionary returned by "load" or
    "read" is actually an OrderedDict from the standard Python
    "collections" module (:class:`collections.OrderedDict`).

    To read into a list:

    >>> names, mats, forms, mtypes = o4.load('testascii.op4',
    ...                                      into='list')

    Check some results:

    >>> print(np.all(r == dct['r'][0]))
    True

    >>> if names[0] == 'c':
    ...     print(np.all(c == mats[0]))
    ... else:
    ...     print(np.all(c == mats[1]))
    True

    To print a 'directory' of an op4 file:

    >>> d = o4.dir('testbin.op4')
    r       ,      3 x 5     , form=2, mtype=2
    c       ,      3 x 5     , form=2, mtype=4

    Clean up:

    >>> import os
    >>> os.remove('testbin.op4')
    >>> os.remove('testascii.op4')
    """

    def __init__(self):
        self._fileh = None
        string = "%.1E" % 1.2
        self._expdigits = len(string) - (string.find("E") + 2)
        self._rows4bigmat = 65536
        # Tunable value ... if number of values exceeds this, read
        # with numpy.fromfile instead of struct.unpack.
        self._rowsCutoff = 3000
        self.save = self.write

    def __del__(self):
        if self._fileh:
            self._fileh.close()
            self._fileh = None

    def _op4close(self):
        if self._fileh:
            self._fileh.close()
            self._fileh = None

    def _op4open_read(self, filename):
        """
        Open binary or ascii op4 file for reading.

        Sets these class variables:

        _fileh : file handle
            Value returned by open(). File is opened in 'r' mode if
            ascii, 'rb' mode if binary.
        _ascii : bool
            True if file is ascii.
        _dformat : bool
            True if an ascii file uses 'D' instead of 'E' (eg, 1.4D3
            instead of 1.4E3)
        _bit64 : True or False
            True if 'key' integers are 64-bit in binary op4 files.
        _endian : string
            Will be '' if no byte-swapping is required; otherwise,
            either '>' or '<' for big-endian and little-endian,
            respectively. Only used for binary files.
        _Str_i4 : struct.Struct object
            Precompiled for reading 4 byte integers
        _Str_i : struct.Struct object
            Precompiled for reading 4 or 8 byte integers
        _bytes_i : integer
            Either 4 or 8, to go with Str_i.
        _str_sr : string
            Either self._endian + '%df' or self._endian + '%dd',
            depending on self._bit64; for reading single precision
            reals.
        _bytes_sr : integer
            Number of bytes in single real.
        _str_dr : string
            self._endian + '%dd', for reading double precision reals.
        _wordsperdouble : integer
            Either 1 or 2; 2 if self._bit64 is False.
        """
        self._fileh = open(filename, "rb")
        bytes = self._fileh.read(16)
        self._endian = ""
        self._dformat = False
        self._matcount = 0

        # Assuming binary, check for a zero byte in the 'type' field;
        # will have one at front or back if binary:
        if bytes[12] == 0 or bytes[15] == 0:
            self._ascii = False
            if sys.byteorder == "little":
                if bytes[12] == 0:
                    self._endian = ">"
            else:
                if bytes[12] != 0:
                    self._endian = "<"
            self._Str_i4 = struct.Struct(self._endian + "i")
            reclen = self._Str_i4.unpack(bytes[:4])[0]
            if reclen == 48:
                self._bit64 = True
                self._Str_i = struct.Struct(self._endian + "q")
                self._bytes_i = 8
                self._Str_ii = struct.Struct(self._endian + "qq")
                self._bytes_ii = 16
                self._Str_iii = struct.Struct(self._endian + "3q")
                self._bytes_iii = 24
                self._Str_iiii = struct.Struct(self._endian + "4q")
                self._bytes_iiii = 32
                self._str_sr = self._endian + "%dd"
                self._str_sr_fromfile = np.dtype(self._endian + "f8")
                self._bytes_sr = 8
                self._wordsperdouble = 1
            else:
                self._bit64 = False
                self._Str_i = self._Str_i4
                self._bytes_i = 4
                self._Str_ii = struct.Struct(self._endian + "ii")
                self._bytes_ii = 8
                self._Str_iii = struct.Struct(self._endian + "3i")
                self._bytes_iii = 12
                self._Str_iiii = struct.Struct(self._endian + "4i")
                self._bytes_iiii = 16
                self._str_sr = self._endian + "%df"
                self._str_sr_fromfile = np.dtype(self._endian + "f4")
                self._bytes_sr = 4
                self._wordsperdouble = 2
            self._str_dr = self._endian + "%dd"
            self._str_dr_fromfile = np.dtype(self._endian + "f8")
            self._fileh.seek(0)
        else:
            self._ascii = True
            self._fileh.readline()
            self._fileh.readline()
            line = self._fileh.readline().decode()
            # sparse formats have integer header line:
            if line.find(".") == -1:
                line = self._fileh.readline().decode()
            if line.find("D") > -1:
                self._dformat = True
            self._fileh.close()
            self._fileh = open(filename, "r")

    def _skipop4_ascii(self, perline, rows, cols, mtype):
        """
        Skip an op4 matrix - ascii.

        Parameters
        ----------
        perline : integer
            Number of elements per line in the file.
        rows : integer
            Number of rows in matrix.
        cols : integer
            Number of columns in matrix.
        mtype : integer
            Nastran matrix type.

        Returns
        -------
        None

        On entry, file is positioned after the title line, but before
        the first column is printed. On exit, the file is positioned
        so the next readline will get the next title line.
        """
        # read until next matrix:
        bigmat = rows < 0 or rows >= self._rows4bigmat
        if mtype & 1:
            wper = 1
        else:
            wper = 2
        line = self._fileh.readline()
        c = int(line[:8]) - 1
        r = int(line[8:16])
        if r > 0:
            while c < cols:
                elems = int(line[16:24])
                nlines = (elems + perline - 1) // perline
                for _ in it.repeat(None, nlines):
                    self._fileh.readline()
                line = self._fileh.readline()
                c = int(line[:8]) - 1
        elif bigmat:
            while c < cols:
                elems = int(line[16:24])
                while elems > 0:
                    line = self._fileh.readline()
                    L = int(line[:8]) - 1  # L
                    elems -= L + 2
                    L //= wper
                    # read column as a long string
                    nlines = (L + perline - 1) // perline
                    for _ in it.repeat(None, nlines):
                        self._fileh.readline()
                line = self._fileh.readline()
                c = int(line[:8]) - 1
        else:
            while c < cols:
                elems = int(line[16:24])
                while elems > 0:
                    line = self._fileh.readline()
                    IS = int(line)  # [:8])
                    L = (IS >> 16) - 1  # L
                    elems -= L + 1
                    L //= wper
                    # read column as a long string
                    nlines = (L + perline - 1) // perline
                    for _ in it.repeat(None, nlines):
                        self._fileh.readline()
                line = self._fileh.readline()
                c = int(line[:8]) - 1
        self._fileh.readline()

    def _check_name(self, name):
        """
        Check name read from op4 file: strip all blanks/nulls; if name
        is not a valid Python identifier, it is set to
        f"m{self._matcount}". self._matcount starts at 0 and is
        incremented in this routine.

        Returns new name (usually the same as the input name).
        """
        name = name.strip(" \x00").replace(" ", "").replace("\x00", "").lower()
        if not name.isidentifier():
            oldname, name = name, f"m{self._matcount}"
            warnings.warn(
                f"Output4 file has matrix name: {oldname!r}. Changing to {name!r}.",
                RuntimeWarning,
            )
        self._matcount += 1
        return name

    def _get_ascii_block(self, L, perline, linelen):
        fh = self._fileh
        nlines = (L - 1) // perline + 1
        blocklist = [ln[:linelen] for ln in it.islice(fh, nlines)]
        s = "".join(blocklist)
        if self._dformat:
            s = s.replace("D", "E")
        return s

    @staticmethod
    def _init_dense_real(rows, cols):
        return np.zeros((rows, cols), dtype=float, order="F")

    @staticmethod
    def _init_dense_complex(rows, cols):
        return np.zeros((rows, cols), dtype=complex, order="F")

    @staticmethod
    def _put_ascii_values(X, r, c, s, L, numlen):
        a = 0
        for i in range(L):
            b = a + numlen
            X[r + i, c] = s[a:b]
            a = b

    @staticmethod
    def _put_ascii_values_c(X, r, c, s, L, numlen):
        a = 0
        for i in range(L // 2):
            b = a + numlen
            real = float(s[a:b])
            a = b
            b = a + numlen
            imag = float(s[a:b])
            a = b
            X[r + i, c] = real + 1j * imag

    @staticmethod
    def _dense_matrix(rows, cols, X):
        return X

    @staticmethod
    def _init_sparse(rows, cols):
        return [], [], []

    @staticmethod
    def _put_ascii_values_sparse(X, r, c, s, L, numlen):
        I, J, V = X
        a = 0
        for i in range(L):
            b = a + numlen
            I.append(r + i)
            J.append(c)
            V.append(float(s[a:b]))
            a = b

    @staticmethod
    def _put_ascii_values_sparse_c(X, r, c, s, L, numlen):
        I, J, V = X
        a = 0
        for i in range(L // 2):
            b = a + numlen
            real = float(s[a:b])
            a = b
            b = a + numlen
            imag = float(s[a:b])
            a = b
            I.append(r + i)
            J.append(c)
            V.append(real + 1j * imag)

    @staticmethod
    def _sparse_matrix(rows, cols, X):
        I, J, V = X
        return sp.coo_matrix((np.array(V), (I, J)), shape=(rows, cols))

    def _rd_dense_ascii(
        self, wper, r, c, rows, cols, line, numlen, perline, linelen, funcs
    ):
        init, put, retrn = funcs
        X = init(rows, cols)
        while c < cols:
            elems = int(line[16:24])
            r -= 1
            # read column as a long string
            s = self._get_ascii_block(elems, perline, linelen)
            put(X, r, c, s, elems, numlen)
            line = self._fileh.readline()
            c = int(line[:8]) - 1
            r = int(line[8:16])
        return retrn(rows, cols, X)

    def _rd_bigmat_ascii(
        self, wper, r, c, rows, cols, line, numlen, perline, linelen, funcs
    ):
        init, put, retrn = funcs
        X = init(rows, cols)
        while c < cols:
            elems = int(line[16:24])
            while elems > 0:
                line = self._fileh.readline()
                L = int(line[:8]) - 1  # L
                r = int(line[8:16]) - 1  # irow-1
                elems -= L + 2
                L //= wper
                s = self._get_ascii_block(L, perline, linelen)
                put(X, r, c, s, L, numlen)
            line = self._fileh.readline()
            c = int(line[:8]) - 1
            # r = int(line[8:16])
        return retrn(rows, cols, X)

    def _rd_nonbigmat_ascii(
        self, wper, r, c, rows, cols, line, numlen, perline, linelen, funcs
    ):
        init, put, retrn = funcs
        X = init(rows, cols)
        while c < cols:
            elems = int(line[16:24])
            while elems > 0:
                line = self._fileh.readline()
                IS = int(line)  # [:8])
                L = (IS >> 16) - 1  # L
                r = IS - ((L + 1) << 16) - 1  # irow-1
                elems -= L + 1
                L //= wper
                s = self._get_ascii_block(L, perline, linelen)
                put(X, r, c, s, L, numlen)
            line = self._fileh.readline()
            c = int(line[:8]) - 1
            # r = int(line[8:16])
        return retrn(rows, cols, X)

    def _get_funcs(self, a_or_b, rows, r, mtype, sparse, allzeros):
        if a_or_b == "ascii":
            rd_dense = self._rd_dense_ascii
            rd_bigmat = self._rd_bigmat_ascii
            rd_nonbigmat = self._rd_nonbigmat_ascii
            put_values = OP4._put_ascii_values
            put_values_c = OP4._put_ascii_values_c
            put_values_sparse = OP4._put_ascii_values_sparse
            put_values_sparse_c = OP4._put_ascii_values_sparse_c
        else:
            rd_dense = self._rd_dense_binary
            rd_bigmat = self._rd_bigmat_binary
            rd_nonbigmat = self._rd_nonbigmat_binary
            put_values = OP4._put_binary_values
            put_values_c = OP4._put_binary_values_c
            put_values_sparse = OP4._put_binary_values_sparse
            put_values_sparse_c = OP4._put_binary_values_sparse_c

        if r > 0:
            if allzeros and rows < 0:
                # assume bigmat sparse format
                # for example:
                #      19     -10       2       4C2      1P,3E22.15
                #      20       1       2
                # 1.000000000000000E+00
                rdfunc = rd_bigmat
                if sparse is None:
                    sparse = True
            else:
                # dense format
                rdfunc = rd_dense
                if sparse is None:
                    sparse = False
        else:
            # either bigmat or nonbigmat sparse format:
            if sparse is None:
                sparse = True
            if rows < 0 or rows >= self._rows4bigmat:
                # bigmat sparse format:
                #       6      -5       2       4C1      1P,3E22.15
                #       1       0      22
                #      21       1
                # 1.233140833282189E+00-1.364201536870681E+00 ...
                rdfunc = rd_bigmat
            else:
                # nonbigmat sparse format
                #       6       5       2       4C1      1P,3E22.15
                #       1       0      21
                #    1376257
                # 1.233140833282189E+00-1.364201536870681E+00 ...
                rdfunc = rd_nonbigmat

        if not sparse:
            if mtype < 3:
                funcs = (OP4._init_dense_real, put_values, OP4._dense_matrix)
            else:
                funcs = (OP4._init_dense_complex, put_values_c, OP4._dense_matrix)
        else:
            if mtype < 3:
                put = put_values_sparse
            else:
                put = put_values_sparse_c
            funcs = (OP4._init_sparse, put, OP4._sparse_matrix)
        return rdfunc, funcs

    @staticmethod
    def _get_sparsefunc(sparse):
        try:
            sparse, sparsefunc = sparse
        except TypeError:
            sparsefunc = None
        return sparse, sparsefunc

    def _loadop4_ascii(self, patternlist=None, listonly=False, sparse=False):
        """
        Reads next matching matrix or returns information on the next
        matrix in the ascii op4 file.

        Parameters
        ----------
        patternlist : list
            List of string patterns; each matrix name is matched
            against this list:  if it matches any of the patterns, it
            is read in.
        listonly : bool
            True if only reading name.
        sparse : bool or None or two-tuple_like; optional
            Specifies whether output matrices will be regular numpy
            arrays or sparse arrays. If not tuple_like:

            ========   ===============================================
            `sparse`   Action
            ========   ===============================================
             None      Auto setting: each matrix will be sparse if and
                       only if it was written in a sparse format
             True      Matrices will be returned in sparse format
             False     Matrices will be returned in regular (dense)
                       numpy arrays
            ========   ===============================================

            If `sparse` is two-tuple_like, the first element is either
            None, True, or False (see table above) and the second
            element is a callable, as in: ``X = callable(X)``. A
            common usage of the callable would be to convert from
            "COO" sparse form (see :class:`scipy.sparse.coo_matrix`)
            to a more desirable form. For example, to ensure *all*
            matrices are returned in CSC form (see
            :class:`scipy.sparse.csc_matrix`) use::

                sparse=(True, scipy.sparse.coo_matrix.tocsc)

            The callable is ignored for non-sparse matrices.

        Returns
        -------
        tuple: (name, matrix, form, mtype)
            name : string
                Lower-case name of matrix.
            matrix : 2d ndarray
                The matrix.
            form : integer
                Nastran form of matrix.
            mtype : integer
                Nastran matrix type.

        Notes
        -----
        - All outputs will be None if reached EOF.
        - The `matrix` output will be [rows, cols] of the matrix if
          the matrix is skipped.
        - The default form for sparse matrices is the "COO" sparse
          form (see :class:`scipy.sparse.coo_matrix`). To override,
          provide a callable in the `sparse` option (see above).
        """
        while 1:
            line = self._fileh.readline()
            line = line.rstrip()
            if line == "":
                return None, None, None, None
            cols = int(line[:8])
            rows = int(line[8:16])
            form = int(line[16:24])
            mtype = int(line[24:32])
            length = len(line)
            name = self._check_name(line[32:40])
            perline = 5
            numlen = 16
            if length > 44:
                # 1P,3E24.16  <-- starts at position 40
                # 3e24.16
                numformat = line[40:].strip().upper()
                if numformat.startswith("1P,"):
                    numformat = numformat[3:]
                p = numformat.replace("D", "E").find("E")
                if p > 0:
                    perline = int(numformat[:p])
                    numlen = int(numformat[p + 1 :].split(".")[0])
            if patternlist and name not in patternlist:
                skip = 1
            else:
                skip = 0
            if listonly or skip:
                self._skipop4_ascii(perline, rows, cols, mtype)
                if listonly:
                    return name, (abs(rows), cols), form, mtype
            else:
                break

        wper = 1 if mtype & 1 else 2
        line = self._fileh.readline()
        linelen = perline * numlen
        c = int(line[:8]) - 1
        r = int(line[8:16])
        sparse, sparsefunc = OP4._get_sparsefunc(sparse)

        rdfunc, funcs = self._get_funcs("ascii", rows, r, mtype, sparse, c >= cols)
        X = rdfunc(wper, r, c, abs(rows), cols, line, numlen, perline, linelen, funcs)

        if sparsefunc and sp.issparse(X):
            X = sparsefunc(X)
        self._fileh.readline()
        return name, X, form, mtype

    def _skipop4_binary(self, cols):
        """
        Skip a binary op4 matrix.

        Parameters
        ----------
        cols : integer
            Number of columns in matrix.
        """
        # Scan matrix by column
        icol = 1
        bi = self._bytes_i
        delta = 4 - bi
        while icol <= cols:
            # Read record length at start of record:
            reclen = self._Str_i4.unpack(self._fileh.read(4))[0]
            # Read column header
            icol = self._Str_i.unpack(self._fileh.read(bi))[0]
            self._fileh.seek(reclen + delta, 1)

    def _get_cutoff_etc(self):
        return (
            self._rowsCutoff,
            self._Str_iii.unpack,
            self._bytes_iii,
            self._Str_i4.unpack,
        )

    def _get_s2(self):
        return self._Str_ii.unpack, self._bytes_ii

    def _get_s1(self):
        return self._Str_i.unpack, self._bytes_i

    @staticmethod
    def _put_binary_values(X, r, c, Y):
        X[r : r + len(Y), c] = Y

    @staticmethod
    def _put_binary_values_c(X, r, c, Y):
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        Y = Y.astype("float", copy=False)
        Y.dtype = complex
        X[r : r + len(Y), c] = Y

    @staticmethod
    def _put_binary_values_sparse(X, r, c, Y):
        I, J, V = X
        for i, v in enumerate(Y):
            I.append(r + i)
            J.append(c)
            V.append(v)

    @staticmethod
    def _put_binary_values_sparse_c(X, r, c, Y):
        I, J, V = X
        for i, j in enumerate(range(0, len(Y), 2)):
            I.append(r + i)
            J.append(c)
            V.append(Y[j] + 1j * Y[j + 1])

    def _rd_dense_binary(
        self,
        fp,
        wper,
        r,
        c,
        rows,
        cols,
        nwords,
        reclen,
        bytesreal,
        numform,
        numform2,
        funcs,
    ):
        init, put, retrn = funcs
        X = init(rows, cols)
        cutoff, s3, b3, s4 = self._get_cutoff_etc()
        while c < cols:
            r -= 1
            nwords //= wper
            if nwords < cutoff:
                Y = struct.unpack(numform % nwords, fp.read(bytesreal * nwords))
            else:
                Y = np.fromfile(fp, numform2, nwords)
            put(X, r, c, Y)
            fp.read(4)
            reclen = s4(fp.read(4))[0]
            c, r, nwords = s3(fp.read(b3))
            c -= 1
        return retrn(rows, cols, X), reclen

    def _rd_bigmat_binary(
        self,
        fp,
        wper,
        r,
        c,
        rows,
        cols,
        nwords,
        reclen,
        bytesreal,
        numform,
        numform2,
        funcs,
    ):
        init, put, retrn = funcs
        X = init(rows, cols)
        cutoff, s3, b3, s4 = self._get_cutoff_etc()
        s2, b2 = self._get_s2()
        while c < cols:
            # bigmat sparse format
            # Read column data, one string of numbers at a time
            # (strings of zeros are skipped)
            while nwords > 0:
                L, r = s2(fp.read(b2))
                nwords -= L + 1
                L = (L - 1) // wper
                r -= 1
                if L < cutoff:
                    Y = struct.unpack(numform % L, fp.read(bytesreal * L))
                else:
                    Y = np.fromfile(fp, numform2, L)
                put(X, r, c, Y)
            fp.read(4)
            reclen = s4(fp.read(4))[0]
            c, r, nwords = s3(fp.read(b3))
            c -= 1
        return retrn(rows, cols, X), reclen

    def _rd_nonbigmat_binary(
        self,
        fp,
        wper,
        r,
        c,
        rows,
        cols,
        nwords,
        reclen,
        bytesreal,
        numform,
        numform2,
        funcs,
    ):
        init, put, retrn = funcs
        X = init(rows, cols)
        cutoff, s3, b3, s4 = self._get_cutoff_etc()
        s1, b1 = self._get_s1()
        while c < cols:
            # non-bigmat sparse format
            # Read column data, one string of numbers at a time
            # (strings of zeros are skipped)
            while nwords > 0:
                IS = s1(fp.read(b1))[0]
                L = (IS >> 16) - 1  # L
                r = IS - ((L + 1) << 16) - 1  # irow-1
                nwords -= L + 1  # words left
                L //= wper
                if L < cutoff:
                    Y = struct.unpack(numform % L, fp.read(bytesreal * L))
                else:
                    Y = np.fromfile(fp, numform2, L)
                put(X, r, c, Y)
            fp.read(4)
            reclen = s4(fp.read(4))[0]
            c, r, nwords = s3(fp.read(b3))
            c -= 1
        return retrn(rows, cols, X), reclen

    def _loadop4_binary(self, patternlist=None, listonly=False, sparse=False):
        """
        Reads next matching matrix or returns information on the next
        matrix in the binary op4 file.

        Parameters
        ----------
        patternlist : list
            List of string patterns; each matrix name is matched
            against this list:  if it matches any of the patterns, it
            is read in.
        listonly : bool
            True if only reading name.
        sparse : bool or None or two-tuple_like; optional
            Specifies whether output matrices will be regular numpy
            arrays or sparse arrays. If not two-tuple_like:

            ========   ===============================================
            `sparse`   Action
            ========   ===============================================
             None      Auto setting: each matrix will be sparse if and
                       only if it was written in a sparse format
             True      Matrices will be returned in sparse format
             False     Matrices will be returned in regular (dense)
                       numpy arrays
            ========   ===============================================

            If `sparse` is two-tuple_like, the first element is either
            None, True, or False (see table above) and the second
            element is a callable, as in: ``X = callable(X)``. A
            common usage of the callable would be to convert from
            "COO" sparse form (see :class:`scipy.sparse.coo_matrix`)
            to a more desirable form. For example, to ensure *all*
            matrices are returned in CSC form (see
            :class:`scipy.sparse.csc_matrix`) use::

                sparse=(True, scipy.sparse.coo_matrix.tocsc)

            The callable is ignored for non-sparse matrices.

        Returns
        -------
        tuple: (name, matrix, form, mtype)
            name : string
                Lower-case name of matrix.
            matrix : 2d ndarray
                The matrix.
            form : integer
                Nastran form of matrix.
            mtype : integer
                Nastran matrix type.

        Notes:
        - All outputs will be None if reached EOF.
        - The `matrix` output will be [rows, cols] of the matrix if
          the matrix is skipped.
        """
        fp = self._fileh
        while 1:
            if len(fp.read(4)) == 0:
                return None, None, None, None
            cols, rows, form, mtype = self._Str_iiii.unpack(fp.read(self._bytes_iiii))
            # Read ascii name of matrix:
            if self._bit64:
                name = fp.read(16).decode()
            else:
                name = fp.read(8).decode()
            name = self._check_name(name)
            fp.read(4)
            if patternlist and name not in patternlist:
                skip = 1
            else:
                skip = 0
            if listonly or skip:
                self._skipop4_binary(cols)
                if listonly:
                    return name, (abs(rows), cols), form, mtype
            else:
                break

        reclen = self._Str_i4.unpack(fp.read(4))[0]
        c, r, nwords = self._Str_iii.unpack(fp.read(self._bytes_iii))
        c -= 1
        sparse, sparsefunc = OP4._get_sparsefunc(sparse)

        rdfunc, funcs = self._get_funcs("binary", rows, r, mtype, sparse, c >= cols)
        if mtype & 1:
            numform = self._str_sr
            numform2 = self._str_sr_fromfile
            bytesreal = self._bytes_sr
            wper = 1
        else:
            numform = self._str_dr
            numform2 = self._str_dr_fromfile
            bytesreal = 8
            wper = self._wordsperdouble

        X, reclen = rdfunc(
            fp,
            wper,
            r,
            c,
            abs(rows),
            cols,
            nwords,
            reclen,
            bytesreal,
            numform,
            numform2,
            funcs,
        )

        if sparsefunc and sp.issparse(X):
            X = sparsefunc(X)

        # read final bytes of record and record marker
        nbytes = reclen - 3 * self._bytes_i + 4
        if len(fp.read(nbytes)) < nbytes:
            warnings.warn(
                f"Premature end-of-file after matrix {name!r}. Nastran "
                "will likely FATAL on this file.",
                RuntimeWarning,
            )
        return name, X, form, mtype

    @staticmethod
    def _sparse_col_stats(r):
        """
        Returns locations of non-zero values and length of each
        series.

        Parameters
        ----------
        r : ndarray
            1d ndarray containing the indices of the values in the
            data column

        Returns
        -------
        ind : ndarray
            m x 2 ndarray. m is number of non-zero sequences in v.
            First column contains the indices to the start of each
            sequence and the second column contains the length of
            the sequence.

        For example, if v is::

          v = [ 0.,  0.,  0.,  7.,  5.,  0.,  6.,  0.,  2.,  3.]

        which makes r be::

          r = [ 3, 4, 6, 8, 9 ]

        Then, ind will be::

          ind = [[3 2]
                 [6 1]
                 [8 2]]
        """
        dr = np.diff(r)
        starts = np.nonzero(dr != 1)[0] + 1
        nrows = len(starts) + 1
        ind = np.zeros((nrows, 2), int)
        ind[0, 0] = r[0]
        if nrows > 1:
            ind[1:, 0] = r[starts]
            ind[0, 1] = starts[0] - 1
            ind[1:-1, 1] = np.diff(starts) - 1
        ind[-1, 1] = len(dr) - len(starts) - sum(ind[:, 1])
        ind[:, 1] += 1
        return ind

    # @staticmethod
    # def _is_symmetric(m, tol=1e-12):
    #     """
    #     returns True if `m` is approx symmetric; sparse or not
    #     """
    #     return abs(m - m.transpose()).max() <= tol * abs(m).max()

    @staticmethod
    def _is_symmetric(m):
        """
        Returns True if matrix `m` is approx symmetric.

        Works for sparse and non-sparse matrices. Only called for
        square matrices, so that check is not done here.

        Note: if m is sparse, what gets input to the routine is a
        tuple: (m, r, c, v) where m is the original sparse matrix and
        r, c, v is the output from scipy.sparse.find(m).
        """
        if isinstance(m, tuple):
            r, c, v = m[1:]
            low = r > c  # values in lower triangle
            upp = c > r  # values in upper triangle

            if np.count_nonzero(low) != np.count_nonzero(upp):
                return False

            rl = r[low]
            cl = c[low]
            vl = v[low]
            ru = r[upp]
            cu = c[upp]
            vu = v[upp]

            sortl = np.lexsort((cl, rl))
            sortu = np.lexsort((ru, cu))
            return (
                np.all(cl[sortl] == ru[sortu])
                and np.all(rl[sortl] == cu[sortu])
                and np.allclose(vl[sortl], vu[sortu])
            )
        return np.allclose(m.transpose(), m)

    @staticmethod
    def _sparse_sort(matrix):
        # sparse matrix:
        m, r, c, v = matrix
        pv = np.lexsort((r, c))  # sort by column, then by row
        rs = r[pv]
        cs = c[pv]
        vs = v[pv]
        cols_with_data = sorted(set(cs))
        return rs, cs, vs, cols_with_data

    @staticmethod
    def _get_header_info(matrix, form=None):
        if isinstance(matrix, tuple):
            mat = matrix[0]
            vals = matrix[3]
        else:
            mat = vals = matrix
        rows, cols = mat.shape
        if form is None:
            if rows == cols:
                if OP4._is_symmetric(matrix):
                    form = 6
                else:
                    form = 1
            else:
                form = 2
        if np.iscomplexobj(vals):
            mtype = 4
            multiplier = 2
        else:
            mtype = 2
            multiplier = 1
        return rows, cols, form, mtype, multiplier

    def _write_ascii_header(self, f, name, matrix, digits, bigmat, form):
        """
        Utility routine that writes the header for ascii matrices.

        Parameters
        ----------
        f : file handle
            Output of open() using binary mode.
        name : string
            Name of matrix.
        matrix : matrix
            Matrix to write.
        digits : integer
            Number of significant digits after the decimal to include
            in the ascii output.
        bigmat : bool
            If true, matrix is to be written in 'bigmat' format.
        form : integer or None
            The matrix form. If None, the form will be determined
            automatically.

        Returns
        -------
        tuple: (cols, multiplier, perline, numlen, numform)
            cols : integer
                Number of columns in matrix.
            multiplier : integer
                2 for complex, 1 for real.
            perline : integer
                Number of values written per row.
            numlen : integer
                Number of characters per value.
            numform : string
                Format string for numbers, eg: '%16.9E'.
        """
        numlen = digits + 5 + self._expdigits  # -1.digitsE-009
        perline = 80 // numlen

        (rows, cols, form, mtype, multiplier) = OP4._get_header_info(matrix, form)

        if bigmat:
            # ~~ if rows < self._rows4bigmat:
            rows = -rows
        f.write(
            f"{cols:8}{rows:8}{form:8}{mtype:8}{name.upper():8s}"
            f"1P,{perline}E{numlen}.{digits}\n"
        )
        numform = f"%{numlen}.{digits}E"
        return cols, multiplier, perline, numlen, numform

    def _write_ascii(self, f, name, matrix, digits, form):
        """
        Write a matrix to a file in ascii, non-sparse format.

        Parameters
        ----------
        f : file handle
            Output of open() using text mode.
        name : string
            Name of matrix.
        matrix : matrix
            Matrix to write.
        digits : integer
            Number of significant digits after the decimal to include
            in the ascii output.
        form : integer or None
            The matrix form. If None, the form will be determined
            automatically.
        """
        (cols, multiplier, perline, numlen, numform) = self._write_ascii_header(
            f, name, matrix, digits, bigmat=False, form=form
        )

        def _write_col_data(f, v, c, s, elems, perline, numform):
            f.write(f"{c + 1:8}{s + 1:8}{elems:8}\n")
            neven = ((elems - 1) // perline) * perline
            for i in range(0, neven, perline):
                for j in range(perline):
                    f.write(numform % v[i + j])
                f.write("\n")
            for i in range(neven, elems):
                f.write(numform % v[i])
            f.write("\n")

        if isinstance(matrix, np.ndarray):
            for c in range(cols):
                v = matrix[:, c]
                if np.any(v):
                    pv = np.nonzero(v)[0]
                    s = pv[0]
                    e = pv[-1]
                    elems = (e - s + 1) * multiplier
                    v = np.asarray(v[s : e + 1]).ravel()
                    v.dtype = float
                    _write_col_data(f, v, c, s, elems, perline, numform)
        else:
            # sparse matrix:
            rs, cs, vs, cols_with_data = OP4._sparse_sort(matrix)
            dt = float if multiplier == 1 else complex
            for c in cols_with_data:
                pv = (cs == c).nonzero()[0]  # find data for column c
                s = rs[pv[0]]  # first row with value
                e = rs[pv[-1]]  # last row with value
                elems = e - s + 1
                vec = np.zeros(elems, dt)
                vec[rs[pv] - s] = vs[pv]
                elems *= multiplier
                vec.dtype = float
                _write_col_data(f, vec, c, s, elems, perline, numform)
        f.write(f"{cols + 1:8}{1:8}{1:8}\n")
        f.write(numform % 2 ** 0.5)
        f.write("\n")

    @staticmethod
    def _write_ascii_sparse(
        f,
        matrix,
        cols,
        _write_col_header,
        _write_data_string,
        multiplier,
        perline,
        numform,
    ):
        if isinstance(matrix, np.ndarray):
            for c in range(cols):
                v = matrix[:, c]
                if np.any(v):
                    v = np.asarray(v).ravel()
                    ind = OP4._sparse_col_stats(v.nonzero()[0])
                    _write_col_header(f, ind, c, multiplier)
                    for r0, r1 in ind:
                        string = v[r0 : r0 + r1]
                        string.dtype = float
                        _write_data_string(
                            f, string, r0, r1, multiplier, perline, numform
                        )
        else:
            # sparse matrix:
            rs, cs, vs, cols_with_data = OP4._sparse_sort(matrix)
            for c in cols_with_data:
                pv = (cs == c).nonzero()[0]  # find data for column c
                ind = OP4._sparse_col_stats(rs[pv])
                _write_col_header(f, ind, c, multiplier)
                coldata = vs[pv]
                j = 0
                for r0, r1 in ind:
                    string = coldata[j : j + r1]
                    j += r1
                    string.dtype = float
                    _write_data_string(f, string, r0, r1, multiplier, perline, numform)
        f.write(f"{cols + 1:8}{1:8}{1:8}\n")
        f.write(numform % 2 ** 0.5)
        f.write("\n")

    def _write_ascii_nonbigmat(self, f, name, matrix, digits, form):
        """
        Write a matrix to a file in ascii, non-bigmat sparse format.

        Parameters
        ----------
        f : file handle
            Output of open() using binary mode.
        name : string
            Name of matrix.
        matrix : matrix
            Matrix to write.
        digits : integer
            Number of significant digits after the decimal to include
            in the ascii output.
        form : integer or None
            The matrix form. If None, the form will be determined
            automatically.

        Note: if rows > 65535, bigmat is turned on and the
        :func:`write_ascii_bigmat` function is called ...
        that's a Nastran rule.
        """
        if isinstance(matrix, tuple):
            rows, cols = matrix[0].shape
        else:
            rows, cols = matrix.shape

        if rows >= self._rows4bigmat:
            self._write_ascii_bigmat(f, name, matrix, digits, form)
            return
        (cols, multiplier, perline, numlen, numform) = self._write_ascii_header(
            f, name, matrix, digits, bigmat=False, form=form
        )

        def _write_col_header(f, ind, c, multiplier):
            nwords = ind.shape[0] + 2 * sum(ind[:, 1]) * multiplier
            f.write(f"{c + 1:8}{0:8}{nwords:8}\n")

        def _write_data_string(f, string, r0, r1, multiplier, perline, numform):
            L = r1 * 2 * multiplier
            IS = (r0 + 1) + ((L + 1) << 16)
            f.write(f"{IS:12}\n")
            elems = L // 2
            neven = ((elems - 1) // perline) * perline
            for i in range(0, neven, perline):
                for j in range(perline):
                    f.write(numform % string[i + j])
                f.write("\n")
            for i in range(neven, elems):
                f.write(numform % string[i])
            f.write("\n")

        OP4._write_ascii_sparse(
            f,
            matrix,
            cols,
            _write_col_header,
            _write_data_string,
            multiplier,
            perline,
            numform,
        )

    def _write_ascii_bigmat(self, f, name, matrix, digits, form):
        """
        Write a matrix to a file in ascii, bigmat sparse format.

        Parameters
        ----------
        f : file handle
            Output of open() using binary mode.
        name : string
            Name of matrix.
        matrix : matrix
            Matrix to write.
        digits : integer
            Number of significant digits after the decimal to include
            in the ascii output.
        form : integer or None
            The matrix form. If None, the form will be determined
            automatically.
        """
        (cols, multiplier, perline, numlen, numform) = self._write_ascii_header(
            f, name, matrix, digits, bigmat=True, form=form
        )

        def _write_col_header(f, ind, c, multiplier):
            nwords = 2 * ind.shape[0] + 2 * sum(ind[:, 1]) * multiplier
            f.write(f"{c + 1:8}{0:8}{nwords:8}\n")

        def _write_data_string(f, string, r0, r1, multiplier, perline, numform):
            L = r1 * 2 * multiplier
            f.write(f"{L + 1:8}{r0 + 1:8}\n")
            elems = L // 2
            neven = ((elems - 1) // perline) * perline
            for i in range(0, neven, perline):
                for j in range(perline):
                    f.write(numform % string[i + j])
                f.write("\n")
            for i in range(neven, elems):
                f.write(numform % string[i])
            f.write("\n")

        OP4._write_ascii_sparse(
            f,
            matrix,
            cols,
            _write_col_header,
            _write_data_string,
            multiplier,
            perline,
            numform,
        )

    def _write_binary_header(self, f, name, matrix, endian, bigmat, form):
        """
        Utility routine that writes the header for binary matrices.

        Parameters
        ----------
        f : file handle
            Output of open() using binary mode.
        name : string
            Name of matrix.
        matrix : matrix
            Matrix to write.
        endian : string
            Endian setting for binary output:  '' for native, '>' for
            big-endian and '<' for little-endian.
        bigmat : bool
            If true, matrix is to be written in 'bigmat' format.
        form : integer or None
            The matrix form. If None, the form will be determined
            automatically.

        Returns
        -------
        tuple: (cols, multiplier)
            cols : integer
                Number of columns in matrix.
            multiplier : integer
                2 for complex, 1 for real.
        """
        (rows, cols, form, mtype, multiplier) = OP4._get_header_info(matrix, form)

        # write 1st record (24 bytes: 4 4-byte ints, 1 8-byte string)
        name = (f"{name.upper():<8}").encode()
        if bigmat:
            # ~~ if rows < self._rows4bigmat:
            rows = -rows
        f.write(struct.pack(endian + "5i8si", 24, cols, rows, form, mtype, name, 24))
        return cols, multiplier

    def _write_binary(self, f, name, matrix, endian, form):
        """
        Write a matrix to a file in double precision binary format.

        Parameters
        ----------
        f : file handle
            Output of open() using binary mode.
        name : string
            Name of matrix.
        matrix : matrix
            Matrix to write.
        endian : string
            Endian setting for binary output:  '' for native, '>' for
            big-endian and '<' for little-endian.
        form : integer or None
            The matrix form. If None, the form will be determined
            automatically.
        """
        cols, multiplier = self._write_binary_header(
            f, name, matrix, endian, bigmat=False, form=form
        )

        def _write_col_data(f, v, c, s, elems, endian, colHeader, colTrailer):
            reclen = 3 * 4 + elems * 8
            f.write(colHeader.pack(reclen, c + 1, s + 1, 2 * elems))
            f.write(struct.pack(endian + ("%dd" % elems), *v))
            f.write(colTrailer.pack(reclen))

        colHeader = struct.Struct(endian + "4i")
        colTrailer = struct.Struct(endian + "i")
        if isinstance(matrix, np.ndarray):
            for c in range(cols):
                v = matrix[:, c]
                if np.any(v):
                    pv = np.nonzero(v)[0]
                    s = pv[0]
                    e = pv[-1]
                    elems = (e - s + 1) * multiplier
                    v = np.asarray(v[s : e + 1]).ravel()
                    v.dtype = float
                    _write_col_data(f, v, c, s, elems, endian, colHeader, colTrailer)
        else:
            # sparse matrix:
            rs, cs, vs, cols_with_data = OP4._sparse_sort(matrix)
            dt = float if multiplier == 1 else complex
            for c in cols_with_data:
                pv = (cs == c).nonzero()[0]  # find data for column c
                s = rs[pv[0]]  # first row with value
                e = rs[pv[-1]]  # last row with value
                elems = e - s + 1
                vec = np.zeros(elems, dt)
                vec[rs[pv] - s] = vs[pv]
                elems *= multiplier
                vec.dtype = float
                _write_col_data(f, vec, c, s, elems, endian, colHeader, colTrailer)
        reclen = 3 * 4 + 8
        f.write(colHeader.pack(reclen, cols + 1, 1, 2))
        f.write(struct.pack(endian + "d", 2 ** 0.5))
        f.write(colTrailer.pack(reclen))

    @staticmethod
    def _write_binary_sparse(
        f,
        matrix,
        cols,
        _write_col_header,
        _write_data_string,
        multiplier,
        colHeader,
        colTrailer,
        LrStruct,
        endian,
    ):
        if isinstance(matrix, np.ndarray):
            for c in range(cols):
                v = matrix[:, c]
                if np.any(v):
                    v = np.asarray(v).ravel()
                    ind = OP4._sparse_col_stats(v.nonzero()[0])
                    reclen = _write_col_header(f, ind, c, multiplier, colHeader)
                    for r0, r1 in ind:
                        string = v[r0 : r0 + r1]
                        string.dtype = float
                        _write_data_string(
                            f, string, r0, r1, multiplier, LrStruct, endian
                        )
                    f.write(colTrailer.pack(reclen))
        else:
            # sparse matrix:
            rs, cs, vs, cols_with_data = OP4._sparse_sort(matrix)
            for c in cols_with_data:
                pv = (cs == c).nonzero()[0]  # find data for column c
                ind = OP4._sparse_col_stats(rs[pv])
                reclen = _write_col_header(f, ind, c, multiplier, colHeader)
                coldata = vs[pv]
                j = 0
                for r0, r1 in ind:
                    string = coldata[j : j + r1]
                    j += r1
                    string.dtype = float
                    _write_data_string(f, string, r0, r1, multiplier, LrStruct, endian)
                f.write(colTrailer.pack(reclen))
        reclen = 3 * 4 + 8
        f.write(colHeader.pack(reclen, cols + 1, 1, 2))
        f.write(struct.pack(endian + "d", 2 ** 0.5))
        f.write(colTrailer.pack(reclen))

    def _write_binary_nonbigmat(self, f, name, matrix, endian, form):
        """
        Write a matrix to a file in double precision binary,
        non-bigmat sparse format.

        Parameters
        ----------
        f : file handle
            Output of open() using binary mode.
        name : string
            Name of matrix.
        matrix : matrix
            Matrix to write.
        endian : string
            Endian setting for binary output:  '' for native, '>' for
            big-endian and '<' for little-endian.
        form : integer or None
            The matrix form. If None, the form will be determined
            automatically.

        Note: if rows > 65535, bigmat is turned on and the
        :func:`write_binary_bigmat` function is called ...
        that's a Nastran rule.
        """
        if isinstance(matrix, tuple):
            rows, cols = matrix[0].shape
        else:
            rows, cols = matrix.shape

        if rows >= self._rows4bigmat:
            self._write_binary_bigmat(f, name, matrix, endian, form)
            return

        cols, multiplier = self._write_binary_header(
            f, name, matrix, endian, bigmat=False, form=form
        )
        colHeader = struct.Struct(endian + "4i")
        colTrailer = struct.Struct(endian + "i")

        def _write_col_header(f, ind, c, multiplier, colHeader):
            nwords = ind.shape[0] + 2 * sum(ind[:, 1]) * multiplier
            reclen = (3 + nwords) * 4
            f.write(colHeader.pack(reclen, c + 1, 0, nwords))
            return reclen

        def _write_data_string(f, string, r0, r1, multiplier, colTrailer, endian):
            L = r1 * 2 * multiplier
            IS = (r0 + 1) + ((L + 1) << 16)
            f.write(colTrailer.pack(IS))
            f.write(struct.pack(endian + ("%dd" % len(string)), *string))

        OP4._write_binary_sparse(
            f,
            matrix,
            cols,
            _write_col_header,
            _write_data_string,
            multiplier,
            colHeader,
            colTrailer,
            colTrailer,
            endian,
        )

    def _write_binary_bigmat(self, f, name, matrix, endian, form):
        """
        Write a matrix to a file in double precision binary, bigmat
        sparse format.

        Parameters
        ----------
        f : file handle
            Output of open() using binary mode.
        name : string
            Name of matrix.
        matrix : matrix
            Matrix to write.
        endian : string
            Endian setting for binary output:  '' for native, '>' for
            big-endian and '<' for little-endian.
        form : integer or None
            The matrix form. If None, the form will be determined
            automatically.
        """
        cols, multiplier = self._write_binary_header(
            f, name, matrix, endian, bigmat=True, form=form
        )
        colHeader = struct.Struct(endian + "4i")
        colTrailer = struct.Struct(endian + "i")
        LrStruct = struct.Struct(endian + "ii")

        def _write_col_header(f, ind, c, multiplier, colHeader):
            nwords = 2 * ind.shape[0] + 2 * sum(ind[:, 1]) * multiplier
            reclen = (3 + nwords) * 4
            f.write(colHeader.pack(reclen, c + 1, 0, nwords))
            return reclen

        def _write_data_string(f, string, r0, r1, multiplier, LrStruct, endian):
            L = r1 * 2 * multiplier
            f.write(LrStruct.pack(L + 1, r0 + 1))
            f.write(struct.pack(endian + ("%dd" % len(string)), *string))

        OP4._write_binary_sparse(
            f,
            matrix,
            cols,
            _write_col_header,
            _write_data_string,
            multiplier,
            colHeader,
            colTrailer,
            LrStruct,
            endian,
        )

    def dctload(self, filename, namelist=None, justmatrix=False, sparse=False):
        """
        Read all matching matrices from op4 file into dictionary.

        Parameters
        ----------
        filename : string
            Name of op4 file to read.
        namelist : list, string, or None; optional
            List of variable names to read in, or string with name of
            the single variable to read in, or None. If None, all
            matrices are read in.
        justmatrix : bool; optional
            If True, only the matrix is stored in the dictionary. If
            False, a tuple of ``(matrix, form, mtype)`` is stored.
        sparse : bool or None or two-tuple_like; optional
            Specifies whether output matrices will be regular numpy
            arrays or sparse arrays. If not two-tuple_like:

            ========   ===============================================
            `sparse`   Action
            ========   ===============================================
             None      Auto setting: each matrix will be sparse if and
                       only if it was written in a sparse format
             True      Matrices will be returned in sparse format
             False     Matrices will be returned in regular (dense)
                       numpy arrays
            ========   ===============================================

            If `sparse` is two-tuple_like, the first element is either
            None, True, or False (see table above) and the second
            element is a callable, as in: ``X = callable(X)``. A
            common usage of the callable would be to convert from
            "COO" sparse form (see :class:`scipy.sparse.coo_matrix`)
            to a more desirable form. For example, to ensure *all*
            matrices are returned in CSC form (see
            :class:`scipy.sparse.csc_matrix`) use::

                sparse=(True, scipy.sparse.coo_matrix.tocsc)

            The callable is ignored for non-sparse matrices.

        Returns
        -------
        dct : :class:`collections.OrderedDict`
            Keys are the lower-case matrix names and the values are
            either just the matrix or a tuple of:
            ``(matrix, form, mtype)`` depending on `justmatrix`.

        Notes
        -----
        The default form for sparse matrices is the "COO" sparse form
        (see :class:`scipy.sparse.coo_matrix`). To override, provide a
        callable in the `sparse` option (see above).

        See also
        --------
        :func:`listload`, :func:`write` (or :func:`save`),
        :func:`dir`.
        """
        if isinstance(namelist, str):
            namelist = [namelist]
        self._op4open_read(filename)
        dct = collections.OrderedDict()
        try:
            if self._ascii:
                loadfunc = self._loadop4_ascii
            else:
                loadfunc = self._loadop4_binary
            while 1:
                name, X, form, mtype = loadfunc(patternlist=namelist, sparse=sparse)
                if not name:
                    break
                if justmatrix:
                    dct[name] = X
                else:
                    dct[name] = X, form, mtype
        finally:
            self._op4close()
        return dct

    def listload(self, filename, namelist=None, sparse=False):
        """
        Read all matching matrices from op4 file into a list; useful
        if op4 file has duplicate names.

        Parameters
        ----------
        filename : string
            Name of op4 file to read.
        namelist : list, string, or None; optional
            List of variable names to read in, or string with name of
            the single variable to read in, or None. If None, all
            matrices are read in.
        sparse : bool or None or two-tuple_like; optional
            Specifies whether output matrices will be regular numpy
            arrays or sparse arrays. If not two-tuple_like:

            ========   ===============================================
            `sparse`   Action
            ========   ===============================================
             None      Auto setting: each matrix will be sparse if and
                       only if it was written in a sparse format
             True      Matrices will be returned in sparse format
             False     Matrices will be returned in regular (dense)
                       numpy arrays
            ========   ===============================================

            If `sparse` is two-tuple_like, the first element is either
            None, True, or False (see table above) and the second
            element is a callable, as in: ``X = callable(X)``. A
            common usage of the callable would be to convert from
            "COO" sparse form (see :class:`scipy.sparse.coo_matrix`)
            to a more desirable form. For example, to ensure *all*
            matrices are returned in CSC form (see
            :class:`scipy.sparse.csc_matrix`) use::

                sparse=(True, scipy.sparse.coo_matrix.tocsc)

            The callable is ignored for non-sparse matrices.

        Returns
        -------
        names : list
            Lower-case list of matrix names in order as read.
        matrices : list
            List of matrices in order as read.
        forms : list
            List of integers specifying the Nastran form of each
            matrix.
        mtypes : list
            List of integers specifying the Nastran type of each
            matrix.

        Notes
        -----
        The default form for sparse matrices is the "COO" sparse form
        (see :class:`scipy.sparse.coo_matrix`). To override, provide a
        callable in the `sparse` option (see above).

        See also
        --------
        :func:`dctload`, :func:`write` (or :func:`save`), :func:`dir`.
        """
        if isinstance(namelist, str):
            namelist = [namelist]
        self._op4open_read(filename)
        names = []
        matrices = []
        forms = []
        mtypes = []
        try:
            if self._ascii:
                loadfunc = self._loadop4_ascii
            else:
                loadfunc = self._loadop4_binary
            while 1:
                name, X, form, mtype = loadfunc(patternlist=namelist, sparse=sparse)
                if not name:
                    break
                names.append(name)
                matrices.append(X)
                forms.append(form)
                mtypes.append(mtype)
        finally:
            self._op4close()
        return names, matrices, forms, mtypes

    def load(self, filename, namelist=None, into="dct", justmatrix=False, sparse=False):
        """
        Read all matching matrices from op4 file into dictionary or
        list; interface to :func:`dctload` and :func:`listload`.

        Parameters
        ----------
        filename : string
            Name of op4 file to read.
        namelist : list, string, or None; optional
            List of variable names to read in, or string with name of
            the single variable to read in, or None. If None, all
            matrices are read in.
        into : string; optional
            Either 'dct' or 'list'. Use 'list' if multiple matrices
            share the same name. See below.
        justmatrix : bool; optional
            If True, only the matrix is stored in the dictionary. If
            False, a tuple of ``(matrix, form, mtype)`` is stored.
            This option is ignored if ``into == 'list'``.
        sparse : bool or None or two-tuple_like; optional
            Specifies whether output matrices will be regular numpy
            arrays or sparse arrays. If not two-tuple_like:

            ========   ===============================================
            `sparse`   Action
            ========   ===============================================
             None      Auto setting: each matrix will be sparse if and
                       only if it was written in a sparse format
             True      Matrices will be returned in sparse format
             False     Matrices will be returned in regular (dense)
                       numpy arrays
            ========   ===============================================

            If `sparse` is two-tuple_like, the first element is either
            None, True, or False (see table above) and the second
            element is a callable, as in: ``X = callable(X)``. A
            common usage of the callable would be to convert from
            "COO" sparse form (see :class:`scipy.sparse.coo_matrix`)
            to a more desirable form. For example, to ensure *all*
            matrices are returned in CSC form (see
            :class:`scipy.sparse.csc_matrix`) use::

                sparse=(True, scipy.sparse.coo_matrix.tocsc)

            The callable is ignored for non-sparse matrices.

        Returns
        -------
        dct : :class:`collections.OrderedDict`, if ``into == 'dct'``
            Keys are the lower-case matrix names and the values are
            either just the matrix or a tuple of:
            ``(matrix, form, mtype)`` depending on `justmatrix`.

        tup : tuple, if ``into == 'list'``
            Tuple of 4 lists: ``(names, matrices, forms, mtypes)``

        Notes
        -----
        The default form for sparse matrices is the "COO" sparse form
        (see :class:`scipy.sparse.coo_matrix`). To override, provide a
        callable in the `sparse` option (see above).

        See also
        --------
        :func:`write` (or :func:`save`), :func:`dir`.
        """
        if into == "dct":
            return self.dctload(filename, namelist, justmatrix, sparse)
        elif into == "list":
            return self.listload(filename, namelist, sparse)
        raise ValueError('invalid "into" option')

    def dir(self, filename, verbose=True):
        """
        Directory of all matrices in op4 file.

        Parameters
        ----------
        filename : string
            Name of op4 file to read.
        verbose : bool; optional
            If true, directory will be printed to screen.

        Returns
        -------
        names : list
            Lower-case list of matrix names in order as read.
        sizes : list
            List of sizes [(r1, c1), (r2, c2), ...], for each
            matrix.
        forms : list
            List of integers specifying the Nastran form of each
            matrix.
        mtypes : list
            List of integers specifying the Nastran type of each
            matrix.

        See also
        --------
        :func:`dctload`, :func:`listload`, :func:`write` (or
        :func:`save`).
        """
        self._op4open_read(filename)
        names = []
        sizes = []
        forms = []
        mtypes = []
        try:
            if self._ascii:
                loadfunc = self._loadop4_ascii
            else:
                loadfunc = self._loadop4_binary
            while 1:
                name, X, form, mtype = loadfunc(listonly=True)
                if not name:
                    break
                names.append(name)
                sizes.append(X)
                forms.append(form)
                mtypes.append(mtype)
            if verbose:
                for n, s, f, m in zip(names, sizes, forms, mtypes):
                    print(f"{n:8}, {s[0]:6} x {s[1]:<6}, form={f}, mtype={m}")
        finally:
            self._op4close()
        return names, sizes, forms, mtypes

    def write(
        self,
        filename,
        names,
        matrices=None,
        binary=True,
        digits=16,
        endian="",
        sparse="auto",
        forms=None,
    ):
        """
        Write op4 file.

        Parameters
        ----------
        filename : string
            Name of file.
        names : dictionary/OrderedDict or list or string
            Dictionary indexed by the matrix names with the values
            being either the matrices or a tuple/list where the first
            two items are ``(matrix, form)``. Alternatively, `names`
            can also be a list of matrix names (strings) or a single
            name (string) if just one matrix is to be saved. If using
            the list inputs, `matrices` is required to specify the
            matrices and, if `forms` needs to specified, that input
            would also be required.
        matrices : array or list; optional
            2d ndarray or list of 2d ndarrays. Ignored if `names` is
            a dictionary. Same length as `names`.
        binary : bool; optional
            If true, a double precision binary file is written;
            otherwise an ascii file is created.
        digits : integer; optional
            Number of significant digits after the decimal to include
            in the ascii output. Ignored for binary files.
        endian : string; optional
            Endian setting for binary output:  '' for native, '>' for
            big-endian and '<' for little-endian.
        sparse : string; optional
            Specifies the output format:

            ===========   ===========================================
             `sparse`     Action
            ===========   ===========================================
            'auto'        Each 2d ndarray will be written in "dense"
                          format and each sparse matrix will be
                          written in "bigmat" sparse format
            'bigmat'      Each matrix (whether sparse or not) is
                          written in "bigmat" sparse format
            'dense'       Each matrix will be written in "dense"
                          format
            'nonbigmat'   Each matrix is written in "nonbigmat"
                          format. Note that if the number of rows is
                          > 65535, then the "bigmat" format is used.
            ===========   ===========================================

        forms : integer or list or None; optional
            The matrix form(s). If None, the forms will be determined
            automatically to be 1, 2, or 6. Ignored if `names` is a
            dictionary (which would contain this information). If not
            None, `forms` must be the same length as `names`, but you
            can use None as the form for one or more matrices (see
            example in :func:`pyyeti.nastran.op4.write`). From Nastran
            documentation:

            ======   ==============================
             form    Matrix format
            ======   ==============================
               1     Square
               2     Rectangular
               3     Diagonal
               4     Lower triangular factor
               5     Upper triangular factor
               6     Symmetric
               8     Identity
               9     Pseudo identity
              10     Cholesky factor
              11     Trapezoidal factor
              13     Sparse lower triangular factor
              15     Sparse upper triangular factor
            ======   ==============================

            .. warning::
                The validity of the values in `forms` is not checked
                in any way.

        Returns
        -------
        None.

        Notes
        -----
        To write multiple matrices that have the same name, `names`
        must be a list, not a dictionary. If a list, the order is
        maintained. If a dictionary, the matrices are written in the
        order they are retrieved from the dictionary; use a
        :class:`collections.OrderedDict` to specify a certain order.

        See the examples in :func:`pyyeti.nastran.op4.write`.

        See also
        --------
        :func:`pyyeti.nastran.op4.write`, :func:`dctload`,
        :func:`listload`, :func:`dir`.
        """

        if isinstance(names, collections.Mapping):
            _names = []
            matrices = []
            forms = []
            for nm, val in names.items():
                if isinstance(val, (list, tuple)):
                    _names.append(nm)
                    matrices.append(val[0])
                    forms.append(val[1])
                else:
                    _names.append(nm)
                    matrices.append(val)
                    forms.append(None)
            names = _names
        else:
            if not isinstance(names, (list, tuple)):
                names = [names]
            if not isinstance(matrices, (list, tuple)):
                matrices = [matrices]
            if forms is not None and not isinstance(forms, (list, tuple)):
                forms = [forms]

        if forms is None:
            forms = [None] * len(names)

        matrices = [_ensure_2d_dp(matrix) for matrix in matrices]
        names = _check_write_names(names)

        if binary:
            if sparse == "dense":
                wrtfunc = self._write_binary
            elif sparse == "bigmat":
                wrtfunc = self._write_binary_bigmat
            elif sparse == "nonbigmat":
                wrtfunc = self._write_binary_nonbigmat
            elif sparse != "auto":
                raise ValueError("invalid sparse option")
            with open(filename, "wb") as f:
                for name, matrix, form in zip(names, matrices, forms):
                    if sparse == "auto":
                        if isinstance(matrix, tuple):
                            wrtfunc = self._write_binary_bigmat
                        else:
                            wrtfunc = self._write_binary
                    wrtfunc(f, name, matrix, endian, form)
        else:
            if sparse == "dense":
                wrtfunc = self._write_ascii
            elif sparse == "bigmat":
                wrtfunc = self._write_ascii_bigmat
            elif sparse == "nonbigmat":
                wrtfunc = self._write_ascii_nonbigmat
            elif sparse != "auto":
                raise ValueError("invalid sparse option")
            with open(filename, "w") as f:
                for name, matrix, form in zip(names, matrices, forms):
                    if sparse == "auto":
                        if isinstance(matrix, tuple):
                            wrtfunc = self._write_ascii_bigmat
                        else:
                            wrtfunc = self._write_ascii
                    wrtfunc(f, name, matrix, digits, form)


def load(filename=None, namelist=None, into="dct", justmatrix=False, sparse=False):
    """
    Read all matching matrices from op4 file into dictionary or list;
    non-member version of :func:`OP4.load`.

    This is a the same as :func:`read` except `justmatrix` default is
    False.

    Parameters
    ----------
    filename : string or None; optional
        Name of op4 file to read. Can also be the name of a directory
        or None; in these cases, a GUI is opened for file selection.
    namelist : list, string, or None; optional
        List of variable names to read in, or string with name of the
        single variable to read in, or None. If None, all matrices
        are read in.
    into : string; optional
        Either 'dct' or 'list'. Use 'list' if multiple matrices share
        the same name. See below.
    justmatrix : bool; optional
        If True, only the matrix is stored in the dictionary. If
        False, a tuple of ``(matrix, form, mtype)`` is stored.
        This option is ignored if ``into == 'list'``.
    sparse : bool or None or two-tuple_like; optional
        Specifies whether output matrices will be regular numpy arrays
        or sparse arrays. If not two-tuple_like:

        ========   ===============================================
        `sparse`   Action
        ========   ===============================================
         None      Auto setting: each matrix will be sparse if and
                   only if it was written in a sparse format
         True      Matrices will be returned in sparse format
         False     Matrices will be returned in regular (dense)
                   numpy arrays
        ========   ===============================================

        If `sparse` is two-tuple_like, the first element is either
        None, True, or False (see table above) and the second element
        is a callable, as in: ``X = callable(X)``. A common usage of
        the callable would be to convert from "COO" sparse form (see
        :class:`scipy.sparse.coo_matrix`) to a more desirable
        form. For example, to ensure *all* matrices are returned in
        CSC form (see :class:`scipy.sparse.csc_matrix`) use::

            sparse=(True, scipy.sparse.coo_matrix.tocsc)

        The callable is ignored for non-sparse matrices.

    Returns
    -------
    dct : :class:`collections.OrderedDict`, if ``into == 'dct'``
        Keys are the lower-case matrix names and the values are
        either just the matrix or a tuple of:
        ``(matrix, form, mtype)`` depending on `justmatrix`.

    tup : tuple, if ``into == 'list'``
        Tuple of 4 lists: ``(names, matrices, forms, mtypes)``

    Notes
    -----
    The default form for sparse matrices is the "COO" sparse form (see
    :class:`scipy.sparse.coo_matrix`). To override, provide a callable
    in the `sparse` option (see above).

    Examples
    --------
    This examples translates a sparse format binary op4 file to a
    simpler ascii format while preserving the matrix forms.

    First, create a file in "bigmat" sparse format and set the form on
    the "m" matrix to be symmetric (form=6):

    >>> import numpy as np
    >>> from pyyeti.nastran import op4
    >>> m = np.array([[1, 2], [2.1, 3]])
    >>> k = np.array([3, 5])
    >>> b = np.array([4, 6])
    >>> names = ['m', 'k', 'b']
    >>> values = [eval(n) for n in names]
    >>> op4.write('mkb.op4', names, values, forms=[6, 2, 2],
    ...           sparse='bigmat')

    Now, translate it to simple ascii, preserving the forms:

    >>> dct = op4.load('mkb.op4')
    >>> op4.write('mkb_ascii.op4', dct, binary=False)

    Check that the order and forms are the same:

    >>> _ = op4.dir('mkb.op4')
    m       ,      2 x 2     , form=6, mtype=2
    k       ,      1 x 2     , form=2, mtype=2
    b       ,      1 x 2     , form=2, mtype=2

    >>> _ = op4.dir('mkb_ascii.op4')
    m       ,      2 x 2     , form=6, mtype=2
    k       ,      1 x 2     , form=2, mtype=2
    b       ,      1 x 2     , form=2, mtype=2

    Clean up:

    >>> import os
    >>> os.remove('mkb.op4')
    >>> os.remove('mkb_ascii.op4')

    See also
    --------
    :func:`read`, :func:`write` (or :func:`save`), :func:`dir`.
    """
    filename = guitools.get_file_name(filename, read=True)
    if into == "dct":
        return OP4().dctload(filename, namelist, justmatrix, sparse)
    elif into == "list":
        return OP4().listload(filename, namelist, sparse)
    raise ValueError('invalid "into" option')


def read(filename=None, namelist=None, into="dct", justmatrix=True, sparse=False):
    """
    Read all matching matrices from op4 file into dictionary or list;
    non-member version of :func:`OP4.load`.

    This is a the same as :func:`load` except `justmatrix` default is
    True.

    Parameters
    ----------
    filename : string or None; optional
        Name of op4 file to read. Can also be the name of a directory
        or None; in these cases, a GUI is opened for file selection.
    namelist : list, string, or None; optional
        List of variable names to read in, or string with name of the
        single variable to read in, or None. If None, all matrices
        are read in.
    into : string; optional
        Either 'dct' or 'list'. Use 'list' if multiple matrices share
        the same name. See below.
    justmatrix : bool; optional
        If True, only the matrix is stored in the dictionary. If
        False, a tuple of ``(matrix, form, mtype)`` is stored.
        This option is ignored if ``into == 'list'``.
    sparse : bool or None or two-tuple_like; optional
        Specifies whether output matrices will be regular numpy arrays
        or sparse arrays. If not two-tuple_like:

        ========   ===============================================
        `sparse`   Action
        ========   ===============================================
         None      Auto setting: each matrix will be sparse if and
                   only if it was written in a sparse format
         True      Matrices will be returned in sparse format
         False     Matrices will be returned in regular (dense)
                   numpy arrays
        ========   ===============================================

        If `sparse` is two-tuple_like, the first element is either
        None, True, or False (see table above) and the second element
        is a callable, as in: ``X = callable(X)``. A common usage of
        the callable would be to convert from "COO" sparse form (see
        :class:`scipy.sparse.coo_matrix`) to a more desirable
        form. For example, to ensure *all* matrices are returned in
        CSC form (see :class:`scipy.sparse.csc_matrix`) use::

            sparse=(True, scipy.sparse.coo_matrix.tocsc)

        The callable is ignored for non-sparse matrices.

    Returns
    -------
    dct : :class:`collections.OrderedDict`, if ``into == 'dct'``
        Keys are the lower-case matrix names and the values are
        either just the matrix or a tuple of:
        ``(matrix, form, mtype)`` depending on `justmatrix`.

    tup : tuple, if ``into == 'list'``
        Tuple of 4 lists: ``(names, matrices, forms, mtypes)``

    Notes
    -----
    The default form for sparse matrices is the "COO" sparse form (see
    :class:`scipy.sparse.coo_matrix`). To override, provide a callable
    in the `sparse` option (see above).

    See also
    --------
    :func:`load`, :func:`write` (or :func:`save`), :func:`dir`.
    """
    return load(filename, namelist, into, justmatrix, sparse)


def dir(filename=None, verbose=True):
    """
    Directory of all matrices in op4 file; non-member version of
    :func:`OP4.dir`.

    Parameters
    ----------
    filename : string or None; optional
        Name of op4 file to read. Can also be the name of a directory
        or None; in these cases, a GUI is opened for file selection.
    verbose : bool; optional
        If true, directory will be printed to screen.

    Returns
    -------
    names : list
        Lower-case list of matrix names in order as read.
    sizes : list
        List of sizes [(r1, c1), (r2, c2), ...], for each
        matrix.
    forms : list
        List of integers specifying the Nastran form of each
        matrix.
    mtypes : list
        List of integers specifying the Nastran type of each
        matrix.

    See also
    --------
    :func:`load`, :func:`write` (or :func:`save`).
    """
    filename = guitools.get_file_name(filename, read=True)
    return OP4().dir(filename, verbose)


def write(
    filename,
    names,
    matrices=None,
    binary=True,
    digits=16,
    endian="",
    sparse="auto",
    forms=None,
):
    """
    Write op4 file; non-member version of :func:`OP4.write`.

    Parameters
    ----------
    filename : string or None
        Name of file. Can also be the name of a directory or None; in
        these cases, a GUI is opened for file selection.
    names : dictionary/OrderedDict or list or string
        Dictionary indexed by the matrix names with the values being
        either the matrices or a tuple/list where the first two items
        are ``(matrix, form)``. Alternatively, `names` can also be a
        list of matrix names (strings) or a single name (string) if
        just one matrix is to be saved. If using the list inputs,
        `matrices` is required to specify the matrices and, if `forms`
        needs to specified, that input would also be required.
    matrices : array or list; optional
        2d ndarray or list of 2d ndarrays. Ignored if `names` is
        a dictionary. Same length as `names`.
    binary : bool; optional
        If true, a double precision binary file is written;
        otherwise an ascii file is created.
    digits : integer; optional
        Number of significant digits after the decimal to include
        in the ascii output. Ignored for binary files.
    endian : string; optional
        Endian setting for binary output:  '' for native, '>' for
        big-endian and '<' for little-endian.
    sparse : string; optional
        Specifies the output format:

        ===========   ===========================================
         `sparse`     Action
        ===========   ===========================================
        'auto'        Each 2d ndarray will be written in "dense"
                      format and each sparse matrix will be
                      written in "bigmat" sparse format
        'bigmat'      Each matrix (whether sparse or not) is
                      written in "bigmat" sparse format
        'dense'       Each matrix will be written in "dense"
                      format
        'nonbigmat'   Each matrix is written in "nonbigmat"
                      format. Note that if the number of rows is
                      > 65535, then the "bigmat" format is used.
        ===========   ===========================================

    forms : integer or list or None; optional
        The matrix form(s). If None, the forms will be determined
        automatically to be 1, 2, or 6. Ignored if `names` is a
        dictionary (which would contain this information). If not
        None, `forms` must be the same length as `names`, but you can
        use None as the form for one or more matrices (see example
        below). From Nastran documentation:

        ======   ==============================
         form    Matrix format
        ======   ==============================
           1     Square
           2     Rectangular
           3     Diagonal
           4     Lower triangular factor
           5     Upper triangular factor
           6     Symmetric
           8     Identity
           9     Pseudo identity
          10     Cholesky factor
          11     Trapezoidal factor
          13     Sparse lower triangular factor
          15     Sparse upper triangular factor
        ======   ==============================

        .. warning::
            The validity of the values in `forms` is not checked in
            any way.

    Returns
    -------
    None.

    Notes
    -----
    To write multiple matrices that have the same name, `names` must
    be a list, not a dictionary. If a list, the order is
    maintained. If a dictionary, the matrices are written in the order
    they are retrieved from the dictionary; use a
    :class:`collections.OrderedDict` to specify a certain order.

    `save` is an alias for `write`.

    Examples
    --------
    To write m, k, b, in that order to a binary file, you could
    use lists or an OrderedDict:

    >>> import numpy as np
    >>> from pyyeti.nastran import op4
    >>> m = np.array([[1, 2], [2, 3]])
    >>> k = np.array([3, 5])
    >>> b = np.array([4, 6])
    >>> names = ['m', 'k', 'b']
    >>> values = [eval(n) for n in names]
    >>> op4.write('mkb.op4', names, values)
    >>> _ = op4.dir('mkb.op4')
    m       ,      2 x 2     , form=6, mtype=2
    k       ,      1 x 2     , form=2, mtype=2
    b       ,      1 x 2     , form=2, mtype=2

    Or, order is also maintained when using an OrderedDict:

    >>> from collections import OrderedDict
    >>> odct = OrderedDict()
    >>> for n in names:
    ...     odct[n] = eval(n)
    >>> op4.write('mkb.op4', odct)
    >>> _ = op4.dir('mkb.op4')
    m       ,      2 x 2     , form=6, mtype=2
    k       ,      1 x 2     , form=2, mtype=2
    b       ,      1 x 2     , form=2, mtype=2

    On the other hand, if you don't care about the order, you could
    use a regular dictionary. (In more recent versions of Python, this
    may behave like an OrderedDict.):

    >>> op4.write('mkb.op4', dict(m=m, k=k, b=b))
    >>> _ = op4.dir('mkb.op4')         # doctest: +SKIP
    m       ,      2 x 2     , form=6, mtype=2
    k       ,      1 x 2     , form=2, mtype=2
    b       ,      1 x 2     , form=2, mtype=2

    To specify the forms, include the `forms` option in either the
    list approach or the dictionary approach. Here, we'll just say
    that "m" is square (not symmetric) for demonstration. The forms
    for "k" and "b" will be automatically detected. First, the list
    approach:

    >>> op4.write('mkb.op4', ['m', 'k', 'b'], [m, k, b],
    ...           forms=[1, None, None])
    >>> _ = op4.dir('mkb.op4')
    m       ,      2 x 2     , form=1, mtype=2
    k       ,      1 x 2     , form=2, mtype=2
    b       ,      1 x 2     , form=2, mtype=2

    Next, the dictionary approach. Since the form is attached to each
    matrix, it is only required if not None:

    >>> odct = OrderedDict()
    >>> odct['m'] = (m, 1)
    >>> odct['k'] = k
    >>> odct['b'] = b
    >>> op4.write('mkb.op4', odct)
    >>> _ = op4.dir('mkb.op4')
    m       ,      2 x 2     , form=1, mtype=2
    k       ,      1 x 2     , form=2, mtype=2
    b       ,      1 x 2     , form=2, mtype=2

    Clean up:

    >>> import os
    >>> os.remove('mkb.op4')

    See also
    --------
    :func:`load`, :func:`dir`.
    """
    filename = guitools.get_file_name(filename, read=False)
    OP4().write(filename, names, matrices, binary, digits, endian, sparse, forms)


# create `save` as an alias for `write`
save = write
