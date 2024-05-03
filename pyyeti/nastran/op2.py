# -*- coding: utf-8 -*-
"""
Some Python tools for reading select data from Nastran .op2 files.
Converted from the Yeti version.

Can read files in big or little endian format.

The functions/classes provided by this module can be accessed by just
importing the "nastran" package. For example, you can access the
:class:`OP2` class in these two ways:

>>> from pyyeti import nastran
>>> from pyyeti.nastran import op2
>>> op2.OP2 is nastran.OP2
True
"""

import sys
import os
import struct
import warnings
from collections import namedtuple
from types import SimpleNamespace
import bisect
import numpy as np
import pandas as pd
from pyyeti import guitools
from pyyeti.nastran import op4, n2p

__all__ = [
    "rdmats",
    "rdnas2cam",
    "procdrm12",
    "rdpostop2",
    "rdparampost",
    "rdparampost_old",
    "OP2",
    "nastran_dr_descriptions",
]

#  Notes on the op2 format.
#
#  DATA BLOCK:
#    All data blocks (including header) start with header 3 elements:
#    [reclen, key, endrec]
#      - reclen = 1 32-bit integer that specifies number of bytes in
#        key (either 4 or 8)
#      - key = 4 or 8 byte integer specifying number of words in next
#        record
#      - endrec = reclen
#
#    DATA SET, can be multiple records:
#        Next is [reclen, data, endrec]
#          - reclen = 1 32-bit integer that specifies number of bytes
#            in data
#          - data = reclen bytes long, variable format; may be part of
#            a data set or the complete set
#          - endrec = reclen
#
#        Next is info about whether we're done with current data set:
#        [reclen, key, endrec]
#          - reclen = 1 32-bit integer that specifies number of bytes
#            in key (either 4 or 8)
#          - key = 4 or 8 byte integer specifying number of words in
#            next record; if 0, done with data set
#          - endrec = reclen
#
#        If not done, we have [reclen, data, endrec] for part 2 (and
#        so on) for the record.
#
#    Once data set is complete, we have: [reclen, key, endrec]
#      - reclen = 1 32-bit integer that specifies number of bytes in
#        key (either 4 or 8)
#      - key = 4 or 8 byte integer specifying number of words in next
#        record (I think ... not useful?)
#      - endrec = reclen
#
#    Then: [reclen, rec_type, endrec]
#      - reclen = 1 32-bit integer that specifies number of bytes in
#        rec_type (either 4 or 8)
#      - rec_type = 0 if table (4 or 8 bytes)
#      - endrec = reclen
#
#    Then, info on whether we're done with data block:
#    [reclen, key, endrec]
#      - reclen = 1 32-bit integer that specifies number of bytes in
#        key (either 4 or 8)
#      - key = 4 or 8 byte integer specifying number of words in next
#        record; if 0, done with data block
#      - endrec = reclen
#
#    If not done, we have [reclen, data, endrec] for record 2 and so
#    on, until data block is read in.


def _expanddof(ids, pvgrids):
    """
    Expands vector of ids to [id, dof].

    Parameters
    ----------
    ids : 1d array_like
        Vector of node ids
    pvgrids : 1d array_like
        True/False vector same length as `ids`. The True entries
        indicate which elements in `ids` are grids; those will get all
        6 DOF while all other ids will just get 0 for the DOF.

    Returns
    -------
    dof : 2d ndarray
        2 column matrix: [id, dof]

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti.nastran import op2
    >>> ids = [1, 2, 3, 4]
    >>> pvgrids = [True, False, False, True]
    >>> _expanddof(ids, pvgrids)
    array([[1, 1],
           [1, 2],
           [1, 3],
           [1, 4],
           [1, 5],
           [1, 6],
           [2, 0],
           [3, 0],
           [4, 1],
           [4, 2],
           [4, 3],
           [4, 4],
           [4, 5],
           [4, 6]])
    """
    ids, pvgrids = np.atleast_1d(ids, pvgrids)
    n = len(ids)
    dof = np.zeros((n, 6), int)
    dof[pvgrids] = np.arange(1, 7)
    V = np.zeros((n, 6), bool)
    V[:, 0] = True
    V[pvgrids, 1:] = True
    expids = np.reshape(ids, (-1, 1)) * V
    V = V.ravel()
    expids = expids.ravel()
    dof = dof.ravel()
    return np.vstack((expids[V], dof[V])).T


class OP2:
    """Class for reading Nastran op2 files and nas2cam data files."""

    def __init__(self, filename):
        self._fileh = None
        self._CodeFuncs = None
        # if isinstance(filename, str):
        self._op2open(filename)

    def __del__(self):
        if self._fileh:
            self._fileh.close()
            self._fileh = None

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        if self._fileh:
            self._fileh.close()
            self._fileh = None
        return False

    @property
    def CodeFuncs(self):
        """See :func:`_check_code`."""
        if self._CodeFuncs is None:

            def _func1(item_code):
                if item_code // 1000 in [2, 3, 6]:
                    return 2
                return 1

            def _func2(item_code):
                return item_code % 100

            def _func3(item_code):
                return item_code % 1000

            def _func4(item_code):
                return item_code // 10

            def _func5(item_code):
                return item_code % 10

            def _func6(item_code):
                if item_code & 8:
                    return 0
                return 1

            def _func7(item_code):
                v = item_code // 1000
                if v in [0, 2]:
                    return 0
                if v in [1, 3]:
                    return 1
                return 2

            def _funcbig(func_code, item_code):
                return item_code & (func_code & 65535)

            self._CodeFuncs = {
                1: _func1,
                2: _func2,
                3: _func3,
                4: _func4,
                5: _func5,
                6: _func6,
                7: _func7,
                "big": _funcbig,
            }
        return self._CodeFuncs

    def _op2open(self, filename):
        """
        Open op2 file in correct endian mode.

        Sets these class variables:

        _fileh : file handle
            Value returned by open().
        _swap : bool
            True if bytes must be swapped to correct endianness.
        _endian : string
            Will be '=' if `swap` is False; otherwise, either '>' or
            '<' for big-endian and little-endian, respectively.
        _intstr : string
            Either `endian` + 'i4' or `endian` + 'i8'.
        _ibytes : integer
            Either 4 or 8 (corresponds to `intstr`)
        _int32str : string
           `endian` + 'i4'.
        _label : string
            The op2 header label or, if none, None.
        _date : vector
            Three element date vector, or None.
        _nastheader : string
            Nastran header for file, or None.
        _postheaderpos : integer
            File position after header.
        dbdct : dictionary
            See :func:`directory` for description. Contains data block
            names, bytes in file, file positions, and for matrices,
            the matrix size.
        dblist : list
            See :func:`directory` for description. Contains same
            information as `dbdct`, but in a file-order list.
        dbstrings : list
            See :func:`directory` for description. Contains same info
            as dbdct, but in a list of ordered and formatted strings.
        names : list
            File-order list of data block names.
        dbstarts : integer 1d ndarray
            File-order list of data block starting byte positions.
        dbstops : integer 1d ndarray
            File-order list of data block stopping byte positions.
        dbtypes : integer 1d ndarray
            File-order list of data block types (0 for table, 1 for
            matrix).
        trailers : list
            File-order list of data block trailers (tuples).
        headers : list
            File-ordered list of header information for records in
            tables. Each entry that corresponds to a table will be a
            list (the length of which equals the number of records in
            the table) of lists::

               [[(3-element record header), record_length],
                [(3-element record header), record_length],
                ...]

            The entry for a matrix is an empty list: ``[]``.
        _Str4 : struct.Struct object
            Precompiled for reading 4 byte integers (corresponds to
            `int32str`).
        _Str : struct.Struct object
            Precompiled for reading 4 or 8 byte integers (corresponds
            to `intstr`).

        File is positioned after the header label (at
        `postheaderpos`).
        """
        self._fileh = open(filename, "rb")
        self.dbdct = {}
        self.dblist = None
        self.dbnames = None
        self.dbstrings = None
        self.names = None
        self.dbstarts = None
        self.dbstops = None
        self.dbtypes = None
        self.headers = None
        self.trailers = None
        bytes = self._fileh.read(4)
        if len(bytes) < 4:
            raise RuntimeError(
                f"{filename!r} is empty or nearly empty (has {len(bytes)} bytes)"
            )
        reclen = struct.unpack("i", bytes)[0]
        self._fileh.seek(0)

        reclen = np.array(reclen, dtype=np.int32)
        if not np.any(reclen == [4, 8]):
            self._swap = True
            reclen = reclen.byteswap()
            if not np.any(reclen == [4, 8]):
                self._fileh.close()
                self._fileh = None
                raise ValueError(
                    "Could not decipher file. First 4-byte integer should be 4 or 8."
                )
            if sys.byteorder == "little":
                self._endian = ">"
            else:
                self._endian = "<"
        else:
            self._swap = False
            self._endian = "="

        self._Str4 = struct.Struct(self._endian + "i")
        if reclen == 4:
            self._ibytes = 4
            self._intstr = self._endian + "i4"
            self._intstru = self._endian + "%di"
            self._i = "i"
            self._Str = self._Str4
            self._rfrmu = self._endian + "%df"
            self._rfrm = self._endian + "f4"
            self._f = "f"
            self._fbytes = 4
        else:
            self._ibytes = 8
            self._intstr = self._endian + "i8"
            self._intstru = self._endian + "%dq"
            self._i = "q"
            self._Str = struct.Struct(self._endian + "q")
            self._rfrmu = self._endian + "%dd"
            self._rfrm = self._endian + "f8"
            self._f = "d"
            self._fbytes = 8

        self._rowsCutoff = 3000
        self._int32str = self._endian + "i4"
        self._int32stru = self._endian + "%di"
        self.rdop2header()
        self._postheaderpos = self._fileh.tell()
        self.directory(verbose=False)

    def _getkey(self):
        """Reads [reclen, key, endrec] triplet and returns key."""
        self._fileh.read(4)
        key = self._Str.unpack(self._fileh.read(self._ibytes))[0]
        self._fileh.read(4)
        return key

    def _skipkey(self, n):
        """Skips `n` key triplets ([reclen, key, endrec])."""
        self._fileh.read(n * (8 + self._ibytes))

    def file_handle(self):
        """Returns the op2 file handle"""
        return self._fileh

    def set_position(self, pos, which=0):
        """
        Set the op2 file position

        Parameters
        ----------
        pos : integer or string
            If integer, it is the desired byte offset in the file. If
            a string, it is the name of the data block to position to;
            in this case, the `which` parameter is also used.
        which : integer; optional
            If `pos` is a string, `which` is the index of the
            data block to seek to. For example, 0 is the first, -1 is
            the last.

        Returns
        -------
        integer
            Byte offset in file (same as `pos` if integer)

        Notes
        -----
        The following two code snippets position to the start of the
        first KAA data block and read it in. The first gets the
        position the "hard way"::

            o2 = op2.OP2('mds.op2')
            fpos = o2.dbdct['KAA'][0].start
            o2.set_position(fpos)
            name, trailer, rectype = o2.rdop2nt()
            kaa = o2.rdop2matrix(trailer)

        The second uses the string feature of this routine::

            o2 = op2.OP2('mds.op2')
            o2.set_position('KAA')
            name, trailer, rectype = o2.rdop2nt()
            kaa = o2.rdop2matrix(trailer)
        """
        if isinstance(pos, str):
            pos = self.dbdct[pos][which].start
        self._fileh.seek(pos)
        return pos

    def rdop2header(self):
        """
        Reads Nastran output2 header label.
        """
        key = self._getkey()
        if key != 3:
            self._fileh.seek(0)
            self._date = self._nastheader = self._label = None
            return

        self._fileh.read(4)  # reclen
        frm = self._intstru % key
        bytes = self._ibytes * key
        self._date = struct.unpack(frm, self._fileh.read(bytes))
        # self._date = np.fromfile(self._fileh, self._intstr, key)
        self._fileh.read(4)  # endrec
        self._getkey()

        reclen = self._Str4.unpack(self._fileh.read(4))[0]
        self._nastheader = self._fileh.read(reclen).decode()
        self._fileh.read(4)  # endrec
        self._getkey()

        reclen = self._Str4.unpack(self._fileh.read(4))[0]
        self._label = self._fileh.read(reclen).decode().strip().replace(" ", "")
        self._fileh.read(4)  # endrec
        self._skipkey(2)

    def _validname(self, bstr):
        """
        Returns a valid variable name from the byte string `bstr`.
        """
        return "".join(
            chr(c)
            for c in bstr
            if (47 < c < 58 or 64 < c < 91 or c == 95 or 96 < c < 123)
        )

    def rdop2eot(self):
        """
        Read Nastran output2 end-of-table marker.

        Returns
        -------
        tuple: (eot, key)
            eot : integer
                1 if end-of-file has been reached and 0 otherwise.
            key : integer
                0 of eot is 1; next key value otherwise.
        """
        bstr = self._fileh.read(4)  # reclen
        if len(bstr) == 4:
            key = self._Str.unpack(self._fileh.read(self._ibytes))[0]
            self._fileh.read(4)  # endrec
        else:
            key = 0
        if key == 0:
            return 1, 0
        return 0, key

    def rdop2nt(self):
        """
        Read Nastran output2 data block name and trailer.

        Returns
        -------
        tuple: (name, trailer, type)
            name : string
                Name of upcoming data block (upper case).
            trailer : tuple
                Data block trailer.
            type : 0 or 1
                0 means table, 1 means matrix.

        All outputs will be None for end-of-file.
        """
        eot, key = self.rdop2eot()
        if key == 0:
            return None, None, None

        reclen = self._Str4.unpack(self._fileh.read(4))[0]
        db_name = self._validname(self._fileh.read(reclen))
        self._fileh.read(4)  # endrec
        self._getkey()
        key = self._getkey()

        self._fileh.read(4)  # reclen
        frm = self._intstru % key
        bytes = self._ibytes * key
        trailer = struct.unpack(frm, self._fileh.read(bytes))
        # trailer = np.fromfile(self._fileh, self._intstr, key)
        self._fileh.read(4)  # endrec
        self._skipkey(4)

        reclen = self._Str4.unpack(self._fileh.read(4))[0]
        db_name2 = self._validname(self._fileh.read(reclen))
        self._fileh.read(4)  # endrec

        self._skipkey(2)
        rec_type = self._getkey()
        return db_name, trailer, rec_type

    def rdop2matrix(self, trailer):
        """
        Read and return Nastran op2 matrix at current file position.

        It is assumed that the name has already been read in via
        :func:`rdop2nt`.

        The size of the matrix is read from trailer:
             rows = trailer[2]
             cols = trailer[1]
        """
        dtype = 1
        rows = trailer[2]
        mtype = trailer[4]
        if mtype > 2:  # complex
            rows *= 2
        if mtype & 1:  # single precision
            frm = self._rfrm
            frmu = self._rfrmu
            bytes_per = self._fbytes
        else:
            frm = self._endian + "f8"
            frmu = self._endian + "%dd"
            bytes_per = 8

        matrix = np.zeros((rows, trailer[1]), order="F")
        intsize = self._ibytes
        col = 0
        while dtype > 0:  # read in matrix columns
            # key is number of elements in next record (row # followed
            # by key-1 real numbers)
            key = self._getkey()
            # read column
            while key > 0:
                reclen = self._Str4.unpack(self._fileh.read(4))[0]
                r = self._Str.unpack(self._fileh.read(intsize))[0] - 1
                if mtype > 2:
                    r *= 2
                n = (reclen - intsize) // bytes_per
                if n < self._rowsCutoff:
                    matrix[r : r + n, col] = struct.unpack(
                        frmu % n, self._fileh.read(n * bytes_per)
                    )
                else:
                    matrix[r : r + n, col] = np.fromfile(self._fileh, frm, n)
                self._fileh.read(4)  # endrec
                key = self._getkey()
            col += 1
            self._getkey()
            dtype = self._getkey()
        self.rdop2eot()
        if mtype > 2:
            matrix = matrix.T.view(complex).T
        return matrix

    def next_db_info(self):
        """
        Get information on the next data block in the file

        See :func:`goto_next` for an example usage.

        Returns
        -------
        SimpleNamespace or None
            Contains the name, start and stop byte positions, trailer,
            etc of the next data block in the file relative to the
            current file postion. This is the same SimpleNamespace
            that is in ``self.dbdct``; see :func:`directory` for a
            detailed description. None is returned if there is no next
            data block.
        """
        curpos = self._fileh.tell()
        j = bisect.bisect_right(self.dblist, curpos, key=lambda sns: sns.start)
        return self.dblist[j] if j < len(self.dblist) else None

    def goto_next(self):
        """
        Set file position at start of next data block

        Position is set to end-of-file if there is no next data block.

        Examples
        --------
        >>> from pathlib import Path
        >>> import inspect
        >>> from pyyeti.nastran import op2
        >>> pth = Path(inspect.getfile(op2))
        >>> fname = (
        ...     pth.parent.parent
        ...     / "tests"
        ...     / "nas2cam_extseout"
        ...     / "assemble.op2"
        ... )
        >>> with op2.OP2(fname) as o2:
        ...    pos = o2.set_position(0)
        ...    while next_info := o2.next_db_info():
        ...        name, start = next_info.name, next_info.start
        ...        print(f"{name} starts @ {start} bytes")
        ...        o2.goto_next()
        PVT0 starts @ 132 bytes
        CASECC starts @ 524 bytes
        GEOM1S starts @ 10548 bytes
        GEOM2S starts @ 11416 bytes
        GEOM4S starts @ 11840 bytes
        BGPDTS starts @ 12384 bytes
        GEOM1S starts @ 13056 bytes
        GEOM2S starts @ 13812 bytes
        GEOM4S starts @ 14180 bytes
        BGPDTS starts @ 14612 bytes
        GEOM1S starts @ 15060 bytes
        GEOM2S starts @ 16440 bytes
        GEOM3S starts @ 16896 bytes
        GEOM4S starts @ 17260 bytes
        DYNAMICS starts @ 17828 bytes
        BGPDTS starts @ 18428 bytes
        EQEXINS starts @ 19244 bytes
        DIT starts @ 20116 bytes
        LAMA starts @ 20508 bytes
        OUGV1 starts @ 22916 bytes
        FRL starts @ 121200 bytes
        KHH starts @ 121464 bytes
        LAMA starts @ 125344 bytes
        OUGV1 starts @ 127752 bytes
        OUGV1 starts @ 130608 bytes
        OUGV1 starts @ 131916 bytes
        """
        nextdb = self.next_db_info()
        if nextdb is None:
            self._fileh.seek(self.dblist[-1].stop)
        else:
            self._fileh.seek(nextdb.start)

    def skipop2matrix(self):
        """
        Skip Nastran op2 matrix at current position.

        It is assumed that the name has already been read in via
        :func:`rdop2nt`.
        """
        # cannot use goto_next here; dblist may not yet be available
        dtype = 1
        while dtype > 0:  # read in matrix columns
            # key is number of elements in next record (row # followed
            # by key-1 real numbers)
            key = self._getkey()
            # skip column
            while key > 0:
                reclen = self._Str4.unpack(self._fileh.read(4))[0]
                self._fileh.seek(reclen, 1)
                self._fileh.read(4)  # endrec
                key = self._getkey()
            self._getkey()
            dtype = self._getkey()
        self.rdop2eot()

    def skipop2table(self):
        """
        Skip over Nastran output2 table.

        .. warning::
            DEPRECATED. This routine will likely be removed in the
            next release. Use :func:`goto_next` instead.
        """
        warnings.warn(
            "`skipop2table` is deprecated. Use `goto_next` instead.",
            FutureWarning,
        )
        self.goto_next()

    @staticmethod
    def _has_match(name, names):
        for patt in names:
            patt = patt.upper()
            if patt[-1] == "*":
                if name.startswith(patt[:-1]):
                    return True
            elif name == patt:
                return True
        return False

    @staticmethod
    def _get_unique(names):
        added = set()
        unique_names = []
        for name in names:
            if name not in added:
                added.add(name)
                unique_names.append(name)
        return unique_names

    def _get_valid_names(self, names, dbtype):
        if names is None or isinstance(next(iter(names)), str):
            # names is None or an iterable of names
            dbs = [sns for sns in self.dblist if sns.dbtype == dbtype]
            all_names = [sns.name for sns in dbs]

            if not names:
                names = all_names
            else:
                # _has_match handles the wildcard; names will have no wildcards
                names = [name for name in all_names if self._has_match(name, names)]
        else:
            # names is an iterable of allowable headers (3-tuples)
            allowed_headers = set(names)
            dbs = [
                sns
                for sns in self.dblist
                if len(sns.headers) > 0 and sns.headers[0][0][:2] in allowed_headers
            ]
            names = [sns.name for sns in dbs]

        # return list of unique names too ... in the order they appear:
        return names, dbs, self._get_unique(names)

    def _rdmat(self, sns, verbose):
        if verbose:
            print(f"Reading matrix {sns.name} at position {sns.start}...")
        self.set_position(sns.start)
        self.rdop2nt()
        return self.rdop2matrix(sns.trailer)

    def rdop2mats(self, names=None, lower=False, verbose=False, which=-1):
        """
        Read all matrices from Nastran output2 file.

        Parameters
        ----------
        names : list_like; optional
            Iterable of names to read in. If None, read all. These can
            be input in lower case. Each name in `names` can end in a
            "*" for simple wildcard matching.
        lower : bool; optional
            If True, returned names will be in lower case.
        verbose : bool; optional
            If True, print matrix names to screen as they are found.
        which : integer or str; optional
            If integer, specifies which occurrence of each matrix to
            read starting at 0. Default is -1; read the last entry for
            each matrix only. Can also be the string "all". In that
            case, each entry in the dictionary is a list of all
            occurrences of each matrix (even if there is only one
            occurrence).

        Returns
        -------
        dict

            Dictionary containing all matrices in the op2 file. If
            `which` is an integer::

                {'NAME1': matrix1, 'NAME2': matrix2, ...}

            If `which` is "all"::

                {'NAME1': [1st matrix1, 2nd matrix1, ...],
                 'NAME2': [1st matrix2, 2nd matrix2, ...],
                 ...}

        Notes
        -----
        Unless `lower` is True, the keys are the names as stored
        in the op2 file (upper case).
        """
        names, matrices, unique_names = self._get_valid_names(names, dbtype=1)
        mats = {}
        for name in unique_names:
            dblist = [sns for sns in matrices if sns.name == name]
            store_name = name if not lower else name.lower()
            if which == "all":
                mats[store_name] = []
                for sns in dblist:
                    mats[store_name].append(self._rdmat(sns, verbose))
            else:
                sns = dblist[which]
                mats[store_name] = self._rdmat(sns, verbose)
        return mats

    def prtdir(self, with_headers=False, with_trailers=False):
        """
        Prints op2 data block directory. See also :func:`directory`.
        """
        if len(self.dbdct) == 0:
            return
        if not with_headers:
            if not with_trailers:
                for s in self.dbstrings:
                    print(s)
            else:
                for s, t in zip(self.dbstrings, self.trailers):
                    print(f"{s}; {t}")
        else:
            if not with_trailers:
                for s, hs in zip(self.dbstrings, self.headers):
                    print(s)
                    for h in hs:
                        print(f"\t{h}")
            else:
                for s, hs, t in zip(self.dbstrings, self.headers, self.trailers):
                    print(f"{s}; {t}")
                    for h in hs:
                        print(f"\t{h}")

    def directory(
        self, verbose=True, redo=False, with_headers=False, with_trailers=False
    ):
        """
        Catalogues and optionally prints contents of op2 file

        Parameters
        ----------
        verbose : bool (or any true/false variable); optional
            If True, print names, sizes, and file offsets to
            screen. Also prints record headers if `with_headers` is
            True.
        redo : bool; optional
            If True, scan through file and redefine ``self.dbdct``
            even if it is already set.
        with_headers : bool; optional
            If True and `verbose` is True, include the table headers
            and record lengths for each record that belongs to the
            table when printing.
        with_trailers : bool; optional
            If True and `verbose` is True, include the table trailers
            when printing.

        Returns
        -------
        dbdct : dictionary
            Dictionary indexed by data block name. Each value is a
            list, one element per occurrence of the data block in the
            op2 file. Each element is a SimpleNamespace with these
            attributes::

              .name : string; name of data block
              .start : integer; file position start
              .stop : integer; file position stop (stop value is start
                      of next data block)
              .nbytes: number of bytes data block consumes in file
              .dbtype : integer; 0 for table, 1 for matrix
              .size : 2-element tuple; for matrices: (rows, cols);
                      for tables: (0, 0)
              .trailer : 7-element tuple; data block trailer
              .headers : list; record headers for data block (see
                         below)

        Attributes
        ----------
        dbdct : dictionary
            Described above.
        dblist : list
            Has the same information as `dbdct`, but in a file-order
            list. Each element is the same SimpleNamespace as in
            `dbdct` (see above).
        dbnames : dictionary
            Here only for backward compatibility; new code should use
            `dbdct` instead. Dictionary indexed by data block
            name. Each value is a list, one element per occurrence of
            the data block in the op2 file. Each element is another
            list that has 3 elements: [fpos, bytes, size]::

                fpos : 2-element list; file position start and stop
                       (stop value is start of next data block)
                bytes: number of bytes data block consumes in file
                size : 2-element list; for matrices, [rows, cols],
                       for tables [0, 0]

        dbstrings : list
            File-order list of strings for printing. Contains
            data block names and file positions.
        names : list
            File-order list of data block names.
        dbstarts : integer 1d ndarray
            File-order list of data block starting byte positions.
        dbstops : integer 1d ndarray
            File-order list of data block stopping byte positions.
        dbtypes : integer 1d ndarray
            File-order list of data block types (0 for table, 1 for
            matrix).
        trailers : list
            File-order list of data block trailers (tuples).
        headers : list
            File-order list of header information for records in
            tables. Each entry that corresponds to a table will be a
            list (the length of which equals the number of records in
            the table) of lists::

               [[(3-element record header), record_length],
                [(3-element record header), record_length],
                ...]

            The entry for a matrix is an empty list: ``[]``.

        Notes
        -----
        As an example of using `dbdct`, to get a list of all sizes of
        matrices named 'KAA'::

            o2 = op2.OP2('mds.op2')
            s = [item.size for item in o2.dbdct['KAA']]

        For another example, to read in first matrix named 'KAA'::

            o2 = op2.OP2('mds.op2')
            o2.set_position('KAA')
            name, trailer, rectype = o2.rdop2nt()
            kaa = o2.rdop2matrix(trailer)
        """
        if len(self.dbdct) > 0 and not redo:
            if verbose:
                self.prtdir(with_headers=with_headers, with_trailers=with_trailers)
            return self.dbdct
        self.dbdct = {}
        self.dblist = []
        self.dbnames = {}
        self.dbstrings = []
        self.names = []
        self.dbstarts = []
        self.dbstops = []
        self.dbtypes = []
        self.headers = []
        self.trailers = []
        self._fileh.seek(self._postheaderpos)
        pos = self._postheaderpos
        while 1:
            name, trailer, dbtype = self.rdop2nt()
            if name is None:
                break
            if dbtype > 0:
                self.skipop2matrix()
                headers = []
                size = (trailer[2], trailer[1])
                s = f"Matrix {name:8}"
            else:
                headers = self.rdop2tabheaders()
                size = (0, 0)
                s = f"Table  {name:8}"
            cur = self._fileh.tell()
            s += f", bytes = {cur-pos-1:10} [{pos:10} to {cur:10}]"
            if size != (0, 0):
                s += f", {size[0]:6} x {size[1]:<}"
            if name not in self.dbdct:
                self.dbdct[name] = []
                self.dbnames[name] = []
            sns = SimpleNamespace(
                name=name,
                start=pos,
                stop=cur,
                nbytes=cur - pos - 1,
                dbtype=dbtype,
                size=size,
                trailer=trailer,
                headers=headers,
            )
            self.dbdct[name].append(sns)
            self.dblist.append(sns)
            self.dbnames[name].append([[pos, cur], cur - pos - 1, size])
            self.dbstrings.append(s)
            self.names.append(name)
            self.dbstarts.append(pos)
            self.dbstops.append(cur)
            self.dbtypes.append(dbtype)
            self.headers.append(headers)
            self.trailers.append(trailer)
            pos = cur

        self.dbstarts = np.array(self.dbstarts)
        self.dbstops = np.array(self.dbstops)
        self.dbtypes = np.array(self.dbtypes)
        if verbose:
            self.prtdir(with_headers=with_headers, with_trailers=with_trailers)
        return self.dbdct

    def rdop2dynamics(self):
        """
        Reads the TLOAD data from a DYNAMICS data block.

        Returns matrix of TLOADS. Rows = 5, 6, 8, or 9. Cols = number
        of TLOADs. TLOAD ids are in first row; other data in matrix
        may not be useful.
        """
        key = self._getkey()
        header_Str = struct.Struct(self._endian + self._i * 3)
        hbytes = 3 * self._ibytes
        eot = 0
        data = np.empty(0, dtype=self._intstr)
        while not eot:
            while key > 0:
                self._fileh.read(4)  # reclen
                header = header_Str.unpack(self._fileh.read(hbytes))
                if header == (7107, 71, 138):
                    if key < self._rowsCutoff:
                        bytes = (key - 3) * self._ibytes
                        ndata = struct.unpack(
                            self._intstru % (key - 3), self._fileh.read(bytes)
                        )
                    else:
                        ndata = np.fromfile(self._fileh, self._intstr, key - 3)
                    data = np.hstack((data, ndata))
                else:
                    self._fileh.seek((key - 3) * self._ibytes, 1)
                self._fileh.read(4)  # endrec
                key = self._getkey()
            self._skipkey(2)
            eot, key = self.rdop2eot()

        if np.any(data):
            L = len(data)
            rows = (5, 6, 8, 9)
            match = [0] * len(rows)
            # see if each col in matrix would match pattern:
            #   [tloadid, exciteid, -, -, tableid, ...]
            # where:
            #   1. tloadid's are all > 0 and always increasing
            #   2. exciteid's are all > 0
            #   3. tableid's are all > 0
            for i, r in enumerate(rows):
                if (
                    L == r * (L // r)
                    and data[::r].min() > 0
                    and (len(data[::r]) == 1 or np.diff(data[::r]).min() > 0)
                    and data[1::r].min() > 0
                    and data[4::r].min() > 0
                ):
                    match[i] = 1
            if sum(match) > 1:
                err = (
                    "Could not determine rows for TLOADs! "
                    f"More than one of {rows} matches. "
                    "Routine needs updating."
                )
                raise ValueError(err)
            if sum(match) < 1:
                err = (
                    "Could not determine rows for TLOADs! "
                    f"None of {rows} matches. "
                    "Routine needs updating."
                )
                raise ValueError(err)
            rows = rows[match.index(1)]
            data = np.reshape(data, (rows, -1), order="F")
        return data

    def rdop2tload(self):
        """
        Returns the TLOAD data from an op2 file.

        This routine scans the op2 file for the DYNAMICS data block and
        then calls :func:`rdop2dynamics` to read the data.
        """
        fpos = self.dbdct["DYNAMICS"][0].start
        self._fileh.seek(fpos)
        name, trailer, dbtype = self.rdop2nt()
        return self.rdop2dynamics()

    def _rdop2gpwg(self):
        self.skipop2record()
        gpwg = {}
        data = self.rdop2record(form="single")
        gpwg["m66"] = np.array(data[0:36], order="F").reshape((6, 6))
        gpwg["s"] = np.array(data[36:45], order="F").reshape((3, 3))
        mass_cg = np.array(data[45:57], order="F").reshape((3, 4))
        gpwg["mass3"] = mass_cg[:, 0].copy()
        gpwg["mass"] = mass_cg[0, 0]
        gpwg["cg33"] = mass_cg[:, 1:4].copy()
        gpwg["cg"] = np.array([mass_cg[1, 1], mass_cg[0, 2], mass_cg[0, 3]])
        gpwg["Is"] = np.array(data[57:66], order="F").reshape((3, 3))
        gpwg["Iq"] = np.array(data[66:69], order="F")
        gpwg["q"] = np.array(data[69:78], order="F").reshape((3, 3))
        self.rdop2eot()
        return gpwg

    def rdop2gpwg(self):
        """
        Read the OGPWG data block, which contains model mass properties.

        Returns
        -------
        dictionary or None if OGPWG not in file

        m66 : 2d ndarray
            6x6 rigid body mass matrix
        s : 2d ndarray
            The 3x3 transformation from principal mass directions to
            the basic csys. This is usually an identity matrix.
        mass3 : 1d ndarray
            A length-3 vector of the model's mass in each principal
            mass direction.
        mass : float
            The model's mass.
        cg33 : 2d ndarray
            The 3x3 center of mass matrix.
        cg : 1d ndarray
            A length-3 vector of the model's center of mass.
        Is : 2d ndarray
            The model's 3x3 inertia matrix at the CG using the
            principal mass directions.
        Iq : 2d ndarray
            A length-3 vector containing the model's inertia matrix
            using the principal axes.
        q : 2d ndarray
            The 3x3 transformation from the basic csys to the principal
            axes.
        """
        try:
            self.set_position("OGPWG")
        except KeyError:
            return None
        self.rdop2nt()
        return self._rdop2gpwg()

    def rdop2opg(self, name="OPG1"):
        """
        Read the OPG data block, which contains applied forces.

        Parameters
        ----------
        name : string; optional
            Name of OPG data block

        Returns
        -------
        opg : dict or None if OPG not in file
            A dictionary containing the applied forces from each static
            subcase. The keys are the subcase IDs and each value is a
            Pandas data frame. The rows of the data frame are the grid
            IDs. There are six columns, which represent three forces
            and three moments.

        Notes
        -----
        The forces are always defined in the grid's output coordinate
        system.
        """
        try:
            self.set_position(name)
        except KeyError:
            return None
        self.rdop2nt()
        output = {}
        while True:
            ident = self.rdop2record(form="int")
            if ident is None:
                break
            data = self.rdop2record(form="single")
            # This uses the generic description of OFP tables, since
            # that matches the actual op2 file contents, rather than
            # the OPG data block description
            acode = ident[0] // 10
            tcode = ident[1] // 1000
            fcode = ident[8]
            numwde = ident[9]
            assert data.shape[0] % numwde == 0

            # Statics, Sort1/Real/Not Random, Real
            if acode != 1 or tcode != 0 or fcode != 1:
                msg = "Unsupported data format in OPG: {}"
                raise ValueError(msg.format((tcode, acode, fcode)))

            loadset_id = ident[4]
            n_grids = data.shape[0] // numwde
            grid_ids = np.frombuffer(data, self._endian + "i4")[::numwde] // 10
            data.shape = (n_grids, numwde)
            forces = data[:, 2:]
            output[loadset_id] = pd.DataFrame(
                forces, index=grid_ids, columns=[1, 2, 3, 4, 5, 6]
            )
        return output

    def _rdop2ogf(self, name):
        startpos = self._fileh.tell()
        _, endpos = self._get_block_bytes(name, startpos)

        col = 0
        while True:
            ident = self.rdop2record(form="int")
            if ident is None:
                break
            data = self.rdop2record(form="single")
            data = data.reshape(-1, ident[9])  # ident[9] is "numwde"

            if col == 0:
                # real eigenvalues, Sort1/Real/Not Random, Real
                acode = ident[0] // 10
                tcode = ident[1] // 1000
                fcode = ident[8]
                if acode != 2 or tcode != 0 or fcode != 1:
                    msg = "Unsupported data format in OGF: {}"
                    raise ValueError(msg.format((tcode, acode, fcode)))

                # determine number of columns by using file bytes
                curpos = self._fileh.tell()
                ncols = (endpos - startpos) // (curpos - startpos)
                node_element = data[:, :2]
                node_element.dtype = np.int32
                eltype = data[:, 2:4]
                eltype.dtype = np.dtype("S8")
                index = [
                    (g // 10, el, desc[0].decode(), f)
                    for (g, el), desc in zip(node_element[:, :2], eltype)
                    for f in ("FX", "FY", "FZ", "MX", "MY", "MZ")
                ]
                indexm = pd.MultiIndex.from_tuples(
                    index, names=("Node", "Element", "Desc", "Comp")
                )
                ogf = np.empty((data.shape[0] * 6, ncols), np.float32)
                freq = np.empty(ncols, np.float32)

            ogf[:, col] = data[:, -6:].ravel()
            ident.dtype = np.float32
            freq[col] = ident[15]
            col += 1

        columns = pd.MultiIndex.from_tuples(
            ((i, f) for i, f in enumerate(freq, 1)), names=("Mode #", "Freq (Hz)")
        )
        return pd.DataFrame(ogf, index=indexm, columns=columns)

    def rdop2ogf(self, name):
        """
        Read an OGF-formatted data block

        Parameters
        ----------
        name : string
            Name of OGF data block

        Returns
        -------
        pandas.DataFrame or None
            If data block is present, the output is a pandas DataFrame
            with multilevel indices. The ``index`` has four levels:
            ``(node, element id, element name, direction)`` with the
            names: ``("Node", "Element", "Desc", "Comp")``. The
            ``columns`` has two levels: ``(mode number, frequency)``
            with the names: ``("Mode #", "Freq (Hz)"). For example,
            here are the first two columns of an ``ofg`` associated
            with node 2693::

                >>> ogf.loc[[2693], :2]   # doctest: +SKIP
                Mode #                                 1             2
                Freq (Hz)                       6.835803      9.240421
                Node Element Desc     Comp
                2693 2648    QUAD4    FX   -6.424288e+00  1.881207e+01
                                      FY   -9.492936e+00  1.947833e+01
                                      FZ   -6.474704e+00  8.646915e+01
                                      MX    3.494281e+00 -1.237239e+01
                                      MY    3.494860e+00 -1.237513e+01
                                      MZ   -2.012388e+00 -1.778058e+00
                     2649    QUAD4    FX    6.426482e+00 -1.881229e+01
                                      FY    9.493129e+00 -1.947879e+01
                                      FZ    6.474766e+00 -8.646799e+01
                                      MX   -3.494281e+00  1.237239e+01
                                      MY   -3.494860e+00  1.237513e+01
                                      MZ    2.012388e+00  1.778058e+00
                     0       *TOTALS* FX    2.193109e-03 -2.204326e-04
                                      FY    1.932395e-04 -4.540352e-04
                                      FZ    6.194949e-05  1.154103e-03
                                      MX   -5.434231e-11  3.802825e-11
                                      MY   -5.434231e-11  3.871037e-11
                                      MZ    1.273293e-10  9.640644e-11
        """
        try:
            self.set_position(name)
        except KeyError:
            return None
        self.rdop2nt()
        return self._rdop2ogf(name)

    def _rdop2oug(self, name):
        startpos = self._fileh.tell()
        _, endpos = self._get_block_bytes(name, startpos)

        col = 0
        while True:
            ident = self.rdop2record(form="int")
            if ident is None:
                break
            data = self.rdop2record(form="single")

            if col == 0:
                # real eigenvalues, Sort1/Real/Not Random, Real
                achk = self._check_code(ident[0], [4], [[2]], "ACODE")
                tchk = self._check_code(
                    ident[1], [1, 2, 7], [[1], [3, 7], [0, 2]], "TCODE"
                )
                if not (achk and tchk):
                    self._fileh.seek(endpos)
                    return

                # determine number of columns by using file bytes
                curpos = self._fileh.tell()
                ncols = (endpos - startpos) // (curpos - startpos)

                # - there are 8 elements per node:
                #   id*10, type, x, y, z, rx, ry, rz
                #   - type == 1 for grids; those have all 6 dof
                id_dof = data.reshape(-1, 8)[:, :2]
                id_dof.dtype = np.int32
                pvgrids = id_dof[:, 1] == 1
                full_dof = _expanddof(id_dof[:, 0] // 10, pvgrids)
                indexm = pd.MultiIndex.from_tuples(
                    (row for row in full_dof), names=("Node", "DOF")
                )

                V = np.zeros((id_dof.shape[0], 8), bool)
                V[:, 2] = True  # all nodes have 'x'
                V[pvgrids, 3:] = True  # only grids have all 6
                V = V.ravel()

                oug = np.empty((full_dof.shape[0], ncols), np.float32)
                eigenval = np.empty(ncols, np.float32)

            oug[:, col] = data[V]
            ident.dtype = np.float32
            eigenval[col] = ident[5]
            col += 1

        frq = np.sqrt(np.abs(eigenval)) / (2 * np.pi)
        columns = pd.MultiIndex.from_tuples(
            ((i, f) for i, f in enumerate(frq, 1)), names=("Mode #", "Freq (Hz)")
        )
        return pd.DataFrame(oug, index=indexm, columns=columns)

    def rdop2oug(self, name):
        """
        Read an OUG-formatted data block

        Parameters
        ----------
        name : string
            Name of OUG data block

        Returns
        -------
        pandas.DataFrame or None
            If data block is present, the output is a pandas DataFrame
            with multilevel indices. The ``index`` has two levels:
            ``(node, dof)``with the names: ``("Node", "DOF")``. The
            ``columns`` has two levels: ``(mode number, frequency)``
            with the names: ``("Mode #", "Freq (Hz)"). For example,
            here are the first four columns of an ``ougv1`` associated
            with node 2693::

                >>> ougv1.loc[[2693], :4]   # doctest: +SKIP
                Mode #            1         2         3         4
                Freq (Hz) 6.835803  9.240421  13.516423 15.768772
                Node DOF
                2693 1    -0.009917  0.000546 -0.029766  0.030169
                     2    -0.000877  0.001123 -0.031812 -0.033568
                     3    -0.000280 -0.002856  0.018068  0.016844
                     4    -0.000128  0.000018  0.000225  0.001604
                     5    -0.000563  0.000115 -0.001383  0.001788
                     6     0.000057  0.000095  0.000393 -0.000192
        """
        try:
            self.set_position(name)
        except KeyError:
            return None
        self.rdop2nt()
        return self._rdop2oug(name)

    def rdop2record(self, form=None, N=0):
        """
        Read Nastran output2 data record.

        Parameters
        ----------
        form : string or None; optional
            String specifying format, or None to read in signed
            integers.  One of::

               'int' (same as None)
               'uint'
               'single'
               'double'
               'bytes' -- raw bytes from file

        N : integer; optional
            Number of elements in final data record; use 0 if unknown.

        Returns
        -------
        rec : 1d ndarray, bytes string, or None
            Typically returns 1d ndarray or, if ``form == 'bytes'``, a
            bytes string. None is returned if the end-of-data block has
            been reached.

        Notes
        -----
        This routine will read in a 'super' record if the data spans
        more than one logical record.
        """
        key = self._getkey()
        if key == 0:
            return None
        f = self._fileh
        if not form or form == "int":
            frm = self._intstr
            frmu = self._intstru
            bytes_per = self._ibytes
        elif form == "uint":
            frm = self._intstr.replace("i", "u")
            frmu = self._intstru.replace("i", "I")
            bytes_per = self._ibytes
        elif form == "double":
            frm = self._endian + "f8"
            frmu = self._endian + "%dd"
            bytes_per = 8
        elif form == "single":
            frm = self._endian + "f4"
            frmu = self._endian + "%df"
            bytes_per = 4
        elif form == "bytes":
            data = []
            while key > 0:
                reclen = self._Str4.unpack(f.read(4))[0]
                data.append(f.read(reclen))
                f.read(4)  # endrec
                key = self._getkey()
            self._skipkey(2)
            data = b"".join(data)
            return data
        else:
            raise ValueError(
                "form must be one of:  None, 'int', "
                "'uint', 'double', 'single' or 'bytes'"
            )
        if N:
            data = np.empty(N, dtype=frm)
            i = 0
            while key > 0:
                reclen = self._Str4.unpack(f.read(4))[0]
                # f.read(4)  # reclen
                n = reclen // bytes_per
                if n < self._rowsCutoff:
                    b = n * bytes_per
                    data[i : i + n] = struct.unpack(frmu % n, f.read(b))
                else:
                    data[i : i + n] = np.fromfile(f, frm, n)
                i += n
                f.read(4)  # endrec
                key = self._getkey()
        else:
            data = []
            while key > 0:
                reclen = self._Str4.unpack(f.read(4))[0]
                # f.read(4)  # reclen
                n = reclen // bytes_per
                if n < self._rowsCutoff:
                    b = n * bytes_per
                    cur = struct.unpack(frmu % n, f.read(b))
                else:
                    cur = np.fromfile(f, frm, n)
                data.extend(cur)
                # data = np.hstack((data, cur))
                f.read(4)  # endrec
                key = self._getkey()
            data = np.array(data, dtype=frm)
        self._skipkey(2)
        return data

    def skipop2record(self):
        """
        Skip over Nastran output2 data record (or super-record).
        """
        key = self._getkey()
        while key > 0:
            reclen = self._Str4.unpack(self._fileh.read(4))[0]
            self._fileh.seek(reclen + 4, 1)
            key = self._getkey()
        self._skipkey(2)

    def rdop2tabheaders(self):
        """
        Read op2 table headers into a list

        Notes
        -----
        File must be positioned after name and trailer block. For
        example, to read the table headers of the last GEOM1S data
        block::

            o2 = op2.OP2('modes.op2')
            o2.set_position('GEOM1S', -1)
            name, trailer, dbtype = o2.rdop2nt()
            o2.rdop2tabheaders()

        """
        eot, key = self.rdop2eot()
        Frm = struct.Struct(self._intstru % 3)
        headers = []
        while not eot:
            while key > 0:
                reclen = self._Str4.unpack(self._fileh.read(4))[0]
                head = Frm.unpack(self._fileh.read(3 * self._ibytes))
                headers.append([head, reclen])
                self._fileh.seek((key - 3) * self._ibytes, 1)
                self._fileh.read(4)
                key = self._getkey()
            self._skipkey(2)
            eot, key = self.rdop2eot()
        return headers

    def _check_code(self, item_code, funcs, vals, name):
        """
        Checks that the code (ACODE or TCODE probably) value is
        acceptable.

        Parameters
        ----------
        item_code : integer
            The ACODE or TCODE (or similar) value for the record.
        funcs : list of integers
            These are the function code values to check for
            `item_code`
        vals : list of lists of integers
            These are the acceptable values for the `code` functions;
            ignored if `item_code` is None.
        name : string
            Name for message; eg: 'TCODE'

        Returns
        -------
        True if all values are acceptable, False otherwise.

        Notes
        -----
        The function codes in `funcs` are:

            ======  ==========================================
            Code    Operation
            ======  ==========================================
               1    if (item_code//1000 = 2,3,6) then return 2
                    else return 1
               2    mod(item_code,100)
               3    mod(item_code,1000)
               4    item_code//10
               5    mod(item_code,10)
               6    if iand(item_code,8)!=0??? then set to 0,
                    else set to 1
               7    if item_code//1000::

                        = 0 or 2, then set to 0
                        = 1 or 3, then set to 1
                        > 3, then set to 2

            >65535  iand(item_code,iand(func_code,65535))
            ======  ==========================================

        where `iand` is the bit-wise AND operation. For example,
        ACODE,4 means that the ACODE value should be integer-divided
        by 10.  So, if ACODE is 22, ACODE,4 is 2 (22 // 10 = 2).
        """
        if len(funcs) != len(vals):
            raise ValueError("len(funcs) != len(vals)!")
        for func, val in zip(funcs, vals):
            if 1 <= func <= 7:
                if self.CodeFuncs[func](item_code) not in val:
                    warnings.warn(
                        f"{name} value {item_code} not acceptable", RuntimeWarning
                    )
                    return False
            elif func > 65535:
                if self.CodeFuncs["big"](func, item_code) not in val:
                    warnings.warn(
                        f"{name} value {item_code} not acceptable", RuntimeWarning
                    )
                    return False
            else:
                raise ValueError(f"Unknown function code: {func}")
        return True

    def _get_block_bytes(self, name, pos):
        """
        Get start/stop bytes of data block

        Parameters
        ----------
        name : string
            Name of data block.
        pos : integer
            Position in file within data block (including start of
            data block)

        Returns
        -------
        start, stop : integers
            Start and stop byte positions of data block. `stop` is
            the start of the next block.
        """
        dbdir = self.dbdct[name]
        for dbdiri in dbdir:
            start = dbdiri.start
            stop = dbdiri.stop
            if start <= pos < stop:
                return start, stop

    def go_to_next_db(self):
        """
        Seek to start of next data block from current position
        """
        pos = self._fileh.tell()
        i = np.searchsorted(self.dbstarts, pos)
        if i == len(self.dbstarts):
            self._fileh.seek(0, os.SEEK_END)
        else:
            self._fileh.seek(self.dbstarts[i])

    def near_db_end(self):
        """
        True if "near" end of data block (there are no more records).
        """
        pos = self._fileh.tell()
        i = np.searchsorted(self.dbstops, pos)
        bytes_to_go = self.dbstops[i] - pos
        return bytes_to_go <= (8 + self._ibytes)

    def _rdop2ougv1(self, name):
        """
        Read op2 OUGV1 mode shape data block.

        Parameters
        ----------
        name : string
            Name of OUGV1 data block.

        Returns
        -------
        ougv1 : dict
            Dictionary with::

               'ougv1' : the OUGV1 matrix
               'lambda' : the eigenvalues; len(lambda) = size(ougv1,2)
               'dof' : 2-column matrix of:  [id, dof];
                       size(dof,1) = size(ougv1,1)

        Notes
        -----
        Can currently only read a real eigenvalue table (ACODE,4 = 2,
        TCODE,1 = 1, TCODE,2 = 7, and TCODE,7 in [0, 2]).
        """
        float2_Str = struct.Struct(self._endian + "ff")
        iif6_int = np.dtype(self._endian + "i4")
        iif6_bytes = 32
        i4_Str = struct.Struct(self._endian + self._i * 4)
        i4_bytes = 4 * self._ibytes
        pos = self._fileh.tell()
        key = self._getkey()
        lam = np.empty(1, float)
        ougv1 = None
        J = 0
        eot = 0
        while not eot:
            if J == 1:
                # compute number of modes by files bytes:
                startpos = pos + 8 + self._ibytes
                bytes_per_mode = self._fileh.tell() - startpos
                _, endpos = self._get_block_bytes(name, startpos)
                nmodes = (endpos - startpos) // bytes_per_mode
                keep = lam
                lam = np.empty(nmodes, float)
                lam[0] = keep[0]
                keep = ougv1
                ougv1 = np.empty((keep.shape[0], nmodes), float, order="F")
                ougv1[:, 0] = keep[:, 0]

            # IDENT record:
            reclen = self._Str4.unpack(self._fileh.read(4))[0]
            header = i4_Str.unpack(self._fileh.read(i4_bytes))
            # header = (ACODE, TCODE, ...)
            achk = self._check_code(header[0], [4], [[2]], "ACODE")
            tchk = self._check_code(
                header[1], [1, 2, 7], [[1], [3, 7], [0, 2]], "TCODE"
            )
            if not (achk and tchk):
                self.goto_next()
                return
            self._fileh.read(self._ibytes)  # mode bytes
            lam[J] = float2_Str.unpack(self._fileh.read(8))[0]
            # ttl bytes = reclen + 4 + 3*(4+ibytes+4)
            #           = reclen + 28 - 3*ibytes
            # read bytes = 4*ibytes + ibytes + 8 = 8 + 5*ibytes
            # self._fileh.seek(reclen-2*self._ibytes+20, 1)  # ... or:
            self._fileh.read(reclen - 2 * self._ibytes + 20)

            # DATA record:
            if ougv1 is None:
                # - process DOF information on first column only
                # - there are 8 elements per node:
                #   id*10, type, x, y, z, rx, ry, rz
                data = self.rdop2record("bytes")  # 1st column
                n = len(data) // iif6_bytes
                data = np.frombuffer(data, iif6_int)
                data1 = (data.reshape(n, 8))[:, :2]
                pvgrids = data1[:, 1] == 1
                dof = _expanddof(data1[:, 0] // 10, pvgrids)
                # form partition vector for modeshape data:
                V = np.zeros((n, 8), bool)
                V[:, 2] = True  # all nodes have 'x'
                V[pvgrids, 3:] = True  # only grids have all 6
                V = V.ravel()
                # initialize ougv1 with first mode shape:
                data.dtype = np.float32  # reinterpret as floats
                ougv1 = data[V].reshape(-1, 1)
            else:
                data = self.rdop2record("single", V.shape[0])
                ougv1[:, J] = data[V]
            J += 1
            eot, key = self.rdop2eot()
        return {"ougv1": ougv1, "lambda": lam, "dof": dof}

    def _rdop2emap(self, nas, nse, trailer):
        """
        Read Nastran output2 EMAP data block.

        Parameters
        ----------
        nas : dict
            Dictionary; has at least {'dnids': {}}.
        nse : integer
            Number of superelements.
        trailer : 1-d array
            The trailer for the EMAP data block.

        Notes
        -----
        Fills in the dnids member of nas.

        See :func:`rdn2cop2`.
        """
        data1 = self.rdop2record()
        # [se bitpos proc_order dnse bitpos_dnse prim_se se_type]
        data1 = np.reshape(data1[: 7 * nse], (-1, 7))

        # skip 2nd record:
        # - This has the superelement connectivity table ... this is
        #   probably not what we want to use to form dnids: a grid can
        #   be on the boundary of se 500, internal to se 100, and yet
        #   still be connected to se 0 (boundary c-set dof I think can
        #   do this). So, the following logic using the connectivity
        #   table would put that grid in the dnids of se 500 AND se
        #   100.
        self.skipop2record()

        # read record 3 and make 'dnids' from it:
        # - this has the list of boundary grids
        for j in range(nse):  # = 1 to nse:
            rec = self.rdop2record()
            se = rec[0]
            if se == 0:
                continue
            ng = rec[2]
            nas["dnids"][se] = rec[8 : 8 + ng]

        self._getkey()

    def _rdop2dnids(self, nas, nse, trailer):
        """
        Read Nastran output2 DNIDS data block.

        Parameters
        ----------
        nas : dict
            Dictionary; has at least {'dnids': {}}.
        nse : integer
            Number of superelements.
        trailer : 1-d array
            The trailer for the EMAP data block.

        Notes
        -----
        Fills in the dnids member of nas. This matrix is an
        alternative to EMAP. NX Nastran 2021 would not always output
        EMAP, complaining about it being an unstructured data block and
        that it would not be byte-swapped. To work around that, but
        work for all other cases as well, I wrote a subdmap that
        extracts the "dnids" information and saves it to a
        matrix::

            dnids = [ nsupers,
                      seid1,
                      ng1,
                      (ng1 grids),
                      seid2,
                      ng2,
                      (ng2 grids),
                      ... ]

        This matrix is read in and processed here.

        See :func:`rdn2cop2`.
        """
        dnids_all = self.rdop2matrix(trailer).astype(int).ravel()
        i = 1
        while i < len(dnids_all):
            se = dnids_all[i]
            ngrids = dnids_all[i + 1]
            nas["dnids"][se] = dnids_all[i + 2 : i + 2 + ngrids]
            i += 2 + ngrids

    def _rdop2bgpdt(self):
        """
        Read record 1 of the Nastran output2 BGPDT data block.

        Returns vector of the BGPDT data or [] if no data found.
        Vector is 9*ngrids in length. For each grid::

          [ coord_id
            internal_id
            external_id
            dof_type;
            permanent_set_constraint
            boundary_grid_id
            x
            y
            z ]

        The x, y, z values are the grid location in basic.

        See :func:`rdn2cop2`.
        """
        nbytes = self._ibytes * 6 + 24
        dtype = np.dtype(
            [("ints", (self._intstr, 6)), ("xyz", (self._endian + "f8", 3))]
        )
        data = self.rdop2record("bytes")
        n = len(data) // nbytes
        if n * nbytes != len(data):
            raise ValueError("incorrect record length for _rdop2bgpdt")
        data = np.frombuffer(data, dtype=dtype)
        return data

    def _rdop2bgpdt68(self):
        """
        Read record 1 of the Nastran output2 BGPDT68 data block.

        Returns vector of the BGPDT data or [] if no data found.
        Vector is 4*ngrids in length. For each grid::

          [ coord_id
            x
            y
            z ]

        The x, y, z values are the grid location in basic.
        """
        nbytes = self._ibytes + 3 * self._fbytes
        dtype = np.dtype([("ints", self._intstr), ("xyz", (self._rfrm, 3))])
        data = self.rdop2record("bytes")
        n = len(data) // nbytes
        if n * nbytes != len(data):
            raise ValueError("incorrect record length for _rdop2bgpdt68")
        self.goto_next()
        data = np.frombuffer(data, dtype=dtype)
        return data

    def _rdop2cstm(self, data_rec1=None):
        """
        Read Nastran output2 CSTM data block.

        Returns 14-column matrix 2-d array of the CSTM data::

          [
           [ id1 type xo yo zo T(1,1:3) T(2,1:3) T(3,1:3) ]
           [ id2 type xo yo zo T(1,1:3) T(2,1:3) T(3,1:3) ]
           ...
          ]

        T is transformation from local to basic for the coordinate
        system.

        See :func:`rdn2cop2`.
        """
        cstm_rec1 = self.rdop2record() if data_rec1 is None else data_rec1
        cstm_rec2 = self.rdop2record("double")
        self.rdop2eot()

        # assemble coordinate system table
        length = len(cstm_rec1)
        cstm = np.zeros((length // 4, 14))
        cstm[:, 0] = cstm_rec1[::4]
        cstm[:, 1] = cstm_rec1[1::4]
        # start index into rec2 for xo, yo, zo, T (12 values) is in
        # last (4th) position in rec1 for each coordinate system:
        pv = range(12)
        for i, j in enumerate(cstm_rec1[3::4]):
            cstm[i, 2:] = cstm_rec2[j + pv - 1]  # -1 for 0 offset
        return cstm

    def _rdop2cstm68(self):
        """
        Read record 1 of Nastran output2 CSTM68 data block.

        Returns vector of the CSTM data or [] if no data
        found. Vector is 14 * number of coordinate systems in
        length. For each coordinate system::

          [ id type xo yo zo T(1,1:3) T(2,1:3) T(3,1:3) ]

        T is transformation from local to basic for the coordinate
        system.
        """
        dtype = np.dtype([("idtype", (self._intstr, 2)), ("xyzT", (self._rfrm, 12))])
        data = self.rdop2record("bytes")
        if not self.near_db_end():
            return self._rdop2cstm(np.frombuffer(data, np.dtype(self._intstr)))
        self.rdop2eot()
        data = np.frombuffer(data, dtype=dtype)
        return np.hstack((data["idtype"], data["xyzT"]))

    def _proc_geom1_record(self, recinfo, record_bytes):
        # the 2d arrays all have fixed number of columns with possibly
        # mixed types
        # - the mixed types arrays will have named fields
        data = np.frombuffer(record_bytes, recinfo.dtype_in)
        if data.dtype.names:
            data = np.hstack([data[name] for name in data.dtype.names])
        else:
            data = data.reshape(recinfo.shape)
        return data

    def _rdop2geom1cord2(self):
        i = self._intstr
        ib = self._ibytes
        f = self._rfrm
        fb = self._fbytes
        Record = namedtuple("Record", "name header shape dtype_in nbytes")

        CORD2R = (2101, 21, 8)
        CORD2C = (2001, 20, 9)
        CORD2S = (2201, 22, 10)
        EXTRN = (1627, 16, 463)
        GRID = (4501, 45, 1)
        SEBULK = (1427, 14, 465)
        SECONCT = (427, 4, 453)
        SELOAD = (1127, 11, 461)

        format_info = {}
        cordinfo = (
            (-1, 13),
            np.dtype([("4i", i, (4,)), ("9f", f, (9,))]),
            4 * ib + 9 * fb,
        )
        grid_frm = np.dtype([("2i", i, (2,)), ("xyz", f, (3,)), ("3i", i, (3,))])
        sebulk_frm = np.dtype([("4i", i, (4,)), ("1f", f, (1,)), ("3i", i, (3,))])
        for name, header, shape, dtype_in, nbytes in (
            ("cord2r", CORD2R, *cordinfo),
            ("cord2c", CORD2C, *cordinfo),
            ("cord2s", CORD2S, *cordinfo),
            ("extrn", EXTRN, (-1, 2), np.dtype(i), 2 * ib),
            ("grid", GRID, (-1, 8), grid_frm, 5 * ib + 3 * fb),
            ("sebulk", SEBULK, (-1, 8), sebulk_frm, 7 * ib + fb),
            ("seconct", SECONCT, (-1,), np.dtype(i), ib),
            ("seload", SELOAD, (-1, 3), np.dtype(i), 3 * ib),
        ):
            format_info[header] = Record(name, header, shape, dtype_in, nbytes)

        header_Str = struct.Struct(self._endian + self._i * 3)
        hbytes = 3 * ib

        data = {}
        eot = 0
        key = self._getkey()
        fh = self._fileh
        while not eot:
            # The following is very much like: rec =
            # self.rdop2record("bytes"). The difference is the call to
            # self._getkey() at the top of rdop2record ... that would
            # mess up this reader logic.

            # ? For speed, should we just read header, check if
            # wanted, and if not, use fh.seek to skip ahead? It's
            # pretty fast as is, so leave it for now.
            rec = []
            while key > 0:
                reclen = self._Str4.unpack(fh.read(4))[0]
                rec.append(fh.read(reclen))
                fh.read(4)  # endrec
                key = self._getkey()
            self._skipkey(2)
            rec = b"".join(rec)

            head = header_Str.unpack(rec[:hbytes])
            if head in format_info:
                data[format_info[head].name] = self._proc_geom1_record(
                    format_info[head], rec[hbytes:]
                )
            eot, key = self.rdop2eot()

        # merge the CORD2* cards:
        cords = np.zeros((0, 13))
        for name in ("cord2r", "cord2c", "cord2s"):
            if name in data:
                cords = np.vstack((cords, data[name]))
                del data[name]
        # remove 3rd column of cords (not used)
        cords = np.delete(cords, 2, axis=1)
        data["cords"] = n2p.build_coords(cords)

        # build selist from seconct:
        if "seconct" in data:
            seconct = data["seconct"]
            pv = np.nonzero(seconct == -1)[0][1:-2:2] + 1
            pv = np.hstack((0, pv))
            u = np.unique(seconct[pv], return_index=True)[1]
            pv = pv[u]
            selist = np.vstack((seconct[pv], seconct[pv + 1])).T
            selist = np.vstack((selist, [[0, 0]]))
            data["selist"] = selist

        return data

    def _rdop2selist(self):
        """
        Read SLIST data block and return `selist` for
        :func:`rdn2cop2`.

        See :func:`rdn2cop2`.
        """
        slist = self.rdop2record()
        slist[1::7] = 0
        self.skipop2record()
        self.rdop2eot()
        return np.vstack((slist[::7], slist[4::7])).T

    def _rdop2lama(self):
        """
        Read LAMA frequency data block.

        Has 7 columns::

            [mode #, ext #, eigenvalue, radians, cycles,
                                               genmass, genstiff]
        """
        self.skipop2record()
        data = self.rdop2record(form="bytes")
        self.rdop2eot()
        dtype = np.dtype([("ints", (self._intstr, 2)), ("reals", (self._rfrm, 5))])
        data = np.frombuffer(data, dtype=dtype)
        return np.hstack((data["ints"], data["reals"]))

    def _rdop2uset(self):
        """
        Read the USET data block.

        Returns 1-d USET array. The 2nd bit is cleared for the S-set.

        See :func:`rdn2cop2`.
        """
        uset = self.rdop2record("uint")
        # clear the 2nd bit for all S-set:
        sset = (uset & n2p.mkusetmask("s")) != 0
        if any(sset):
            uset[sset] = uset[sset] & ~np.array(2, uset.dtype)
        self.rdop2eot()
        return uset

    def _rdop2eqexin(self):
        """
        Read the EQEXIN data block.

        Returns (EQEXIN1, EQEXIN) tuple.

        See :func:`rdn2cop2`.
        """
        eqexin1 = self.rdop2record()
        eqexin = self.rdop2record()
        self.rdop2eot()
        return eqexin1, eqexin

    def _proc_bgpdt(self, eqexin1, eqexin, ver68=False, bgpdtin=None):
        """
        Reads and processes the BGPDT data block for :func:`rdn2cop2`,
        :func:`rdparampost`, and :func:`rdpostop2`.

        Returns (xyz, cid, dof, doftype, nid, upids)

        See :func:`rdn2cop2`, :func:`rdparampost`, :func:`rdpostop2`.
        """
        if ver68:
            bgpdt_rec1 = bgpdtin
        else:
            bgpdt_rec1 = self._rdop2bgpdt()
            self.goto_next()

        xyz = bgpdt_rec1["xyz"]
        if ver68:
            cid = bgpdt_rec1["ints"]
        else:
            cid = bgpdt_rec1["ints"][:, 0]

        # assemble dof table:
        dof = eqexin[1::2] // 10
        doftype = eqexin[1::2] - 10 * dof
        nid = eqexin[::2]

        # eqexin is in external sort, so sort it
        i = eqexin1[1::2].argsort()
        dof = dof[i]
        doftype = doftype[i]
        nid = nid[i]
        if ver68:
            upids = None
        else:
            upids = bgpdt_rec1["ints"][:, 5]
        return xyz, cid, dof, doftype, nid, upids

    def _buildUset(
        self,
        se,
        dof,
        doftype,
        nid,
        uset,
        xyz,
        cid,
        cstm=None,
        cstm2=None,
        print_warning=True,
    ):
        """
        Builds the 6-column uset table for :func:`rdn2cop2`,
        :func:`rdparampost`, and :func:`rdpostop2`.

        Returns: (uset, cstm, cstm2).
        """
        # Fill in all dof use -1 as default and set dof as
        # appropriate ... make it big enough for grids (6 cols).
        # The -1s will be partitioned out below.
        rd = len(dof)
        rb = len(cid)
        if rd != rb:
            raise ValueError(
                f"RDOP2USET: BGPDTS incompatible with EQEXINS for superelement {se}.\n"
                "  Guess:  residual run clobbered EQEXINS\n"
                '    Fix:  add the "fxphase0" alter to your residual run'
            )
        coordinfo = np.zeros((rd, 18))
        coordinfo[:, :3] = xyz
        if cstm is None:
            n = len(cstm2)
            cstm = np.zeros((n, 14))
            for i, key in enumerate(cstm2):
                cstm[i, :2] = cstm2[key][0, :2]
                cstm[i, 2:] = (cstm2[key].ravel())[3:]
        cref = cstm[:, 0].astype(int)
        c_all = cid
        i = np.argsort(cref)
        pv = i[np.searchsorted(cref, c_all, sorter=i)]
        coordinfo[:, 3] = cid
        coordinfo[:, 4] = cstm[pv, 1]
        coordinfo[:, 6:] = cstm[pv, 2:]

        grids = doftype == 1
        ngrids = np.count_nonzero(grids)
        nongrids = rd - ngrids
        doflist = np.zeros((rd, 6), int) - 1
        if ngrids > 0:
            doflist[grids] = np.arange(1, 7)
        if nongrids > 0:
            doflist[~grids, 0] = 0
        doflist = doflist.ravel()
        idlist = np.dot(nid.reshape(-1, 1), np.ones((1, 6), nid.dtype)).ravel()
        coordinfo = coordinfo.reshape((rd * 6, 3))

        # partition out -1s:
        pv = doflist != -1
        doflist = doflist[pv]
        idlist = idlist[pv]
        coordinfo = coordinfo[pv, :]
        if uset is None:
            if print_warning:
                warnings.warn(
                    "uset information not found. Putting all DOF in b-set.",
                    RuntimeWarning,
                )
            b = n2p.mkusetmask("b")
            uset = np.zeros(len(doflist), int) + b
        uset = n2p.make_uset(np.column_stack((idlist, doflist)), uset, coordinfo)

        # make sure cstm2 has everything:
        cstm3 = {}
        for row in cstm:
            m = np.zeros((5, 3))
            m[0, :2] = row[:2]
            m[1:, :] = row[2:].reshape((4, 3))
            cstm3[int(row[0])] = m

        if cstm2 is None:
            cstm2 = cstm3
        else:
            cstm2 = {**cstm2, **cstm3}
        return uset, cstm, cstm2

    def _build_uset(self, eqexin1, eqexin, bgpdt0, uset, cstm, cstm2):
        (xyz, cid, dof, doftype, nid, upids) = self._proc_bgpdt(
            eqexin1,
            eqexin,
            True,
            bgpdt0,
        )
        return self._buildUset(
            0,
            dof,
            doftype,
            nid,
            uset,
            xyz,
            cid,
            cstm,
            cstm2,
            print_warning=False,
        )

    def _rdgeom1(op2obj):
        geom1 = op2obj._rdop2geom1cord2()
        if 0 not in geom1["cords"]:
            geom1["cords"][0] = np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        if -1 not in geom1["cords"]:
            # dummy for spoints
            geom1["cords"][-1] = np.zeros((5, 3))
            geom1["cords"][-1][0, 0] = -1
        return geom1

    def _rdcstm(self, verbose):
        if "CSTMS" in self.dbdct:
            name = "CSTMS"
        elif "CSTM" in self.dbdct:
            name = "CSTM"
        else:
            return None

        pos = self.set_position(name, which=-1)
        if verbose:
            print(f"Reading table {name} at position {pos}...")
        self.rdop2nt()
        cstm = self._rdop2cstm68()
        bc = np.array(
            [
                [+0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        return np.vstack((bc, cstm))

    def _rddatablock(
        self, names, rdfunc, verbose, mats=None, *, which=-1, include_name=False
    ):
        names, tables, unique_names = self._get_valid_names(names, dbtype=0)
        for name in unique_names:
            dblist = [sns for sns in tables if sns.name == name]
            if which == "all":
                lst = []
                for sns in dblist:
                    if verbose:
                        print(f"Reading table {name} at position {sns.start}...")
                    self.set_position(sns.start)
                    self.rdop2nt()
                    db = rdfunc(name) if include_name else rdfunc()
                    lst.append(db)
                if mats is not None and len(lst) > 0:
                    mats[name.lower()] = lst
            else:
                sns = dblist[which]
                if verbose:
                    print(f"Reading table {name} at position {sns.start}...")
                self.set_position(sns.start)
                self.rdop2nt()
                db = rdfunc(name) if include_name else rdfunc()
                if mats is not None:
                    mats[name.lower()] = db
                else:
                    return db, name
        return None, None

    def _rd_dr_table(self, name, pos, verbose):
        self.set_position(pos)
        self.rdop2nt()
        self.rdop2record()
        rec2 = self.rdop2record("bytes")
        if (rec2[:40] == b"TYPE  IDCOMP ROW    TYPE  IDCOMP ROW    ") or (
            rec2[:40] == b"TYPE      ID    COMP     ROW            "
        ):
            if verbose:
                print(f"Reading table {name} at position {pos}...")
            if rec2[:8] == b"TYPE  ID":
                offset = 40
                iform = "i4"
            else:
                offset = 80
                iform = "i8"
            type_id_ndof = np.frombuffer(rec2[offset:], self._endian + iform).reshape(
                -1, 5
            )[:, :3]
            mat = np.array(
                [
                    [i, dof, etype]
                    for etype, i, ndof in type_id_ndof
                    for dof in range(1, ndof + 1)
                ]
            )
            return mat
        return None

    def _rd_datarecovery_tables(self, names, verbose, which=-1):
        names, tables, unique_names = self._get_valid_names(names, dbtype=0)
        mats = {}
        for name in unique_names:
            dblist = [sns for sns in tables if sns.name == name]
            store_name = name.lower()
            if which == "all":
                mats[store_name] = []
                for sns in dblist:
                    mats[store_name].append(
                        self._rd_dr_table(sns.name, sns.start, verbose)
                    )
            else:
                sns = dblist[which]
                mats[store_name] = self._rd_dr_table(sns.name, sns.start, verbose)
        return mats

    def rdparampost(
        self,
        *,
        verbose=False,
        get_all=False,
        get_mats=True,
        get_ougs=False,
        get_ogfs=False,
        get_drms=False,
        get_dr_tables=True,
        which=-1,
    ):
        """
        Reads PARAM,POST,-1 op2 file and returns dictionary of data.

        Parameters
        ----------
        verbose : bool; optional
            If True, echo names of tables and matrices to screen
        get_all; bool; optional
            If True, sets all other `get_*` parameters to
            True. Ignored if False. It won't override `get_mats` or
            `get_dr_tables` if they are lists or tuples.
        get_mats : bool or list/tuple; optional
            If True (or False), read (or do not read) matrices. If a
            list/tuple, it is an iterable of data blocks to look for
            and read if possible. Each name in the list can end with a
            "*" for simple wildcard matching. For example::

                get_mats = ["K4HH", "M*"]

            would read in "K4HH" and all matrices that start with
            "M". In the output dictionary of this routine, each matrix
            is stored by its name in lower case. If `which` is "all",
            then each entry is a list of all the matrices of each name
            in the file (see :func:`OP2.rdop2mats`)
        get_ougs : bool; optional
            If True, read all "OUG" formatted datablocks. See also
            `which`.
        get_ogfs : bool; optional
            If True, read all "OGF" formatted datablocks. See also
            `which`.
        get_drms : bool; optional
            If True, read all "OEF" and "OES" formatted datablocks.
            See also `which`.
        get_dr_tables : bool or list/tuple; optional
            If True (or False), read (or do not read) data recovery
            tables. If True, any data block that starts with "T" is
            checked and if it is a data recovery table, it is read
            in. If `get_dr_tables` is a list/tuple, it is an iterable
            of data blocks to look for and read if possible. Each name
            in the list can end with a "*" for simple wildcard
            matching. For example::

                get_dr_tables = ["TUG*"]

            would read in all data recovery tables that match that
            pattern, and only those. Setting `get_dr_tables` to True
            is equivalent to setting to ``["T*"]``. In the output
            dictionary of this routine, each matrix is stored by its
            name in lower case. Each matrix is 3-columns::

                [id, dof, type]

            See also `which`.
        which : integer or str; optional
            If integer, specifies which occurrence of each matrix to
            read starting at 0. Default is -1; read the last entry for
            each matrix only. Can also be the string "all". In that
            case, each entry in the dictionary is a list of all
            occurrences of each matrix (even if there is only one
            occurrence).

        Returns
        -------
        dictionary

        'uset' : pandas DataFrame
            A DataFrame as output by :func:`OP2.rdn2cop2`
        'cstm' : array
            14-column matrix containing the coordinate system
            transformation matrix for each coordinate system. See
            description in class OP2, member function
            :func:`OP2.rdn2cop2`.
        'cstm2' : dictionary
            Dictionary indexed by the coordinate system id
            number. This has the same information as 'cstm', but in a
            different format.  See description in class OP2, member
            function :func:`OP2.rdn2cop2`.
        'lama' : ndarray or list of ndarrays
            The "LAMA" table; # modes x 7. List if `which` is "all".
        'dynamics' : ndarray or list of ndarrays
            The "tload" part of the DYNAMICS data block. See also
            "tload". List if `which` is "all".
        'tload' : ndarray
            Only present if the "DYNAMICS" data block is present. This
            is the "tload" part of the last "DYNAMICS" data block read
            in. (Same as "dynamics" if `which` is not "all".)
        'ogpwg' : dictionary or list of dictionaries
            Only present if the "OGPWG" table is present in the op2
            file. The dictionary is the output of
            :func:`OP2.rdop2gpwg`. List if `which` is "all".
        'selist' : 2d ndarray
            2-columns matrix: [ seid, dnseid ] where, for each row,
            dnseid is the downstream superelement for seid. (dnseid =
            0 if seid = 0).
        'sebulk' : 2d ndarray
            output record from GEOM1 of SE 0
        'seload' : 2d ndarray
            output record from GEOM1 of SE 0
        'seconct' : 1d ndarray
            output record from GEOM1 of SE 0
        'geom1' : dictionary
            Dictionary of GEOM1(S) data blocks; key is SE. Does not
            depend on `which`. Will be empty if no GEOM1(S) data
            blocks are in the op2 file.
        'bgpdt' : dictionary
            Dictionary of BGPDT(S) data blocks; key is SE. Does not
            depend on `which`. Will be empty if no BGPDT(S) data
            blocks are in the op2 file.

        Notes
        -----
        Other data blocks may be present in the output depending on the
        input parameters and the contents of the op2 file.
        """
        # read last geom1s or geom1 data block
        # - if se tree, this would be for residual and will have "selist"
        # - if upstream cb creation run, will have "extrn"
        # - if non-se run, will have neither of those
        nms = ["GEOM1S", "GEOM1"]
        geom1_0, name = self._rddatablock(nms, self._rdgeom1, verbose)
        addon = "S"
        nsupers = 0
        ses = []
        if name == "GEOM1S":
            if "selist" in geom1_0:
                # se tree run
                nsupers = geom1_0["selist"].shape[0]
                ses = geom1_0["selist"][:, 0]
            else:
                # single upstream se run
                nsupers = 1
                ses = [0]  # treat as SE 0
        elif name == "GEOM1":
            # non-se run
            nsupers = 1
            ses = [0]
            addon = ""

        gname = f"GEOM1{addon}"
        bname = f"BGPDT{addon}"
        ng = len(self.dbdct[gname]) if gname in self.dbdct else 0
        nb = len(self.dbdct[bname]) if bname in self.dbdct else 0
        if ng != nsupers or nb != nsupers:
            warnings.warn(
                f"number of SEs from 'selist' is {nsupers}; however have "
                f"{ng} {gname!r} data blocks and {nb} {bname!r} data blocks."
                " Output may be garbled.",
                RuntimeWarning,
            )

        geom1 = {}
        bgpdt = {}
        build_uset = 0
        for j, se in enumerate(ses):
            for dbname, dct, rdfunc in (
                (gname, geom1, self._rdgeom1),
                (bname, bgpdt, self._rdop2bgpdt68),
            ):
                if se == 0 and dbname.startswith("GEOM1"):
                    geom1[0] = geom1_0
                    continue
                try:
                    db, name = self._rddatablock([dbname], rdfunc, verbose, which=j)
                except KeyError:
                    pass
                else:
                    dct[se] = db
                    if se == 0 and dbname == bname:
                        build_uset += 1

        cstm = self._rdcstm(verbose)
        cstm2 = geom1[0]["cords"] if 0 in geom1 and "cords" in geom1[0] else None

        dct = {
            "cstm": cstm,
            "cstm2": cstm2,
            "geom1": geom1,
            "bgpdt": bgpdt,
        }

        eqexin, name = self._rddatablock(
            ["EQEXINS", "EQEXIN"], self._rdop2eqexin, verbose
        )
        if eqexin:
            eqexin1, eqexin = eqexin
            build_uset += 1

        uset, name = self._rddatablock(["USET"], self._rdop2uset, verbose)

        if build_uset == 2:
            Uset, cstm, cstm2 = self._build_uset(
                eqexin1, eqexin, bgpdt[0], uset, cstm, cstm2
            )
            dct["uset"] = Uset

        # read other miscellaneous data blocks:
        if get_mats or get_all:
            if isinstance(get_mats, (list, tuple)):
                names = get_mats
            else:
                names = None
            dct.update(
                self.rdop2mats(names=names, verbose=verbose, lower=True, which=which)
            )

        for names, rdfunc in (
            (["LAMA*"], self._rdop2lama),
            (["DYNAMICS", "DYNAMIC"], self.rdop2dynamics),
            (["OGPWG*"], self._rdop2gpwg),
        ):
            self._rddatablock(names, rdfunc, verbose, dct, which=which)

        # search by (acode, tcode)
        for flag, names, rdfunc in (
            (get_ougs, {(22, 7), (22, 3)}, self._rdop2oug),
            (get_drms, {(22, 4), (22, 5)}, self._rdop2drm),
            (get_ogfs, {(22, 19)}, self._rdop2ogf),
        ):
            if flag or get_all:
                self._rddatablock(
                    names, rdfunc, verbose, dct, which=which, include_name=True
                )

        # read data recovery tables like tug1 and tef1
        if not isinstance(get_dr_tables, (list, tuple)):
            if get_dr_tables or get_all:
                get_dr_tables = ["T*"]

        if get_dr_tables:
            dct.update(
                self._rd_datarecovery_tables(get_dr_tables, verbose, which=which)
            )

        # some special se 0 data block handling for backward
        # compatibility:
        if 0 in geom1:
            for name in ("selist", "sebulk", "seload", "seconct"):
                if name in geom1[0]:
                    dct[name] = geom1[0][name]

        if "dynamics" in dct:
            if which == "all":
                dct["tload"] = dct["dynamics"][-1]
            else:
                dct["tload"] = dct["dynamics"]

        return dct

    def rdparampost_old(
        self,
        verbose=False,
        get_all=False,
        get_mats=True,
        get_ougv1=False,
        get_oef1=False,
        get_oes1=False,
        get_dr_tables=True,
        which=-1,
    ):
        """
        Reads PARAM,POST,-1 op2 file and returns dictionary of data.

        .. warning::
            DEPRECATED. This routine will likely be removed in the
            next release. Use :func:`rdparampost` instead.

        Parameters
        ----------
        verbose : bool; optional
            If True, echo names of tables and matrices to screen
        get_all; bool; optional
            If True, sets all other `get_*` parameters to
            True. Ignored if False. It won't override `get_mats` or
            `get_dr_tables` if they are lists or tuples.
        get_mats : bool or list/tuple; optional
            If True (or False), read (or do not read) matrices. If a
            list/tuple, it is an iterable of data blocks to look for
            and read if possible. Each name in the list can end with a
            "*" for simple wildcard matching. For example::

                get_mats = ["K4HH", "M*"]

            would read in "K4HH" and all matrices that start with
            "M". In the output dictionary of this routine, each matrix
            is stored by its name in lower case. If `which` is "all",
            then each entry is a list of all the matrices of each name
            in the file (see :func:`OP2.rdop2mats`)
        get_ougv1 : bool; optional
            If True, read the OUGV1, OUG1, or BOPHIG matrix, if
            present. See also `which`.
        get_oef1 : bool; optional
            If True, read the OEF1 matrix, if any. See also `which`.
        get_oes1 : bool; optional
            If True, read the OES1 matrix, if any. See also `which`.
        get_dr_tables : bool or list/tuple; optional
            If True (or False), read (or do not read) data recovery
            tables. Any data block that starts with "T" is checked and
            if it is a data recovery table, it is read in. If a
            list/tuple, it is an iterable of data blocks to look for
            and read if possible. Each name in the list can end with a
            "*" for simple wildcard matching. For example::

                get_dr_tables = ["TUG*"]

            would read in all data recovery tables that match that
            pattern, and only those. Setting `get_dr_tables` to True
            is equivalent to setting to ``["T*"]``. In the output
            dictionary of this routine, each matrix is stored by its
            name in lower case. Each matrix is 3-columns::

                [id, dof, type]

            See also `which`.
        which : integer or str; optional
            If integer, specifies which occurrence of each matrix to
            read starting at 0. Default is -1; read the last entry for
            each matrix only. Can also be the string "all". In that
            case, each entry in the dictionary is a list of all
            occurrences of each matrix (even if there is only one
            occurrence).

        Returns
        -------
        dictionary

        'uset' : pandas DataFrame
            A DataFrame as output by :func:`OP2.rdn2cop2`
        'cstm' : array
            14-column matrix containing the coordinate system
            transformation matrix for each coordinate system. See
            description in class OP2, member function
            :func:`OP2.rdn2cop2`.
        'cstm2' : dictionary
            Dictionary indexed by the coordinate system id
            number. This has the same information as 'cstm', but in a
            different format.  See description in class OP2, member
            function :func:`OP2.rdn2cop2`.
        'lama' : ndarray or list of ndarrays
            The "LAMA" table; # modes x 7. List if `which` is "all".
        'dynamics' : ndarray or list of ndarrays
            The "tload" part of the DYNAMICS data block. See also
            "tload". List if `which` is "all".
        'tload' : ndarray
            Only present if the "DYNAMICS" data block is present. This
            is the "tload" part of the last "DYNAMICS" data block read
            in. (Same as "dynamics" if `which` is not "all".)
        'ogpwg' : dictionary or list of dictionaries
            Only present if the "OGPWG" table is present in the op2
            file. The dictionary is the output of
            :func:`OP2.rdop2gpwg`. List if `which` is "all".
        'selist' : 2d ndarray
            2-columns matrix: [ seid, dnseid ] where, for each row,
            dnseid is the downstream superelement for seid. (dnseid =
            0 if seid = 0).
        'sebulk' : 2d ndarray
            output record from GEOM1 of SE 0
        'seload' : 2d ndarray
            output record from GEOM1 of SE 0
        'seconct' : 1d ndarray
            output record from GEOM1 of SE 0
        'geom1' : dictionary
            Dictionary of GEOM1(S) data blocks; key is SE. Does not
            depend on `which`. Will be empty if no GEOM1(S) data
            blocks are in the op2 file.
        'bgpdt' : dictionary
            Dictionary of BGPDT(S) data blocks; key is SE. Does not
            depend on `which`. Will be empty if no BGPDT(S) data
            blocks are in the op2 file.

        Notes
        -----
        Other data blocks may be present in the output depending on the
        input parameters and the contents of the op2 file.
        """
        warnings.warn(
            "`rdparampost_old` will be removed soon. Use `rdparampost` instead.",
            FutureWarning,
        )

        # read last geom1s or geom1 data block
        # - if se tree, this would be for residual and will have "selist"
        # - if upstream cb creation run, will have "extrn"
        # - if non-se run, will have neither of those
        nms = ["GEOM1S", "GEOM1"]
        geom1_0, name = self._rddatablock(nms, self._rdgeom1, verbose)
        addon = "S"
        nsupers = 0
        ses = []
        if name == "GEOM1S":
            if "selist" in geom1_0:
                # se tree run
                nsupers = geom1_0["selist"].shape[0]
                ses = geom1_0["selist"][:, 0]
            else:
                # single upstream se run
                nsupers = 1
                ses = [0]  # treat as SE 0
        elif name == "GEOM1":
            # non-se run
            nsupers = 1
            ses = [0]
            addon = ""

        gname = f"GEOM1{addon}"
        bname = f"BGPDT{addon}"
        ng = len(self.dbdct[gname]) if gname in self.dbdct else 0
        nb = len(self.dbdct[bname]) if bname in self.dbdct else 0
        if ng != nsupers or nb != nsupers:
            warnings.warn(
                f"number of SEs from 'selist' is {nsupers}; however have "
                f"{ng} {gname!r} data blocks and {nb} {bname!r} data blocks."
                " Output may be garbled.",
                RuntimeWarning,
            )

        geom1 = {}
        bgpdt = {}
        build_uset = 0
        for j, se in enumerate(ses):
            for dbname, dct, rdfunc in (
                (gname, geom1, self._rdgeom1),
                (bname, bgpdt, self._rdop2bgpdt68),
            ):
                if se == 0 and dbname.startswith("GEOM1"):
                    geom1[0] = geom1_0
                    continue
                try:
                    db, name = self._rddatablock([dbname], rdfunc, verbose, which=j)
                except KeyError:
                    pass
                else:
                    dct[se] = db
                    if se == 0 and dbname == bname:
                        build_uset += 1

        cstm = self._rdcstm(verbose)
        cstm2 = geom1[0]["cords"] if 0 in geom1 and "cords" in geom1[0] else None

        dct = {
            "cstm": cstm,
            "cstm2": cstm2,
            "geom1": geom1,
            "bgpdt": bgpdt,
        }

        eqexin, name = self._rddatablock(
            ["EQEXINS", "EQEXIN"], self._rdop2eqexin, verbose
        )
        if eqexin:
            eqexin1, eqexin = eqexin
            build_uset += 1

        uset, name = self._rddatablock(["USET"], self._rdop2uset, verbose)

        if build_uset == 2:
            Uset, cstm, cstm2 = self._build_uset(
                eqexin1, eqexin, bgpdt[0], uset, cstm, cstm2
            )
            dct["uset"] = Uset

        # read other miscellaneous data blocks:
        if get_mats or get_all:
            if isinstance(get_mats, (list, tuple)):
                names = get_mats
            else:
                names = None
            dct.update(
                self.rdop2mats(names=names, verbose=verbose, lower=True, which=which)
            )

        for names, rdfunc in (
            (["LAMA*"], self._rdop2lama),
            (["DYNAMICS", "DYNAMIC"], self.rdop2dynamics),
            (["OGPWG*"], self._rdop2gpwg),
        ):
            self._rddatablock(names, rdfunc, verbose, dct, which=which)

        for flag, names, rdfunc in (
            (get_ougv1, ["OUGV1", "BOPHIG", "OUG1"], self._rdop2ougv1),
            (get_oef1, ["OEF1*"], self._rdop2drm),
            (get_oes1, ["OES1*"], self._rdop2drm),
        ):
            if flag or get_all:
                self._rddatablock(
                    names, rdfunc, verbose, dct, which=which, include_name=True
                )

        # read data recovery tables like tug1 and tef1
        if not isinstance(get_dr_tables, (list, tuple)):
            if get_dr_tables or get_all:
                get_dr_tables = ["T*"]

        if get_dr_tables:
            dct.update(
                self._rd_datarecovery_tables(get_dr_tables, verbose, which=which)
            )

        # some special se 0 data block handling for backward
        # compatibility:
        if 0 in geom1:
            for name in ("selist", "sebulk", "seload", "seconct"):
                if name in geom1[0]:
                    dct[name] = geom1[0][name]

        if "dynamics" in dct:
            if which == "all":
                dct["tload"] = dct["dynamics"][-1]
            else:
                dct["tload"] = dct["dynamics"]

        return dct

    def _rdop2maps(self, trailer, se):
        """
        This is a matrix with a single 1.0 in each row. Here (for now) we
        are only interested in the positions of the 1.0 values.

        It is assumed that the name has already been read in via
        :func:`rdop2nt`.

        The size of the matrix is read from trailer:
             rows = trailer[2]
             cols = trailer[1]
        """
        dtype = 1
        rows = trailer[2]
        mtype = trailer[4]
        if mtype > 2:  # complex
            rows *= 2
        if mtype & 1:  # single precision
            frmu = self._endian + self._f
            bytes_per = self._fbytes
        else:
            frmu = self._endian + "d"
            bytes_per = 8

        # maps matrix will be rows x 2, and each value in col 1 is the
        # column where the 1.0 goes for the corresponding row; the
        # second column is all 1.0's (or at least, that's the
        # expectation)
        matrix = np.zeros((rows, 2))
        intsize = self._ibytes
        col = 0
        while dtype > 0:  # read in matrix columns
            # key is number of elements in next record (row # followed
            # by key-1 real numbers)
            key = self._getkey()
            # read column
            while key > 0:
                reclen = self._Str4.unpack(self._fileh.read(4))[0]
                r = self._Str.unpack(self._fileh.read(intsize))[0] - 1
                if mtype > 2:
                    r *= 2
                n = (reclen - intsize) // bytes_per

                if n != 1:  # pragma: no cover
                    raise ValueError(
                        "expected 1 value in each column of MAPS matrix "
                        f"but for SE {se}, found {n} values in column {col}"
                    )

                matrix[r, 0] = col
                matrix[r, 1] = struct.unpack(frmu, self._fileh.read(bytes_per))[0]
                self._fileh.read(4)  # endrec
                key = self._getkey()
            col += 1
            self._getkey()
            dtype = self._getkey()
        self.rdop2eot()
        if mtype > 2:
            matrix.dtype = complex
        return matrix

    def _rdop2drm(self, name, oldstyle=False):
        """
        Read Nastran output2 DRM SORT1 data block (table).

        Parameters
        ----------
        name : string
            Name of data block.
        oldstyle : bool; optional
            If True, return 2-tuple: ``(drm, elem_info)``, otherwise
            return pandas.DataFrame where the index is element id and
            type.

        Returns
        -------
        drm : ndarray; optional
            The drm matrix. Returned if `oldstyle` is True.
        elem_info : ndarray; optional
            2-column matrix of [id, element_type]. Returned if
            `oldstyle` is True.
        drm_df : pandas.DataFrame; optional
            Returned if `oldstyle` is False (the default).

        Notes
        -----
        Reads OEF1 and OES1 type data blocks. This routine is beta --
        check output carefully.
        """
        # Expect IDENT/DATA record pairs. They repeat for each element
        # type for each mode.
        # This routine assumes all values are written, even the zeros.

        def _getdrm(pos, e, s, eids, etypes, ibytes):
            bytes_per_col = pos - s
            nrows = len(eids)
            ncols = (e - s) // bytes_per_col
            dtype = np.int32 if ibytes == 4 else np.int64
            drm = np.empty((nrows, ncols), dtype, order="F")
            elem_info = np.column_stack((eids, etypes))
            elem_info[:, 0] //= 10
            return drm, elem_info

        s = self._fileh.tell()
        _, e = self._get_block_bytes(name, s)

        # read first IDENT above loop & check ACODE/TCODE:
        ident = self.rdop2record()
        n_ident = len(ident)
        achk = self._check_code(ident[0], [4], [[2]], "ACODE")
        tchk = self._check_code(ident[1], [1, 7], [[1], [0, 2]], "TCODE")
        if not (achk and tchk):
            raise ValueError("invalid ACODE and/or TCODE value")

        eids = []
        etypes = []
        column = []
        drm = None
        n_data = []
        r = 0
        j = 0
        while ident is not None:
            elemtype = ident[2]
            mode = ident[4]
            nwords = ident[9]  # number of words/entry

            if mode == 1:
                # DATA record:
                data = self.rdop2record().reshape(-1, nwords).T
                n_data.append(data.size)
                pos = self._fileh.tell()
                z = np.zeros((nwords - 1, 1), data.dtype)
                eids.extend((data[0] + z).T.ravel())
                etypes.extend([elemtype] * data.shape[1] * (nwords - 1))
                column.extend(data[1:].T.ravel())
            else:
                # DATA record:
                data = self.rdop2record(N=n_data[j])
                data = data.reshape(-1, nwords).T
                if drm is None:
                    drm, elem_info = _getdrm(pos, e, s, eids, etypes, self._ibytes)
                    drm[:, mode - 2] = column
                n = (nwords - 1) * data.shape[1]
                drm[r : r + n, mode - 1] = data[1:].T.ravel()
                j += 1
                r += n
                if r == drm.shape[0]:
                    j = 0
                    r = 0

            # IDENT record:
            ident = self.rdop2record(N=n_ident)

        if drm is None:
            drm, elem_info = _getdrm(pos, e, s, eids, etypes, self._ibytes)
            drm[:, mode - 1] = column
        drm.dtype = np.float32 if self._ibytes == 4 else np.float64
        if oldstyle:
            return drm, elem_info

        indexm = pd.MultiIndex.from_tuples(
            (row for row in elem_info), names=("Eid", "Etype")
        )
        return pd.DataFrame(drm, index=indexm)

    def rddrm2op2(self, verbose=False):
        """
        Read op2 file output by DRM2 DMAP.

        Parameters
        ----------
        verbose : bool
            If true, echo names of tables and matrices to screen.

        Returns
        -------
        drmkeys : dictionary

            - 'dr' : data recovery items in order requested (from
              XYCDBDRS)
            - 'drs' : sorted version of 'dr' (from XYCDBDRS)
            - 'tougv1', 'tougs1', etc : directories corresponding to
              the data recovery matrices (which are written to op4).
              All of these start with 'to' (lower case).

        Notes
        -----
        File is created with a header and then these data blocks are
        written:

        .. code-block:: none

            OUTPUT2  XYCDBDRS//0/OP2UNIT $
            OUTPUT2  TOUGV1,TOUGS1,TOUGD1//0/OP2UNIT $
            OUTPUT2  TOQGS1,TOQGD1,TOEFS1,TOEFD1//0/OP2UNIT $
            OUTPUT2  TOESS1,TOESD1//0/OP2UNIT $

        """
        self._fileh.seek(self._postheaderpos)
        drmkeys = {}
        while 1:
            name, trailer, rectype = self.rdop2nt()
            if name is None:
                break
            if rectype > 0:
                if verbose:
                    print(f"Skipping matrix {name}...")
                self.goto_next()
                # self.skipop2matrix()
            elif len(name) > 2 and name.find("TO") == 0:
                if verbose:
                    print(f"Reading {name}...")
                # skip record 1
                self.rdop2record()
                # record 2 contains directory
                # - starting at 10: type, id, number, row, 0
                info = self.rdop2record()[10:]
                drmkeys[name.lower()] = (info.reshape(-1, 5).T)[:-1]
                self.rdop2eot()
            elif len(name) > 4 and name[:4] == "XYCD":
                if verbose:
                    print(f"Reading {name}...")
                # record 1 contains order of request info
                drmkeys["dr"] = self.rdop2record()
                # record 2 contains sorted list
                drmkeys["drs"] = self.rdop2record().reshape(-1, 6).T
                self.rdop2eot()
            else:
                if verbose:
                    print(f"Skipping table {name}...")
                self.goto_next()
        return drmkeys

    def rdn2cop2(self):
        """
        Read Nastran output2 file written by DMAP NAS2CAM; usually
        called by :func:`rdnas2cam`.

        Returns
        -------
        dictionary

        'selist' : array
            2-columns matrix: [ seid, dnseid ] where, for each row,
            dnseid is the downstream superelement for seid. (dnseid =
            0 if seid = 0).
        'uset' : dictionary
            Indexed by the SE number. Each member is a pandas
            DataFrame described below.
        'cstm' : dictionary
            Indexed by the SE number. Each member is a 14-column
            matrix containing the coordinate system transformation
            matrix for each coordinate system. See description below.
        'cstm2' : dictionary
            Indexed by the SE number. Each member is another
            dictionary indexed by the coordinate system id
            number. This has the same information as 'cstm', but in a
            different format. See description below.
        'maps' : dictionary
            Indexed by the SE number. Each member is a mapping table
            for mapping the A-set order from upstream to downstream;
            see below.
        'dnids' : dictionary
            Indexed by the SE number. Each member is a vector of ids
            of the A-set ids of grids and spoints for SE in the
            downstream superelement. (Does not have each DOF, just
            ids.) When using the CSUPER entry, these will be the ids
            on that entry. When using SECONCT entry, the ids are
            internally generated and will be a subset of the 'upids'
            entry for the downstream SE.
        'upids' : dictionary
            Indexed by the SE number. Each member is a vector of ids
            of the A-set grids and spoints for upstream se's that
            connect to SE. (Does not have each DOF, just ids.) This
            will only be present for SECONCT type SEs. These ids are
            internally generated and will contain all the values in
            the 'dnids' of each upstream SE. This allows, for example,
            the routine :func:`pyyeti.nastran.n2p.upasetpv` to
            work. Has 0's for downstream ids (in P-set) that are not
            part of upstream SEs.

        Notes
        -----
        The module `nastran` has many routines that use the data
        created by this routine.

        *'uset' description*

        Each `uset` variable is a 4-column pandas DataFrame with ID
        and DOF forming the row MultiIndex. The column index is:
        ``['nasset', 'x', 'y', 'z']``. The order of the degrees of
        freedom is in Nastran internal sort. GRIDs have 6 rows each
        and SPOINTs have 1 row. Here is an example `uset` variable
        with some notes for the ['x', 'y', 'z'] columns (described in
        more detail below)::

                     nasset    x    y    z
            id dof
            1  1    2097154  1.0  2.0  3.0    # grid location in basic
               2    2097154  0.0  1.0  0.0    # coord system info
               3    2097154  0.0  0.0  0.0    # coord system origin
               4    2097154  1.0  0.0  0.0    # | transform to basic
               5    2097154  0.0  1.0  0.0    # | for coord system
               6    2097154  0.0  0.0  1.0    # |
            2  0    4194304  0.0  0.0  0.0    # spoint

        That example was formed by using
        :func:`pyyeti.nastran.n2p.make_uset`::

            from pyyeti.nastran import n2p
            uset = n2p.make_uset(dof=[[1, 123456], [2, 0]],
                                 nasset=[n2p.mkusetmask('b'),
                                         n2p.mkusetmask('q')],
                                 xyz=[[1, 2, 3], [0, 0, 0]])

        The "nasset" column specifies Nastran set membership. It is a
        32-bit integer for each DOF. The integer has bits set to
        specify which Nastran set that particular DOF belongs to. For
        example, if the integer has the 1 bit set, the DOF is in the
        m-set. See the source code in
        :func:`pyyeti.nastran.n2p.mkusetmask` for all the bit
        positions. Note that you rarely (if ever) need to call
        :func:`pyyeti.nastran.n2p.mkusetmask` directly since the
        function :func:`pyyeti.nastran.n2p.mksetpv` does this behind
        the scenes to make partition vectors.

        For grids, the ['x', 'y', 'z'] part of the DataFrame is a 6
        row by 3 column matrix::

            Coord_Info = [[x   y    z]   # location of node in basic
                          [id  type 0]   # coord. id and type
                          [xo  yo  zo]   # origin of coord. system
                          [    T     ]]  # 3x3 transformation to basic
                                         #  for coordinate system

        For spoints, the ['x', 'y', 'z'] part of the DataFrame is a 1
        row by 3 column matrix::

            Coord_Info = [0.0, 0.0, 0.0]

        *'cstm' description*

        Each `cstm` contains all the coordinate system information for
        the superelement. Some or all of this info is in the `uset`
        table, but if a coordinate system is not used as an output
        system of any grid, it will not show up in `uset`. That is why
        `cstm` is here. `cstm` has 14 columns::

            cstm = [ id type xo yo zo T(1,:) T(2,:) T(3,:) ]

        Note that each `cstm` always starts with the two ids 0 and -1.
        The 0 is the basic coordinate system and the -1 is a dummy for
        SPOINTs. Note the T is transformation between coordinate
        systems as defined (not necessarily the same as the
        transformation for a particular grid ... which, for
        cylindrical and spherical, depends on grid location). This is
        the same T as in the `uset` table.

        For example, to convert coordinates from global to basic::

          Rectangular (type = 1):
             [x; y; z] = T*[xg; yg; zg] + [xo; yo; zo]

          Cylindrical (type = 2):
             % c = cos(theta); s = sin(theta)
             [x; y; z] = T*[R c; R s; zg] + [xo; yo; zo]

          Spherical (type = 3):
             % s1 = sin(theta); s2 = sin(phi)
             [x; y; z] = T*[r s1 c2; r s1 s2; r c1] + [xo; yo; zo]

        *'cstm2' description*

        Each `cstm2` is a dictionary with the same 5x3 that the
        'Coord_Info' listed above has (doesn't include the first row
        which is the node location). The dictionary is indexed by the
        coordinate id.

        *'maps' description*

        `maps` will be [] for superelements whose A-set dof did not
        get rearranged going downstream (on the CSUPER entry.)  For
        other superelements, `maps` will contain two columns: [order,
        scale].  The first column reorders upstream A-set to be in the
        order that they appear in the downstream:
        ``down = up[maps[:, 0]]``. The second column is typically 1.0;
        if not, these routines will print an error message and
        stop. Together with `dnids`, a partition vector can be formed
        for the A-set of an upstream superelement (see
        :func:`pyyeti.nastran.n2p.upasetpv`).

        The op2 file that this routine reads is written by the Nastran
        DMAP NAS2CAM. The data in the file are expected to be in this
        order::

            SLIST & EMAP or  SUPERID
            For each superelement:
              USET
              EQEXINS
              CSTMS    (if required)
              BGPDTS
              MAPS     (if required)

        Note: The 2nd bit for the DOF column of all `uset` tables is
        cleared for all S-set. See
        :func:`pyyeti.nastran.n2p.mkusetmask` for more information.

        Example usage::

            from pyyeti import nastran
            # list superelement 100 DOF that are in the B set:
            o2 = nastran.OP2('nas2cam.op2')
            nas = nastran.rdn2cop2()
            bset = nastran.mksetpv(nas['uset'][100], 'p', 'b')
            print('bset of se100 =', nas['uset'][100][bset, :2])

        See also
        --------
        :func:`rdnas2cam`, :func:`pyyeti.nastran.bulk.bulk2uset`.
        """
        # setup basic coordinate system info and a dummy for spoints:
        bc = np.array(
            [
                [+0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        nas = {
            "uset": {},
            "cstm": {},
            "cstm2": {},
            "maps": {},
            "dnids": {},
            "upids": {},
        }
        self._fileh.seek(self._postheaderpos)
        # read data block (slist) header record:
        name, trailer, dbtype = self.rdop2nt()
        if name not in ("SUPERID", "SLIST"):
            raise ValueError(
                "expected 'SUPERID' or 'SLIST' as first data "
                f"block in nas2cam op2 file, but got {name}"
            )
        if dbtype > 0:
            selist = np.hstack((self.rdop2matrix(trailer), [[0]]))
            selist = selist.astype(int)
            name, trailer, dbtype = self.rdop2nt()
        else:
            selist = self._rdop2selist()
            nse = np.size(selist, 0)
            name, trailer, dbtype = self.rdop2nt()
            if name == "EMAP":
                self._rdop2emap(nas, nse, trailer)
                name, trailer, dbtype = self.rdop2nt()
            elif name == "DNIDS":
                self._rdop2dnids(nas, nse, trailer)
                name, trailer, dbtype = self.rdop2nt()

        # read uset and eqexins tables and do some processing:
        for se in selist[:, 0]:
            if not name:
                break
            uset = self._rdop2uset()
            name, trailer, dbtype = self.rdop2nt()
            eqexin1, eqexin = self._rdop2eqexin()
            name, trailer, dbtype = self.rdop2nt()
            if name == "CSTMS":
                cstm = np.vstack((bc, self._rdop2cstm()))
                name, trailer, dbtype = self.rdop2nt()
            else:
                cstm = bc
            (xyz, cid, dof, doftype, nid, upids) = self._proc_bgpdt(eqexin1, eqexin)
            nas["upids"][se] = upids
            Uset, cstm, cstm2 = self._buildUset(
                se,
                dof,
                doftype,
                nid,
                uset,
                xyz,
                cid,
                cstm,
                None,
                print_warning=True,
            )
            nas["uset"][se] = Uset
            nas["cstm"][se] = cstm
            nas["cstm2"][se] = cstm2
            name, trailer, dbtype = self.rdop2nt()
            if name == "MAPS":
                nas["maps"][se] = self._rdop2maps(trailer, se)
                name, trailer, dbtype = self.rdop2nt()
            else:
                nas["maps"][se] = []
        nas["selist"] = selist
        return nas


def rdmats(filename=None, names=None, lower=False, verbose=False):
    """
    Read all matrices from Nastran output2 file.

    Parameters
    ----------
    filename : string or None; optional
        Name of op2 file to read. Can also be the name of a directory
        or None; in these cases, a GUI is opened for file selection.
    names : list_like; optional
        Iterable of names to read in. If None, read all. These can
        be input in lower case.
    lower : bool; optional
        If True, returned names will be in lower case.
    verbose : bool; optional
        If True, print matrix names to screen as they are found.

    Returns
    -------
    dict
        Dictionary containing all matrices in the op2 file:
        {'NAME1': matrix1, 'NAME2': matrix2, ...}

    Notes
    -----
    This routine is for convenience; this is what it does::

        filename = guitools.get_file_name(filename, read=True)
        return OP2(filename).rdop2mats(
            names, lower=lower, verbose=verbose
        )
    """
    filename = guitools.get_file_name(filename, read=True)
    return OP2(filename).rdop2mats(names, lower=lower, verbose=verbose)


def _get_op2_op4(op2file, op4file):
    if op2file is None:  # pragma: no cover
        op2file = guitools.get_file_name(None, read=True)
    else:
        op2file = os.fspath(op2file)
        if not os.path.exists(op2file):
            op2file = op2file + ".op2"
        if not op4file:
            op4file = op2file.replace(".op2", ".op4")
    return op2file, op4file


def rdnas2cam(op2file="nas2cam", op4file=None):
    """
    Read op2/op4 data written by the DMAP NAS2CAM.

    Parameters
    ----------
    op2file : string or None
        Either the basename of the .op2 and .op4 files, or the full
        name of the .op2 file. If None, a GUI is opened for file
        selection.
    op4file : string or None
        The name of the .op4 file or, if None, builds name from the
        `op2file` input.

    Returns
    -------
    nas : dictionary
        Dictionary with all members created by :func:`OP2.rdn2cop2`
        (see that routine's help) and the following additional
        members.

    'nrb' : integer
        The number of rigid-body modes for residual.
    'ulvs' : dictionary indexed by SE
        The ULVS matrices (row partitions of residual modes to the
        A-set DOF of the SE).
    'lambda' : dictionary indexed by SE
        The eigenvalues for each SE.
    'gm' : dictionary indexed by SE
        N-set to M-set transformation matrix GM:  M = GM N.
    'got' : dictionary indexed by SE
        constraint modes
    'goq' : dictionary indexed by SE
        normal modes
    'rfmodes' : dictionary indexed by SE
        index partition vector for res-flex modes
    'maa' : dictionary indexed by SE
        A-set mass
    'baa' : dictionary indexed by SE
        A-set damping
    'kaa' : dictionary indexed by SE
        A-set stiffness
    'pha' : dictionary indexed by SE
        A-set modes
    'mdd' : dictionary indexed by SE
        D-set mass
    'bdd' : dictionary indexed by SE
        D-set damping
    'kdd' : dictionary indexed by SE
        D-set stiffness
    'pdt' : dictionary indexed by SE
        D-set loads
    'mgg' : dictionary indexed by SE
        G-set mass
    'kgg' : dictionary indexed by SE
        G-set stiffness
    'phg' : dictionary indexed by SE
        G-set mode shape matrix
    'rbg' : dictionary indexed by SE
        G-set rigid-body modes; see also drg output and rbgeom_uset
    'drg' : dictionary indexed by SE
        G-set transpose of rigid-body modes; see also 'rbg' and
        :func:`pyyeti.nastran.n2p.rbgeom_uset`. `drg` = `rbg.T` if
        both are present.
    'pg' : dictionary indexed by SE
        G-set loads
    'fgravh' : array
        gravity on generalized dof for se 0
    'fgravg' : array
        gravity on G-set physical dof for se 0

    Notes
    -----
    See :func:`OP2.rdn2cop2` for a description of what is expected of
    the `op2file`. The `op4file` is expected to contain certain
    marker matrices. Scalar SE_START starts each superelement and can
    be followed by any matrices for that superelement. The end of the
    superelement input is marked by a matrix named LOOP_END.

    See also the Nastran DMAP NAS2CAM.
    """
    op2file, op4file = _get_op2_op4(op2file, op4file)

    # read op2 file:
    with OP2(op2file) as o2:
        nas = o2.rdn2cop2()

    # read op4 file:
    op4names, op4vars = op4.load(op4file, into="list")[:2]

    # loop over superelements:
    j = 0
    for se in nas["selist"][:, 0]:
        if op4names[j] != "se_start":
            raise ValueError(
                "matrices are not in understandable order. Expected 'se_start', got "
                f"'{op4names[j]}'"
            )
        # read all matrices for this se
        j += 1
        while 1:
            name = op4names[j]
            if name == "loop_end" or name == "se_start":
                # go on to next se or to residual
                break
            if name not in nas:
                nas[name] = {}
            if se == 0 and name == "lambda":
                # count number of rigid body modes
                nrb = sum(op4vars[j] < 0.005)[0]
                nas["nrb"] = nrb
                nas["lambda"][0] = abs(op4vars[j].ravel())
            elif name == "lambda":
                nas[name][se] = op4vars[j].ravel()
            elif name == "rfmodes":
                nas[name][se] = np.nonzero(op4vars[j])[0]
            else:
                nas[name][se] = op4vars[j]
            j += 1
        if name == "loop_end":
            j += 1
            break
    while j < len(op4vars):
        nas[op4names[j]] = op4vars[j]
        j += 1
    return nas


def nastran_dr_descriptions():
    """
    Get dictionary of descriptions for Nastran data recovery items.

    Normally called by :func:`procdrm12`.

    Returns
    -------
    desc : dictionary
        Has keys: 'acce', 'spcf', 'force', 'stress':

        .. code-block:: none

            desc['acce'] : numpy string array
                ['T1', 'T2', 'T3',  'R1', 'R2', 'R3']
            desc['spcf'] : numpy string array
                ['Fx', 'Fy', 'Fz',  'Mx', 'My', 'Mz']
            desc['force'] : dict
                Dictionary with element numbers as keys to numpy
                string arrays.
            desc['stress'] : dict
                Dictionary with element numbers as keys to numpy
                string arrays.

    Notes
    -----
    The force and stress dictionaries are indexed by the element
    id. For example, for the CBAR (which is element 34)::

        desc['force'][34] = ['CBAR Bending Moment 1 - End A',
                             'CBAR Bending Moment 2 - End A',
                             ...]
        desc['stress'][34] = ['CBAR Bending Stress 1 - End A',
                              'CBAR Bending Stress 2 - End A',
                              ...]
    """
    #   Acceleration, Velocity, Displacement Recovery Items:
    accedesc = ["T1", "T2", "T3", "R1", "R2", "R3"]
    spcfdesc = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    stress = {}
    force = {}

    #  CBAR Recovery Items (element 34):                Item code
    stress[34] = [
        "CBAR Bending Stress 1 - End A",  # 2
        "CBAR Bending Stress 2 - End A",  # 3
        "CBAR Bending Stress 3 - End A",  # 4
        "CBAR Bending Stress 4 - End A",  # 5
        "CBAR Axial Stress",  # 6
        "CBAR Max. Bend. Stress -End A",  # 7
        "CBAR Min. Bend. Stress -End A",  # 8
        "CBAR M.S. Tension",  # 9
        "CBAR Bending Stress 1 - End B",  # 10
        "CBAR Bending Stress 2 - End B",  # 11
        "CBAR Bending Stress 3 - End B",  # 12
        "CBAR Bending Stress 4 - End B",  # 13
        "CBAR Max. Bend. Stress -End B",  # 14
        "CBAR Min. Bend. Stress -End B",  # 15
        "CBAR M.S. Compression",
    ]  # 16

    force[34] = [
        "CBAR Bending Moment 1 - End A",  # 2
        "CBAR Bending Moment 2 - End A",  # 3
        "CBAR Bending Moment 1 - End B",  # 4
        "CBAR Bending Moment 2 - End B",  # 5
        "CBAR Shear 1",  # 6
        "CBAR Shear 2",  # 7
        "CBAR Axial Force",  # 8
        "CBAR Torque",
    ]  # 9

    #   CBEAM Recovery Items (element 2):               Item code
    stress2_main = [
        "CBEAM External grid pt. ID",  # 2
        "CBEAM Station dist./length",  # 3
        "CBEAM Long. Stress at Pt. C",  # 4
        "CBEAM Long. Stress at Pt. D",  # 5
        "CBEAM Long. Stress at Pt. E",  # 6
        "CBEAM Long. Stress at Pt. F",  # 7
        "CBEAM Maximum stress",  # 8
        "CBEAM Minimum stress",  # 9
        "CBEAM M.S. Tension",  # 10
        "CBEAM M.S. Compression",
    ]  # 11

    # expand and append station id for all 11 stations:
    stress2 = [i + " End-A" for i in stress2_main]
    for K in range(2, 11):
        id_string = f" K={K:2}"
        stress2 += [i + id_string for i in stress2_main]
    stress2 += [i + " End-B" for i in stress2_main]
    stress[2] = stress2

    force2_main = [
        "CBEAM External grid pt. ID",  # 2
        "CBEAM Station dist./length",  # 3
        "CBEAM Bending moment plane 1",  # 4
        "CBEAM Bending moment plane 2",  # 5
        "CBEAM Web shear plane 1",  # 6
        "CBEAM Web shear plane 2",  # 7
        "CBEAM Axial force",  # 8
        "CBEAM Total torque",  # 9
        "CBEAM Warping torque",
    ]  # 10

    # expand and append station id for all 11 stations:
    force2 = [i + " End-A" for i in force2_main]
    for K in range(2, 11):
        id_string = f" K={K:2}"
        force2 += [i + id_string for i in force2_main]
    force2 += [i + " End-B" for i in force2_main]
    force[2] = force2

    #   CBUSH Recovery Items (element 102):             Item code
    stress[102] = [
        "CBUSH Translation-x",  # 2
        "CBUSH Translation-y",  # 3
        "CBUSH Translation-z",  # 4
        "CBUSH Rotation-x",  # 5
        "CBUSH Rotation-y",  # 6
        "CBUSH Rotation-z",
    ]  # 7

    force[102] = [
        "CBUSH Force-x",  # 2
        "CBUSH Force-y",  # 3
        "CBUSH Force-z",  # 4
        "CBUSH Moment-x",  # 5
        "CBUSH Moment-y",  # 6
        "CBUSH Moment-z",
    ]  # 7

    #   CROD Recovery Items (element 10=CONROD, 1=CROD):
    stress1 = [
        "Axial Stress",  # 2
        "M.S. Axial Stress",  # 3
        "Torsional Stress",  # 4
        "M.S. Torsional Stress",
    ]  # 5
    force1 = ["Axial Force", "Torque"]  # 2  # 3
    stress[1] = ["CROD " + i + "  " for i in stress1]
    force[1] = ["CROD " + i + "  " for i in force1]
    stress[10] = ["CONROD " + i for i in stress1]
    force[10] = ["CONROD " + i for i in force1]

    #   CELAS1, 2, 3 Recovery Items (elements 11, 12, 13):
    stress[11] = "CELAS1 Stress"
    stress[12] = "CELAS2 Stress"
    stress[13] = "CELAS3 Stress"
    force[11] = "CELAS1 Force"
    force[12] = "CELAS2 Force"
    force[13] = "CELAS3 Force"

    #   CQUAD4 Recovery Items (element 33):
    stress[33] = [
        "CQUAD4 Fiber distance Z1",  # 2
        "CQUAD4 Z1 Normal x",  # 3
        "CQUAD4 Z1 Normal y",  # 4
        "CQUAD4 Z1 Shear xy",  # 5
        "CQUAD4 Z1 Shear angle",  # 6
        "CQUAD4 Z1 Major principal",  # 7
        "CQUAD4 Z1 Minor principal",  # 8
        "CQUAD4 Z1 von Mises or max shear",  # 9
        "CQUAD4 Fiber distance Z2",  # 10
        "CQUAD4 Z2 Normal x",  # 11
        "CQUAD4 Z2 Normal y",  # 12
        "CQUAD4 Z2 Shear xy",  # 13
        "CQUAD4 Z2 Shear angle",  # 14
        "CQUAD4 Z2 Major principal",  # 15
        "CQUAD4 Z2 Minor principal",  # 16
        "CQUAD4 Z2 von Mises or max shear",
    ]  # 17

    force[33] = [
        "CQUAD4 Membrane force x",  # 2
        "CQUAD4 Membrane force y",  # 3
        "CQUAD4 Membrane force xy",  # 4
        "CQUAD4 Bending moment x",  # 5
        "CQUAD4 Bending moment y",  # 6
        "CQUAD4 Bending moment xy",  # 7
        "CQUAD4 Shear x",  # 8
        "CQUAD4 Shear y",
    ]  # 9

    #   CQUADR Recovery Items (element 82, and CQUAD8-64):
    stress[82] = [
        "CQUADR EID                         ",  # 1
        "CQUADR CEN/                        ",  # 2
        "CQUADR 4                           ",  # 3
        "CQUADR Fiber distance Z1           ",  # 4
        "CQUADR Z1 Normal x                 ",  # 5
        "CQUADR Z1 Normal y                 ",  # 6
        "CQUADR Z1 Shear xy                 ",  # 7
        "CQUADR Z1 Shear angle              ",  # 8
        "CQUADR Z1 Major principal          ",  # 9
        "CQUADR Z1 Minor principal          ",  # 10
        "CQUADR Z1 von Mises or max shear   ",  # 11
        "CQUADR Fiber distance Z2           ",  # 12
        "CQUADR Z2 Normal x                 ",  # 13
        "CQUADR Z2 Normal y                 ",  # 14
        "CQUADR Z2 Shear xy                 ",  # 15
        "CQUADR Z2 Shear angle              ",  # 16
        "CQUADR Z2 Major principal          ",  # 17
        "CQUADR Z2 Minor principal          ",  # 18
        "CQUADR Z2 von Mises or max shear   ",  # 19
        "CQUADR Grid 1                      ",  # 20
        "CQUADR Fiber distance Z1         c1",  # 21
        "CQUADR Z1 Normal x               c1",  # 22
        "CQUADR Z1 Normal y               c1",  # 23
        "CQUADR Z1 Shear xy               c1",  # 24
        "CQUADR Z1 Shear angle            c1",  # 25
        "CQUADR Z1 Major principal        c1",  # 26
        "CQUADR Z1 Minor principal        c1",  # 27
        "CQUADR Z1 von Mises or max shear c1",  # 28
        "CQUADR Fiber distance Z2         c1",  # 29
        "CQUADR Z2 Normal x               c1",  # 30
        "CQUADR Z2 Normal y               c1",  # 31
        "CQUADR Z2 Shear xy               c1",  # 32
        "CQUADR Z2 Shear angle            c1",  # 33
        "CQUADR Z2 Major principal        c1",  # 34
        "CQUADR Z2 Minor principal        c1",  # 35
        "CQUADR Z2 von Mises or max shear c1",  # 36
        "CQUADR Grid 2                      ",  # 37
        "CQUADR Fiber distance Z1         c2",  # 38
        "CQUADR Z1 Normal x               c2",  # 39
        "CQUADR Z1 Normal y               c2",  # 40
        "CQUADR Z1 Shear xy               c2",  # 41
        "CQUADR Z1 Shear angle            c2",  # 42
        "CQUADR Z1 Major principal        c2",  # 43
        "CQUADR Z1 Minor principal        c2",  # 44
        "CQUADR Z1 von Mises or max shear c2",  # 45
        "CQUADR Fiber distance Z2         c2",  # 46
        "CQUADR Z2 Normal x               c2",  # 47
        "CQUADR Z2 Normal y               c2",  # 48
        "CQUADR Z2 Shear xy               c2",  # 49
        "CQUADR Z2 Shear angle            c2",  # 50
        "CQUADR Z2 Major principal        c2",  # 51
        "CQUADR Z2 Minor principal        c2",  # 52
        "CQUADR Z2 von Mises or max shear c2",  # 53
        "CQUADR Grid 3                      ",  # 54
        "CQUADR Fiber distance Z1         c3",  # 55
        "CQUADR Z1 Normal x               c3",  # 56
        "CQUADR Z1 Normal y               c3",  # 57
        "CQUADR Z1 Shear xy               c3",  # 58
        "CQUADR Z1 Shear angle            c3",  # 59
        "CQUADR Z1 Major principal        c3",  # 60
        "CQUADR Z1 Minor principal        c3",  # 61
        "CQUADR Z1 von Mises or max shear c3",  # 62
        "CQUADR Fiber distance Z2         c3",  # 63
        "CQUADR Z2 Normal x               c3",  # 64
        "CQUADR Z2 Normal y               c3",  # 65
        "CQUADR Z2 Shear xy               c3",  # 66
        "CQUADR Z2 Shear angle            c3",  # 67
        "CQUADR Z2 Major principal        c3",  # 68
        "CQUADR Z2 Minor principal        c3",  # 69
        "CQUADR Z2 von Mises or max shear c3",  # 70
        "CQUADR Grid 4                      ",  # 71
        "CQUADR Fiber distance Z1         c4",  # 72
        "CQUADR Z1 Normal x               c4",  # 73
        "CQUADR Z1 Normal y               c4",  # 74
        "CQUADR Z1 Shear xy               c4",  # 75
        "CQUADR Z1 Shear angle            c4",  # 76
        "CQUADR Z1 Major principal        c4",  # 77
        "CQUADR Z1 Minor principal        c4",  # 78
        "CQUADR Z1 von Mises or max shear c4",  # 79
        "CQUADR Fiber distance Z2         c4",  # 80
        "CQUADR Z2 Normal x               c4",  # 81
        "CQUADR Z2 Normal y               c4",  # 82
        "CQUADR Z2 Shear xy               c4",  # 83
        "CQUADR Z2 Shear angle            c4",  # 84
        "CQUADR Z2 Major principal        c4",  # 85
        "CQUADR Z2 Minor principal        c4",  # 86
        "CQUADR Z2 von Mises or max shear c4",
    ]  # 87

    force[82] = [
        "CQUADR Membrane force x            ",  # 4
        "CQUADR Membrane force y            ",  # 5
        "CQUADR Membrane force xy           ",  # 6
        "CQUADR Bending moment x            ",  # 7
        "CQUADR Bending moment y            ",  # 8
        "CQUADR Bending moment xy           ",  # 9
        "CQUADR Shear x                     ",  # 10
        "CQUADR Shear y                     ",  # 11
        "CQUADR   (non-documented item)     ",  # 12
        "CQUADR Membrane force x          c1",  # 13
        "CQUADR Membrane force y          c1",  # 14
        "CQUADR Membrane force xy         c1",  # 15
        "CQUADR Bending moment x          c1",  # 16
        "CQUADR Bending moment y          c1",  # 17
        "CQUADR Bending moment xy         c1",  # 18
        "CQUADR Shear x                   c1",  # 19
        "CQUADR Shear y                   c1",  # 20
        "CQUADR   (non-documented item)     ",  # 21
        "CQUADR Membrane force x          c2",  # 22
        "CQUADR Membrane force y          c2",  # 23
        "CQUADR Membrane force xy         c2",  # 24
        "CQUADR Bending moment x          c2",  # 25
        "CQUADR Bending moment y          c2",  # 26
        "CQUADR Bending moment xy         c2",  # 27
        "CQUADR Shear x                   c2",  # 28
        "CQUADR Shear y                   c2",  # 29
        "CQUADR   (non-documented item)     ",  # 30
        "CQUADR Membrane force x          c3",  # 31
        "CQUADR Membrane force y          c3",  # 32
        "CQUADR Membrane force xy         c3",  # 33
        "CQUADR Bending moment x          c3",  # 34
        "CQUADR Bending moment y          c3",  # 35
        "CQUADR Bending moment xy         c3",  # 36
        "CQUADR Shear x                   c3",  # 37
        "CQUADR Shear y                   c3",  # 38
        "CQUADR   (non-documented item)     ",  # 39
        "CQUADR Membrane force x          c4",  # 40
        "CQUADR Membrane force y          c4",  # 41
        "CQUADR Membrane force xy         c4",  # 42
        "CQUADR Bending moment x          c4",  # 43
        "CQUADR Bending moment y          c4",  # 44
        "CQUADR Bending moment xy         c4",  # 45
        "CQUADR Shear x                   c4",  # 46
        "CQUADR Shear y                   c4",
    ]  # 47
    stress[64] = [i.replace("CQUADR", "CQ8-64") for i in stress[82]]
    force[64] = [i.replace("CQUADR", "CQ8-64") for i in force[82]]

    #   CTRIAR Recovery Items (element 70, and CTRIA6-75):
    stress[70] = [
        "CTRIAR Z1 Normal x                 ",  # 5
        "CTRIAR Z1 Normal y                 ",  # 6
        "CTRIAR Z1 Shear xy                 ",  # 7
        "CTRIAR Z1 Q shear angle            ",  # 8
        "CTRIAR Z1 Major principal          ",  # 9
        "CTRIAR Z1 Minor principal          ",  # 10
        "CTRIAR Z1 von Mises or max shear   ",  # 11
        "CTRIAR   (non-documented item)     ",  # 12
        "CTRIAR Z2 Normal x                 ",  # 13
        "CTRIAR Z2 Normal y                 ",  # 14
        "CTRIAR Z2 Shear xy                 ",  # 15
        "CTRIAR Z2 Q shear angle            ",  # 16
        "CTRIAR Z2 Major principal          ",  # 17
        "CTRIAR Z2 Minor principal          ",  # 18
        "CTRIAR Z2 von Mises or max shear   ",  # 19
        "CTRIAR   (non-documented item)     ",  # 20
        "CTRIAR   (non-documented item)     ",  # 21
        "CTRIAR Z1 Normal x               c1",  # 22
        "CTRIAR Z1 Normal y               c1",  # 23
        "CTRIAR Z1 Shear xy               c1",  # 24
        "CTRIAR Z1 Q shear angle          c1",  # 25
        "CTRIAR Z1 Major principal        c1",  # 26
        "CTRIAR Z1 Minor principal        c1",  # 27
        "CTRIAR Z1 von Mises or max shear c1",  # 28
        "CTRIAR   (non-documented item)   c1",  # 29
        "CTRIAR Z2 Normal x               c1",  # 30
        "CTRIAR Z2 Normal y               c1",  # 31
        "CTRIAR Z2 Shear xy               c1",  # 32
        "CTRIAR Z2 Q shear angle          c1",  # 33
        "CTRIAR Z2 Major principal        c1",  # 34
        "CTRIAR Z2 Minor principal        c1",  # 35
        "CTRIAR Z2 von Mises or max shear c1",  # 36
        "CTRIAR   (non-documented item)     ",  # 37
        "CTRIAR   (non-documented item)     ",  # 38
        "CTRIAR Z1 Normal x               c2",  # 39
        "CTRIAR Z1 Normal y               c2",  # 40
        "CTRIAR Z1 Shear xy               c2",  # 41
        "CTRIAR Z1 Q shear angle          c2",  # 42
        "CTRIAR Z1 Major principal        c2",  # 43
        "CTRIAR Z1 Minor principal        c2",  # 44
        "CTRIAR Z1 von Mises or max shear c2",  # 45
        "CTRIAR   (non-documented item)   c2",  # 46
        "CTRIAR Z2 Normal x               c2",  # 47
        "CTRIAR Z2 Normal y               c2",  # 48
        "CTRIAR Z2 Shear xy               c2",  # 49
        "CTRIAR Z2 Q shear angle          c2",  # 50
        "CTRIAR Z2 Major principal        c2",  # 51
        "CTRIAR Z2 Minor principal        c2",  # 52
        "CTRIAR Z2 von Mises or max shear c2",  # 53
        "CTRIAR   (non-documented item)     ",  # 54
        "CTRIAR   (non-documented item)     ",  # 55
        "CTRIAR Z1 Normal x               c3",  # 56
        "CTRIAR Z1 Normal y               c3",  # 57
        "CTRIAR Z1 Shear xy               c3",  # 58
        "CTRIAR Z1 Q shear angle          c3",  # 59
        "CTRIAR Z1 Major principal        c3",  # 60
        "CTRIAR Z1 Minor principal        c3",  # 61
        "CTRIAR Z1 von Mises or max shear c3",  # 62
        "CTRIAR   (non-documented item)   c3",  # 63
        "CTRIAR Z2 Normal x               c3",  # 64
        "CTRIAR Z2 Normal y               c3",  # 65
        "CTRIAR Z2 Shear xy               c3",  # 66
        "CTRIAR Z2 Q shear angle          c3",  # 67
        "CTRIAR Z2 Major principal        c3",  # 68
        "CTRIAR Z2 Minor principal        c3",  # 69
        "CTRIAR Z2 von Mises or max shear c3",
    ]  # 70

    force[70] = [
        "CTRIAR Membrane force x            ",  # 4
        "CTRIAR Membrane force y            ",  # 5
        "CTRIAR Membrane force xy           ",  # 6
        "CTRIAR Bending moment x            ",  # 7
        "CTRIAR Bending moment y            ",  # 8
        "CTRIAR Bending moment xy           ",  # 9
        "CTRIAR Shear x                     ",  # 10
        "CTRIAR Shear y                     ",  # 11
        "CTRIAR   (non-documented item)     ",  # 12
        "CTRIAR Membrane force x          c1",  # 13
        "CTRIAR Membrane force y          c1",  # 14
        "CTRIAR Membrane force xy         c1",  # 15
        "CTRIAR Bending moment x          c1",  # 16
        "CTRIAR Bending moment y          c1",  # 17
        "CTRIAR Bending moment xy         c1",  # 18
        "CTRIAR Shear x                   c1",  # 19
        "CTRIAR Shear y                   c1",  # 20
        "CTRIAR   (non-documented item)     ",  # 21
        "CTRIAR Membrane force x          c2",  # 22
        "CTRIAR Membrane force y          c2",  # 23
        "CTRIAR Membrane force xy         c2",  # 24
        "CTRIAR Bending moment x          c2",  # 25
        "CTRIAR Bending moment y          c2",  # 26
        "CTRIAR Bending moment xy         c2",  # 27
        "CTRIAR Shear x                   c2",  # 28
        "CTRIAR Shear y                   c2",  # 29
        "CTRIAR   (non-documented item)     ",  # 30
        "CTRIAR Membrane force x          c3",  # 31
        "CTRIAR Membrane force y          c3",  # 32
        "CTRIAR Membrane force xy         c3",  # 33
        "CTRIAR Bending moment x          c3",  # 34
        "CTRIAR Bending moment y          c3",  # 35
        "CTRIAR Bending moment xy         c3",  # 36
        "CTRIAR Shear x                   c3",  # 37
        "CTRIAR Shear y                   c3",
    ]  # 38

    stress[75] = [i.replace("CTRIAR", "CT6-75") for i in stress[70]]
    force[75] = [i.replace("CTRIAR", "CT6-75") for i in force[70]]
    for i in stress:
        stress[i] = np.array(stress[i])
        force[i] = np.array(force[i])
    return {
        "acce": np.array(accedesc),
        "spcf": np.array(spcfdesc),
        "stress": stress,
        "force": force,
    }


def _get_tinr(iddof, idj):
    """
    Called by get_drm.

    Parameters
    ----------
    iddof : 2d array
        Each col has [type, id, number of rows, start row]
    idj : integer
        Id to return info for.

    Returns tuple of (type, start row)

    Note: start row return value starts at 0, not at 1.
    """
    i = np.nonzero(iddof[1] == idj)[0]
    tinr = iddof[:, i]
    return tinr[0, 0], tinr[3, 0] - 1


def _get_drm(drminfo, otm, drms, drmkeys, dr, desc):
    """
    Called by :func:`procdrm12` to add displacement-dependent data
    recovery items to the otm input.

    Parameters
    ----------
    drminfo : tuple
        DRM Information; (output drm name, 3 or 5 character Nastran
        name, description index).

        - The first element of tuple is used to name the output (which
          is put in `otm`)

        - If the second element of tuple is 3 chars, say '---', this
          routine uses the following members of `drms` and `drmkeys`::

            'm---d1', 'm---s1' and 't---d1' if available (mode-acce)

            or:

            'm---x1', 't---x1' if not (mode-disp)

        - If the second element of tuple is 5 chars, say '-----', this
          routine uses 'm-----' and 't-----'

        - The "description index is" used to get info from `desc`.

    otm : input/output dictionary
        Filled in with 'DTM' (or 'DTMA', 'DTMD') and 'DTM_id_dof',
        'DTM_desc'.
    drms : dictionary
        Contains all drms from op4 file.
    drmkeys : dictionary
        Contains the keys (directories) to the drms.
    dr : array
        Matrix 3 x number of data recovery items: [type; id; dof].
        Type is 1 for displacements.
    desc : dictionary
        Output of :func:`nastran_dr_descriptions`.

    Notes
    -----
    Example usages::

        _get_drm(('DTM', 'oug', 'acce'), otm, drms, drmkeys, dr, desc)
        _get_drm(('ATM', 'ougv1', 'acce'), ...)
        _get_drm(('LTM', 'oef', 'force'), ...)
        _get_drm(('SPCF', 'oqg', 'spcf'), ...)
        _get_drm(('STM', 'oes', 'stress'), ...)
    """
    drc = dr.shape[1]
    ID = dr[1, :]
    DOF = dr[2, :]
    nm, nasnm, desci = drminfo
    otm[nm + "_id_dof"] = np.vstack((ID, DOF)).T

    # arg offset is for translating between Nastran argument to
    # matrix index; eg 'x' recovery for a grid is arg 3, so offset
    # is 3
    if nasnm.find("oug") > -1 or nasnm.find("oqg") > -1:
        offset = 3
        otm[nm + "_id_dof"][:, 1] -= 2
    else:
        offset = 2

    if not isinstance(desc[desci], dict):
        otm[nm + "_desc"] = desc[desci][DOF - offset]
        getdesc = False
    else:
        getdesc = True
        _desc = nm + "_desc"
        otm[_desc] = [""] * drc
        _dct = desc[desci]
        _name = desci.capitalize()

    if len(nasnm) == 3 and "m" + nasnm + "d1" in drms:
        d1 = drms["m" + nasnm + "d1"]
        s1 = drms["m" + nasnm + "s1"]
        iddof = drmkeys["t" + nasnm + "d1"]
        acce = nm + "A"
        disp = nm + "D"
        otm[acce] = np.zeros((drc, d1.shape[1]))
        otm[disp] = np.zeros((drc, s1.shape[1]))
        lastid = -1
        for j in range(drc):  # loop over requests
            # find rows corresponding to requested grid
            if ID[j] != lastid:
                eltype, srow = _get_tinr(iddof, ID[j])
                lastid = ID[j]
            otm[acce][j] = d1[srow + DOF[j] - offset]
            otm[disp][j] = s1[srow + DOF[j] - offset]
            if getdesc:
                if eltype in _dct:
                    otm[_desc][j] = _dct[eltype][DOF[j] - offset]
                else:
                    _s = f"EL-{_name}, El. Type {eltype:3}, Code {DOF[j]:3}  "
                    otm[_desc][j] = _s
    else:
        if len(nasnm) == 3:
            matname = "m" + nasnm + "x1"
            tabname = "t" + nasnm + "x1"
        else:
            matname = "m" + nasnm
            tabname = "t" + nasnm
        x1 = drms[matname]
        iddof = drmkeys[tabname]
        otm[nm] = np.zeros((drc, x1.shape[1]))
        lastid = -1
        for j in range(drc):  # loop over requests
            # find rows corresponding to requested grid
            if ID[j] != lastid:
                eltype, srow = _get_tinr(iddof, ID[j])
                lastid = ID[j]
            otm[nm][j] = x1[srow + DOF[j] - offset]
            if getdesc:
                if eltype in _dct:
                    otm[_desc][j] = _dct[eltype][DOF[j] - offset]
                else:
                    _s = f"EL-{_name}, El. Type {eltype:3}, Code {DOF[j]:3}  "
                    otm[_desc][j] = _s


def procdrm12(op2file=None, op4file=None, dosort=True):
    """
    Process op2/op4 file2 output from DRM1/DRM2 DMAPs to form data
    recovery matrices.

    Parameters
    ----------
    op2file : string or None
        Either the basename of the .op2 and .op4 files, or the full
        name of the .op2 file. If None, a GUI is opened for file
        selection.
    op4file : string or None
        The name of the .op4 file or, if None, builds name from the
        `op2file` input.
    dosort : bool
        If True, sort data recovery rows in ascending order by ID/DOF.
        Otherwise, return in order requested in Nastran run.

    Returns
    -------
    otm : dictionary
        Has data recovery matrices (DRMs), id/dof info, and generic
        descriptions. The potential DRM keys are::

            'ATM'  : acceleration DRM

          For mode-displacement:
            'DTM'  : displacement DRM
            'LTM'  : element force (loads) DRM
            'SPCF' : SPC forces DRM
            'STM'  : element stress DRM

          For mode-acceleration:
            'DTMD' : displacement-dependent part of displacement DRM
            'DTMA' : acceleration-dependent part of displacement DRM
            'LTMD' : displacement-dependent part of element force DRM
            'LTMA' : acceleration-dependent part of element force DRM
            'SPCFD': displacement-dependent part of SPCF forces DRM
            'SPCFA': acceleration-dependent part of SPCF forces DRM
            'STMD' : displacement-dependent part of element stress DRM
            'STMA' : displacement-dependent part of element stress DRM

        The id/dof matrices are each 2 columns of [id, dof] with
        number of rows equal to the number of rows in corresponding
        DRM. The keys are the applicable strings from::

            'ATM_id_dof'
            'DTM_id_dof'
            'LTM_id_dof' - dof is actually the Nastran item code
            'SPCF_id_dof'
            'STM_id_dof' - dof is actually the Nastran item code

        The descriptions are arrays of strings with generic
        descriptions for each data recovery item. Length is equal to
        number of rows in corresponding DRM. See
        :func:`nastran_dr_descriptions` for more information. The keys
        are the applicable strings from::

            'ATM_desc'
            'DTM_desc'
            'LTM_desc',
            'SPCF_desc'
            'STM_desc'.

    Notes
    -----
    Currently, only displacements, accelerations, SPC forces, element
    forces and element stresses (for some elements) are implemented.

    Example usage::

        import op2
        otm = op2.procdrm12('drm2')
    """
    op2file, op4file = _get_op2_op4(op2file, op4file)

    # read op4 file:
    drms = op4.read(op4file)

    with OP2(op2file) as o2:
        drmkeys = o2.rddrm2op2()
    N = drmkeys["drs"].shape[1]

    # drs format:
    # 6 elements per recovery item:
    #    1  -  Subcase number (0 for all)
    #    2  -  Vector request type
    #    3  -  Point or Element ID
    #    4  -  Component
    #    5  -  XY output type
    #    6  -  Destination code

    # Vector request type:
    Vreq = [
        "Displacement",  # 1
        "Velocity",  # 2
        "Acceleration",  # 3
        "SPC Force",  # 4
        "Load",  # 5
        "Stress",  # 6
        "Element Force",  # 7
        "SDisplacement",  # 8
        "SVelocity",  # 9
        "SAcceleration",  # 10
        "Nonlinear Force",  # 11
        "Total",
    ]  # 12

    #   XY output type:
    #      1 = Response
    #      2 = PSDF
    #      3 = AUTO
    #
    #   Destination code:
    #      0 = XYpeak only   (from DRMEXT)
    #      1 = Print
    #      2 = Plot
    #      3 = Print, Plot
    #      4 = Punch
    #      5 = Print, Punch
    #      6 = Plot, Punch
    #      7 = Print, Plot, Punch

    if not dosort:
        # reshape dr:
        dr = drmkeys["dr"]
        r = np.nonzero(dr == dr[0])[0]
        r = np.hstack((r, len(dr)))
        n = len(r) - 1
        # dr(r) = ? -- starts every XYPEAK card
        # dr(r+1:3) = 0, 0, 0  ?
        # dr(r+4) = 1  ?
        # dr(r+5) = request type
        # dr(r+6:8) = 0, 0, #(?)
        # dr(r+9) = id 1
        # dr(r+10) = dof 1
        # dr(r+11) = 0
        # ... r + 9, 10, 11 can repeat until three -1's are reached
        #     These 3 values repeat when there is a comma: 1(T1),1(T2)
        # dr(X:X+2) = -1, -1, -1
        # 8-X+2 repeat until all dof for an XYPEAK are listed
        #     This section repeats when there is a slash: 1(T1)/1(T2)
        DR = np.zeros((3, N), dtype=int)  # [type; id; dof]
        R = 0  # index into DR columns
        for j in range(n):  # loop over XYPEAK cards
            curtype = dr[r[j] + 5]
            J = r[j] + 9  # index to first id
            while J < r[j + 1]:
                while dr[J] != -1:
                    DR[:, R] = curtype, dr[J], dr[J + 1]
                    R += 1
                    J += 3
                J += 4  # jump over [-1,-1,-1,#]
    else:
        DR = drmkeys["drs"][1:4]  # use sorted version

    desc = nastran_dr_descriptions()
    drminfo = {
        1: ("DTM", "oug", "acce"),
        3: ("ATM", "ougv1", "acce"),
        4: ("SPCF", "oqg", "spcf"),
        6: ("STM", "oes", "stress"),
        7: ("LTM", "oef", "force"),
    }
    otm = {}
    types = np.array([1, 3, 4, 6, 7])
    for drtype in range(1, 13):
        pv = np.nonzero(DR[0] == drtype)[0]
        if pv.size > 0:
            if np.any(drtype == types):
                print(f'Processing "{Vreq[drtype - 1]}" requests...')
                _get_drm(drminfo[drtype], otm, drms, drmkeys, DR[:, pv], desc)
            else:
                print(
                    f'Skipping "{Vreq[drtype - 1]}" requests. Needs to be added '
                    "to procdrm12()."
                )
    return otm


def rdpostop2(
    op2file=None,
    verbose=False,
    getougv1=False,
    getoef1=False,
    getoes1=False,
):
    """
    Reads PARAM,POST,-1 op2 file and returns dictionary of data.

    .. warning::
        This routine may be removed in the future in favor of
        :func:`rdparampost`.

    Parameters
    ----------
    op2file : string or None
        Name of op2 file. If None, a GUI is opened for file selection.
    verbose : bool
        If True, echo names of tables and matrices to screen
    getougv1 : bool
        If True, read the OUGV1, OUG1, or BOPHIG matrices, if any
    getoef1 : bool
        If True, read the OEF1* matrices, if any
    getoes1 : bool
        If True, read the OES1* matrices, if any

    Returns
    -------
    dictionary

    'uset' : pandas DataFrame
        A DataFrame as output by :func:`OP2.rdn2cop2`
    'cstm' : array
        14-column matrix containing the coordinate system
        transformation matrix for each coordinate system. See
        description in class OP2, member function
        :func:`OP2.rdn2cop2`.
    'cstm2' : dictionary
        Dictionary indexed by the coordinate system id number. This
        has the same information as 'cstm', but in a different format.
        See description in class OP2, member function
        :func:`OP2.rdn2cop2`.
    'gpwg' : dictionary; optional
        Only present if the "OGPWG" table is present in the op2
        file. The dictionary is the output of :func:`OP2.rdop2gpwg`.
    'mats' : dictionary
        Dictionary of matrices read from op2 file and indexed by the
        name. The 'tload' entry is a typical entry. Will also
        contain lists of 'OUGV1', 'EOF1*', and 'EOS1*' matrices if
        the respective `get*` flag is set and those entries are
        present.
    'selist' : 2d ndarray
        2-columns matrix: [ seid, dnseid ] where, for each row, dnseid
        is the downstream superelement for seid. (dnseid = 0 if seid =
        0).
    'sebulk' : 2d ndarray
        output record from GEOM1 of SE 0
    'seload' : 2d ndarray
        output record from GEOM1 of SE 0
    'seconct' : 1d ndarray
        output record from GEOM1 of SE 0
    'geom1' : dictionary
        Dictionary of GEOM1 data blocks; key is SE. Will not be
        present if number of GEOM1 data blocks did not line up with
        the `sebulk` array. See 'geom1_list'.
    'geom1_list': list
        List of all GEOM1 data blocks in order. Data also in 'geom1'
        dictionary.
    """
    warnings.warn(
        "`rdpostop2` may be removed in the future; use `rdparampost` instead.",
        FutureWarning,
    )
    # read op2 file:
    op2file = guitools.get_file_name(op2file, read=True)
    with OP2(op2file) as o2:
        mats = {}
        geom1 = {}
        geom1_list = []
        uset = None
        gpwg = None
        se = 0
        o2._fileh.seek(o2._postheaderpos)

        eqexin1 = eqexin = None
        bgpdt_rec1 = None
        dof = None
        Uset = None
        cstm, cstm2 = None, None

        while 1:
            name, trailer, dbtype = o2.rdop2nt()
            if name is None:
                break
            if dbtype > 0:
                if verbose:
                    print(f"Reading matrix {name}...")
                if name not in mats:
                    mats[name] = []
                mats[name] += [o2.rdop2matrix(trailer)]
            else:
                if name.find("BGPDT") == 0:
                    if verbose:
                        print(f"Reading table {name}...")
                    bgpdt_rec1 = o2._rdop2bgpdt68()
                    continue

                if name.find("CSTM") == 0:
                    if verbose:
                        print(f"Reading table {name}...")
                    cstm = o2._rdop2cstm68()
                    bc = np.array(
                        [
                            [+0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                            [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        ],
                        dtype=float,
                    )
                    cstm = np.vstack((bc, cstm))
                    continue

                if name.find("GEOM1") == 0:
                    if verbose:
                        print(f"Reading table {name}...")
                    geom1 = o2._rdop2geom1cord2()
                    geom1_list += [geom1]

                    if 0 not in geom1["cords"]:
                        geom1["cords"][0] = np.array(
                            [
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ]
                        )
                    if -1 not in geom1["cords"]:
                        # dummy for spoints
                        geom1["cords"][-1] = np.zeros((5, 3))
                        geom1["cords"][-1][0, 0] = -1
                    cstm2 = geom1["cords"]
                    continue

                if name.find("DYNAMIC") == 0:
                    if verbose:
                        print(f"Reading table {name}...")
                    mats["tload"] = o2.rdop2dynamics()
                    continue

                if name.find("EQEXIN") == 0:
                    if verbose:
                        print(f"Reading table {name}...")
                    eqexin1, eqexin = o2._rdop2eqexin()
                    continue

                if name.find("USET") == 0:
                    if verbose:
                        print(f"Reading table {name}...")
                    uset = o2._rdop2uset()
                    continue

                if name.find("OGPWG") == 0:
                    if verbose:
                        print(f"Reading table {name}...")
                    gpwg = o2._rdop2gpwg()
                    continue

                if getougv1 and (
                    name.find("OUGV1") == 0
                    or name.find("BOPHIG") == 0
                    or name.find("OUG1") == 0
                ):
                    if verbose:
                        print(f"Reading table {name}...")
                    try:
                        mo = mats["ougv1"]
                    except KeyError:
                        mo = mats["ougv1"] = []
                    mo += [o2._rdop2ougv1(name)]
                    continue

                if getoef1 and name.startswith("OEF1"):
                    if verbose:
                        print(f"Reading table {name}...")
                    try:
                        mo = mats["oef1"]
                    except KeyError:
                        mo = mats["oef1"] = []
                    mo += [o2._rdop2drm(name, True)]
                    continue

                if getoes1 and name.startswith("OES1"):
                    if verbose:
                        print(f"Reading table {name}...")
                    try:
                        mo = mats["oes1"]
                    except KeyError:
                        mo = mats["oes1"] = []
                    mo += [o2._rdop2drm(name, True)]
                    continue

                if name.find("LAMA") == 0:
                    if verbose:
                        print(f"Reading table {name}...")
                    try:
                        mo = mats["lama"]
                    except KeyError:
                        mo = mats["lama"] = []
                    mo += [o2._rdop2lama()]
                    continue

                if verbose:
                    print(f"Skipping table {name}...")
                o2.goto_next()

        if eqexin1 is not None and eqexin is not None and bgpdt_rec1 is not None:
            (xyz, cid, dof, doftype, nid, upids) = o2._proc_bgpdt(
                eqexin1, eqexin, True, bgpdt_rec1
            )
            Uset, cstm, cstm2 = o2._buildUset(
                se,
                dof,
                doftype,
                nid,
                uset,
                xyz,
                cid,
                cstm,
                cstm2,
                print_warning=False,
            )

    dct = {
        "uset": Uset,
        "cstm": cstm,
        "cstm2": cstm2,
        "mats": mats,
        "geom1_list": geom1_list,
    }
    if gpwg is not None:
        dct["gpwg"] = gpwg

    # make geom1 dictionary where the se id is the key:
    # - This method of finding the se id is a guess, but a logical one
    #   ... I just hope my logic matches Nastran's
    if "sebulk" in geom1:
        sebulk = geom1["sebulk"]
        seids = np.r_[sebulk[:, 0].astype(np.int64), 0]
        geom1_dict = {}
        nse = len(seids)
        ngeom1s = len(geom1_list)
        if nse > ngeom1s:  # pragma: no cover
            warnings.warn(
                f"number SEs in SE 0 'sebulk' record ({nse-1}) is greater "
                "than the number of upstream 'geom1(s)' data blocks in .op2 file "
                f"({ngeom1s-1}). Not creating the `geom1` dictionary.",
                RuntimeWarning,
            )
        else:
            # fill in dictionary assuming everything lines up:
            for seid, se_geom1 in zip(seids, geom1_list):
                geom1_dict[seid] = se_geom1

            # check the lining up assumption:
            if nse < ngeom1s:  # pragma: no cover
                warnings.warn(
                    f"number SEs in SE 0 'sebulk' record ({nse-1}) is less "
                    "than the number of upstream 'geom1(s)' data blocks in .op2 file "
                    f"({ngeom1s-1}).",
                    RuntimeWarning,
                )

                # check to see if geom1_list[nse-1] (which should be
                # for se 0) has sebulk:
                if "sebulk" not in se_geom1:
                    geom1_dict = {0: geom1}

            dct["geom1"] = geom1_dict

        # some special se 0 data block handling for backward
        # compatibility:
        for name in ("selist", "sebulk", "seload", "seconct"):
            if name in geom1:
                dct[name] = geom1[name]

    elif len(geom1_list) == 1:
        dct["geom1"] = {0: geom1}

    return dct


def rdparampost(
    op2file=None,
    *,
    verbose=False,
    get_all=False,
    get_mats=True,
    get_ougs=False,
    get_ogfs=False,
    get_drms=False,
    get_dr_tables=True,
    which=-1,
):
    """
    Reads PARAM,POST,-1 op2 file and returns dictionary of data.

    Parameters
    ----------
    verbose : bool; optional
        If True, echo names of tables and matrices to screen
    get_all; bool; optional
        If True, sets all other `get_*` parameters to True. Ignored if
        False. It won't override `get_mats` or `get_dr_tables` if they
        are lists or tuples.
    get_mats : bool or list/tuple; optional
        If True (or False), read (or do not read) matrices. If a
        list/tuple, it is an iterable of data blocks to look for and
        read if possible. Each name in the list can end with a "*" for
        simple wildcard matching. For example::

            get_mats = ["K4HH", "M*"]

        would read in "K4HH" and all matrices that start with
        "M". In the output dictionary of this routine, each matrix
        is stored by its name in lower case. If `which` is "all",
        then each entry is a list of all the matrices of each
        name in the file (see :func:`OP2.rdop2mats`)
    get_ougs : bool; optional
        If True, read all "OUG" formatted datablocks. See also
        `which`.
    get_ogfs : bool; optional
        If True, read all "OGF" formatted datablocks. See also
        `which`.
    get_drms : bool; optional
        If True, read all "OEF" and "OES" formatted datablocks.  See
        also `which`.
    get_dr_tables : bool or list/tuple; optional
        If True (or False), read (or do not read) data recovery
        tables. If True, any data block that starts with "T" is
        checked and if it is a data recovery table, it is read in. If
        `get_dr_tables` is a list/tuple, it is an iterable of data
        blocks to look for and read if possible. Each name in the list
        can end with a "*" for simple wildcard matching. For example::

            get_dr_tables = ["TUG*"]

        would read in all data recovery tables that match that
        pattern, and only those. Setting `get_dr_tables` to True is
        equivalent to setting to ``["T*"]``. In the output dictionary
        of this routine, each matrix is stored by its name in lower
        case. Each matrix is 3-columns::

            [id, dof, type]

        See also `which`.
    which : integer or str; optional
        If integer, specifies which occurrence of each matrix to
        read starting at 0. Default is -1; read the last entry for
        each matrix only. Can also be the string "all". In that
        case, each entry in the dictionary is a list of all
        occurrences of each matrix (even if there is only one
        occurrence).

    Returns
    -------
    dictionary

    'uset' : pandas DataFrame
        A DataFrame as output by :func:`OP2.rdn2cop2`
    'cstm' : array
        14-column matrix containing the coordinate system
        transformation matrix for each coordinate system. See
        description in class OP2, member function
        :func:`OP2.rdn2cop2`.
    'cstm2' : dictionary
        Dictionary indexed by the coordinate system id number. This
        has the same information as 'cstm', but in a different format.
        See description in class OP2, member function
        :func:`OP2.rdn2cop2`.
    'lama' : ndarray or list of ndarrays
        The "LAMA" table; # modes x 7. List if `which` is "all".
    'dynamics' : ndarray or list of ndarrays
        The "tload" part of the DYNAMICS data block. See also
        "tload". List if `which` is "all".
    'tload' : ndarray
        Only present if the "DYNAMICS" data block is present. This
        is the "tload" part of the last "DYNAMICS" data block read
        in. (Same as "dynamics" if `which` is not "all".)
    'ogpwg' : dictionary or list of dictionaries
        Only present if the "OGPWG" table is present in the op2
        file. The dictionary is the output of
        :func:`OP2.rdop2gpwg`. List if `which` is "all".
    'selist' : 2d ndarray
        2-columns matrix: [ seid, dnseid ] where, for each row, dnseid
        is the downstream superelement for seid. (dnseid = 0 if seid =
        0).
    'sebulk' : 2d ndarray
        output record from GEOM1 of SE 0
    'seload' : 2d ndarray
        output record from GEOM1 of SE 0
    'seconct' : 1d ndarray
        output record from GEOM1 of SE 0
    'geom1' : dictionary
        Dictionary of GEOM1(S) data blocks; key is SE. Does not
        depend on `which`. Will be empty if no GEOM1(S) data
        blocks are in the op2 file.
    'bgpdt' : dictionary
        Dictionary of BGPDT(S) data blocks; key is SE. Does not
        depend on `which`. Will be empty if no BGPDT(S) data
        blocks are in the op2 file.

    Notes
    -----
    Other data blocks may be present in the output depending on the
    input parameters and the contents of the op2 file.

    This routine is for convenience; this is what it does::

        op2file = guitools.get_file_name(op2file, read=True)
        with OP2(op2file) as o2:
            return o2.rdparampost(
                verbose=verbose,
                get_all=get_all,
                get_mats=get_mats,
                get_ougs=get_ougs,
                get_ogfs=get_ogfs,
                get_drms=get_drms,
                get_dr_tables=get_dr_tables,
                which=which,
            )
    """
    op2file = guitools.get_file_name(op2file, read=True)
    with OP2(op2file) as o2:
        return o2.rdparampost(
            verbose=verbose,
            get_all=get_all,
            get_mats=get_mats,
            get_ougs=get_ougs,
            get_ogfs=get_ogfs,
            get_drms=get_drms,
            get_dr_tables=get_dr_tables,
            which=which,
        )


def rdparampost_old(
    op2file=None,
    verbose=False,
    get_all=False,
    get_mats=True,
    get_ougv1=False,
    get_oef1=False,
    get_oes1=False,
    get_dr_tables=True,
    which=-1,
):
    """
    Reads PARAM,POST,-1 op2 file and returns dictionary of data.

    .. warning::
        DEPRECATED. This routine will likely be removed in the next
        release. Use :func:`rdparampost` instead.

    Parameters
    ----------
    verbose : bool; optional
        If True, echo names of tables and matrices to screen
    get_all; bool; optional
        If True, sets all other `get_*` parameters to True. Ignored if
        False. It won't override `get_mats` or `get_dr_tables` if they
        are lists or tuples.
    get_mats : bool or list/tuple; optional
        If True (or False), read (or do not read) matrices. If a
        list/tuple, it is an iterable of data blocks to look for and
        read if possible. Each name in the list can end with a "*" for
        simple wildcard matching. For example::

            get_mats = ["K4HH", "M*"]

        would read in "K4HH" and all matrices that start with
        "M". In the output dictionary of this routine, each matrix
        is stored by its name in lower case. If `which` is "all",
        then each entry is a list of all the matrices of each
        name in the file (see :func:`OP2.rdop2mats`)
    get_ougv1 : bool; optional
        If True, read the OUGV1, OUG1, or BOPHIG matrix, if
        present. See also `which`.
    get_oef1 : bool; optional
        If True, read the OEF1 matrix, if any. See also `which`.
    get_oes1 : bool; optional
        If True, read the OES1 matrix, if any. See also `which`.
    get_dr_tables : bool or list/tuple; optional
        If True (or False), read (or do not read) data recovery
        tables. Any data block that starts with "T" is checked and if
        it is a data recovery table, it is read in. If a list/tuple,
        it is an iterable of data blocks to look for and read if
        possible. Each name in the list can end with a "*" for
        simple wildcard matching. For example::

            get_dr_tables = ["TUG*"]

        would read in all data recovery tables that match that
        pattern, and only those. Setting `get_dr_tables` to True is
        equivalent to setting to ``["T*"]``. In
        the output dictionary of this routine, each matrix is stored
        by its name in lower case. Each matrix is 3-columns::

            [id, dof, type]

        See also `which`.
    which : integer or str; optional
        If integer, specifies which occurrence of each matrix to
        read starting at 0. Default is -1; read the last entry for
        each matrix only. Can also be the string "all". In that
        case, each entry in the dictionary is a list of all
        occurrences of each matrix (even if there is only one
        occurrence).

    Returns
    -------
    dictionary

    'uset' : pandas DataFrame
        A DataFrame as output by :func:`OP2.rdn2cop2`
    'cstm' : array
        14-column matrix containing the coordinate system
        transformation matrix for each coordinate system. See
        description in class OP2, member function
        :func:`OP2.rdn2cop2`.
    'cstm2' : dictionary
        Dictionary indexed by the coordinate system id number. This
        has the same information as 'cstm', but in a different format.
        See description in class OP2, member function
        :func:`OP2.rdn2cop2`.
    'lama' : ndarray or list of ndarrays
        The "LAMA" table; # modes x 7. List if `which` is "all".
    'dynamics' : ndarray or list of ndarrays
        The "tload" part of the DYNAMICS data block. See also
        "tload". List if `which` is "all".
    'tload' : ndarray
        Only present if the "DYNAMICS" data block is present. This
        is the "tload" part of the last "DYNAMICS" data block read
        in. (Same as "dynamics" if `which` is not "all".)
    'ogpwg' : dictionary or list of dictionaries
        Only present if the "OGPWG" table is present in the op2
        file. The dictionary is the output of
        :func:`OP2.rdop2gpwg`. List if `which` is "all".
    'selist' : 2d ndarray
        2-columns matrix: [ seid, dnseid ] where, for each row, dnseid
        is the downstream superelement for seid. (dnseid = 0 if seid =
        0).
    'sebulk' : 2d ndarray
        output record from GEOM1 of SE 0
    'seload' : 2d ndarray
        output record from GEOM1 of SE 0
    'seconct' : 1d ndarray
        output record from GEOM1 of SE 0
    'geom1' : dictionary
        Dictionary of GEOM1(S) data blocks; key is SE. Does not
        depend on `which`. Will be empty if no GEOM1(S) data
        blocks are in the op2 file.
    'bgpdt' : dictionary
        Dictionary of BGPDT(S) data blocks; key is SE. Does not
        depend on `which`. Will be empty if no BGPDT(S) data
        blocks are in the op2 file.

    Notes
    -----
    Other data blocks may be present in the output depending on the
    input parameters and the contents of the op2 file.

    This routine is for convenience; this is what it does::

        op2file = guitools.get_file_name(op2file, read=True)
        with OP2(op2file) as o2:
            return o2.rdparampost_old(
                verbose,
                get_all,
                get_mats,
                get_ougv1,
                get_oef1,
                get_oes1,
                get_dr_tables,
                which,
            )
    """
    warnings.warn(
        "`rdparampost_old` will be removed soon. Use `rdparampost` instead.",
        FutureWarning,
    )
    op2file = guitools.get_file_name(op2file, read=True)
    with OP2(op2file) as o2:
        return o2.rdparampost_old(
            verbose,
            get_all,
            get_mats,
            get_ougv1,
            get_oef1,
            get_oes1,
            get_dr_tables,
            which,
        )
