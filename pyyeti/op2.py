# -*- coding: utf-8 -*-
"""
Some Python tools for reading select data from Nastran .op2 files.
Converted from the Yeti version.

Can read files in big or little endian format.
"""

import numpy as np
import sys
from pyyeti import n2p, op4
import struct
import warnings
import itertools as it
# from collections import namedtuple

#  Notes on the op2 format.
#
#  DATA BLOCK:
#      All data blocks (including header) start with header 3 elements:
#      [reclen, key, endrec]
#        - reclen = 1 32-bit integer that specifies number of bytes in
#          key (either 4 or 8)
#        - key = 4 or 8 byte integer specifying number of words in next
#          record
#        - endrec = reclen
#
#      DATA SET, can be multiple records:
#          Next is [reclen, data, endrec]
#            - reclen = 1 32-bit integer that specifies number of bytes
#              in data
#            - data = reclen bytes long, variable format; may be part of
#              a data set or the complete set
#            - endrec = reclen
#
#          Next is info about whether we're done with current data set:
#          [reclen, key, endrec]
#            - reclen = 1 32-bit integer that specifies number of bytes
#              in key (either 4 or 8)
#            - key = 4 or 8 byte integer specifying number of words in
#              next record; if 0, done with data set
#            - endrec = reclen
#
#          If not done, we have [reclen, data, endrec] for part 2 (and
#          so on) for the record.
#
#      Once data set is complete, we have: [reclen, key, endrec]
#        - reclen = 1 32-bit integer that specifies number of bytes in
#          key (either 4 or 8)
#        - key = 4 or 8 byte integer specifying number of words in next
#          record (I think ... not useful?)
#        - endrec = reclen
#
#      Then: [reclen, rec_type, endrec]
#        - reclen = 1 32-bit integer that specifies number of bytes in
#          rec_type (either 4 or 8)
#        - rec_type = 0 if table (4 or 8 bytes)
#        - endrec = reclen
#
#      Then, info on whether we're done with data block:
#      [reclen, key, endrec]
#        - reclen = 1 32-bit integer that specifies number of bytes in
#          key (either 4 or 8)
#        - key = 4 or 8 byte integer specifying number of words in next
#          record; if 0, done with data block
#        - endrec = reclen
#
#      If not done, we have [reclen, data, endrec] for record 2 and so
#      on, until data block is read in.


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
    >>> from pyyeti import op2
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


class OP2(object):
    """Class for reading Nastran op2 files and nas2cam data files."""

    def __init__(self, filename=None):
        self._fileh = None
        self._CodeFuncs = None
        if isinstance(filename, str):
            self._op2open(filename)

    def __del__(self):
        if self._fileh:
            self._fileh.close()
            self._fileh = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self._fileh:
            self._fileh.close()
            self._fileh = None
        return False

    @property
    def CodeFuncs(self):
        """See :func:`_check_code`."""
        if self._CodeFuncs is None:
            def func1(item_code):
                if item_code // 1000 in [2, 3, 6]:
                    return 2
                return 1

            def func2(item_code):
                return item_code % 100

            def func3(item_code):
                return item_code % 1000

            def func4(item_code):
                return item_code // 10

            def func5(item_code):
                return item_code % 10

            def func6(item_code):
                if item_code & 8:
                    return 0
                return 1

            def func7(item_code):
                v = item_code // 1000
                if v in [0, 2]:
                    return 0
                if v in [1, 3]:
                    return 1
                return 2

            def funcbig(func_code, item_code):
                return item_code & (func_code & 65535)

            self._CodeFuncs = {1: func1, 2: func2, 3: func3, 4: func4,
                               5: func5, 6: func6, 7: func7,
                               'big': funcbig}
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
            Will be '=' if `swap` is False; otherwise, either '>' or '<'
            for big-endian and little-endian, respectively.
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
        dbnames : dictionary
            See :func:`directory` for description.  Contains data block
            names, bytes in file, file positions, and for matrices, the
            matrix size.
        dblist : list
            See :func:`directory` for description.  Contains same info
            as dbnames, but in a list of ordered and formatted strings.
        _Str4 : struct.Struct object
            Precompiled for reading 4 byte integers (corresponds to
            `int32str`).
        _Str : struct.Struct object
            Precompiled for reading 4 or 8 byte integers (corresponds
            to `intstr`).

        File is positioned after the header label (at `postheaderpos`).
        """
        self._fileh = open(filename, 'rb')
        self.dbnames = []
        self.dblist = []
        reclen = struct.unpack('i', self._fileh.read(4))[0]
        self._fileh.seek(0)

        reclen = np.array(reclen, dtype=np.int32)
        if not np.any(reclen == [4, 8]):
            self._swap = True
            reclen = reclen.byteswap()
            if not np.any(reclen == [4, 8]):
                self._fileh.close()
                self._fileh = None
                raise ValueError('Could not decipher file.  First'
                                 '4-byte integer should be 4 or 8.')
            if sys.byteorder == 'little':
                self._endian = '>'
            else:
                self._endian = '<'
        else:
            self._swap = False
            self._endian = '='

        self._Str4 = struct.Struct(self._endian + 'i')
        if reclen == 4:
            self._ibytes = 4
            self._intstr = self._endian + 'i4'
            self._intstru = self._endian + '%di'
            self._i = 'i'
            self._Str = self._Str4
            self._rfrmu = self._endian + '%df'
            self._rfrm = self._endian + 'f4'
            self._f = 'f'
            self._fbytes = 4
        else:
            self._ibytes = 8
            self._intstr = self._endian + 'i8'
            self._intstru = self._endian + '%dq'
            self._i = 'q'
            self._Str = struct.Struct(self._endian + 'q')
            self._rfrmu = self._endian + '%dd'
            self._rfrm = self._endian + 'f8'
            self._f = 'd'
            self._fbytes = 8

        self._rowsCutoff = 3000
        self._int32str = self._endian + 'i4'
        self._int32stru = self._endian + '%di'
        self._rdop2header()
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
        self._fileh.read(n*(8+self._ibytes))

    def _rdop2header(self):
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
        bytes = self._ibytes*key
        self._date = struct.unpack(frm, self._fileh.read(bytes))
        # self._date = np.fromfile(self._fileh, self._intstr, key)
        self._fileh.read(4)  # endrec
        self._getkey()

        reclen = self._Str4.unpack(self._fileh.read(4))[0]
        self._nastheader = self._fileh.read(reclen).decode()
        self._fileh.read(4)  # endrec
        self._getkey()

        reclen = self._Str4.unpack(self._fileh.read(4))[0]
        self._label = self._fileh.read(reclen).decode().\
            strip().replace(' ', '')
        self._fileh.read(4)  # endrec
        self._skipkey(2)

    def _validname(self, bstr):
        """
        Returns a valid variable name from the byte string `bstr`.
        """
        return ''.join(chr(c) for c in bstr if (
              47 < c < 58 or 64 < c < 91 or c == 95 or 96 < c < 123))

    def _rdop2eot(self):
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

    def _rdop2nt(self):
        """
        Read Nastran output2 datablock name and trailer.

        Returns
        -------
        tuple: (name, trailer, type)
            name : string
                Name of upcoming data block (upper case).
            trailer : tuple
                Data block trailer.
            type : 0 or 1
                0 means table, 1 means matrix.  I think.

        All outputs will be None for end-of-file.
        """
        eot, key = self._rdop2eot()
        if key == 0:
            return None, None, None

        reclen = self._Str4.unpack(self._fileh.read(4))[0]
        db_name = self._validname(self._fileh.read(reclen))
        self._fileh.read(4)  # endrec
        self._getkey()
        key = self._getkey()

        self._fileh.read(4)  # reclen
        frm = self._intstru % key
        bytes = self._ibytes*key
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
        if mtype > 2:   # complex
            rows *= 2
        if mtype & 1:   # single precision
            frm = self._endian + 'f4'
            frmu = self._endian + '%df'
            bytes_per = 4
        else:
            frm = self._endian + 'f8'
            frmu = self._endian + '%dd'
            bytes_per = 8

        matrix = np.zeros((rows, trailer[1]), order='F')
        intsize = self._ibytes
        col = 0
        while dtype > 0:  # read in matrix columns
            # key is number of elements in next record (row # followed
            # by key-1 real numbers)
            key = self._getkey()
            # read column
            while key > 0:
                reclen = self._Str4.unpack(self._fileh.read(4))[0]
                r = self._Str.unpack(self._fileh.read(intsize))[0]-1
                if mtype > 2:
                    r *= 2
                n = (reclen - intsize) // bytes_per
                if n < self._rowsCutoff:
                    matrix[r:r+n, col] = struct.unpack(
                        frmu % n, self._fileh.read(n*bytes_per))
                else:
                    matrix[r:r+n, col] = np.fromfile(self._fileh,
                                                     frm, n)
                self._fileh.read(4)  # endrec
                key = self._getkey()
            col += 1
            self._getkey()
            dtype = self._getkey()
        self._rdop2eot()
        if mtype > 2:
            op4._view_as_complex(matrix)
        return matrix

    def skipop2matrix(self, trailer):
        """
        Skip Nastran op2 matrix at current position.

        It is assumed that the name has already been read in via
        :func:`rdop2nt`.

        The size of the matrix is read from trailer:
             rows = trailer[2]
             cols = trailer[1]
        """
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
        self._rdop2eot()

    def skipop2table(self):
        """Skip over Nastran output2 table."""
        eot, key = self._rdop2eot()
        if key == 0:
            return
        while key > 0:
            while key > 0:
                reclen = self._Str4.unpack(self._fileh.read(4))[0]
                self._fileh.seek(8+reclen, 1)
                key = self._Str.unpack(self._fileh.read(self._ibytes))[0]
                self._fileh.read(4)  # endrec
            self._skipkey(2)
            eot, key = self._rdop2eot()

    def rdop2mats(self):
        """Read all matrices from Nastran output2 file.

        Returns dictionary containing all matrices in the op2 file:
        {'NAME1': matrix1, 'NAME2': matrix2, ...}

        The keys are the names as stored (upper case).
        """
        self._fileh.seek(self._postheaderpos)
        mats = {}
        while 1:
            name, trailer, rectype = self._rdop2nt()
            if name is None:
                break
            if rectype > 0:
                print("Reading matrix {}...".format(name))
                mats[name] = self.rdop2matrix(trailer)
            else:
                self.skipop2table()
        return mats

    def prtdir(self):
        """
        Prints op2 data block directory.  See also :func:`directory`.
        """
        if len(self.dblist) == 0:
            self.directory(verbose=False)
        for s in self.dblist:
            print(s)

    def directory(self, verbose=True, redo=False):
        """
        Return list of data block names in op2 file.

        Parameters
        ----------
        verbose : bool (or any true/false variable)
            If True, print names, sizes, and file offsets to screen.
        redo : bool
            If True, scan through file and redefine self.dbnames even
            if it is already set.

        Returns
        -------
        dbnames : Dictionary
            Dictionary indexed by data block name.  Each value is a
            list, one element per occurrence of the data block in the
            op2 file.  Each element is another list that has 3
            elements: [fpos, bytes, size]::

               fpos : 2-element list; file position start and stop
                      (stop value is start of next data block)
               bytes: number of bytes data block consumes in file
               size : 2-element list; for matrices, [rows, cols],
                      for tables [0, 0]

        dblist : list
            List of strings for printing.  Contains the info above
            in formatted and sorted (in file position order) strings.

        Notes
        -----
        As an example of using dbnames, to get a list of all sizes of
        matrices named 'KAA'::

            o2 = op2.OP2('mds.op2')
            s = [item[2] for item in o2.dbnames['KAA']]

        For another example, to read in first matrix named 'KAA'::

            o2 = op2.OP2('mds.op2')
            fpos = o2.dbnames['KAA'][0][0][0]
            o2._fileh.seek(fpos)
            name, trailer, rectype = o2._rdop2nt()
            kaa = o2.rdop2matrix(trailer)

        This routine also sets self.dbnames = dbnames.
        """
        if len(self.dbnames) > 0 and not redo:
            if verbose:
                self.prtdir()
            return self.dbnames, self.dblist
        dbnames = {}
        dblist = []
        self._fileh.seek(self._postheaderpos)
        pos = self._postheaderpos
        while 1:
            name, trailer, dbtype = self._rdop2nt()
            if name is None:
                break
            if dbtype > 0:
                self.skipop2matrix(trailer)
                size = [trailer[2], trailer[1]]
                s = 'Matrix {:8}'.format(name)
            else:
                self.skipop2table()
                size = [0, 0]
                s = 'Table  {:8}'.format(name)
            cur = self._fileh.tell()
            s += (', bytes = {:10} [{:10} to {:10}]'.
                  format(cur-pos-1, pos, cur))
            if size != [0, 0]:
                s += (', {:6} x {:<}'.
                      format(size[0], size[1]))
            if name not in dbnames:
                dbnames[name] = []
            dbnames[name].append([[pos, cur], cur-pos-1, size])
            dblist.append(s)
            pos = cur
        self.dbnames = dbnames
        self.dblist = dblist
        if verbose:
            self.prtdir()
        return dbnames, dblist

    def rdop2dynamics(self):
        """
        Reads the TLOAD data from a DYNAMICS datablock.

        Returns matrix of TLOADS.  Rows = 5, 6, or 8, Cols = number
        of TLOADs.  TLOAD ids are in first row; other data in matrix
        may not be useful.
        """
        key = self._getkey()
        header_Str = struct.Struct(self._endian + self._i*3)
        hbytes = 3*self._ibytes
        eot = 0
        data = np.empty(0, dtype=self._intstr)
        while not eot:
            while key > 0:
                self._fileh.read(4)  # reclen
                header = header_Str.unpack(self._fileh.read(hbytes))
                if header == (7107, 71, 138):
                    if key < self._rowsCutoff:
                        bytes = (key-3)*self._ibytes
                        ndata = struct.unpack(self._intstru % (key-3),
                                              self._fileh.read(bytes))
                    else:
                        ndata = np.fromfile(self._fileh,
                                            self._intstr, key-3)
                    data = np.hstack((data, ndata))
                else:
                    self._fileh.seek((key-3)*self._ibytes, 1)
                self._fileh.read(4)  # endrec
                key = self._getkey()
            self._skipkey(2)
            eot, key = self._rdop2eot()

        if np.any(data):
            L = len(data)
            rows = (5, 6, 8)
            match = [0, 0, 0]
            # see if each col in matrix would match pattern:
            #   [tloadid, exciteid, -, -, tableid, ...]
            # where:
            #   1. tloadid's are all > 0 and always increasing
            #   2. exciteid's are all > 0
            #   3. tableid's are all > 0
            for i, r in enumerate(rows):
                if (L == r*(L // r) and
                        data[::r].min() > 0 and
                        np.diff(data[::r]).min() > 0 and
                        data[1::r].min() > 0 and
                        data[4::r].min() > 0):
                    match[i] = 1
            if sum(match) > 1:
                err = ('Could not determine rows for TLOADs! '
                       'More than one of 5, 6, or 8 matches. '
                       'Routine needs updating.')
                raise ValueError(err)
            if sum(match) < 1:
                err = ('Could not determine rows for TLOADs! '
                       'None of 5, 6, or 8 matches. '
                       'Routine needs updating.')
                raise ValueError(err)
            rows = rows[match.index(1)]
            data = np.reshape(data, (rows, -1), order='F')
        return data

    def rdop2tload(self):
        """
        Returns the TLOAD data from an op2 file.

        This routine scans the op2 file for the DYNAMICS datablock and
        then calls :func:`rdop2dynamics` to read the data.
        """
        fpos = self.dbnames['DYNAMICS'][0][0][0]
        self._fileh.seek(fpos)
        name, trailer, dbtype = self._rdop2nt()
        return self.rdop2dynamics()

    def rdop2record(self, form=None, N=0):
        """
        Read Nastran output2 data record.

        Parameters
        ----------
        form : string or None; optional
            String specifying format, or None to read in signed integers.
            One of::

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
            bytes string. None is returned if the end-of-datablock has
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
        if not form or form == 'int':
            frm = self._intstr
            frmu = self._intstru
            bytes_per = self._ibytes
        elif form == 'uint':
            frm = self._intstr.replace('i', 'u')
            frmu = self._intstru.replace('i', 'I')
            bytes_per = self._ibytes
        elif form == 'double':
            frm = self._endian + 'f8'
            frmu = self._endian + '%dd'
            bytes_per = 8
        elif form == 'single':
            frm = self._endian + 'f4'
            frmu = self._endian + '%df'
            bytes_per = 4
        elif form == 'bytes':
            data = []
            while key > 0:
                reclen = self._Str4.unpack(f.read(4))[0]
                data.append(f.read(reclen))
                f.read(4)  # endrec
                key = self._getkey()
            self._skipkey(2)
            data = b''.join(data)
            return data
        else:
            raise ValueError("form must be one of:  None, 'int', "
                             "'uint', 'double', 'single' or 'bytes'")
        if N:
            data = np.empty(N, dtype=frm)
            i = 0
            while key > 0:
                reclen = self._Str4.unpack(f.read(4))[0]
                # f.read(4)  # reclen
                n = reclen // bytes_per
                if n < self._rowsCutoff:
                    b = n * bytes_per
                    data[i:i+n] = struct.unpack(frmu % n, f.read(b))
                else:
                    data[i:i+n] = np.fromfile(f, frm, n)
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
            self._fileh.seek(reclen+4, 1)
            key = self._getkey()
        self._skipkey(2)

    def rdop2tabheaders(self, name):
        """
        Read op2 table headers and echo them to the screen.

        Parameters
        ----------
        name : string
            Name of data block that headers are being read for.

        Notes
        -----
        File must be positioned after name and trailer block.  For
        example, to read the table headers of the last GEOM1S data
        block::

            o2 = op2.OP2('modes.op2')
            fpos = o2.dbnames['GEOM1S'][-1][0][0]
            o2._fileh.seek(fpos)
            name, trailer, dbtype = o2._rdop2nt()
            o2.rdop2tabheaders('GEOM1S')

        """
        key = self._getkey()
        print("{} Headers:".format(name))
        Frm = struct.Struct(self._intstru % 3)
        eot = 0
        while not eot:
            while key > 0:
                reclen = self._Str4.unpack(self._fileh.read(4))[0]
                head = Frm.unpack(self._fileh.read(3*self._ibytes))
                print(np.hstack((head, reclen)))
                self._fileh.seek((key-3)*self._ibytes, 1)
                self._fileh.read(4)
                key = self._getkey()
            self._skipkey(2)
            eot, key = self._rdop2eot()

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

        where `iand` is the bit-wise AND operation. For example, ACODE,4
        means that the ACODE value should be integer divided it by 10.
        So, if ACODE is 22, ACODE,4 is 2.
        """
        if len(funcs) != len(vals):
            raise ValueError('len(funcs) != len(vals)!')
        for func, val in zip(funcs, vals):
            if 1 <= func <= 7:
                if self.CodeFuncs[func](item_code) not in val:
                    warnings.warn('{} value {} not acceptable'.
                                  format(name, item_code),
                                  RuntimeWarning)
                    return False
            elif func > 65535:
                if self.CodeFuncs['big'](func, item_code) not in val:
                    warnings.warn('{} value {} not acceptable'.
                                  format(name, item_code),
                                  RuntimeWarning)
                    return False
            else:
                raise ValueError('Unknown function code: {}'.
                                 format(func))
        return True

    def _get_block_bytes(self, name, pos):
        """
        Get start/stop bytes of datablock

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
        dbdir = self.dbnames[name]
        for i in range(len(dbdir)):
            start = dbdir[i][0][0]
            stop = dbdir[i][0][1]
            if start <= pos < stop:
                return start, stop

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
        float2_Str = struct.Struct(self._endian + 'ff')
        iif6_int = np.dtype(self._endian+'i4')
        iif6_bytes = 32
        i4_Str = struct.Struct(self._endian + self._i*4)
        i4_bytes = 4*self._ibytes
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
                lam[0] = keep
                keep = ougv1
                ougv1 = np.empty((keep.shape[0], nmodes), float,
                                 order='F')
                ougv1[:, 0] = keep[:, 0]

            # IDENT record:
            reclen = self._Str4.unpack(self._fileh.read(4))[0]
            header = i4_Str.unpack(self._fileh.read(i4_bytes))
            # header = (ACODE, TCODE, ...)
            achk = self._check_code(header[0], [4], [[2]], 'ACODE')
            tchk = self._check_code(header[1], [1, 2, 7],
                                    [[1], [7], [0, 2]], 'TCODE')
            if not (achk and tchk):
                self._fileh.seek(pos)
                self.skipop2table()
                return
            self._fileh.read(self._ibytes)  # mode bytes
            lam[J] = float2_Str.unpack(self._fileh.read(8))[0]
            # ttl bytes = reclen + 4 + 3*(4+ibytes+4)
            #           = reclen + 28 - 3*ibytes
            # read bytes = 4*ibytes + ibytes + 8 = 8 + 5*ibytes
            # self._fileh.seek(reclen-2*self._ibytes+20, 1)  # ... or:
            self._fileh.read(reclen-2*self._ibytes+20)

            # DATA record:
            if ougv1 is None:
                # - process DOF information on first column only
                # - there are 8 elements per node:
                #   id*10, type, x, y, z, rx, ry, rz
                data = self.rdop2record('bytes')  # 1st column
                n = len(data) // iif6_bytes
                data = np.fromstring(data, iif6_int)
                data1 = (data.reshape(n, 8))[:, :2]
                pvgrids = data1[:, 1] == 1
                dof = _expanddof(data1[:, 0] // 10, pvgrids)
                # form partition vector for modeshape data:
                V = np.zeros((n, 8), bool)
                V[:, 2] = True          # all nodes have 'x'
                V[pvgrids, 3:] = True   # only grids have all 6
                V = V.ravel()
                # initialize ougv1 with first mode shape:
                data.dtype = np.float32  # reinterpret as floats
                ougv1 = data[V].reshape(-1, 1)
            else:
                data = self.rdop2record('single', V.shape[0])
                ougv1[:, J] = data[V]
            J += 1
            eot, key = self._rdop2eot()
        return {'ougv1': ougv1, 'lambda': lam, 'dof': dof}

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
        words4bits = trailer[4]
        data1 = self.rdop2record()
        # [se bitpos proc_order dnse bitpos_dnse prim_se se_type]
        data1 = np.reshape(data1[:7*nse], (-1, 7))

        # read 2nd record:
        key = self._getkey()
        data2 = np.empty(0, dtype='u4')
        frm = self._endian + 'u4'
        frmu = self._endian + '%dI'
        mult = self._ibytes // 4
        while key > 0:
            self._fileh.read(4)  # reclen
            if mult*key < self._rowsCutoff:
                cur = struct.unpack(frmu % (mult*key),
                                    self._fileh.read(4*mult*key))
            else:
                cur = np.fromfile(self._fileh, frm, mult*key)
            data2 = np.hstack((data2, cur))
            self._fileh.read(4)  # endrec
            key = self._getkey()
        if self._ibytes == 8:
            data2 = np.reshape(data2, (4, -1))
            data2 = data2[[0, 3], :].ravel()
        self._skipkey(2)

        # [ grid_id [bitmap] ]
        data2 = np.reshape(data2, (-1, words4bits))
        # 1 in front need to skip over grid_id (vars are col indices)
        word4bit_up = 1 + data1[:, 1] // 32
        word4bit_dn = 1 + data1[:, 4] // 32
        bitpos_up = 31 - data1[:, 1] % 32
        bitpos_dn = 31 - data1[:, 4] % 32
        for j in range(nse-1):
            se = data1[j, 0]
            bitdn = 1 << bitpos_dn[j]
            bitup = 1 << bitpos_up[j]
            connected = np.logical_and(data2[:, word4bit_dn[j]] & bitdn,
                                       data2[:, word4bit_up[j]] & bitup)
            grids = data2[connected, 0]
            nas['dnids'][se] = grids
        for j in range(nse):  # = 1 to nse:
            self.skipop2record()
        self._getkey()

    def _rdop2bgpdt(self):
        """
        Read record 1 of the Nastran output2 BGPDT data block.

        Returns vector of the BGPDT data or [] if no data found.
        Vector is 9*ngrids in length.  For each grid::

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
        nbytes = self._ibytes*6 + 24
        dtype = np.dtype([('ints', (self._intstr, 6)),
                          ('xyz', (self._endian + 'f8', 3))])
        data = self.rdop2record('bytes')
        n = len(data) // nbytes
        if n*nbytes != len(data):
            raise ValueError("incorrect record length for"
                             " _rdop2bgpdt")
        data = np.fromstring(data, dtype=dtype)
        return data

    def _rdop2bgpdt68(self):
        """
        Read record 1 of the Nastran output2 BGPDT68 data block.

        Returns vector of the BGPDT data or [] if no data found.
        Vector is 4*ngrids in length.  For each grid::

          [ coord_id
            x
            y
            z ]

        The x, y, z values are the grid location in basic.
        """
        nbytes = self._ibytes + 12
        dtype = np.dtype([('ints', (self._intstr, 1)),
                          ('xyz', (self._endian + 'f4', 3))])
        data = self.rdop2record('bytes')
        n = len(data) // nbytes
        if n*nbytes != len(data):
            raise ValueError("incorrect record length for"
                             " _rdop2bgpdt68")
        data = np.fromstring(data, dtype=dtype)
        return data

    def _rdop2cstm(self):
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
        cstm_rec1 = self.rdop2record()
        cstm_rec2 = self.rdop2record('double')
        self._rdop2eot()

        # assemble coordinate system table
        length = len(cstm_rec1)
        cstm = np.zeros((length//4, 14))
        cstm[:, 0] = cstm_rec1[::4]
        cstm[:, 1] = cstm_rec1[1::4]
        # start index into rec2 for xo, yo, zo, T (12 values) is in
        # last (4th) position in rec1 for each coordinate system:
        pv = range(12)
        for i, j in enumerate(cstm_rec1[3::4]):
            cstm[i, 2:] = cstm_rec2[j+pv-1]  # -1 for 0 offset
        return cstm

#     def _rdop2cstm68(self):
#         """
#         Read record 1 of Nastran output2 CSTM68 data block.
#
#         Returns vector of the CSTM data or [] if no data found.  Vector
#         is 14 * number of coordinate systems in length.  For each
#         coordinate system::
#
#           [ id type xo yo zo T(1,1:3) T(2,1:3) T(3,1:3) ]
#
#         T is transformation from local to basic for the coordinate
#         system.
#         """
#         nbytes = 2*self._ibytes + 4*12
#         dtype = np.dtype([('idtype', (self._intstr, 2)),
#                           ('xyzT', (self._endian + 'f4', 12))])
#         data = self.rdop2record('bytes')
#         n = len(data) // nbytes
#         if n*nbytes != len(data):
#             raise ValueError("incorrect record length for"
#                              " _rdop2cstm68")
#         data = np.fromstring(data, dtype=dtype)
#         return data

    def _rdop2geom1cord2(self):
        e = self._endian
        i = self._i
        ib = self._ibytes
        f = self._f
        fb = self._fbytes
        header_Str = struct.Struct(e + i*3)
        cord2_Str = struct.Struct(e + '4' + i + '9' + f)
        sebulk_Str = struct.Struct(e + '4' + i + f + '3' + i)
        hbytes = 3*ib
        cbytes = 4*ib + 9*fb
        bbytes = 7*ib + fb

        CORD2R = (2101, 21, 8)
        CORD2C = (2001, 20, 9)
        CORD2S = (2201, 22, 10)
        SEBULK = (1427, 14, 465)
        SECONCT = (427, 4, 453)

        cord2 = np.zeros((0, 13))
        sebulk = np.zeros((1, 8))
        selist = np.array([[0, 0]], int)
        key = self._getkey()
        eot = 0
        # data = np.zeros(0, dtype=self._intstr)
        while not eot:
            while key > 0:
                self._fileh.read(4)  # reclen
                # reclen = self._Str4.unpack(self._fileh.read(4))[0]
                head = header_Str.unpack(self._fileh.read(hbytes))
                if head == CORD2R or head == CORD2C or head == CORD2S:
                    n = (key-3) // 13
                    data = np.empty((n, 13))
                    for i in range(n):
                        data[i] = cord2_Str.unpack(self._fileh.read(cbytes))
                    cord2 = np.vstack((cord2, data))
                elif head == SEBULK:
                    n = (key-3) // 8
                    sebulk = np.empty((n, 8))
                    for i in range(n):
                        sebulk[i] = sebulk_Str.unpack(self._fileh.read(bbytes))
                elif head == SECONCT:
                    n = key - 3
                    if n < self._rowsCutoff:
                        nbytes = n * ib
                        seconct = np.empty(n, int)
                        seconct[:] = struct.unpack(self._intstru % n,
                                                   self._fileh.read(nbytes))
                    else:
                        seconct = np.fromfile(self._fileh, self._intstr, n)
                    pv = np.nonzero(seconct == -1)[0][1:-2:2] + 1
                    pv = np.hstack((0, pv))
                    u = np.unique(seconct[pv], return_index=True)[1]
                    pv = pv[u]
                    selist = np.vstack((seconct[pv], seconct[pv+1])).T
                    selist = np.vstack((selist, [0, 0]))
                else:
                    self._fileh.seek((key-3)*ib, 1)
                self._fileh.read(4)  # endrec
                key = self._getkey()
            self._skipkey(2)
            eot, key = self._rdop2eot()
        cord2 = np.delete(cord2, 2, axis=1)
        return n2p.build_coords(cord2), sebulk, selist

    def _rdop2selist(self):
        """
        Read SLIST data block and return `selist` for :func:`rdn2cop2`.

        See :func:`rdn2cop2`.
        """
        slist = self.rdop2record()
        slist[1::7] = 0
        self.skipop2record()
        self._rdop2eot()
        return np.vstack((slist[::7], slist[4::7])).T

    def _rdop2uset(self):
        """
        Read the USET data block.

        Returns 1-d USET array.  The 2nd bit is cleared for the S-set.

        See :func:`rdn2cop2`.
        """
        uset = self.rdop2record('uint')
        # clear the 2nd bit for all S-set:
        sset = 0 != (uset & n2p.mkusetmask("s"))
        if any(sset):
            uset[sset] = uset[sset] & ~2
        self._rdop2eot()
        return uset

    def _rdop2eqexin(self):
        """
        Read the EQEXIN data block.

        Returns (EQEXIN1, EQEXIN) tuple.

        See :func:`rdn2cop2`.
        """
        eqexin1 = self.rdop2record()
        eqexin = self.rdop2record()
        self._rdop2eot()
        return eqexin1, eqexin

    def _proc_bgpdt(self, eqexin1, eqexin, ver68=False, bgpdtin=None):
        """
        Reads and processes the BGPDT data block for :func:`rdn2cop2`
        and :func:`rdpostop2`.

        Returns (xyz, cid, dof, doftype, nid, upids)

        See :func:`rdn2cop2`, :func:`rdpostop2`.
        """
        if ver68:
            bgpdt_rec1 = bgpdtin
        else:
            bgpdt_rec1 = self._rdop2bgpdt()
            self.rdop2record()
            self.skipop2table()

        xyz = bgpdt_rec1['xyz']
        if ver68:
            cid = bgpdt_rec1['ints']
        else:
            cid = bgpdt_rec1['ints'][:, 0]

        # assemble dof table:
        dof = eqexin[1::2] // 10
        doftype = eqexin[1::2] - 10*dof
        nid = eqexin[::2]

        # eqexin is in external sort, so sort it
        i = eqexin1[1::2].argsort()
        dof = dof[i]
        doftype = doftype[i]
        nid = nid[i]
        if ver68:
            upids = None
        else:
            upids = bgpdt_rec1['ints'][:, 5]
        return xyz, cid, dof, doftype, nid, upids

    def _buildUset(self, se, dof, doftype, nid, uset, xyz, cid,
                   cstm=None, cstm2=None):
        """
        Builds the 6-column uset table for :func:`rdn2cop2` and
        :func:`rdpostop2`.

        Returns: (uset, cstm, cstm2).

        See :func:`rdn2cop2`.
        """
        # Fill in all dof use -1 as default and set dof as
        # appropriate ... make it big enough for grids (6 cols).
        # The -1s will be partitioned out below.
        rd = len(dof)
        rb = len(cid)
        if rd != rb:
            raise ValueError(
                'RDOP2USET:  BGPDTS incompatible with '
                'EQEXINS for superelement {}.\n'
                '  Guess:  residual run clobbered EQEXINS\n'
                '    Fix:  add the "fxphase0" alter to your '
                'residual run'.format(se))
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
        ngrids = np.sum(grids)
        nongrids = rd - ngrids
        doflist = np.zeros((rd, 6)) - 1
        if ngrids > 0:
            doflist[grids] = np.arange(1, 7)
        if nongrids > 0:
            doflist[~grids, 0] = 0
        doflist = doflist.ravel()
        idlist = np.dot(nid.reshape(-1, 1), np.ones((1, 6))).ravel()
        coordinfo = coordinfo.reshape((rd*6, 3))

        # partition out -1s:
        pv = doflist != -1
        doflist = doflist[pv]
        idlist = idlist[pv]
        coordinfo = coordinfo[pv, :]
        if uset is None:
            warnings.warn('uset information not found.  Putting all '
                          'DOF in b-set.', RuntimeWarning)
            from pyyeti import n2p
            b = n2p.mkusetmask('b')
            uset = np.zeros(len(doflist), int) + b
        uset = np.hstack((np.vstack((idlist, doflist, uset)).T,
                          coordinfo))
        if cstm2 is None:
            cstm2 = {}
            for row in cstm:
                m = np.zeros((5, 3))
                m[0, :2] = row[:2]
                m[1:, :] = row[2:].reshape((4, 3))
                cstm2[int(row[0])] = m
        return uset, cstm, cstm2

    def _rdop2maps(self):
        """
        Reads and returns the MAPS information for :func:`rdn2cop2`.
        """
        id_Str = struct.Struct(self._endian + self._i + 'd')
        id_bytes = self._ibytes + 8
        key = 1
        maps = np.zeros((0, 2))
        while key:
            key = self._getkey()  # 2 (1 integer, 1 double)
            self._fileh.read(4)  # reclen 12 or 16 bytes
            curmap = id_Str.unpack(self._fileh.read(id_bytes))
            maps = np.vstack((maps, curmap))
            self._fileh.read(4)  # endrec
            self._skipkey(2)  # 1st key is mystery negative
            key = self._getkey()  # 1 if cont, 0 if done
        self._getkey()
        maps[:, 0] -= 1
        return maps

#     # appears to work, but so confusing ...
#     def _rdop2drm_old(self):
#         """
#         Read Nastran output2 DRM data block (table).
# 
#         Returns
#         -------
#         drm : ndarray
#             The drm matrix.
#         iddof : ndarray
#             2-column matrix of [id, dof].
# 
#         This routine is beta -- check output carefully.
#         """
#         def getStr(iprev, elemtype, ir_Str, ir_bytes):
#             if np.any(elemtype == np.array([4, 5])):
#                 ints_rec2 = 1
#             else:
#                 ints_rec2 = 2
#             if ints_rec2 != iprev:
#                 ir_Str = struct.Struct(self._endian + self._i*ints_rec2)
#                 ir_bytes = self._ibytes*ints_rec2
#             return ir_Str, ir_bytes, ints_rec2
# 
#         rfrm = self._rfrm
#         rfrmu = self._rfrmu
#         rsize = self._fbytes
#         u1 = self.rdop2record()
#         elemtype = u1[1]
#         elemid = u1[2]
#         ir_Str, ir_bytes, ints_rec2 = getStr(0, elemtype, None, None)
#         nwords = u1[9]
#         key = self._getkey()
#         block = 7*4+3*self._ibytes
# 
#         # determine records/column by scanning first column:
#         rpc = 0
#         fp = self._fileh
#         pos = fp.tell()
#         id1 = -1
#         drmrow = 0
#         blocksize = 500   # number of rows or cols to grow iddof and drm
#         drmrows = blocksize
#         iddof = np.zeros((drmrows, 2), int)
#         KN = key, nwords
#         while key >= nwords:
#             L = nwords - ints_rec2
#             fp.read(4)   # reclen
#             dataint = ir_Str.unpack(fp.read(ir_bytes))
#             id_cur = dataint[0] // 10
#             if id1 == -1:
#                 id1 = id_cur
#             elif id1 == id_cur:
#                 break
#             rpc += 1
#             if drmrow+L >= drmrows:
#                 iddof = np.vstack((iddof,
#                                    np.zeros((blocksize, 2), int)))
#                 drmrows += blocksize
#             iddof[drmrow:drmrow+L, 0] = id_cur
#             iddof[drmrow:drmrow+L, 1] = elemid
#             fp.seek(self._ibytes*L, 1)
#             drmrow += L
# 
#             # read rest of record:
#             for i in range(1, key // nwords):
#                 dataint = ir_Str.unpack(fp.read(ir_bytes))
#                 id_cur = dataint[0] // 10
#                 if drmrow+L >= drmrows:
#                     iddof = np.vstack((iddof,
#                                        np.zeros((blocksize, 2), int)))
#                     drmrows += blocksize
#                 iddof[drmrow:drmrow+L, 0] = id_cur
#                 iddof[drmrow:drmrow+L, 1] = elemid
#                 fp.seek(self._ibytes*L, 1)
#                 drmrow += L
#             fp.seek(block, 1)
#             key = self._getkey()
#             if key > 0:
#                 fp.read(4)   # reclen
#                 if key < self._rowsCutoff:
#                     u1 = struct.unpack(self._intstru % key,
#                                        fp.read(key*self._ibytes))
#                 else:
#                     u1 = np.fromfile(fp, self._intstr, key)
#                 if u1[1] != elemtype:
#                     raise ValueError('u1[1] != elemtype')
#                 elemid = u1[2]
#                 nwords = u1[9]
#                 fp.seek(block, 1)
#                 key = self._getkey()
# 
#         drmrows = drmrow
#         iddof = iddof[:drmrows]
#         drmcols = blocksize
#         fp.seek(pos)
#         B = np.zeros((drmrows, drmcols), order='F')
#         drm = B.copy()
#         drmcol = 0
#         key, nwords = KN
#         while key >= nwords:
#             drmrow = 0
#             if drmcol == drmcols:
#                 drm = np.asfortranarray(np.hstack((drm, B)))
#                 drmcols += blocksize
#             for _ in it.repeat(None, rpc):
#                 L = nwords - ints_rec2
#                 fp.read(4)   # reclen
#                 for i in range(key // nwords):
#                     # dataint = ir_Str.unpack(fp.read(ir_bytes))
#                     fp.read(ir_bytes)
#                     if L < self._rowsCutoff:
#                         drm[drmrow:drmrow+L,
#                             drmcol] = struct.unpack(rfrmu % L,
#                                                     fp.read(rsize*L))
#                     else:
#                         drm[drmrow:drmrow+L,
#                             drmcol] = np.fromfile(fp, rfrm, L)
#                     drmrow += L
# 
#                 fp.seek(block, 1)
#                 key = self._getkey()
#                 if key > 0:
#                     fp.read(4)   # reclen
#                     if key < self._rowsCutoff:
#                         u1 = struct.unpack(self._intstru % key,
#                                            fp.read(key*self._ibytes))
#                     else:
#                         u1 = np.fromfile(fp, self._intstr, key)
#                     nwords = u1[9]
#                 else:
#                     break
#                 fp.seek(block, 1)
#                 key = self._getkey()
#             drmcol += 1
#         return drm[:, :drmcol], iddof

    def _rdop2drm(self, name):
        """
        Read Nastran output2 DRM SORT1 data block (table).

        Parameters
        ----------
        name : string
            Name of data block.

        Returns
        -------
        drm : ndarray
            The drm matrix.
        elem_info : ndarray
            2-column matrix of [id, element_type].

        Notes
        -----
        Reads OEF1 and OES1 type data blocks. This routine is beta --
        check output carefully.
        """
        # Expect IDENT/DATA record pairs. They repeat for each element
        # type for each mode.
        # This routine assumes all values are written, even the zeros.

        def _getdrm(pos, e, s, nwords, eids, etypes, ibytes):
            bytes_per_col = pos - s
            nrows = len(eids)
            ncols = (e - s) // bytes_per_col
            dtype = np.int32 if ibytes == 4 else np.int64
            drm = np.empty((nrows, ncols), dtype, order='F')
            elem_info = np.column_stack((eids, etypes))
            elem_info[:, 0] //= 10
            return drm, elem_info

        s = self._fileh.tell()
        _, e = self._get_block_bytes(name, s)

        # read first IDENT above loop & check ACODE/TCODE:
        ident = self.rdop2record()
        n_ident = len(ident)
        achk = self._check_code(ident[0], [4], [[2]], 'ACODE')
        tchk = self._check_code(ident[1], [1, 7], [[1], [0, 2]],
                                'TCODE')
        if not (achk and tchk):
            raise ValueError('invalid ACODE and/or TCODE value')

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
                z = np.zeros((nwords-1, 1), data.dtype)
                eids.extend((data[0]+z).T.ravel())
                etypes.extend([elemtype]*data.shape[1]*(nwords-1))
                column.extend(data[1:].T.ravel())
            else:
                # DATA record:
                data = self.rdop2record(N=n_data[j])
                data = data.reshape(-1, nwords).T
                if drm is None:
                    drm, elem_info = _getdrm(
                        pos, e, s, nwords, eids, etypes, self._ibytes)
                    drm[:, mode-2] = column
                n = (nwords-1) * data.shape[1]
                drm[r:r+n, mode-1] = data[1:].T.ravel()
                j += 1
                r += n
                if r == drm.shape[0]:
                    j = 0
                    r = 0

            # IDENT record:
            ident = self.rdop2record(N=n_ident)

        if drm is None:
            drm, elem_info = _getdrm(
                pos, e, s, eids, etypes, self._ibytes)
            drm[:, mode-1] = column
        drm.dtype = np.float32 if self._ibytes == 4 else np.float64
        return drm, elem_info
#        DRM = namedtuple('DRM', ['drm', 'elem_info'])
#        return DRM(drm=drm, elem_info=elem_info)

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
        written::

            OUTPUT2  XYCDBDRS//0/OP2UNIT $
            OUTPUT2  TOUGV1,TOUGS1,TOUGD1//0/OP2UNIT $
            OUTPUT2  TOQGS1,TOQGD1,TOEFS1,TOEFD1//0/OP2UNIT $
            OUTPUT2  TOESS1,TOESD1//0/OP2UNIT $

        """
        self._fileh.seek(self._postheaderpos)
        drmkeys = {}
        while 1:
            name, trailer, rectype = self._rdop2nt()
            if name is None:
                break
            if rectype > 0:
                if verbose:
                    print("Skipping matrix {}...".format(name))
                self.skipop2matrix(trailer)
            elif len(name) > 2 and name.find('TO') == 0:
                if verbose:
                    print("Reading {}...".format(name))
                # skip record 1
                self.rdop2record()
                # record 2 contains directory
                # - starting at 10: type, id, number, row, 0
                info = self.rdop2record()[10:]
                drmkeys[name.lower()] = (info.reshape(-1, 5).T)[:-1]
                self._rdop2eot()
            elif len(name) > 4 and name[:4] == 'XYCD':
                if verbose:
                    print("Reading {}...".format(name))
                # record 1 contains order of request info
                drmkeys['dr'] = self.rdop2record()
                # record 2 contains sorted list
                drmkeys['drs'] = self.rdop2record().reshape(-1, 6).T
                self._rdop2eot()
            else:
                if verbose:
                    print("Skipping table {}...".format(name))
                self.skipop2table()
        return drmkeys

    def rdn2cop2(self):
        """
        Read Nastran output2 file written by DMAP NAS2CAM; usually
        called by :func:`rdnas2cam`.

        Returns
        -------
        nasop2 : dictionary

        'selist' : array
            2-columns matrix:  [ seid, dnseid ] where, for each row,
            dnseid is the downstream superelement for seid. (dnseid = 0
            if seid = 0).
        'uset' : dictionary
            Indexed by the SE number.  Each member is a 6-column matrix
            described below.
        'cstm' : dictionary
            Indexed by the SE number.  Each member is a 14-column matrix
            containing the coordinate system transformation matrix for
            each coordinate system.  See description below.
        'cstm2' : dictionary
            Indexed by the SE number.  Each member is another dictionary
            indexed by the coordinate system id number.  This has the
            same information as 'cstm', but in a different format.  See
            description below.
        'maps' : dictionary
            Indexed by the SE number.  Each member is a mapping table
            for mapping the A-set order from upstream to downstream;
            see below.
        'dnids' : dictionary
            Indexed by the SE number.  Each member is a vector of ids of
            the A-set ids of grids and spoints for SE in the downstream
            superelement.  When using the CSUPER entry, these will be
            the ids on that entry.  (Does not have each DOF, just ids.)
        'upids' : dictionary
            Indexed by the SE number.  Each member is a vector of ids of
            the A-set grids and spoints for upstream se's that connect
            to SE.  These ids are internally generated and should match
            with 'dnids'.  This allows, for example, the routine
            :func:`n2p.upasetpv` to work.  (Does not have each DOF, just
            ids.)

        Notes
        -----
        The module `n2p` has many routines that use the data created by
        this routine.

        *'uset' description*

        Each USET variable is a 6-column matrix where the rows
        correspond to the DOF in Nastran internal sort, and the columns
        are::

            USET = [ ID DOF Set_Membership Coord_Info ]

        ID is the node id and DOF is 1, 2, 3, 4, 5, or 6 for a grid
        and 0 for an SPOINT.

        Set_Membership is a single column that contains a 32-bit
        integer for each DOF. The integer has bits set to specify
        which Nastran set that particular DOF belongs to. For example,
        if the integer has the 1 bit set, the DOF is in the m-set. See
        the source code in :func:`n2p.mkusetmask` for all the bit
        positions. Note that you rarely (if ever) need to call
        :func:`n2p.mkusetmask` directly since the function
        :func:`n2p.mksetpv` does this behind the scenes to make
        partition vectors.

        For grids, Coord_Info is a 6 row by 3 column matrix::

            Coord_Info = [[x   y    z]    # location of node in basic
                          [id  type 0]    # coord. id and type
                          [xo  yo  zo]    # origin of coord. system
                          [    T     ]]   # 3x3 transformation to basic
                                          #  for coordinate system

            Coord_Info = [ 0 0 0 ]        # for SPOINTs


        *'cstm' description*

        Each CSTM contains all the coordinate system information for
        the superelement.  Some or all of this info is in the USET
        table, but if a coordinate system is not used as an output
        system of any grid, it will not show up in USET.  That is why
        CSTM is here.  CSTM has 14 columns::

            CSTM = [ id type xo yo zo T(1,:) T(2,:) T(3,:) ]

        Note that each CSTM always starts with the two ids 0 and -1.
        The 0 is the basic coordinate system and the -1 is a dummy for
        SPOINTs.  Note the T is transformation between coordinate
        systems as defined (not necessarily the same as the
        transformation for a particular grid ... which, for
        cylindrical and spherical, depends on grid location).  This is
        the same T as in the USET table.

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

        Each CSTM2 is a dictionary with the same 5x3 that the
        'Coord_Info' listed above has (doesn't include the first row
        which is the node location).  The dictionary is indexed by the
        coordinate id.

        *'maps' description*

        MAPS will be [] for superelements whose A-set dof did not get
        rearranged going downstream (on the CSUPER entry.)  For other
        superelements, MAPS will contain two columns: [order, scale].
        The first column reorders upstream A-set to be in the order
        that they appear in the downstream: Down = Up(MAPS(:,1)).  The
        second column is typically 1.0; if not, these routines will
        print an error message and stop.  Together with DNIDS, a
        partition vector can be formed for the A-set of an upstream
        superelement (see :func:`n2p.upasetpv`).

        The op2 file that this routine reads is written by the Nastran
        DMAP NAS2CAM.  The data in the file are expected to be in this
        order::

            SLIST & EMAP or  SUPERID
            For each superelement:
              USET
              EQEXINS
              CSTMS    (if required)
              BGPDTS
              MAPS     (if required)

        Note: The 2nd bit for the DOF column of all USET tables is
        cleared for all S-set.  See :func:`n2p.mkusetmask` for more
        information.

        Example::

          from pyyeti import op2
          from pyyeti import n2p
          # list superelement 100 DOF that are in the B set:
          o2 = op2.OP2('nas2cam.op2')
          nas = op2.rdn2cop2()
          bset = n2p.mksetpv(nas['uset'][100], 'p', 'b')
          print('bset of se100 = ', nas['uset'][100][bset, :2])

        See also
        --------
        :func:`rdnas2cam`, :mod:`n2p`.
        """
        # setup basic coordinate system info and a dummy for spoints:
        bc = np.array([[+0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                       [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        nas = {'uset': {},
               'cstm': {},
               'cstm2': {},
               'maps': {},
               'dnids': {},
               'upids': {}}
        self._fileh.seek(self._postheaderpos)
        # read datablock (slist) header record:
        name, trailer, dbtype = self._rdop2nt()
        if dbtype > 0:
            selist = np.hstack((self.rdop2matrix(trailer), [[0]]))
            selist = selist.astype(int)
            name, trailer, dbtype = self._rdop2nt()
        else:
            selist = self._rdop2selist()
            nse = np.size(selist, 0)
            name, trailer, dbtype = self._rdop2nt()
            if name == "EMAP":
                self._rdop2emap(nas, nse, trailer)
                name, trailer, dbtype = self._rdop2nt()

        # read uset and eqexins tables and do some processing:
        for se in selist[:, 0]:
            if not name:
                break
            uset = self._rdop2uset()
            name, trailer, dbtype = self._rdop2nt()
            eqexin1, eqexin = self._rdop2eqexin()
            name, trailer, dbtype = self._rdop2nt()
            if name == "CSTMS":
                cstm = np.vstack((bc, self._rdop2cstm()))
                name, trailer, dbtype = self._rdop2nt()
            else:
                cstm = bc
            (xyz, cid, dof,
             doftype, nid, upids) = self._proc_bgpdt(eqexin1, eqexin)
            nas['upids'][se] = upids
            Uset, cstm, cstm2 = self._buildUset(se, dof, doftype, nid,
                                                uset, xyz, cid, cstm, None)
            nas['uset'][se] = Uset
            nas['cstm'][se] = cstm
            nas['cstm2'][se] = cstm2
            name, trailer, dbtype = self._rdop2nt()
            if name == "MAPS":
                nas['maps'][se] = self._rdop2maps()
                name, trailer, dbtype = self._rdop2nt()
            else:
                nas['maps'][se] = []
        nas['selist'] = selist
        return nas


def rdnas2cam(op2file='nas2cam', op4file=None):
    """
    Read op2/op4 data written by the DMAP NAS2CAM.

    Parameters
    ----------
    op2file : string
        Either the basename of the .op2 and .op4 files, or the full
        name of the .op2 file
    op4file : string or None
        The name of the .op4 file or, if None, builds name from the
        `op2file` input.

    Returns
    -------
    nas : dictionary
        Dictionary with all members created by :func:`rdn2cop2` (see
        that routine's help) and the following additional members.

    'nrb' : integer
        The number of rigid-body modes for residual.
    'ulvs' : dictionary indexed be SE
        The ULVS matrices (row partitions of residual modes to the
        A-set DOF of the SE).
    'lambda' : dictionary indexed be SE
        The eigenvalues for each SE.
    'gm' : dictionary indexed be SE
        N-set to M-set transformation matrix GM:  M = GM N.
    'got' : dictionary indexed be SE
        constraint modes
    'goq' : dictionary indexed be SE
        normal modes
    'rfmodes' : dictionary indexed be SE
        index partition vector for res-flex modes
    'maa' : dictionary indexed be SE
        A-set mass
    'baa' : dictionary indexed be SE
        A-set damping
    'kaa' : dictionary indexed be SE
        A-set stiffness
    'pha' : dictionary indexed be SE
        A-set modes
    'mdd' : dictionary indexed be SE
        D-set mass
    'bdd' : dictionary indexed be SE
        D-set damping
    'kdd' : dictionary indexed be SE
        D-set stiffness
    'pdt' : dictionary indexed be SE
        D-set loads
    'mgg' : dictionary indexed be SE
        G-set mass
    'kgg' : dictionary indexed be SE
        G-set stiffness
    'phg' : dictionary indexed be SE
        G-set mode shape matrix
    'rbg' : dictionary indexed be SE
        G-set rigid-body modes; see also drg output and rbgeom_uset
    'drg' : dictionary indexed be SE
        G-set transpose of rigid-body modes; see also 'rbg' and
        :func:`n2p.rbgeom_uset`.  `drg` = `rbg.T` if both are
        present.
    'pg' : dictionary indexed be SE
        G-set loads
    'fgravh' : array
        gravity on generalized dof for se 0
    'fgravg' : array
        gravity on G-set physical dof for se 0

    Notes
    -----
    See :func:`rdn2cop2` for a description of what is expected of the
    `op2file`.  The `op4file` is expected to contain certain marker
    matrices.  Scalar SE_START starts each superelement and can be
    followed by any matrices for that superelement.  The end of the
    superelement input is marked by a matrix named LOOP_END.

    See also the Nastran DMAP NAS2CAM.
    """
    if not op4file:
        op4file = op2file+'.op4'
        op2file = op2file+'.op2'

    # read op2 file:
    with OP2(op2file) as o2:
        nas = o2.rdn2cop2()

    # read op4 file:
    op4names, op4vars = op4.load(op4file, into='list')[:2]

    # loop over superelements:
    j = 0
    for se in nas['selist'][:, 0]:
        if op4names[j] != "se_start":
            raise ValueError("matrices are not in understandable"
                             " order.  Expected 'se_start', got "
                             "'{}'".format(op4names[j]))
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
                nrb = sum(op4vars[j] < .005)[0]
                nas['nrb'] = nrb
                nas['lambda'][0] = abs(op4vars[j].ravel())
            elif name == 'lambda':
                nas[name][se] = op4vars[j].ravel()
            elif name == 'rfmodes':
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


def get_dof_descs():
    """
    Returns dictionary of descriptions for Nastran data recovery items.
    Normally called by :func:`procdrm12`.

    Returns
    -------
    desc : dictionary
        Has keys: 'acce', 'spcf', 'stress', 'force'::

            desc['acce'] : numpy string array
                ['T1', 'T2', 'T3',  'R1', 'R2', 'R3']
            desc['spcf'] : numpy string array
                ['Fx', 'Fy', 'Fz',  'Mx', 'My', 'Mz']
            desc['stress'] : dict
                Dictionary with element numbers as keys to numpy
                string arrays.
            desc['stress'] : dict
                Dictionary with element numbers as keys to numpy
                string arrays.

    Notes
    -----
    The stress and force returns are dictionaries indexed by the
    element id.  For example, for the CBAR (which is element 34)::

        desc['stress'][34] = ['CBAR Bending Stress 1 - End A',
                              'CBAR Bending Stress 2 - End A',
                              ...]
        desc['force'][34] = ['CBAR Bending Moment 1 - End A',
                             'CBAR Bending Moment 2 - End A',
                             ...]

    """
    #   Acceleration, Velocity, Displacement Recovery Items:
    accedesc = ["T1", "T2", "T3", "R1", "R2", "R3"]
    spcfdesc = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    stress = {}
    force = {}

    #  CBAR Recovery Items (element 34):                        Item code
    stress[34] = ["CBAR Bending Stress 1 - End A",         # 2
                  "CBAR Bending Stress 2 - End A",         # 3
                  "CBAR Bending Stress 3 - End A",         # 4
                  "CBAR Bending Stress 4 - End A",         # 5
                  "CBAR Axial Stress",                     # 6
                  "CBAR Max. Bend. Stress -End A",         # 7
                  "CBAR Min. Bend. Stress -End A",         # 8
                  "CBAR M.S. Tension",                     # 9
                  "CBAR Bending Stress 1 - End B",         # 10
                  "CBAR Bending Stress 2 - End B",         # 11
                  "CBAR Bending Stress 3 - End B",         # 12
                  "CBAR Bending Stress 4 - End B",         # 13
                  "CBAR Max. Bend. Stress -End B",         # 14
                  "CBAR Min. Bend. Stress -End B",         # 15
                  "CBAR M.S. Compression"]                 # 16

    force[34] = ["CBAR Bending Moment 1 - End A",         # 2
                 "CBAR Bending Moment 2 - End A",         # 3
                 "CBAR Bending Moment 1 - End B",         # 4
                 "CBAR Bending Moment 2 - End B",         # 5
                 "CBAR Shear 1",                          # 6
                 "CBAR Shear 2",                          # 7
                 "CBAR Axial Force",                      # 8
                 "CBAR Torque"]                           # 9

    #   CBEAM Recovery Items (element 2):                        Item code
    stress2_main = ["CBEAM External grid pt. ID",          # 2
                    "CBEAM Station dist./length",          # 3
                    "CBEAM Long. Stress at Pt. C",         # 4
                    "CBEAM Long. Stress at Pt. D",         # 5
                    "CBEAM Long. Stress at Pt. E",         # 6
                    "CBEAM Long. Stress at Pt. F",         # 7
                    "CBEAM Maximum stress",                # 8
                    "CBEAM Minimum stress",                # 9
                    "CBEAM M.S. Tension",                  # 10
                    "CBEAM M.S. Compression"]              # 11

    # expand and append station id for all 11 stations:
    stress2 = [i+' End-A' for i in stress2_main]
    for K in range(2, 11):
        id_string = ' K={:2}'.format(K)
        stress2 += [i+id_string for i in stress2_main]
    stress2 += [i+' End-B' for i in stress2_main]
    stress[2] = stress2

    force2_main = ["CBEAM External grid pt. ID",             # 2
                   "CBEAM Station dist./length",             # 3
                   "CBEAM Bending moment plane 1",           # 4
                   "CBEAM Bending moment plane 2",           # 5
                   "CBEAM Web shear plane 1",                # 6
                   "CBEAM Web shear plane 2",                # 7
                   "CBEAM Axial force",                      # 8
                   "CBEAM Total torque",                     # 9
                   "CBEAM Warping torque"]                   # 10

    # expand and append station id for all 11 stations:
    force2 = [i+' End-A' for i in force2_main]
    for K in range(2, 11):
        id_string = ' K={:2}'.format(K)
        force2 += [i+id_string for i in force2_main]
    force2 += [i+' End-B' for i in force2_main]
    force[2] = force2

    #   CBUSH Recovery Items (element 102):                        Item code
    stress[102] = ["CBUSH Translation-x",         # 2
                   "CBUSH Translation-y",         # 3
                   "CBUSH Translation-z",         # 4
                   "CBUSH Rotation-x",            # 5
                   "CBUSH Rotation-y",            # 6
                   "CBUSH Rotation-z"]            # 7

    force[102] = ["CBUSH Force-x",          # 2
                  "CBUSH Force-y",          # 3
                  "CBUSH Force-z",          # 4
                  "CBUSH Moment-x",         # 5
                  "CBUSH Moment-y",         # 6
                  "CBUSH Moment-z"]         # 7

    #   CROD Recovery Items (element 10=CONROD, 1=CROD):
    stress1 = ["Axial Stress",              # 2
               "M.S. Axial Stress",         # 3
               "Torsional Stress",          # 4
               "M.S. Torsional Stress"]     # 5
    force1 = ["Axial Force",        # 2
              "Torque"]             # 3
    stress[1] = ['CROD '+i+'  ' for i in stress1]
    force[1] = ['CROD '+i+'  ' for i in force1]
    stress[10] = ['CONROD '+i for i in stress1]
    force[10] = ['CONROD '+i for i in force1]

    #   CELAS1, 2, 3 Recovery Items (elements 11, 12, 13):
    stress[11] = 'CELAS1 Stress'
    stress[12] = 'CELAS2 Stress'
    stress[13] = 'CELAS3 Stress'
    force[11] = 'CELAS1 Force'
    force[12] = 'CELAS2 Force'
    force[13] = 'CELAS3 Force'

    #   CQUAD4 Recovery Items (element 33):
    stress[33] = ["CQUAD4 Fiber distance Z1",           # 2
                  "CQUAD4 Z1 Normal x",                 # 3
                  "CQUAD4 Z1 Normal y",                 # 4
                  "CQUAD4 Z1 Shear xy",                 # 5
                  "CQUAD4 Z1 Shear angle",              # 6
                  "CQUAD4 Z1 Major principal",          # 7
                  "CQUAD4 Z1 Minor principal",          # 8
                  "CQUAD4 Z1 von Mises or max shear",   # 9
                  "CQUAD4 Fiber distance Z2",           # 10
                  "CQUAD4 Z2 Normal x",                 # 11
                  "CQUAD4 Z2 Normal y",                 # 12
                  "CQUAD4 Z2 Shear xy",                 # 13
                  "CQUAD4 Z2 Shear angle",              # 14
                  "CQUAD4 Z2 Major principal",          # 15
                  "CQUAD4 Z2 Minor principal",          # 16
                  "CQUAD4 Z2 von Mises or max shear"]   # 17

    force[33] = ["CQUAD4 Membrane force x",         # 2
                 "CQUAD4 Membrane force y",         # 3
                 "CQUAD4 Membrane force xy",        # 4
                 "CQUAD4 Bending moment x",         # 5
                 "CQUAD4 Bending moment y",         # 6
                 "CQUAD4 Bending moment xy",        # 7
                 "CQUAD4 Shear x",                  # 8
                 "CQUAD4 Shear y"]                  # 9

    #   CQUADR Recovery Items (element 82, and CQUAD8-64):
    stress[82] = ["CQUADR EID                         ",      # 1
                  "CQUADR CEN/                        ",      # 2
                  "CQUADR 4                           ",      # 3
                  "CQUADR Fiber distance Z1           ",      # 4
                  "CQUADR Z1 Normal x                 ",      # 5
                  "CQUADR Z1 Normal y                 ",      # 6
                  "CQUADR Z1 Shear xy                 ",      # 7
                  "CQUADR Z1 Shear angle              ",      # 8
                  "CQUADR Z1 Major principal          ",      # 9
                  "CQUADR Z1 Minor principal          ",      # 10
                  "CQUADR Z1 von Mises or max shear   ",      # 11
                  "CQUADR Fiber distance Z2           ",      # 12
                  "CQUADR Z2 Normal x                 ",      # 13
                  "CQUADR Z2 Normal y                 ",      # 14
                  "CQUADR Z2 Shear xy                 ",      # 15
                  "CQUADR Z2 Shear angle              ",      # 16
                  "CQUADR Z2 Major principal          ",      # 17
                  "CQUADR Z2 Minor principal          ",      # 18
                  "CQUADR Z2 von Mises or max shear   ",      # 19

                  "CQUADR Grid 1                      ",      # 20
                  "CQUADR Fiber distance Z1         c1",      # 21
                  "CQUADR Z1 Normal x               c1",      # 22
                  "CQUADR Z1 Normal y               c1",      # 23
                  "CQUADR Z1 Shear xy               c1",      # 24
                  "CQUADR Z1 Shear angle            c1",      # 25
                  "CQUADR Z1 Major principal        c1",      # 26
                  "CQUADR Z1 Minor principal        c1",      # 27
                  "CQUADR Z1 von Mises or max shear c1",      # 28
                  "CQUADR Fiber distance Z2         c1",      # 29
                  "CQUADR Z2 Normal x               c1",      # 30
                  "CQUADR Z2 Normal y               c1",      # 31
                  "CQUADR Z2 Shear xy               c1",      # 32
                  "CQUADR Z2 Shear angle            c1",      # 33
                  "CQUADR Z2 Major principal        c1",      # 34
                  "CQUADR Z2 Minor principal        c1",      # 35
                  "CQUADR Z2 von Mises or max shear c1",      # 36

                  "CQUADR Grid 2                      ",      # 37
                  "CQUADR Fiber distance Z1         c2",      # 38
                  "CQUADR Z1 Normal x               c2",      # 39
                  "CQUADR Z1 Normal y               c2",      # 40
                  "CQUADR Z1 Shear xy               c2",      # 41
                  "CQUADR Z1 Shear angle            c2",      # 42
                  "CQUADR Z1 Major principal        c2",      # 43
                  "CQUADR Z1 Minor principal        c2",      # 44
                  "CQUADR Z1 von Mises or max shear c2",      # 45
                  "CQUADR Fiber distance Z2         c2",      # 46
                  "CQUADR Z2 Normal x               c2",      # 47
                  "CQUADR Z2 Normal y               c2",      # 48
                  "CQUADR Z2 Shear xy               c2",      # 49
                  "CQUADR Z2 Shear angle            c2",      # 50
                  "CQUADR Z2 Major principal        c2",      # 51
                  "CQUADR Z2 Minor principal        c2",      # 52
                  "CQUADR Z2 von Mises or max shear c2",      # 53

                  "CQUADR Grid 3                      ",      # 54
                  "CQUADR Fiber distance Z1         c3",      # 55
                  "CQUADR Z1 Normal x               c3",      # 56
                  "CQUADR Z1 Normal y               c3",      # 57
                  "CQUADR Z1 Shear xy               c3",      # 58
                  "CQUADR Z1 Shear angle            c3",      # 59
                  "CQUADR Z1 Major principal        c3",      # 60
                  "CQUADR Z1 Minor principal        c3",      # 61
                  "CQUADR Z1 von Mises or max shear c3",      # 62
                  "CQUADR Fiber distance Z2         c3",      # 63
                  "CQUADR Z2 Normal x               c3",      # 64
                  "CQUADR Z2 Normal y               c3",      # 65
                  "CQUADR Z2 Shear xy               c3",      # 66
                  "CQUADR Z2 Shear angle            c3",      # 67
                  "CQUADR Z2 Major principal        c3",      # 68
                  "CQUADR Z2 Minor principal        c3",      # 69
                  "CQUADR Z2 von Mises or max shear c3",      # 70

                  "CQUADR Grid 4                      ",      # 71
                  "CQUADR Fiber distance Z1         c4",      # 72
                  "CQUADR Z1 Normal x               c4",      # 73
                  "CQUADR Z1 Normal y               c4",      # 74
                  "CQUADR Z1 Shear xy               c4",      # 75
                  "CQUADR Z1 Shear angle            c4",      # 76
                  "CQUADR Z1 Major principal        c4",      # 77
                  "CQUADR Z1 Minor principal        c4",      # 78
                  "CQUADR Z1 von Mises or max shear c4",      # 79
                  "CQUADR Fiber distance Z2         c4",      # 80
                  "CQUADR Z2 Normal x               c4",      # 81
                  "CQUADR Z2 Normal y               c4",      # 82
                  "CQUADR Z2 Shear xy               c4",      # 83
                  "CQUADR Z2 Shear angle            c4",      # 84
                  "CQUADR Z2 Major principal        c4",      # 85
                  "CQUADR Z2 Minor principal        c4",      # 86
                  "CQUADR Z2 von Mises or max shear c4"]      # 87

    force[82] = ["CQUADR Membrane force x            ",      # 4
                 "CQUADR Membrane force y            ",      # 5
                 "CQUADR Membrane force xy           ",      # 6
                 "CQUADR Bending moment x            ",      # 7
                 "CQUADR Bending moment y            ",      # 8
                 "CQUADR Bending moment xy           ",      # 9
                 "CQUADR Shear x                     ",      # 10
                 "CQUADR Shear y                     ",      # 11

                 "CQUADR   (non-documented item)     ",      # 12

                 "CQUADR Membrane force x          c1",      # 13
                 "CQUADR Membrane force y          c1",      # 14
                 "CQUADR Membrane force xy         c1",      # 15
                 "CQUADR Bending moment x          c1",      # 16
                 "CQUADR Bending moment y          c1",      # 17
                 "CQUADR Bending moment xy         c1",      # 18
                 "CQUADR Shear x                   c1",      # 19
                 "CQUADR Shear y                   c1",      # 20

                 "CQUADR   (non-documented item)     ",      # 21

                 "CQUADR Membrane force x          c2",      # 22
                 "CQUADR Membrane force y          c2",      # 23
                 "CQUADR Membrane force xy         c2",      # 24
                 "CQUADR Bending moment x          c2",      # 25
                 "CQUADR Bending moment y          c2",      # 26
                 "CQUADR Bending moment xy         c2",      # 27
                 "CQUADR Shear x                   c2",      # 28
                 "CQUADR Shear y                   c2",      # 29

                 "CQUADR   (non-documented item)     ",      # 30

                 "CQUADR Membrane force x          c3",      # 31
                 "CQUADR Membrane force y          c3",      # 32
                 "CQUADR Membrane force xy         c3",      # 33
                 "CQUADR Bending moment x          c3",      # 34
                 "CQUADR Bending moment y          c3",      # 35
                 "CQUADR Bending moment xy         c3",      # 36
                 "CQUADR Shear x                   c3",      # 37
                 "CQUADR Shear y                   c3",      # 38

                 "CQUADR   (non-documented item)     ",      # 39

                 "CQUADR Membrane force x          c4",      # 40
                 "CQUADR Membrane force y          c4",      # 41
                 "CQUADR Membrane force xy         c4",      # 42
                 "CQUADR Bending moment x          c4",      # 43
                 "CQUADR Bending moment y          c4",      # 44
                 "CQUADR Bending moment xy         c4",      # 45
                 "CQUADR Shear x                   c4",      # 46
                 "CQUADR Shear y                   c4"]      # 47
    stress[64] = [i.replace('CQUADR', 'CQ8-64') for i in stress[82]]
    force[64] = [i.replace('CQUADR', 'CQ8-64') for i in force[82]]

    #   CTRIAR Recovery Items (element 70, and CTRIA6-75):
    stress[70] = ["CTRIAR Z1 Normal x                 ",       # 5
                  "CTRIAR Z1 Normal y                 ",       # 6
                  "CTRIAR Z1 Shear xy                 ",       # 7
                  "CTRIAR Z1 Q shear angle            ",       # 8
                  "CTRIAR Z1 Major principal          ",       # 9
                  "CTRIAR Z1 Minor principal          ",       # 10
                  "CTRIAR Z1 von Mises or max shear   ",       # 11
                  "CTRIAR   (non-documented item)     ",       # 12
                  "CTRIAR Z2 Normal x                 ",       # 13
                  "CTRIAR Z2 Normal y                 ",       # 14
                  "CTRIAR Z2 Shear xy                 ",       # 15
                  "CTRIAR Z2 Q shear angle            ",       # 16
                  "CTRIAR Z2 Major principal          ",       # 17
                  "CTRIAR Z2 Minor principal          ",       # 18
                  "CTRIAR Z2 von Mises or max shear   ",       # 19

                  "CTRIAR   (non-documented item)     ",       # 20
                  "CTRIAR   (non-documented item)     ",       # 21

                  "CTRIAR Z1 Normal x               c1",       # 22
                  "CTRIAR Z1 Normal y               c1",       # 23
                  "CTRIAR Z1 Shear xy               c1",       # 24
                  "CTRIAR Z1 Q shear angle          c1",       # 25
                  "CTRIAR Z1 Major principal        c1",       # 26
                  "CTRIAR Z1 Minor principal        c1",       # 27
                  "CTRIAR Z1 von Mises or max shear c1",       # 28
                  "CTRIAR   (non-documented item)   c1",       # 29
                  "CTRIAR Z2 Normal x               c1",       # 30
                  "CTRIAR Z2 Normal y               c1",       # 31
                  "CTRIAR Z2 Shear xy               c1",       # 32
                  "CTRIAR Z2 Q shear angle          c1",       # 33
                  "CTRIAR Z2 Major principal        c1",       # 34
                  "CTRIAR Z2 Minor principal        c1",       # 35
                  "CTRIAR Z2 von Mises or max shear c1",       # 36

                  "CTRIAR   (non-documented item)     ",       # 37
                  "CTRIAR   (non-documented item)     ",       # 38

                  "CTRIAR Z1 Normal x               c2",       # 39
                  "CTRIAR Z1 Normal y               c2",       # 40
                  "CTRIAR Z1 Shear xy               c2",       # 41
                  "CTRIAR Z1 Q shear angle          c2",       # 42
                  "CTRIAR Z1 Major principal        c2",       # 43
                  "CTRIAR Z1 Minor principal        c2",       # 44
                  "CTRIAR Z1 von Mises or max shear c2",       # 45
                  "CTRIAR   (non-documented item)   c2",       # 46
                  "CTRIAR Z2 Normal x               c2",       # 47
                  "CTRIAR Z2 Normal y               c2",       # 48
                  "CTRIAR Z2 Shear xy               c2",       # 49
                  "CTRIAR Z2 Q shear angle          c2",       # 50
                  "CTRIAR Z2 Major principal        c2",       # 51
                  "CTRIAR Z2 Minor principal        c2",       # 52
                  "CTRIAR Z2 von Mises or max shear c2",       # 53

                  "CTRIAR   (non-documented item)     ",       # 54
                  "CTRIAR   (non-documented item)     ",       # 55

                  "CTRIAR Z1 Normal x               c3",       # 56
                  "CTRIAR Z1 Normal y               c3",       # 57
                  "CTRIAR Z1 Shear xy               c3",       # 58
                  "CTRIAR Z1 Q shear angle          c3",       # 59
                  "CTRIAR Z1 Major principal        c3",       # 60
                  "CTRIAR Z1 Minor principal        c3",       # 61
                  "CTRIAR Z1 von Mises or max shear c3",       # 62
                  "CTRIAR   (non-documented item)   c3",       # 63
                  "CTRIAR Z2 Normal x               c3",       # 64
                  "CTRIAR Z2 Normal y               c3",       # 65
                  "CTRIAR Z2 Shear xy               c3",       # 66
                  "CTRIAR Z2 Q shear angle          c3",       # 67
                  "CTRIAR Z2 Major principal        c3",       # 68
                  "CTRIAR Z2 Minor principal        c3",       # 69
                  "CTRIAR Z2 von Mises or max shear c3"]       # 70

    force[70] = ["CTRIAR Membrane force x            ",      # 4
                 "CTRIAR Membrane force y            ",      # 5
                 "CTRIAR Membrane force xy           ",      # 6
                 "CTRIAR Bending moment x            ",      # 7
                 "CTRIAR Bending moment y            ",      # 8
                 "CTRIAR Bending moment xy           ",      # 9
                 "CTRIAR Shear x                     ",      # 10
                 "CTRIAR Shear y                     ",      # 11

                 "CTRIAR   (non-documented item)     ",      # 12

                 "CTRIAR Membrane force x          c1",      # 13
                 "CTRIAR Membrane force y          c1",      # 14
                 "CTRIAR Membrane force xy         c1",      # 15
                 "CTRIAR Bending moment x          c1",      # 16
                 "CTRIAR Bending moment y          c1",      # 17
                 "CTRIAR Bending moment xy         c1",      # 18
                 "CTRIAR Shear x                   c1",      # 19
                 "CTRIAR Shear y                   c1",      # 20

                 "CTRIAR   (non-documented item)     ",      # 21

                 "CTRIAR Membrane force x          c2",      # 22
                 "CTRIAR Membrane force y          c2",      # 23
                 "CTRIAR Membrane force xy         c2",      # 24
                 "CTRIAR Bending moment x          c2",      # 25
                 "CTRIAR Bending moment y          c2",      # 26
                 "CTRIAR Bending moment xy         c2",      # 27
                 "CTRIAR Shear x                   c2",      # 28
                 "CTRIAR Shear y                   c2",      # 29

                 "CTRIAR   (non-documented item)     ",      # 30

                 "CTRIAR Membrane force x          c3",      # 31
                 "CTRIAR Membrane force y          c3",      # 32
                 "CTRIAR Membrane force xy         c3",      # 33
                 "CTRIAR Bending moment x          c3",      # 34
                 "CTRIAR Bending moment y          c3",      # 35
                 "CTRIAR Bending moment xy         c3",      # 36
                 "CTRIAR Shear x                   c3",      # 37
                 "CTRIAR Shear y                   c3"]      # 38

    stress[75] = [i.replace('CTRIAR', 'CT6-75') for i in stress[70]]
    force[75] = [i.replace('CTRIAR', 'CT6-75') for i in force[70]]
    for i in stress:
        stress[i] = np.array(stress[i])
        force[i] = np.array(force[i])
    return {'acce': np.array(accedesc),
            'spcf': np.array(spcfdesc),
            'stress': stress,
            'force': force}


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
    return tinr[0, 0], tinr[3, 0]-1


def get_drm(drminfo, otm, drms, drmkeys, dr, desc):
    """
    Called by :func:`procdrm12` to add displacement-dependent data
    recovery items to the otm input.

    Parameters
    ----------
    drminfo : tuple
        DRM Information; (output drm name, 3 or 5 character Nastran
        name, description index).

        - if the second input is 3 chars, say '---', this routine
          uses the following members of `drms` and `drmkeys`::

            'm---d1', 'm---s1' and 't---d1' if available (mode-acce), or
            'm---x1', 't---x1' if not (mode-disp)

        - if the second input is 5 chars, say '-----', this routine
          uses 'm-----' and 't-----'
        - the description index is used to get info from `desc`.

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
        Output of :func:`get_dof_descs`.

    Examples usages::

        get_drm(('DTM', 'oug', 'acce'), otm, drms, drmkeys, dr, desc)
        get_drm(('ATM', 'ougv1', 'acce'), ...)
        get_drm(('LTM', 'oef', 'force'), ...)
        get_drm(('SPCF', 'oqg', 'spcf'), ...)
        get_drm(('STM', 'oes', 'stress'), ...)

    """
    drc = dr.shape[1]
    ID = dr[1, :]
    DOF = dr[2, :]
    nm, nasnm, desci = drminfo
    otm[nm+'_id_dof'] = np.vstack((ID, DOF)).T

    # arg offset is for translating between Nastran argument to
    # matrix index; eg 'x' recovery for a grid is arg 3, so offset
    # is 3
    if nasnm.find('oug') > -1 or nasnm.find('oqg') > -1:
        offset = 3
        otm[nm+'_id_dof'][:, 1] -= 2
    else:
        offset = 2

    if not isinstance(desc[desci], dict):
        otm[nm+'_desc'] = desc[desci][DOF-offset]
        getdesc = False
    else:
        getdesc = True
        _desc = nm+'_desc'
        otm[_desc] = [''] * drc
        _dct = desc[desci]
        _name = desci.capitalize()

    if len(nasnm) == 3 and 'm'+nasnm+'d1' in drms:
        d1 = drms['m'+nasnm+'d1'][0]
        s1 = drms['m'+nasnm+'s1'][0]
        iddof = drmkeys['t'+nasnm+'d1']
        acce = nm+'A'
        disp = nm+'D'
        otm[acce] = np.zeros((drc, d1.shape[1]))
        otm[disp] = np.zeros((drc, s1.shape[1]))
        lastid = -1
        for j in range(drc):  # loop over requests
            # find rows corresponding to requested grid
            if ID[j] != lastid:
                eltype, srow = _get_tinr(iddof, ID[j])
                lastid = ID[j]
            otm[acce][j] = d1[srow+DOF[j]-offset]
            otm[disp][j] = s1[srow+DOF[j]-offset]
            if getdesc:
                if eltype in _dct:
                    otm[_desc][j] = _dct[eltype][DOF[j]-offset]
                else:
                    otm[_desc][j] = ('EL-{}, El. Type {:3}, '
                                     'Code {:3}  ').format(_name,
                                                           eltype,
                                                           DOF[j])
    else:
        if len(nasnm) == 3:
            matname = 'm'+nasnm+'x1'
            tabname = 't'+nasnm+'x1'
        else:
            matname = 'm'+nasnm
            tabname = 't'+nasnm
        x1 = drms[matname][0]
        iddof = drmkeys[tabname]
        otm[nm] = np.zeros((drc, x1.shape[1]))
        lastid = -1
        for j in range(drc):  # loop over requests
            # find rows corresponding to requested grid
            if ID[j] != lastid:
                eltype, srow = _get_tinr(iddof, ID[j])
                lastid = ID[j]
            otm[nm][j] = x1[srow+DOF[j]-offset]
            if getdesc:
                if eltype in _dct:
                    otm[_desc][j] = _dct[eltype][DOF[j]-offset]
                else:
                    otm[_desc][j] = ('EL-{}, El. Type {:3}, '
                                     'Code {:3}  ').format(_name,
                                                           eltype,
                                                           DOF[j])


def procdrm12(op2file, op4file=None, dosort=True):
    """
    Process op2/op4 file2 output from DRM1/DRM2 DMAPs to form data
    recovery matrices.

    Parameters
    ----------
    op2file : string
        Either the basename of the .op2 and .op4 files, or the full
        name of the .op2 file
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
        descriptions.  The potential DRM keys are::

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

        The id/dof matrices are each 2 columns of [id, dof] with number
        of rows equal to the number of rows in corresponding DRM.  The
        keys are the applicable strings from::

            'ATM_id_dof'
            'DTM_id_dof'
            'LTM_id_dof' - dof is actually the Nastran item code
            'SPCF_id_dof'
            'STM_id_dof' - dof is actually the Nastran item code

        The descriptions are arrays of strings with generic descriptions
        for each data recovery item.  Length is equal to number of rows
        in corresponding DRM. See :func:`get_dof_descs` for more
        information.  The keys are the applicable strings from::

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
    if not op4file:
        op4file = op2file+'.op4'
        op2file = op2file+'.op2'
    # read op4 file:
    drms = op4.load(op4file)

    with OP2(op2file) as o2:
        drmkeys = o2.rddrm2op2()
    N = drmkeys['drs'].shape[1]

    # drs format:
    # 6 elements per recovery item:
    #    1  -  Subcase number (0 for all)
    #    2  -  Vector request type
    #    3  -  Point or Element ID
    #    4  -  Component
    #    5  -  XY output type
    #    6  -  Destination code

    # Vector request type:
    Vreq = ["Displacement",     # 1
            "Velocity",         # 2
            "Acceleration",     # 3
            "SPC Force",        # 4
            "Load",             # 5
            "Stress",           # 6
            "Element Force",    # 7
            "SDisplacement",    # 8
            "SVelocity",        # 9
            "SAcceleration",    # 10
            "Nonlinear Force",  # 11
            "Total"]            # 12

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
        dr = drmkeys['dr']
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
        #     ... r + 9, 10, 11 can repeat until three -1's are reached
        #        These 3 values repeat when there is a comma: 1(T1),1(T2)
        # dr(X:X+2) = -1, -1, -1
        # 8-X+2 repeat until all dof for an XYPEAK are listed
        #        This section repeats when there is a slash: 1(T1)/1(T2)
        DR = np.zeros((3, N), dtype=int)  # [type; id; dof]
        R = 0  # index into DR columns
        for j in range(n):  # loop over XYPEAK cards
            curtype = dr[r[j]+5]
            J = r[j] + 9  # index to first id
            while J < r[j+1]:
                while dr[J] != -1:
                    DR[:, R] = curtype, dr[J], dr[J+1]
                    R += 1
                    J += 3
                J += 4  # jump over [-1,-1,-1,#]
    else:
        DR = drmkeys['drs'][1:4]  # use sorted version

    desc = get_dof_descs()
    drminfo = {1: ('DTM', 'oug', 'acce'),
               3: ('ATM', 'ougv1', 'acce'),
               4: ('SPCF', 'oqg', 'spcf'),
               6: ('STM', 'oes', 'stress'),
               7: ('LTM', 'oef', 'force')}
    otm = {}
    types = np.array([1, 3, 4, 6, 7])
    for drtype in range(1, 13):
        pv = np.nonzero(DR[0] == drtype)[0]
        if pv.size > 0:
            if np.any(drtype == types):
                print('Processing "{}" requests...'.
                      format(Vreq[drtype-1]))
                get_drm(drminfo[drtype], otm, drms,
                        drmkeys, DR[:, pv], desc)
            else:
                print('Skipping "{}" requests.  Needs to be added '
                      'to procdrm12().'.format(Vreq[drtype-1]))
    return otm


def rdpostop2(op2file, verbose=False, getougv1=False, getoef1=False,
              getoes1=False):
    """
    Reads PARAM,POST,-1 op2 file and returns dictionary of data.

    Parameters
    ----------
    op2file : string
        Name of op2 file.
    verbose : bool
        If True, echo names of tables and matrices to screen
    getougv1 : bool
        If True, read the OUGV1 or BOPHIG matrices, if any
    getoef1 : bool
        If True, read the OEF1* matrices, if any
    getoes1 : bool
        If True, read the OES1* matrices, if any

    Returns
    -------
    pop2 : dictionary
        With following members.
    'uset' : array
        6-column matrix as described in class OP2, member function
        :func:`rdn2cop2`.
    'cstm' : array
        14-column matrix containing the coordinate system
        transformation matrix for each coordinate system.  See
        description in class OP2, member function :func:`rdn2cop2`.
    'cstm2' : dictionary
        Dictionary indexed by the coordinate system id number.  This
        has the same information as 'cstm', but in a different format.
        See description in class OP2, member function :func:`rdn2cop2`.
    'mats' : dictionary
        Dictionary of matrices read from op2 file and indexed by the
        name.  The 'tload' entry is a typical entry.  Will also
        contain lists of 'OUGV1', 'EOF1*', and 'EOS1*' matrices if
        the respective `get*` flag is set and those entries are
        present.
    """
    # read op2 file:
    with OP2(op2file) as o2:
        mats = {}
        selist = uset = cstm2 = sebulk = None
        se = 0
        o2._fileh.seek(o2._postheaderpos)

        eqexin1 = eqexin = None
        bgpdt_rec1 = None
        dof = None
        Uset = None
        cstm = None
        while 1:
            name, trailer, dbtype = o2._rdop2nt()
            if name is None:
                break
            if dbtype > 0:
                if verbose:
                    print("Reading matrix {}...".format(name))
                if name not in mats:
                    mats[name] = []
                mats[name] += [o2.rdop2matrix(trailer)]
            else:
                if name.find('BGPDT') == 0:
                    if verbose:
                        print("Reading table {}...".format(name))
                    bgpdt_rec1 = o2._rdop2bgpdt68()
                    o2.skipop2table()
                    continue

                # if name.find('CSTM') == 0:
                #     if verbose:
                #         print("Reading table {}...".format(name))
                #     cstm = o2._rdop2cstm68().reshape((-1, 14))
                #     cstm = np.vstack((bc, cstm))
                #     continue

                if name.find('GEOM1') == 0:
                    if verbose:
                        print("Reading table {}...".format(name))
                    cords, sebulk, selist = o2._rdop2geom1cord2()
                    if 0 not in cords:
                        cords[0] = np.array([[0., 1., 0.],
                                             [0., 0., 0.],
                                             [1., 0., 0.],
                                             [0., 1., 0.],
                                             [0., 0., 1.]])
                    if -1 not in cords:
                        cords[-1] = np.zeros((5, 3))  # dummy for spoints
                        cords[-1][0, 0] = -1
                    cstm2 = cords
                    continue

                if name.find('DYNAMIC') == 0:
                    if verbose:
                        print("Reading table {}...".format(name))
                    mats['tload'] = o2.rdop2dynamics()
                    continue

                if name.find('EQEXIN') == 0:
                    if verbose:
                        print("Reading table {}...".format(name))
                    eqexin1, eqexin = o2._rdop2eqexin()
                    continue

                if name.find('USET') == 0:
                    if verbose:
                        print("Reading table {}...".format(name))
                    uset = o2._rdop2uset()
                    continue

                if getougv1 and (name.find('OUGV1') == 0 or
                                 name.find('BOPHIG') == 0):
                    if verbose:
                        print("Reading table {}...".format(name))
                    try:
                        mo = mats['ougv1']
                    except KeyError:
                        mo = mats['ougv1'] = []
                    mo += [o2._rdop2ougv1(name)]
                    continue

                if getoef1 and name.startswith('OEF1'):
                    if verbose:
                        print("Reading table {}...".format(name))
                    try:
                        mo = mats['oef1']
                    except KeyError:
                        mo = mats['oef1'] = []
                    # mo += [o2._rdop2drm_old()]
                    mo += [o2._rdop2drm(name)]
                    continue

                if getoes1 and name.startswith('OES1'):
                    if verbose:
                        print("Reading table {}...".format(name))
                    try:
                        mo = mats['oes1']
                    except KeyError:
                        mo = mats['oes1'] = []
                    # mo += [o2._rdop2drm_old()]
                    mo += [o2._rdop2drm(name)]
                    continue

                if verbose:
                    print("Skipping table {}...".format(name))
                o2.skipop2table()

        if (eqexin1 is not None and
            eqexin is not None and
            bgpdt_rec1 is not None):
            (xyz, cid, dof,
             doftype, nid, upids) = o2._proc_bgpdt(
                 eqexin1, eqexin, True, bgpdt_rec1)
            Uset, cstm, cstm2 = o2._buildUset(
                se, dof, doftype, nid, uset, xyz, cid, None, cstm2)

    return {'uset': Uset,
            'cstm': cstm,
            'cstm2': cstm2,
            'mats': mats,
            'selist': selist,
            'sebulk': sebulk}
