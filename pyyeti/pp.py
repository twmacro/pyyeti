# -*- coding: utf-8 -*-
"""
A pretty printer.
"""
import numpy as np
import h5py
from types import SimpleNamespace

class PP:
    """
    A very simple hack for pretty printing data structures.
    """
    def __init__(self, var=None, depth=5, tab=4,
                 keylen=40, strlen=80):
        """
        Initializer for `PP`.

        Parameters
        ----------
        var : any; optional
            If not None, the variable to pretty print.
        depth : integer; optional
            Maximum number of levels to print.
        tab : integer; optional
            Number of additional indent spaces for each level.
        keylen : integer; optional
            Maximum length for dictionary (and similar) keys.
        strlen : integer; optional
            Maximum length for dictionary (and similar) values.

        Notes
        -----
        A variable can be printed during instantiation or via
        :func:`PP.pp`. For example::

            PP(var, depth=3, tab=8)

        gives the same output as::

            p = PP(depth=3, tab=8)
            p.pp(var)

        Examples
        --------
        >>> import numpy as np
        >>> from types import SimpleNamespace
        >>> from pyyeti.pp import PP
        >>> r = np.arange(4, dtype=np.int16)
        >>> s = np.random.randn(4, 4, 4)
        >>> t = np.array(9, dtype=np.uint8)
        >>> d = {'asdf': 4,
        ...      '34': 'value',
        ...      'r': r,
        ...      'longer name': {1: 2,
        ...                      2: 3,
        ...                      3: s,
        ...                      4: SimpleNamespace(a=6,
        ...                                         b=[1, 23],
        ...                                         var='string',
        ...                                         t=t)
        ...      }
        ... }
        >>> PP(d)       # doctest: +ELLIPSIS
        dict[n=4]
            '34'         : 'value'
            'asdf'       : 4
            'longer name': dict[n=4]
                1: 2
                2: 3
                3: float64 ndarray 64 elems: (4, 4, 4)
                4: namespace[n=4]
                    .a  : 6
                    .b  : [1, 23]
                    .t  : uint8 ndarray 1 elems: () 9
                    .var: 'string'
            'r'          : int16 ndarray 4 elems: (4,) [0 1 2 3]
        <BLANKLINE>
        <...>
        >>> PP(d, depth=1)       # doctest: +ELLIPSIS
        dict[n=4]
            '34'         : 'value'
            'asdf'       : 4
            'longer name': dict[n=4]
            'r'          : int16 ndarray 4 elems: (4,) [0 1 2 3]
        <BLANKLINE>
        <...>
        """
        self._tab = tab
        self._depth = depth
        self._keylen = keylen
        self._strlen = strlen
        self._functions = {
            dict: self._dict_string,
            SimpleNamespace: self._ns_string,
            np.ndarray: self._array_string,
            h5py._hl.dataset.Dataset: self._h5data_string,
            }
        if var is not None:
            self.pp(var)

    def _lead_string(self, level):
        self.output = self.output + ' '*self._tab*level

    def _key_string(self, val, typename):
        if isinstance(val, str):
            if typename == 'namespace':
                s = "." + val
            else:
                s = "'" + val + "'"
        else:
            s = str(val)
        if len(s) > self._keylen:
            s = s[:self._keylen-4] + ' ...'
        return s

    def _value_string(self, val):
        if isinstance(val, str):
            s = "'" + val + "'"
        else:
            s = str(val)
        if len(s) > self._strlen:
            s = s[:self._strlen-4] + ' ...'
        return s

    def _getarrhdr(self, arr):
        s = str(arr.dtype) + ' ndarray '
        s = s + str(arr.size) + ' elems: ' + str(arr.shape)
        return s

    def _getarrstr(self, arr):
        s = ' ' + str(arr).replace('\n', '')
        if len(s) > 4*(self._strlen//5):
            n = self._strlen//3
            s = s[:n-3] + ' <...> ' + s[-(n+3):]
        return s

    def _array_string(self, arr, level):
        s = self._getarrhdr(arr)
        if arr.size <= 10:
            s = s + self._getarrstr(arr)
        self.output = self.output + s + '\n'

    def _h5data_string(self, arr, level):
        s = 'H5 ' + self._getarrhdr(arr)
        if arr.size <= 10:
            s = s + self._getarrstr(arr.value)
        self.output = self.output + s + '\n'

    def _dict_string(self, dct, level, typename='dict'):
        self.output = (self.output + '{}[n={}]\n'
                       .format(typename, len(dct)))
        if level < self._depth:
            keys = list(dct)
            try:
                keys = sorted(keys)
            except TypeError:
                pass
            level += 1
            # get max key length for pretty printing:
            n = 0
            for k in keys:
                n = max(n, len(self._key_string(k, typename)))
            frm = '{:<' + str(n) + 's}: '
            for k in keys:
                self._lead_string(level)
                s = self._key_string(k, typename)
                self.output = self.output + frm.format(s)
                self._print_var(dct[k], level)

    def _ns_string(self, ns, level):
        self._dict_string(ns.__dict__, level, typename='namespace')

    def _print_var(self, var, level):
        try:
            self._functions[type(var)](var, level)
        except KeyError:
            typename = str(type(var))
            if isinstance(var, (h5py._hl.files.File,
                                h5py._hl.files.Group)):
                self._dict_string(var, level, typename=typename)
            else:
                s = self._value_string(var)
                self.output = self.output + s + '\n'

    def pp(self, var):
        """
        Pretty print variable `var`.
        """
        self.output = ''
        self._print_var(var, 0)
        print(self.output)
