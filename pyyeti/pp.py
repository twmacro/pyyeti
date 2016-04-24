# -*- coding: utf-8 -*-
"""
A pretty printer.
"""
import numpy as np
import h5py

class PP(object):
    """
    A simple class for pretty printing data structures.
    """
    def __init__(self, var=None, depth=5, tab=4,
                 keylen=40, strlen=80, show_hidden=False):
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
        show_hidden : bool; optional
            Many objects (classes, class instances, namespaces, etc)
            are printed via looping over the ``.__dict__`` method. For
            those variables, if `show_hidden` is True, show members
            that start with '_'. Default is to not show them.

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
        ...                                         t=(s,))
        ...      }
        ... }
        >>> PP(d)       # doctest: +ELLIPSIS
        <class 'dict'>[n=4]
            '34'         : 'value'
            'asdf'       : 4
            'longer name': <class 'dict'>[n=4]
                1: 2
                2: 3
                3: float64 ndarray 64 elems: (4, 4, 4)
                4: <class 'types.SimpleNamespace'>[n=4]
                    .a  : 6
                    .b  : [n=2]: [1, 23]
                    .t  : [n=1]: (float64 ndarray: (4, 4, 4),)
                    .var: 'string'
            'r'          : int16 ndarray 4 elems: (4,) [0 1 2 3]
        <BLANKLINE>
        <...>
        >>> PP(d, depth=1)       # doctest: +ELLIPSIS
        <class 'dict'>[n=4]
            '34'         : 'value'
            'asdf'       : 4
            'longer name': <class 'dict'>[n=4]
            'r'          : int16 ndarray 4 elems: (4,) [0 1 2 3]
        <BLANKLINE>
        <...>

        To demonstrate the `show_hidden` option:

        >>> class A:
        ...     a = 9
        ...     b = {'variable': [1, 2, 3]}
        >>> PP(A)       # doctest: +ELLIPSIS
        <class 'type'>[n=6]
            .a: 9
            .b: <class 'dict'>[n=1]
                'variable': [n=3]: [1, 2, 3]
        <BLANKLINE>
        <...>
        >>> PP(A, show_hidden=1)       # doctest: +ELLIPSIS
        <class 'type'>[n=6]
            .__dict__   : <attribute '__dict__' of 'A' objects>
            .__doc__    : None
            .__module__ : 'pyyeti.pp'
            .__weakref__: <attribute '__weakref__' of 'A' objects>
            .a          : 9
            .b          : <class 'dict'>[n=1]
                'variable': [n=3]: [1, 2, 3]
        <BLANKLINE>
        <...>
        """
        self._tab = tab
        self._depth = depth
        self._keylen = keylen
        self._strlen = strlen
        self._show_hidden = show_hidden
        self._functions = {
            np.ndarray: self._array_string,
            h5py._hl.dataset.Dataset: self._h5data_string,
            }
        if var is not None:
            self.pp(var)

    def _lead_string(self, level):
        self.output = self.output + ' '*self._tab*level

    def _key_string(self, val, isns):
        if isinstance(val, str):
            if isns:
                s = "." + val
            else:
                s = "'" + val + "'"
        else:
            s = str(val)
        if len(s) > self._keylen:
            s = s[:self._keylen-4] + ' ...'
        return s

    def _lst_tup_string(self, lst, level):
        s = '[n={}]: '.format(len(lst))
        be = '[]' if isinstance(lst, list) else '()'
        s = s + be[0]
        for item in lst:
            if isinstance(item, np.ndarray):
                s = s + self._shortarrhdr(item) + ', '
            else:
                s = s + self._value_string(item, level+1) + ', '
        if s.endswith(', '):
            if len(lst) == 1 and isinstance(lst, tuple):
                s = s[:-1]
            else:
                s = s[:-2]
        return s + be[1]

    def _value_string(self, val, level):
        if isinstance(val, str):
            s = "'" + val + "'"
        elif level < self._depth and isinstance(val, (list, tuple)):
            s = self._lst_tup_string(val, level)
        else:
            s = str(val)
        if len(s) > self._strlen:
            s = s[:self._strlen-4] + ' ...'
        return s

    def _shortarrhdr(self, arr):
        return str(arr.dtype) + ' ndarray: ' + str(arr.shape)
        
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
            s = s + self._getarrstr(arr[...])
        self.output = self.output + s + '\n'

    def _get_keys(self, dct, showhidden):
        if not showhidden:
            keys = [k for k in dct if k[0] != '_']
        else:
            keys = list(dct.keys())
        return keys

    def _dict_string(self, dct, level, typename, isns=False,
                     showhidden=True):
        self.output = (self.output + '{}[n={}]\n'
                       .format(typename, len(dct)))
        if level < self._depth:
            keys = self._get_keys(dct, showhidden)
            try:
                keys = sorted(keys)
            except TypeError:
                pass
            level += 1
            # get max key length for pretty printing:
            n = 0
            for k in keys:
                n = max(n, len(self._key_string(k, isns)))
            frm = '{:<' + str(n) + 's}: '
            for k in keys:
                self._lead_string(level)
                s = self._key_string(k, isns)
                self.output = self.output + frm.format(s)
                self._print_var(dct[k], level)

    def _print_var(self, var, level):
        try:
            self._functions[type(var)](var, level)
        except KeyError:
            typename = str(type(var))
            if isinstance(var, (dict,
                                h5py._hl.files.File,
                                h5py._hl.files.Group)):
                self._dict_string(var, level, typename=typename)
            elif '__dict__' in dir(var):
                self._dict_string(var.__dict__, level,
                                  typename=typename, isns=True,
                                  showhidden=self._show_hidden)
            else:
                s = self._value_string(var, level)
                self.output = self.output + s + '\n'

    def pp(self, var):
        """
        Pretty print variable `var`. See :class:`PP`.
        """
        self.output = ''
        self._print_var(var, 0)
        print(self.output)
