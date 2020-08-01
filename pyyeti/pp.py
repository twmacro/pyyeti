# -*- coding: utf-8 -*-
"""
A pretty printer.
"""
import collections
import numpy as np
import h5py
import pandas as pd


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


class PP(object):
    """
    A simple class for pretty printing data structures.
    """

    def __init__(
        self, var=None, depth=1, tab=4, keylen=40, strlen=80, show_hidden=False
    ):
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
        >>> import pandas as pd
        >>> from types import SimpleNamespace
        >>> from pyyeti.pp import PP
        >>> r = np.arange(4, dtype=np.int16)
        >>> s = np.random.randn(4, 4, 4)
        >>> t = np.array(9, dtype=np.uint8)
        >>> d = {'asdf': 4,
        ...      '34': 'value',
        ...      'r': r,
        ...      'dataframe': pd.DataFrame(np.ones((3, 3))),
        ...      'series': pd.Series(np.ones(8)),
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
        <class 'dict'>[n=6]
            'asdf'       : 4
            '34'         : 'value'
            'r'          : int16 ndarray 4 elems: (4,) [0 1 2 3]
            'dataframe'  : pandas DataFrame: (3, 3)
            'series'     : pandas Series: (8,)
            'longer name': <class 'dict'>[n=4]
        <BLANKLINE>
        <...>
        >>> PP(d, 5)    # doctest: +ELLIPSIS
        <class 'dict'>[n=6]
            'asdf'       : 4
            '34'         : 'value'
            'r'          : int16 ndarray 4 elems: (4,) [0 1 2 3]
            'dataframe'  : pandas DataFrame: (3, 3)
            'series'     : pandas Series: (8,)
            'longer name': <class 'dict'>[n=4]
                1: 2
                2: 3
                3: float64 ndarray 64 elems: (4, 4, 4)
                4: <class 'types.SimpleNamespace'>[n=4]
                    .a  : 6
                    .b  : [n=2]: [1, 23]
                    .var: 'string'
                    .t  : [n=1]: (float64 ndarray: (4, 4, 4),)
        <BLANKLINE>
        <...>

        To demonstrate the `show_hidden` option:

        >>> class A:
        ...     a = 9
        ...     b = {'variable': [1, 2, 3]}
        >>> PP(A, 2)       # doctest: +ELLIPSIS
        <class 'A'>[n=6]
            .a: 9
            .b: <class 'dict'>[n=1]
                'variable': [n=3]: [1, 2, 3]
        <BLANKLINE>
        <...>
        >>> PP(A, 2, show_hidden=1)        # doctest: +SKIP
        <class 'A'>[n=6]
            .__module__ : 'pyyeti.pp'
            .a          : 9
            .b          : <class 'dict'>[n=1]
                'variable': [n=3]: [1, 2, 3]
            .__dict__   : <attribute '__dict__' of 'A' objects>
            .__weakref__: <attribute '__weakref__' of 'A' objects>
            .__doc__    : None
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
        return " " * self._tab * level

    def _key_string(self, val, isns):
        if isinstance(val, str):
            if isns:
                s = f".{val}"
            else:
                s = f"'{val}'"
        else:
            s = str(val)
        if len(s) > self._keylen:
            s = f"{s[: self._keylen - 4]} ..."
        return s

    def _lst_tup_string(self, lst, list_level):
        s = [f"[n={len(lst)}]: "]
        be = "[]" if isinstance(lst, list) else "()"
        s.append(be[0])
        if list_level > self._depth:
            s.append(f"...{be[1]}")
            return s
        list_level += 1
        for i, item in enumerate(lst):
            if i > self._strlen // 3:
                # each entry is at least 3 chars: "1, "
                s.append(" ...")
                break
            if isinstance(item, np.ndarray):
                s.extend(self._shortarrhdr(item))
            else:
                s.extend(self._value_string(item, list_level))
            s.append(", ")
        if s[-1] == ", ":
            if len(lst) == 1 and isinstance(lst, tuple):
                s[-1] = ","
            else:
                s = s[:-1]
        s.append(be[1])
        return s

    def _value_string(self, val, list_level):
        if isinstance(val, str):
            s = [f"'{val}'"]
        elif isinstance(val, (list, tuple)):
            s = self._lst_tup_string(val, list_level)
        else:
            s = [f"{val}".replace("\n", " ")]
        s = "".join(s)
        if len(s) > self._strlen:
            n = self._strlen // 2 - 3
            s = f"{s[:n]} ... {s[-n:]}"
        return [s]

    def _shortarrhdr(self, arr):
        return [f"{arr.dtype} ndarray: {arr.shape}"]

    def _getarrhdr(self, arr):
        return [f"{arr.dtype} ndarray {arr.size} elems: {arr.shape}"]

    def _getarrstr(self, arr):
        s = f" {arr}".replace("\n", "")
        if len(s) > 4 * (self._strlen // 5):
            n = self._strlen // 3
            s = f"{s[: n - 3]} <...> {s[-(n + 3) :]}"
        return [s]

    def _array_string(self, arr, level):
        s = self._getarrhdr(arr)
        if arr.size <= 10:
            s.extend(self._getarrstr(arr))
        s.append("\n")
        return s

    def _h5data_string(self, arr, level):
        s = ["H5 "]
        s.extend(self._getarrhdr(arr))
        if arr.size <= 10:
            s.extend(self._getarrstr(arr[...]))
        s.append("\n")
        return s

    def _get_keys(self, dct, showhidden):
        if not showhidden:
            keys = [k for k in dct if k[0] != "_"]
        else:
            keys = list(dct.keys())
        return keys

    def _dict_string(self, dct, level, typename, isns=False, showhidden=True):
        s = [f"{typename}[n={len(dct)}]\n"]
        if level < self._depth:
            keys = self._get_keys(dct, showhidden)
            level += 1
            # get max key length for pretty printing:
            n = 0
            for k in keys:
                n = max(n, len(self._key_string(k, isns)))
            frm = f"{{:<{n}s}}: "
            for k in keys:
                s.append(self._lead_string(level))
                s.append(frm.format(self._key_string(k, isns)))
                s.extend(self._print_var(dct[k], level))
        return s

    def _pandas_string(self, var, level, typename):
        return [f"pandas {var.__class__.__name__}: {var.shape}\n"]

    def _print_var(self, var, level):
        try:
            s = self._functions[type(var)](var, level)
        except KeyError:
            cls = type(var)
            if cls is type:
                typename = f"<class '{var.__name__}'>"
            else:
                typename = str(cls)
            if isinstance(var, collections.Mapping):
                s = self._dict_string(var, level, typename=typename)
            elif isinstance(var, pd.core.base.PandasObject):
                s = self._pandas_string(var, level, typename=typename)
            elif hasattr(var, "__dict__") and not hasattr(var, "keys"):
                s = self._dict_string(
                    var.__dict__,
                    level,
                    typename=typename,
                    isns=True,
                    showhidden=self._show_hidden,
                )
            else:
                s = self._value_string(var, 0)
                s.append("\n")
        return s

    def pp(self, var):
        """
        Pretty print variable `var`. See :class:`PP`.
        """
        s = self._print_var(var, 0)
        self.s = s
        self.output = "".join(s)
        for line in self.output.split("\n"):
            print(repr(line)[1:-1])
