# -*- coding: utf-8 -*-
"""
DR_Def: data recovery definitions
"""
import os
import sys
import copy
import numbers
import inspect
from collections import abc, OrderedDict
from types import SimpleNamespace
import warnings
import copyreg
import re
import numpy as np
import pandas as pd
import xlsxwriter
from pyyeti.nastran import n2p
from ._utilities import _merge_uf_reds, _is_valid_identifier, _compile_strfunc


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


class DR_Def(OrderedDict):
    """
    Data recovery definitions.

    This class inherits from :class:`collections.OrderedDict` and
    defines how data recovery will be done. The entries are created
    through repeated calls to member function :func:`add` (typically
    from a "prepare_4_cla.py" script). Within a single instance of
    this class, the order that items are added also defines the data
    recovery order (as done in :func:`DR_Results.time_data_recovery`,
    for example). When using multiple instances, the data recovery
    order of the instances is determined by the order they are added
    to an instance of :class:`DR_Event` via :func:`DR_Event.add`.
    :func:`DR_Event.set_dr_order` can be used to modify the final
    order. :func:`pyyeti.ytools.reorder_dict` can also be used for
    that purpose.

    Attributes
    ----------
    defaults : dict or None; optional
        Dictionary with any desired defaults for the parameters
        listed in :func:`add`. If None, it is initialized to an
        empty dictionary.
    ncats : integer
        The number of data recovery categories defined. This is a
        static class variable. This means that if you have multiple
        instances of this class, `ncats` will end up being the total
        number of categories over all instances (and there is nothing
        wrong with that).

    Notes
    -----
    The keys, with the exception of the key '_vars', are the short
    names of each data recovery category. These names are defined by
    the `name` parameter in successive calls to :func:`add`.
    The values for these categories are SimpleNamespaces that fully
    describe the category containing the description, units, row
    labels, shock response spectra requests, etc, and even provides
    (directly or indirectly) the function for performing data recovery
    for the category.

    As noted above, there is one other entry in the dictionary: the
    '_vars' entry. This entry is also a SimpleNamespace with two
    attributes: `drms` and `nondrms`. The `drms` attribute contains
    the data recovery matrices; it is a regular Python dictionary
    indexed by the superelement ID number. The `nondrms` attribute
    contains any other information needed for performing data
    recovery; it is also a regular Python dictionary indexed by the
    superelement ID number.

    For illustration, we'll use :func:`pyyeti.pp.PP` to display
    sections of this dictionary for an example mission:

    PP(drdefs):

    .. code-block:: none

        <class 'cla.DR_Def'>[n=9]
            'SC_atm'    : <class 'types.SimpleNamespace'>[n=20]
            'SC_dtm'    : <class 'types.SimpleNamespace'>[n=20]
            'SC_ifl'    : <class 'types.SimpleNamespace'>[n=20]
            'SC_ltma'   : <class 'types.SimpleNamespace'>[n=20]
            'SC_ltmd'   : <class 'types.SimpleNamespace'>[n=20]
            'SC_cg'     : <class 'types.SimpleNamespace'>[n=20]
            'SC_ifa'    : <class 'types.SimpleNamespace'>[n=20]
            'SC_ifa_0rb': <class 'types.SimpleNamespace'>[n=20]
            '_vars'     : <class 'types.SimpleNamespace'>[n=2]

    PP(drdefs['SC_ifa'], 2):

    .. code-block:: none

        <class 'types.SimpleNamespace'>[n=20]
            .desc      : 'S/C Interface Accelerations'
            .drfile    : '/loads/CLA/Rocket/missions/.../drfuncs.py'
            .drfunc    : 'SC_ifa'
            .filterval : 1e-06
            .histlabels: [n=12]: ['I/F Axial Accel     X sc', ... lv']
            .histpv    : slice(None, 12, None)
            .histunits : 'G, rad/sec^2'
            .ignorepv  : None
            .labels    : [n=12]: ['I/F Axial Accel     X sc', ... lv']
            .misc      : None
            .se        : 500
            .srsQs     : [n=2]: (25, 50)
            .srsconv   : 1.0
            .srsfrq    : float64 ndarray 990 elems: (990,)
            .srslabels : [n=12]: ['$X_{SC}$', '$Y_{SC}$', '$Z_ ...}$']
            .srsopts   : <class 'dict'>[n=2]
                'eqsine': 1
                'ic'    : 'steady'
            .srspv     : slice(None, 12, None)
            .srsunits  : 'G, rad/sec^2'
            .uf_reds   : [n=4]: (1, 1, 1.25, 1)
            .units     : 'G, rad/sec^2'

    PP(drdefs['_vars'], 3):

    .. code-block:: none

        <class 'types.SimpleNamespace'>[n=2]
            .drms   : <class 'dict'>[n=1]
                500: <class 'dict'>[n=7]
                    'SC_ifl_drm': float64 ndarray 1296 ... (12, 108)
                    'SC_cg_drm' : float64 ndarray 1512 ... (14, 108)
                    'SC_ifa_drm': float64 ndarray 1296 ... (12, 108)
                    'scatm'     : float64 ndarray 28944 ... (268, 108)
                    'scdtmd'    : float64 ndarray 9396 ... (87, 108)
                    'scltma'    : float64 ndarray 15768 ... (146, 108)
                    'scltmd'    : float64 ndarray 15768 ... (146, 108)
            .nondrms: <class 'dict'>[n=1]
                500: <class 'dict'>[n=0]
    """

    # static variable:
    ncats = 0

    @staticmethod
    def addcat(func):
        """
        Decorator to ensure :func:`add` is called to add category.

        Notes
        -----
        Example of typical usage::

            @cla.DR_Def.addcat
            def _():
                name = 'SC_atm'
                desc = 'S/C Internal Accelerations'
                units = 'G'
                drms = {name: sc_atm}
                drfunc = f"Vars[se]['{name}'] @ sol.a"
                # ... other variables defined; see :func:`add`.
                drdefs.add(**locals())

        """
        nc = DR_Def.ncats
        func()
        if DR_Def.ncats <= nc:
            msg = (
                "function must call the `DR_Def.add` "
                "method (eg: ``drdefs.add(**locals())``)"
            )
            raise RuntimeError(msg)

    def __init__(self, defaults=None):
        """
        Parameters
        ----------
        defaults : dict or None; optional
            Sets the `defaults` attribute; see :class:`DR_Def`.
        """
        super().__init__()
        self.defaults = {} if defaults is None else defaults
        self["_vars"] = SimpleNamespace(drms={}, nondrms={})
        self._drfilemap = {}

    def __repr__(self):
        cats = ", ".join(f"'{name}'" for name in self)
        return (
            f"{type(self).__name__} ({hex(id(self))}) with "
            f"{len(self) - 1} categories: [{cats}]"
        )

    # add drms and nondrms to self:
    def _add_vars(self, name, drms, nondrms):
        def _add_drms(d1, d2):
            for key in d2:
                if key in d1:
                    # print warning or error out:
                    msg = (
                        f'"{key}" already included for a previously '
                        "defined data recovery matrix"
                    )
                    if d1[key] is d2[key]:
                        warnings.warn(msg, RuntimeWarning)
                    else:
                        raise ValueError("A different " + msg)
                else:
                    d1[key] = d2[key]

        _vars = self["_vars"]
        se = self[name].se
        for curdrms, newdrms in ((_vars.drms, drms), (_vars.nondrms, nondrms)):
            if se not in curdrms:
                curdrms[se] = {}
            _add_drms(curdrms[se], newdrms)

    def _check_for_drfunc(self, name, filename, funcname):
        def _get_msg():
            s0 = f'When checking `drfunc` for category "{name}",'
            s1 = (
                "This can be safely ignored if you plan to "
                "implement the data recovery functions later."
            )
            return s0, s1

        if _is_valid_identifier(funcname):
            try:
                # search for "def funcname("
                with open(filename, "r") as fin:
                    for line in fin:
                        if re.match(rf" *def +{funcname} *\(", line):
                            break
                    else:
                        s0, s1 = _get_msg()
                        msg = (
                            f'{s0} function "{funcname}" not found in: '
                            f"{filename}. {s1}"
                        )
                        warnings.warn(msg, RuntimeWarning)
            except FileNotFoundError as e:
                s0, s1 = _get_msg()
                msg = (
                    f'{s0} could not open "{filename}". {s1}\nThe exception was: '
                    f"{type(e).__name__}: {e!s}"
                )
                warnings.warn(msg, RuntimeWarning)
        else:
            try:
                _compile_strfunc(funcname, False)
            except SyntaxError as e:
                s0, s1 = _get_msg()
                msg = (
                    f'{s0} failed to compile string: "{funcname}".\n'
                    f"The exception was: {type(e).__name__}: {e!s}"
                )
                warnings.warn(msg, RuntimeWarning)

    @staticmethod
    def _get_pv(pv, name, length):
        if pv is None:
            return pv
        if isinstance(pv, str) and pv == "all":
            return slice(length)
        if isinstance(pv, slice):
            return pv
        if not isinstance(pv, str) and isinstance(pv, (abc.Sequence, numbers.Integral)):
            pv = np.atleast_1d(pv)
        if isinstance(pv, np.ndarray):
            pv = pv.ravel()
            if pv.dtype == bool:
                if len(pv) != length:
                    msg = (
                        f"length of `{name}` ({len(pv)}) does not "
                        f"match length of labels ({length})"
                    )
                    raise ValueError(msg)
                return pv.nonzero()[0]
            elif issubclass(pv.dtype.type, np.integer):
                if pv.max() >= length or pv.min() < 0:
                    msg = (
                        f"values in `{name}` range from [{pv.min()}, {pv.max()}], "
                        f"but should be in range [0, {length}] for this "
                        "category"
                    )
                    raise ValueError(msg)
                return pv
        raise TypeError(f"`{name}` input not understood")

    @staticmethod
    def _get_labels(pv, labels):
        if isinstance(pv, slice):
            return labels[pv]
        return [labels[i] for i in pv]

    def _handle_defaults(self, name):
        """Handle default values and take default actions"""
        # first, the defaults:
        ns = self[name]

        if ns.drfunc is None:
            ns.drfunc = name

        if ns.drfile is None:
            # this is set to defaults only if it is in defaults,
            # otherwise, leave it None
            ns.drfile = self.defaults.get("drfile", None)

        if ns.se is None:
            ns.se = self.defaults.get("se", 0)

        if ns.uf_reds is None:
            ns.uf_reds = self.defaults.get("uf_reds", (1, 1, 1, 1))

        if ns.srsQs is None:
            for k, v in ns.__dict__.items():
                if k.startswith("srs") and v is not None:
                    ns.srsQs = self.defaults
                    break

        if ns.srsfrq is None and ns.srsQs is not None:
            ns.srsfrq = self.defaults

        dct = ns.__dict__
        for key in dct:
            if dct[key] is self.defaults:
                try:
                    dct[key] = self.defaults[key]
                except KeyError:
                    msg = f"{key} set to `defaults` but is not found in `defaults`!"
                    raise ValueError(msg)

        if ns.drfile == ".":
            ns.drfile = None

        # add path to `drfile` if needed:
        if _is_valid_identifier(ns.drfunc):
            try:
                ns.drfile = self._drfilemap[ns.drfile]
            except KeyError:
                calling_file = os.path.realpath(inspect.stack()[2][1])
                if ns.drfile is None:
                    ns.drfile = drfile = calling_file
                else:
                    drfile = ns.drfile
                    callerdir = os.path.dirname(calling_file)
                    ns.drfile = os.path.join(callerdir, drfile)
                self._drfilemap[drfile] = ns.drfile

        self._check_for_drfunc(name, ns.drfile, ns.drfunc)

        # ensure uf_reds has no None values:
        ns.uf_reds = _merge_uf_reds((1, 1, 1, 1), ns.uf_reds)

        # next, the default actions:
        if ns.desc is None:
            ns.desc = name

        if isinstance(ns.labels, numbers.Integral):
            ns.labels = [f"Row {i + 1:6d}" for i in range(ns.labels)]

        # check filter value:
        try:
            nf = len(ns.filterval)
        except TypeError:
            pass
        else:
            if nf != len(ns.labels):
                raise ValueError(
                    f"length of `filterval` ({nf}) does not match length of"
                    f"labels ({len(ns.labels)})"
                )
            ns.filterval = np.atleast_1d(ns.filterval)

        ns.ignorepv = self._get_pv(ns.ignorepv, "ignorepv", len(ns.labels))

        if ns.srsQs is not None:
            ns.srsQs = n2p._ensure_iter(ns.srsQs)
            if ns.srspv is None:
                ns.srspv = slice(len(ns.labels))

        if ns.srsconv is None and ns.srsQs is not None:
            ns.srsconv = self.defaults.get("srsconv", 1.0)

        if ns.srspv is not None and ns.srsopts is None:
            ns.srsopts = self.defaults.get("srsopts", {})

        # fill in hist-labels/units and srs-labels/units if needed:
        for i in ("hist", "srs"):
            pv = i + "pv"
            dct[pv] = self._get_pv(dct[pv], pv, len(ns.labels))
            if dct[pv] is not None:
                lbl = i + "labels"
                unt = i + "units"
                if dct[lbl] is None:
                    dct[lbl] = self._get_labels(dct[pv], ns.labels)
                if dct[unt] is None:
                    dct[unt] = ns.units

        # ensure that the labels are lists (or None):
        for i in ("", "hist", "srs"):
            lbl = i + "labels"
            if not isinstance(dct[lbl], (list, type(None))):
                dct[lbl] = list(dct[lbl])

    def add(
        self,
        *,
        name,
        labels,
        active="yes",
        drms=None,
        drfunc=None,
        drfile=None,
        se=None,
        desc=None,
        units="Not specified",
        uf_reds=None,
        filterval=1.0e-6,
        histlabels=None,
        histpv=None,
        histunits=None,
        misc=None,
        ignorepv=None,
        nondrms=None,
        srsQs=None,
        srsfrq=None,
        srsconv=None,
        srslabels=None,
        srsopts=None,
        srspv=None,
        srsunits=None,
        **kwargs,
    ):
        """
        Adds a data recovery category.

        All inputs to :func:`add` must be named.

        .. note::

            Any of the inputs can be set to `self.defaults`. In this
            case, the value for the parameter will be taken from the
            `self.defaults` dictionary (which is defined during
            instantiation). A ValueError is raised if `self.defaults`
            does not contain a value for the parameter.

        .. note::

            Using None for some inputs (which is default for many)
            will cause a default action to be taken. These are listed
            below as applicable after "DA:".

        Parameters
        ----------
        name : string
            Short name of data recovery category, eg: 'SC_atm'. This
            typically also defines `drfunc`, the name of the function
            in `drfile` that is called to do data recovery; in that
            case, it must be a valid Python variable name.
        labels : list_like or integer
            List_like of strings describing each row. Can also be an
            integer specifying number of rows being recovered; in this
            case, the list is formed internally as:
            ``['Row 1', 'Row 2', ...]``. This input is used to
            determine number of rows being recovered. If not a list,
            it is converted to a list via :class:`list`.
        active : string; optional
            If 'yes', this category will be included when the
            :func:`DR_Event.add` function is called to add categories
            for the event simulation. Otherwise, this category will be
            ignored during the simulation by default. These tools only
            check for 'yes', but user written tools can make use of
            other settings, perhaps by monkey-patching the
            :class:`DR_Event` class to replace or augment the
            :func:`DR_Event.add` method. As an example, for a category
            that calculates the fairing-to-spacecraft
            loss-of-clearance, you might set `active` to 'no' to
            unconditionally ignore this category but use 'loc' to
            include this category if a fairing is present but ignore
            it if not.
        drms : dict or None; optional
            Dictionary of data recovery matrices for this category;
            keys are matrix names and must match what is used in the
            data recovery function in `drfile`. In the data recovery
            function, these variables are accessed through
            ``Vars[se]``. For example: ``drms = {'scatm': scatm}``
            means that ``Vars[se]['scatm']`` would be used in the
            function (see also `drfunc`). If `se` is greater than 0,
            each matrix, before being stored in ``Vars[se]`` via
            :func:`DR_Event.add`, will be multiplied by the
            appropriate "ULVS" matrix (see
            :func:`pyyeti.nastran.op2.rdnas2cam`) during event
            simulation. If `se` is 0, it is used as is and is likely
            added during event simulation since system modes are often
            needed. Note that when `se` is 0, using `drms` is equivalent
            to using `nondrms`.
        drfunc : string or None; optional
            A string that either defines the data recovery
            calculations directly (a "Type 1" `drfunc`) or, if it is a
            valid Python identifier, it is the name of the data
            recovery function in `drfile` (a "Type 2" `drfunc`). If it
            is Type 2, as a check, this routine will attempt to import
            the function; if not found, a warning is printed.

            Assume that `drfile` is specified properly and the file it
            indicates has::

                def SC_atm(sol, nas, Vars, se):
                    return Vars[se]['atm'] @ sol.a

            (The inputs to the data recovery function are defined
            below in the Notes section.) Further, assume that `name`
            is 'SC_atm'. With these assumptions, we can use either a
            Type 1 or a Type 2 `drfunc`. That is, the following
            settings for `drfunc` are all functionally equivalent:

              =========================  =============================
                     `drfunc`                    Description
              =========================  =============================
              "Vars[se]['atm'] @ sol.a"  Type 1. This defines the
                                         return statement directly
                                         (the function is defined
                                         internally, inside module
                                         :mod:`cla`).
                     'SC_atm'            Type 2. Uses the `SC_atm`
                                         function in `drfile`.
                       None              Type 2. Same as 'SC_atm'.
                                         (Takes the default action ...
                                         see 'DA' below)
              =========================  =============================

            Note that, currently, if the data recovery function needs
            more than just the typical return statement calculation or
            if a PSD-specific data recovery function is needed, then
            `drfunc` must be Type 2.

            DA: set to `name` and `drfile` is used.
        drfile : string or '.' or None; optional
            Only used if needed according to `drfunc` (if using a
            "Type 2" `drfunc`). If needed, `drfile` is the name of
            file that contains the data recovery function named
            `drfunc`. It can optionally also have a PSD-specific data
            recovery function; this must have the same name but with
            "_psd" appended. See notes below for example
            functions. `drfile` is imported during event simulation.
            If not already provided in `drfile`, the full path of
            `drfile` is defined to be relative to the path of the file
            that contains the function that called this routine. If
            input as '.', `drfile` is set to the full name of the file
            that called this routine.

            DA: if `drfile` is set in `self.defaults`, that is used;
            otherwise, it is set to the full name of the file that
            called this routine (as if `drfile` was input as '.').
        se : integer or None; optional
            The superelement number.

            DA: get value from `self.defaults` if present; otherwise
            set to 0.
        desc : string or None; optional
            One line description of category; eg:
            ``desc = 'S/C Internal Accelerations'``.

            DA: set to `name`.
        units : string; optional
            Specifies the units.
        uf_reds : 4-element tuple or None; optional
            The uncertainty factors in "reds" order:
            [rigid, elastic, dynamic, static]. Examples::

                (1, 1, 1.25, 1)  - Typical; DUF = 1.25
                (0, 1, 1.25, 1)  - Same, but without rigid-body part
                (1, 1, 0, 1)     - Recover static part only
                (1, 1, 1, 0)     - Recover dynamic part only

            DA: get value from `self.defaults` if present; otherwise
            set to (1, 1, 1, 1). Also, any of the four entries in the
            tuple can be None; these get reset to the corresponding
            entry from the `self.defaults` or, if that's None too, 1.
        filterval : scalar or 1d array_like; optional
            Response values smaller than `filterval` will be skipped
            during comparison to another set of results. If 1d
            array_like, length must be ``len(labels)`` allowing for a
            unique filter value for each row.
        histlabels : list_like or None; optional
            Analogous to `labels` but just for the `histpv` rows.

            DA: derive from `labels` according to `histpv` if
            needed; otherwise, leave it None.
        histpv : 1d array_like or 'all' or slice or None; optional
            Specifies which rows to save the response histories of.
            See note below about inputting partition vectors.
        histunits : string or None; optional
            Units string for the `histpv`.

            DA: set to `units` if `histpv` is not None; otherwise,
            leave it None.
        ignorepv : 1d array_like or 'all' or slice or None; optional
            Typically, this is left as None (default) so that all rows
            are compared. `ignorepv` specifies rows that need to be
            ignored during comparisons against a reference data set.
            If set to 'all', all rows will ignored ... meaning no
            comparisons will be done for this category. See note below
            about inputting partition vectors.
        misc : any object; optional
            Available for storing miscellaneous information for the
            category. It is not used within this module. This option
            is similar to `nondrms` in functionality; the differences
            are that the `misc` input will be stored with the results,
            and that the `nondrms` contents will be put in ``Vars[se]``
            (making those values available in the the data recovery
            function).
        nondrms : dict or None; optional
            With one important exception, this input is used
            identically to `drms`. The exception is that the values in
            `nondrms` are not multiplied by ULVS. Therefore, `nondrms`
            can contain any variables you need for data recovery. An
            alternative option is to include data you need in
            `drfile` with the data recovery functions.
        srsQs : scalar or 1d array_like or None; optional
            Q values for SRS calculation or None.

            DA: set to `self.defaults` if any other `srs*` option is
            not None; otherwise, leave it None.
        srsfrq : 1d array_like or None; optional
            Frequency vector for SRS.

            DA: get value from `self.defaults` if `srsQs` is not None
            (after its default action, if applicable); otherwise,
            leave it None.
        srsconv : scalar or 1d array_like or None; optional
            Conversion factor for the SRS; scalar or vector same
            length as `srspv`.

            DA: if `srsQs` is not None (after its default action, if
            applicable), either get value for `srsconv` from
            `self.defaults` (if possible), or set to 1.0. If
            `srsQs` is None, leave `srsconv` None.
        srslabels : list_like or None; optional
            Analogous to `labels` but just for the `srspv` rows.

            DA: derive from `labels` according to `srspv` if needed;
            otherwise, leave it None.
        srsopts : dict or None; optional
            This dictionary can specify options for all three of these
            SRS calculation routines: :func:`pyyeti.srs.srs`,
            :func:`pyyeti.srs.srs_frf`, and
            :func:`pyyeti.srs.vrs`. The "eqsine" option is special in
            that it enforces equivalent sine output for all three
            routines even if the routine does not accept "eqsine" as
            input. Other options are only passed if the function
            accepts it. For example, to specify equivalent sine for
            all three ("eqsine" is special, as noted), steady-state
            initial conditions for :func:`pyyeti.srs.srs` and
            `scale_by_Q_only` for :func:`pyyeti.srs.srs_frf`, use:
            ``dict(eqsine=True, ic='steady', scale_by_Q_only=True)``

            DA: if `srsQs` is not None (after its default action, if
            applicable), either get value for `srsopts` from
            `self.defaults` (if possible), or set to ``{}``. If
            `srsQs` is None, leave `srsopts` None.
        srspv : 1d array_like or 'all' or slice or None; optional
            Specifies which rows to compute SRS for. See note below
            about inputting partition vectors.

            DA: if `srsQs` is not None (after its default action, if
            applicable), set to ``slice(len(labels))``; otherwise,
            leave it None.
        srsunits : string or None; optional
            Units string for the `srspv`.

            DA: if `srsQs` is not None (after its default action, if
            applicable), set to `units`; otherwise, leave it None.
        **kwargs : dict; optional
            All other inputs are quietly ignored.

        Returns
        -------
        None

        Notes
        -----
        See :class:`DR_Def` for a discussion about how the order of
        data recovery is determined. In summary: :func:`DR_Event.add`
        determines the order of :class:`DR_Def` instances, and
        :func:`DR_Def.add` determines the order of data recovery
        categories within each :class:`DR_Def` instance.
        :func:`DR_Event.set_dr_order` can be used to modify the final
        order. That routine uses the more general
        :func:`.pyyeti.ytools.reorder_dict` to reorder the
        categories.

        **Entering partition vectors.** The `histpv`, `srspv` and
        `ignorepv` inputs are all handled similarly. They can be input
        as an index or boolean style partition vector, as a slice, or
        as "all". If input as 'all', they are reset internally to
        ``slice(len(labels))``. The stored versions are either a slice
        or an index vector (uses 0-offset for standard Python
        indexing).

        **`drfunc`, `drfile` notes.** If `drfunc` is a valid Python
        identifier (called "Type 2" above), then `drfile` must contain
        the appropriate data recovery function(s) named ``name`` and,
        optionally, ``name_psd``. For a typical data recovery
        category, only one data recovery function would be
        needed. Here are some examples:

        For a typical ATM::

            def SC_atm(sol, nas, Vars, se):
                return Vars[se]['atm'] @ sol.a

        For a typical mode-displacement DTM::

            def SC_dtm(sol, nas, Vars, se):
                return Vars[se]['dtm'] @ sol.d

        For a typical mode-acceleration LTM::

            def SC_ltm(sol, nas, Vars, se):
                return (Vars[se]['ltma'] @ sol.a +
                        Vars[se]['ltmd'] @ sol.d)

        Note that those three examples could also be implemented more
        directly by using the following 3 "Type 1" `drfunc`
        definitions::

            drfunc = "Vars[se]['atm'] @ sol.a"        # ATM
            drfunc = "Vars[se]['dtm'] @ sol.d"        # DTM
            drfunc = \"\"\"(Vars[se]['ltma'] @ sol.a +
                     Vars[se]['ltmd'] @ sol.d)\"\"\"  # Mode-acce LTM

        For a more complicated data recovery category, you'll need to
        use `drfile` and you might also need to include a special PSD
        version. This function has the same name except with
        "_psd" tacked on. For example, to compute a time-consistent
        (or phase-consistent) root-sum-square (RSS), you need to
        provide both functions. Here is a pair of functions that
        recover 3 rows (for `name` "x_y_rss"): acceleration response
        in the 'x' and 'y' directions and the consistent RSS between
        them for the 3rd row. In this example, it is assumed the data
        recovery matrix has 3 rows where the 3rd row could be all
        zeros::

            def x_y_rss(sol, nas, Vars, se):
                resp = Vars[se]['xyrss'] @ sol.a
                xr = 0             # 'x' row(s)
                yr = 1             # 'y' row(s)
                rr = 2             # rss  rows
                resp[rr] = np.sqrt(resp[xr]**2 + resp[yr]**2)
                return resp

            def x_y_rss_psd(sol, nas, Vars, se, freq, forcepsd,
                            drmres, case, i):
                # drmres is a results namespace, eg:
                #   drmres = results['x_y_rss']
                # i is psd force number
                resp = Vars[se]['xyrss'] @ sol.a
                xr = 0             # 'x' row(s)
                yr = 1             # 'y' row(s)
                rr = 2             # rss  rows
                cla.PSD_consistent_rss(
                    resp, xr, yr, rr, freq, forcepsd,
                    drmres, case, i)

        Ultimately, the PSD recovery function must provide the final
        PSD solution in ``drmres[case]`` after the final force is
        analyzed (when ``i == forcepsd.shape[0]-1``). In the example
        above, :func:`PSD_consistent_rss` takes care of that job.

        Data recovery function inputs:

            =====  ===================================================
            Input  Description
            =====  ===================================================
            sol    ODE modal solution namespace with uncertainty
                   factors applied. Typically has at least .a, .v and
                   .d members (modal accelerations, velocities and
                   displacements). See
                   :func:`pyyeti.ode.SolveUnc.tsolve`.
            nas    The nas2cam dict:
                   ``nas = pyyeti.nastran.op2.rdnas2cam()``
            DR     Defines data recovery for an event simulation (and
                   is created in the simulation script via
                   ``DR = cla.DR_Event()``). It is an event specific
                   version of all combined :class:`DR_Def` objects
                   with all ULVS matrices applied.
            se     Superelement number. Used as key into `DR`.
            etc    See :func:`PSD_consistent_rss` for description of
                   `freq`, `forcepsd`, etc.
            =====  ===================================================
        """
        # get dictionary of inputs and trim out specially handled
        # entries:
        frame = inspect.currentframe()
        co = frame.f_code
        nargs = co.co_argcount + co.co_kwonlyargcount
        args = co.co_varnames[:nargs]
        values = frame.f_locals
        dr_def_args = {
            i: values[i]
            for i in args
            if i not in ("self", "name", "drms", "nondrms", "kwargs")
        }
        if name in self:
            raise ValueError(f'data recovery for "{name}" already defined')

        if drms is None:
            drms = {}

        if nondrms is None:
            nondrms = {}

        # check for overlapping keys in drms and nondrms:
        overlapping_names = set(nondrms) & set(drms)
        if overlapping_names:
            raise ValueError(
                f"`drms` and `nondrms` have overlapping names: {overlapping_names}"
            )

        # define self[name] entry:
        self[name] = SimpleNamespace(**dr_def_args)

        # hand defaults and default actions:
        self._handle_defaults(name)

        # add drms and nondrms to self:
        self._add_vars(name, drms, nondrms)

        # increment static variable ncats for error checking:
        DR_Def.ncats += 1

    def amend(self, *, name, overwrite_drms=False, **kwargs):
        """
        Amend a category

        Parameters
        ----------
        name : string
            Name of category to amend, eg: 'SC_atm'. Must already
            exist (if it doesn't, just use :func:`add` to create it as
            desired).
        overwrite_drms : bool; optional
            Allow replacement of original `drms` and `nondrms`. This
            can be dangerous, since multiplie categories can use the
            same `drms` and `nondrms`; use with caution.
        **kwargs : dict; optional
            Any inputs to :func:`add` that are to be amended. Any
            unrecognized entries are quietly ignored.

        Returns
        -------
        None

        Notes
        -----
        This routine collects all original settings for category
        `name`, replaces any of them by whatever is specified in
        ``**kwargs``, and calls :func:`add` to add the category to a
        temporary DR_Def instance. If there are no errors, the old
        category is replaced with the new one. Original data recovery
        order is maintained in case that is important.
        """
        ns = self[name]

        # up-front check for drms/nondrms:
        for drm in ("drms", "nondrms"):
            if drm in kwargs and not overwrite_drms:
                raise ValueError(
                    f"to overwrite the '{drm}' entry, set `overwrite_drms` to True"
                )

        # work with a temporary instance:
        temp_drdefs = DR_Def(self.defaults)

        # copy old dictionary:
        settings = ns.__dict__.copy()
        settings.update(kwargs)

        # add updated category to temporary DR_Def:
        temp_drdefs.add(name=name, **settings)

        # replace old category with updated version:
        self[name] = temp_drdefs[name]

        # check for drms/nondrms:
        se = self[name].se
        for drm in ("drms", "nondrms"):
            if drm in kwargs:
                new_dct = getattr(temp_drdefs["_vars"], drm)[se]
                if len(new_dct) > 0:
                    old_dct = getattr(self["_vars"], drm)[se]
                    old_dct.update(new_dct)

    def copycat(self, categories, name_addon, **kwargs):
        """
        Copy a category with optional modifications.

        Parameters
        ----------
        categories : string or list
            Name or names of category(s) to copy.
        name_addon : string or list
            If string, it will be appended to all category names to
            make the new names. If list, contains the new names with
            no regard to old names.
        **kwargs : misc
            Any options to modify. See :func:`add` for complete list.
            For `uf_reds`, set any of the four values to None to keep
            the original value.

        Returns
        -------
        None
            Adds a data recovery category.

        Notes
        -----
        The data recovery categories are copies of the originals with
        the `name` changed (according to `name_addon`) and new values
        set according to `**kwargs`.

        One very common usage is to add a zero-rigid-body version of a
        category; for example::

            drdefs.copycat(['SC_ifa', 'SC_atm'], '_0rb',
                             uf_reds=(0, None, None, None))

        would add 'SC_ifa_0rb' and 'SC_atm_0rb' copy categories
        without the rigid-body component. (Note that, as a
        convenience, :func:`add_0rb` exists for this specific task.)

        For another example, recover the 'SC_cg' (cg load factors)
        in "static" and "dynamic" pieces::

            drdefs.copycat('SC_cg', '_static',
                             uf_reds=(None, None, 0, None))
            drdefs.copycat('SC_cg', '_dynamic',
                             uf_reds=(None, None, None, 0))

        As a final example to show the alternate use of `name_addon`,
        here is an equivalent call for the static example::

            drdefs.copycat('SC_cg', ['SC_cg_static'],
                             uf_reds=(None, None, 0, None))

        Raises
        ------
        ValueError
            When the new category name already exists.
        """
        if isinstance(categories, str):
            categories = [categories]

        for name in categories:
            if name not in self:
                raise ValueError(f"{name} not found")

        for key in kwargs:
            if key not in self[categories[0]].__dict__:
                raise ValueError(f'{key} not found in `self["{categories[0]}"]`')

        for i, name in enumerate(categories):
            if isinstance(name_addon, str):
                new_name = name + name_addon
            else:
                new_name = name_addon[i]
            if new_name in self:
                raise ValueError(f'"{new_name}" category already defined')
            self[new_name] = copy.copy(self[name])
            cat = self[new_name]
            for key, value in kwargs.items():
                if key == "uf_reds":
                    cat.__dict__[key] = _merge_uf_reds(self[name].uf_reds, value)
                else:
                    cat.__dict__[key] = value

    def add_0rb(self, *args):
        """
        Add zero-rigid-body versions of selected categories.

        Parameters
        ----------
        *args : strings
            Category names for which to make '_0rb' versions.

        Notes
        -----
        This is a convenience function that uses :func:`copycat` to do
        the work::

            copycat(args, '_0rb', uf_reds=(0, None, None, None),
                    desc=desc+' w/o RB')

        where `desc` is the current value. See :func:`copycat` for
        more information.

        For example::

            drdefs.add_0rb('SC_ifa', 'SC_atm')

        would add 'SC_ifa_0rb' and 'SC_atm_0rb' categories with the
        first element in `uf_reds` set to 0.
        """
        for arg in args:
            if arg not in self:
                raise ValueError(f"{arg} not found")
            desc = self[arg].desc + " w/o RB"
            self.copycat(arg, "_0rb", uf_reds=(0, None, None, None), desc=desc)

    def excel_summary(self, excel_file="dr_summary.xlsx"):
        """
        Make excel file with summary of data recovery information.

        Parameters
        ----------
        excel_file : string or None; optional
            Name of excel file to create; if None, no file will be
            created.

        Returns
        -------
        drinfo : pandas DataFrame
            Summary table of data recovery information. The index
            values are those set via :func:`add` (name, desc, etc).
            The columns are the categories (eg: 'SC_atm', 'SC_ltm',
            etc).
        """
        cats = sorted([i for i in self if not i.startswith("_")])
        if not cats:
            raise RuntimeError("add data recovery categories first")
        vals = sorted(self[cats[0]].__dict__)
        df = pd.DataFrame(index=vals, columns=cats)

        def _issame(old, new):
            if new is old:
                return True
            # if type(new) != type(old):
            if not isinstance(new, type(old)):
                return False
            if isinstance(new, np.ndarray):
                if new.shape != old.shape:
                    return False
                return (new == old).all()
            return new == old

        fill_char = "-"
        # fill in DataFrame, use `fill_char` for "same as previous"
        for i, cat in enumerate(cats):
            for val in vals:
                new = self[cat].__dict__[val]
                s = None
                if i > 0:
                    old = self[cats[i - 1]].__dict__[val]
                    if _issame(old, new):
                        self[cat].__dict__[val] = old
                        s = fill_char
                if s is None:
                    s = str(new)
                    if len(s) > 80:
                        s = s[:35] + " ... " + s[-35:]
                    if not isinstance(new, str):
                        try:
                            slen = len(new)
                        except TypeError:
                            pass
                        else:
                            s = f"{slen}: {s}"
                # df[cat].loc[val] = s
                df.loc[val, cat] = s

        if excel_file is not None:
            with xlsxwriter.Workbook(excel_file) as workbook:
                hform = workbook.add_format(
                    {"bold": True, "align": "center", "valign": "vcenter", "border": 1}
                )
                tform = workbook.add_format({"border": 1, "text_wrap": True})
                worksheet = workbook.add_worksheet("DR_Def")
                worksheet.set_column("A:A", 10)
                worksheet.set_column(1, len(cats), 25)
                # worksheet.set_default_row(20)
                # write header:
                for i, cat in enumerate(df.columns):
                    worksheet.write(0, i + 1, cat, hform)
                # write labels:
                for i, lbl in enumerate(df.index):
                    worksheet.write(i + 1, 0, lbl, hform)
                # write table:
                for i, cat in enumerate(df.columns):
                    for j, lbl in enumerate(df.index):
                        worksheet.write(j + 1, i + 1, df[cat].loc[lbl], tform)
                # write notes at bottom:
                bold = workbook.add_format({"bold": True})
                worksheet.write(df.shape[0] + 2, 1, "Notes:", bold)
                tab = "    "
                msg = fill_char + " = same as previous category"
                worksheet.write(df.shape[0] + 3, 1, tab + msg)
                msg = (
                    "The partition vector variables (*pv) "
                    "use 0-offset (or are slices)"
                )
                worksheet.write(df.shape[0] + 4, 1, tab + msg)
                # freeze row 0 and column 0:
                worksheet.freeze_panes(1, 1)
        return df

    @staticmethod
    def merge(first, *args):
        """
        Merge DR_Def instances together to make a new instance

        Parameters
        ----------
        first : DR_Def instance
            The first DR_Def instance to merge with other instances
        *args : any number of DR_Def instances
            The other DR_Def instances to merge with `first`

        Returns
        -------
        DR_Def instance
            The merger of all input DR_Def instances

        Notes
        -----
        The final order of categories is as input. That is, the
        categories of `first` are first and in the order they were
        defined. The others are treated similarly. Note that the order
        can be changed via :func:`pyyeti.ytools.reorder_dict`.
        :func:`DR_Event.set_dr_order` can also be used to modify the
        final data recovery order.

        The ``+`` operator can also be used to merge DR_Def instances.
        That is, this::

            drdefs_new = DR_Def.merge(drdefs1, drdefs2, drdefs3)

        is equivalent to this::

            drdefs_new = drdefs1 + drdefs2 + drdefs3

        except calling :func:`DR_Def.merge` directly is more efficient
        if there are more than two instances being merged.

        Raises
        ------
        ValueError
            When the there are duplicate category names.

        Examples
        --------
        For demonstration, create two data recovery categories using
        two DR_Def instances instead of just one, and them merge them
        together:

        >>> from pyyeti import cla
        >>>
        >>> drdefs1 = cla.DR_Def()
        >>> @cla.DR_Def.addcat
        ... def _():
        ...     se = 101
        ...     name = 'atm'
        ...     desc = 'description'
        ...     labels = 12
        ...     drms = {'atm': 1}
        ...     nondrms = {'radius': 10}
        ...     drfunc = f"Vars[se]['{name}'] @ sol.a"
        ...     drdefs1.add(**locals())
        >>>
        >>> drdefs2 = cla.DR_Def()
        >>> @cla.DR_Def.addcat
        ... def _():
        ...     se = 102
        ...     name = 'dtm'
        ...     desc = 'description'
        ...     labels = 6
        ...     drms = {'dtm': 2}
        ...     nondrms = {'station': 20}
        ...     drfunc = f"Vars[se]['{name}'] @ sol.d"
        ...     drdefs2.add(**locals())
        >>>
        >>> drdefs = cla.DR_Def.merge(drdefs1, drdefs2)
        >>> drdefs    # doctest: +ELLIPSIS
        DR_Def (...) with 2 categories: ['_vars', 'atm', 'dtm']

        You can also use the ``+`` operator:

        >>> drdefs = drdefs1 + drdefs2
        >>> drdefs    # doctest: +ELLIPSIS
        DR_Def (...) with 2 categories: ['_vars', 'atm', 'dtm']
        """
        # use a set to check for duplicate category names:
        cats = set(first)
        cats.remove("_vars")
        duplicate_cats = set()
        for arg in args:
            newcats = set(arg)
            newcats.remove("_vars")
            duplicate_cats |= cats.intersection(newcats)
            cats |= newcats

        # were there any duplicate categories?:
        if duplicate_cats:
            raise ValueError(
                f"there were duplicate categories:\n\t{sorted(duplicate_cats)!r}"
            )

        # category names are unique, need to check drms & nondrms:
        data = {}
        for attr_name in ("drms", "nondrms"):
            duplicate_drms = {}
            drms = getattr(first["_vars"], attr_name).copy()
            for arg in args:
                new_drms = getattr(arg["_vars"], attr_name)
                for se, drm_dict in new_drms.items():
                    if se in drms:
                        # check for duplicate drm names:
                        current = set(drms[se])
                        new = set(drm_dict)
                        if se not in duplicate_drms:
                            duplicate_drms[se] = set()
                        duplicate_drms[se] |= current.intersection(new)
                        drms[se].update(drm_dict)
                    else:
                        drms[se] = drm_dict.copy()

            # were there any duplicate drm names?:
            dup_names = "".join(
                f"{se}: {sorted(dups)}\n" for se, dups in duplicate_drms.items() if dups
            )

            if dup_names:
                raise ValueError(
                    f'there were duplicate "{attr_name}" names. By SE:\n{dup_names}'
                )

            data[attr_name] = drms

        # if here, no duplicates, so merge:
        drdefs = DR_Def()
        drdefs.update(first)
        for arg in args:
            drdefs.update(arg)
        drdefs["_vars"] = SimpleNamespace(drms=data["drms"], nondrms=data["nondrms"])
        return drdefs

    def __add__(self, other):
        """
        Syntactic sugar for the :func:`DR_Def.merge` method.
        """
        return DR_Def.merge(self, other)


# setup pickling for a little bit of future-proofing:
def unpickle_drdefs(kwargs):
    # pickle_version is not used yet
    kwargs.pop("__pickle_version", 0)

    # Cannot use old drdefaults. For example, the drdefs may have been
    # merged from multiple different sets, all with their own
    # "drdefaults". So, get it but don't use it
    drdefaults = kwargs.pop("__defaults", None)

    # Currently, "srsQs" and "srsfrq" are set to "defaults" if None
    # (which is handy when defining categories originally, but less
    # handy for unpickling already-processed categories) ... then, if
    # any other "srs*" value is set (like "srsconv" used to be), the
    # "add" will fail because they're not in the "defaults". This is
    # avoided here simply by setting these two values to None in a new
    # "defaults". (This feels messy ... I'd like a better way to do
    # handle this ....)
    new_drdefs = DR_Def({"srsQs": None, "srsfrq": None})
    for k, v in kwargs.items():
        if k.startswith("__"):
            continue
        if k.startswith("_"):
            new_drdefs[k] = v
        else:
            new_drdefs.add(name=k, **v.__dict__)

    if drdefaults:
        new_drdefs.defaults = drdefaults

    return new_drdefs


def pickle_drdefs(drdefs):
    odct = OrderedDict(drdefs)
    odct["__defaults"] = drdefs.defaults
    odct["__pickle_version"] = 1
    return unpickle_drdefs, (odct,)


copyreg.pickle(DR_Def, pickle_drdefs)
