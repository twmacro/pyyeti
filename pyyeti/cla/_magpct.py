# -*- coding: utf-8 -*-
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from ._utilities import _proc_filterval, get_marker_cycle


def magpct(M1, M2, Ref=None, ismax=None, symbols=None,
           filterval=None, symlog=True, ax=None):
    """
    Plot percent differences in two sets of values vs magnitude.

    Parameters
    ----------
    M1, M2 : 1d or 2d array_like
        The two sets of values to compare. Must have the same shape.
        If 2d, each column is compared.
    Ref : 1d or 2d array_like or None; optional
        Same size as `M1` and `M2` and is used as the reference
        values. If None, ``Ref = M2``.
    ismax : bool or None; optional
        If None, the sign of the percent differences is determined by
        ``M1 - M2`` (the normal way). Otherwise, the sign is set to
        be positive where `M1` is more extreme than `M2`. More extreme
        is higher if `ismax` is True (comparing maximums), and lower
        if `ismax` is False (comparing minimums).
    symbols : iterable or None; optional
        Plot marker iterable (eg: string, list, tuple) that specifies
        the marker for each column. Values in `symbols` are reused if
        necessary. For example, ``symbols = 'ov^'``. If None,
        :func:`get_marker_cycle` is used to get the symbols.
    filterval : scalar, 1d array_like, or None; optional
        If None, no filtering is done and all percent differences are
        shown on a linear y-axis scale. If not None, percent
        differences for small numbers are handled in one of two ways
        depending on the `symlog` option (described next).
    symlog : bool; optional
        If `filterval` is not None, this option determines how
        `filterval` is used:

          ========   =================================================
          `symlog`   Description
          ========   =================================================
           True      Plot all percent differences, but use
                     :func:`matplotlib.pyplot.yscale` with the
                     "symlog" option. The "filtered region" is plotted
                     on a linear y-axis while percent differences for
                     the small values are potentially plotted outside
                     that region on a log y-axis. This is so all data
                     is shown, while emphasizing the important data.
           False     Plot only values larger (in the absolute sense)
                     than `filterval`.
          ========   =================================================
    ax : Axes object or None
        The axes to plot on. If None, ``ax = plt.gca()``.

    Returns
    -------
    pds : list
        List of 1d percent differences, one numpy array for each
        column in `M1` and `M2`. Each 1d array contains only the
        percent differences where ``M2 != 0.0``. If `M2` is all zero
        for a column (or all filtered out and `symlog` is False), the
        corresponding entry in `pds` is None.

    Notes
    -----
    The percent differences, ``(M1-M2)/Ref*100``, are plotted against
    the magnitude of `Ref`. If `ismax` is not None, signs are set as
    defined above so that positive percent differences indicate where
    `M1` is more extreme than `M2`.

    If desired, setup the plot axes before calling this routine.

    This routine is called by :func:`rptpct1`.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyyeti import cla
        >>> n = 500
        >>> m1 = 5 + np.arange(n)[:, None]/5 + np.random.randn(n, 2)
        >>> m2 = m1 + np.random.randn(n, 2)
        >>> fig = plt.figure('magpct demo', figsize=(6.4, 8))
        >>> ax = fig.subplots(3, 1, sharex=True)
        >>> _ = ax[0].set_title('No Filter')
        >>> pds = cla.magpct(m1, m2, symbols='ox', ax=ax[0])
        >>> _ = ax[1].set_title('Filter = 45, symlog=True')
        >>> pds = cla.magpct(m1, m2, symbols='ox',
        ...                  filterval=45, ax=ax[1])
        >>> _ = ax[2].set_title('Filter = 45, symlog=False')
        >>> pds = cla.magpct(m1, m2, symbols='ox',
        ...                  filterval=45, symlog=False, ax=ax[2])
        >>> fig.tight_layout()
    """
    if Ref is None:
        M1, M2 = np.atleast_1d(M1, M2)
        Ref = M2
    else:
        M1, M2, Ref = np.atleast_1d(M1, M2, Ref)

    if M1.shape != M2.shape or M1.shape != Ref.shape:
        raise ValueError('`M1`, `M2` and `Ref` must all have the'
                         ' same shape')

    filterval = _proc_filterval(filterval, M1.shape[0])

    def _get_next_pds(M1, M2, Ref, ismax):
        def _getpds(m1, m2, ref, ismax):
            if not symlog and filterval is not None:
                pv = ((ref != 0) & (abs(m1) > filterval) &
                      (abs(m2) > filterval))
            else:
                pv = ref != 0.0
            if not np.any(pv):
                return None, None, None
            a = m1[pv]
            b = m2[pv]
            pdiff = (a - b) / ref[pv] * 100.0
            if ismax is not None:
                pdiff = abs(pdiff)
                neg = a < b if ismax else a > b
                pdiff[neg] *= -1.0
            if symlog and filterval is not None:
                if len(filterval) > len(a):
                    filt = filterval[pv]
                else:
                    filt = filterval
                pvf = ((abs(a) > filt) &
                       (abs(b) > filt))
                if pvf.any():
                    max_linear_pdiff = abs(pdiff[pvf]).max()
                else:
                    max_linear_pdiff = None
            else:
                max_linear_pdiff = None
            return pdiff, ref[pv], max_linear_pdiff

        if M1.ndim == 1:
            yield _getpds(M1, M2, Ref, ismax)
        else:
            for c in range(M1.shape[1]):
                yield _getpds(M1[:, c], M2[:, c], Ref[:, c], ismax)

    if symbols:
        marker = itertools.cycle(symbols)
    else:
        marker = get_marker_cycle()

    if ax is None:
        ax = plt.gca()

    pds = []
    linthreshy, logthreshy = 0.0, 0.0
    apd = None
    for curpd, ref, mlp in _get_next_pds(M1, M2, Ref, ismax):
        pds.append(curpd)
        if curpd is not None:
            apd = abs(curpd)
            _marker = next(marker)
            for pv, c in [(apd <= 5, 'b'),
                          ((apd > 5) & (apd <= 10), 'm'),
                          (apd > 10, 'r')]:
                ax.plot(ref[pv], curpd[pv], c + _marker)
        if mlp is not None:
            logthreshy = max(logthreshy, abs(curpd).max())
            linthreshy = max(linthreshy, mlp)

    if linthreshy > 0.0 and logthreshy > 2 * linthreshy:
        lty = linthreshy
        # linthreshy = np.ceil(linthreshy)
        # linthreshy = 10 * np.ceil(linthreshy / 10)
        ax.set_yscale('symlog', linthreshy=linthreshy, linscaley=3,
                      subsy=[2, 3, 4, 5, 6, 7, 8, 9])
        ax.set_facecolor('lightgray')
        if len(filterval) == 1:
            ax.axvline(filterval, color='gray', linestyle='--',
                       linewidth=2.0, zorder=-1)
        ax.axhspan(-lty, lty, facecolor='white', zorder=-2)
        trans = transforms.blended_transform_factory(
            ax.transAxes, ax.transData)
        ax.text(0.98, 0.98 * lty, 'Filtered region', va='top',
                ha='right', transform=trans, fontstyle='italic')

    if apd is not None:
        ax.set_xlabel('Reference Magnitude')
        ax.set_ylabel('% Difference')
    return pds
