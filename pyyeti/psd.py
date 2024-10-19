# -*- coding: utf-8 -*-
"""
Power spectral density tools.
"""

from warnings import warn
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from pyyeti import dsp


# temporary patch for numpy < 2.0
try:
    np.trapezoid
except AttributeError:
    np.trapezoid = np.trapz


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def _set_frange(frange, low, high):
    s = frange[0] if frange[0] > 0.0 else low
    e = frange[-1] if frange[-1] <= high else high
    return s, e


def get_freq_oct(n, frange=(1.0, 10000.0), exact=False, trim="outside", anchor=None):
    r"""
    Get frequency vector on an octave scale.

    Parameters
    ----------
    n : scalar
        Specify octave band:  1 for full octave, 3 for 1/3 octave, 6
        for 1/6, etc.
    frange : 1d array_like; optional
        Specifies bounds for the frequencies according to input
        `trim`. Only the first and last elements are used. If the
        first element <= 0.0, 1.0 is used instead. See also the `trim`
        input.
    trim : string; optional
        Specify how to trim frequency vector to `frange`:

        =========   ================================================
         `trim`     Description
        =========   ================================================
        'inside'    All frequencies inside `frange`:
                        ``F_lower[0] >= frange[0]``;
                        ``F_upper[-1] <= frange[-1]``
        'center'    Center frequencies inside `frange`:
                        ``F[0] >= frange[0]``;
                        ``F[-1] <= frange[-1]``
        'outside'   First band includes ``frange[0]`` and last band
                    includes ``frange[-1]``
        'band'      Same as 'outside'
        =========   ================================================

    exact : bool; optional
        If False, return an approximate octave scale so that it hits
        the power of 10s, achored at 1 Hz by default (see
        `anchor`). If True, return an exact octave scale, anchored at
        1000 Hz by default.
    anchor : scalar or None; optional
        If scalar, it specifies the anchor. If None, the anchor used
        is specified under `exact` above (1 or 1000).

    Returns
    -------
    F : 1d ndarray
        Contains the band center frequencies on an octave scale.
    F_lower : 1d ndarray
        Same size as `F`, band lower frequencies.
    F_upper : 1d ndarray
        Same size as `F`, band upper frequencies.

    Notes
    -----
    If `exact` is False, the center, lower, and upper frequencies are
    computed from (where :math:`i` and :math:`j` are integers
    according to `frange` and `trim`):

    .. math::
        \begin{aligned}
        F &= anchor \cdot 10^{3 [i, i+1, i+2, ..., j] / (10 n)} \\
        F_{lower} &= F/10^{3/(20 n)} \\
        F_{upper} &= F \cdot 10^{3/(20 n)}
        \end{aligned}

    If `exact` is True:

    .. math::
        \begin{aligned}
        F &= anchor \cdot 2^{[i, i+1, i+2, ..., j] / n} \\
        F_{lower} &= F/2^{1/(2 n)} \\
        F_{upper} &= F \cdot 2^{1/(2 n)}
        \end{aligned}

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import psd
    >>> np.set_printoptions(precision=4, linewidth=75)
    >>> np.array(psd.get_freq_oct(3, [505, 900]))
    array([[  501.1872,   630.9573,   794.3282,  1000.    ],
           [  446.6836,   562.3413,   707.9458,   891.2509],
           [  562.3413,   707.9458,   891.2509,  1122.0185]])
    >>> np.array(psd.get_freq_oct(3, [505, 900], trim='center'))
    array([[ 630.9573,  794.3282],
           [ 562.3413,  707.9458],
           [ 707.9458,  891.2509]])
    >>> np.array(psd.get_freq_oct(3, [505, 900], exact=True))
    array([[  500.    ,   629.9605,   793.7005,  1000.    ],
           [  445.4494,   561.231 ,   707.1068,   890.8987],
           [  561.231 ,   707.1068,   890.8987,  1122.462 ]])
    >>> psd.get_freq_oct(6, [.8, 2.6])[0]
    array([ 0.7943,  0.8913,  1.    ,  1.122 ,  1.2589,  1.4125,  1.5849,
            1.7783,  1.9953,  2.2387,  2.5119])
    >>> psd.get_freq_oct(6, [.8, 2.6], anchor=2)[0]
    array([ 0.7962,  0.8934,  1.0024,  1.1247,  1.2619,  1.4159,  1.5887,
            1.7825,  2.    ,  2.244 ,  2.5179])
    >>> psd.get_freq_oct(6, [.8, 2.6], exact=True)[0]
    array([ 0.7751,  0.87  ,  0.9766,  1.0962,  1.2304,  1.3811,  1.5502,
            1.74  ,  1.9531,  2.1923,  2.4608])
    >>> psd.get_freq_oct(6, [.8, 2.6], exact=True, anchor=2)[0]
    array([ 0.7937,  0.8909,  1.    ,  1.1225,  1.2599,  1.4142,  1.5874,
            1.7818,  2.    ,  2.2449,  2.5198])
    """
    s = frange[0] if frange[0] > 0.0 else 1.0
    e = frange[-1]
    if exact:
        if not anchor:
            anchor = 1000.0
        var1 = np.floor(np.log2(s / anchor) * n)
        var2 = np.log2(e / anchor) * n + 1
        bands = np.arange(var1, var2)
        F = anchor * 2 ** (bands / n)
        factor = 2 ** (1 / (2 * n))
    else:
        if not anchor:
            anchor = 1.0
        var1 = np.floor(np.log10(s / anchor) * 10 * n / 3)
        var2 = np.log10(e / anchor) * 10 * n / 3 + 1
        bands = np.arange(var1, var2)
        F = anchor * 10 ** (3 * bands / (10 * n))
        factor = 10 ** (3 / (20 * n))
    FL, FU = F / factor, F * factor
    if trim in ("outside", "band"):
        Nmax = np.max(np.nonzero(FL <= e)[0]) + 1
        Nmin = np.min(np.nonzero(FU >= s)[0])
    elif trim == "center":
        Nmax = np.max(np.nonzero(F <= e)[0]) + 1
        Nmin = np.min(np.nonzero(F >= s)[0])
    elif trim == "inside":
        Nmax = np.max(np.nonzero(FU <= e)[0]) + 1
        Nmin = np.min(np.nonzero(FL >= s)[0])
    else:
        raise ValueError("input `trim` has invalid value")
    F = F[Nmin:Nmax]
    FL = FL[Nmin:Nmax]
    FU = FU[Nmin:Nmax]
    return F, FL, FU


def proc_psd_spec(spec):
    """
    Process PSD specification for other routines

    Parameters
    ----------
    spec : 2d ndarray or 2-element tuple/list
        If ndarray, its columns are ``[Freq, PSD1, PSD2, ... PSDn]``.
        Otherwise, it must be a 2-element tuple or list, eg:
        ``(Freq, PSD)`` where PSD is: ``[PSD1, PSD2, ... PSDn]``. In
        the second usage, PSD can be 1d; in the first usage, PSD is
        always considered 2d.

    Returns
    -------
    freq : 1d ndarray
        The frequency vector in `spec`.
    PSD : 1d or 2d ndarray
        The PSD matrix or vector as noted above. Will be 1d only if
        the second usage of `spec` was used and `PSD` is 1d. Shape is
        either ``(len(freq), n)`` or ``(len(freq),)``.
    npsds : integer
        Number of PSDs in `PSD`.

    Notes
    -----
    Any NaNs in the specification frequency are deleted (along with
    the corresponding PSD values).
    """
    if isinstance(spec, np.ndarray):
        if spec.ndim != 2 or spec.shape[1] <= 1:
            raise ValueError(
                "incorrectly sized ndarray for "
                "`spec` input (must be 2d with more"
                " than 1 column)"
            )
        Freq = spec[:, 0]
        PSD = spec[:, 1:]
        npsds = PSD.shape[1]
    else:
        if len(spec) != 2:
            msg = (
                "incorrectly sized `spec` input: for tuple/"
                f"list input, must have length 2, not {len(spec)}"
            )
            raise ValueError(msg)
        Freq, PSD = np.atleast_1d(*spec)
        if len(Freq) != PSD.shape[0]:
            raise ValueError("Freq and PSD inputs in `spec` are incompatibly sized")
        if PSD.ndim > 2:
            raise ValueError("the PSD input in `spec` has more than 2 dimensions.")
        npsds = 1 if PSD.ndim == 1 else PSD.shape[1]
    # check for nans in Freq:
    pv = np.isnan(Freq)
    if pv.any():
        pv = ~pv
        Freq = Freq[pv]
        PSD = PSD[pv]
    return Freq, PSD, npsds


def area(spec):
    r"""
    Compute the area under a PSD random specification.

    Parameters
    ----------
    spec : 2d ndarray or 2-element tuple/list
        If ndarray, its columns are ``[Freq, PSD1, PSD2, ... PSDn]``.
        Otherwise, it must be a 2-element tuple or list, eg:
        ``(Freq, PSD)`` where PSD is: ``[PSD1, PSD2, ... PSDn]``. In
        the second usage, PSD can be 1d; in the first usage, PSD is
        always considered 2d. The frequency vector must be
        monotonically increasing.

    Returns
    -------
    area : 1d array
        Area(s) under the PSD specification.

    Notes
    -----
    This routine assumes a constant db/octave slope for all segments.
    With this assumption, it computes the area for each segment with
    the following formula (derivation below). The segment goes from
    (f1, p1) to (f2, p2)::

         s = log(p2/p1)/log(f2/f1)
         if s == -1:
             area_segment = p1*f1*log(f2/f1)
         else:
             area_segment = (f2*p2 - f1*p1)/(s + 1)

    .. warning::
        This routine is only for specifications with all constant
        db/octave segments. Do not use for general freq vs. psd
        curves, such as output from analysis; use something like
        :func:`numpy.trapezoid` (or :func:`trapz` for NumPy versions <
        2.0).

    The following derives the equations for computing the area under
    the curve. Each segment is assumed to have a constant db/octave
    slope. We'll start by computing the slope (:math:`m`) of each
    segment. To get the slope, we'll need the number of dbs :math:`d`
    and the number of octaves :math:`n`. First, the number of dbs:

    .. math::
        d = 10 \log_{10} (p_2 / p_1)

    The number of octaves (:math:`n`) is defined by :math:`f_2 = 2^n
    f_1`. Solving for :math:`n` gives:

    .. math::
        n = \frac{\log_{10} (f_2 / f_1)}{\log_{10} (2)}

    Therefore, the slope :math:`m = d / n` for the segment is:

    .. math::
        m = 10 \log_{10} (2) \frac{\log_{10} (p_2 / p_1)}
            {\log_{10} (f_2 / f_1)}

    To simplify the equations, we'll define the variable :math:`s` as:

    .. math::
        s = \frac{m}{10 \log_{10} (2)} = \frac{\log_{10} (p_2 / p_1)}
            {\log_{10} (f_2 / f_1)}

    Solving for :math:`p_2` gives:

    .. math::
        p_2 = p_1 ( f_2 / f_1 )^s

    Since the segment has a constant db/octave slope, that equation
    can be generalized for any frequency value :math:`x` from
    :math:`f_1` to :math:`f_2`:

    .. math::
        p(x) = p_1 ( x / f_1 )^s

    The area under the segment is simply the integral of that
    equation:

    .. math::
        area_{segment} = \int_{f_1}^{f_2} p_1 ( x / f_1 )^s dx

    If :math:`s = -1`:

    .. math::
        area_{segment} = p_1 f_1 \ln ( f_2 / f_1 )

    Otherwise, if :math:`s \neq -1`:

    .. math::
        \begin{aligned}
        area_{segment} &= \frac{p_1 ( f_2^{s+1} - f_1^{s+1} )}
        {f_1^s (s+1)} \\
        &= ( p_2 f_2 - p_1 f_1 ) / (s + 1)
        \end{aligned}

    Examples
    --------
    Compute a 3-sigma peak value given a test spec:

    >>> import numpy as np
    >>> from pyyeti import psd
    >>> spec = np.array([[20, .0053],
    ...                  [150, .04],
    ...                  [600, .04],
    ...                  [2000, .0036]])
    >>> 3*np.sqrt(psd.area(spec))   # doctest: +ELLIPSIS
    array([ 18.43...])

    For comparison, use a brute-force technique:

    >>> f = np.arange(20, 2000.1, 0.1)
    >>> p = psd.interp(spec, f, linear=False)
    >>> 3*np.sqrt(np.trapezoid(p, f, axis=0))   # doctest: +ELLIPSIS
    array([ 18.43...])
    """
    Freq, PSD, _ = proc_psd_spec(spec)
    if PSD.ndim == 1:
        PSD = PSD[:, None]
    _area = np.zeros(PSD.shape[1])

    # accumulate the areas of all segments for each curve:
    for i in range(Freq.size - 1):
        f1 = Freq[i]
        f2 = Freq[i + 1]
        for j in range(PSD.shape[1]):
            p1 = PSD[i, j]
            p2 = PSD[i + 1, j]

            s = np.log(p2 / p1) / np.log(f2 / f1)
            if abs(s + 1.0) < 1e-5:
                # happens when p2/p1 = f1/f2
                #   slope = -10*log10(2) db/octave
                intarea = p1 * f1 * np.log(f2 / f1)
            else:
                intarea = (f2 * p2 - f1 * p1) / (s + 1.0)
            _area[j] += intarea
    return _area


def interp(spec, freq, linear=False):
    """
    Interpolate values on a PSD specification (or analysis curve).

    Parameters
    ----------
    spec : 2d ndarray or 2-element tuple/list
        If ndarray, its columns are ``[Freq, PSD1, PSD2, ... PSDn]``.
        Otherwise, it must be a 2-element tuple or list, eg:
        ``(Freq, PSD)`` where PSD is: ``[PSD1, PSD2, ... PSDn]``. In
        the second usage, PSD can be 1d; in the first usage, PSD is
        always considered 2d. The frequency vector must be
        monotonically increasing.
    freq : 1d array
        Vector of frequencies to interpolate the specification to.
    linear : bool
        If True, use linear interpolation to expand `spec` to the
        frequencies in `freq`. If False, `spec` is expanded via
        interpolation in log space. In other words:

        ================   ==========================================
        Use:               When:
        ================   ==========================================
        ``linear=False``   `spec` is an actual PSD test specification
                           -- that is, it uses constant db/octave
                           slopes
        ``linear=True``    `spec` doesn't use constant db/octave
                           slopes (eg, an analysis curve)
        ================   ==========================================

    Returns
    -------
    psdfull : 1d or 2d array
        Matrix of the interpolated PSD values. If 2d, has "n" columns
        (which will be one less than `spec` in the first usage above
        because the frequency column is not included). Will be 1d if
        PSD was 1d on input.

    Notes
    -----
    For PSD data that is defined on a center-band frequency scale, use
    :func:`rescale` instead.

    Zeros are used to fill in PSD values for frequencies outside the
    specification(s).

    Any NaNs in the specification frequency are deleted (along with
    the corresponding PSD values) before interpolation. This is done
    by the routine :func:`proc_psd_spec`.

    See also
    --------
    :func:`rescale`

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import psd
    >>> spec = np.array([[20, .0053],
    ...                  [150, .04],
    ...                  [600, .04],
    ...                  [2000, .0036]])
    >>> freq = [100, 200, 600, 1200]
    >>> np.set_printoptions(precision=3)
    >>> psd.interp(spec, freq).ravel()
    array([ 0.027,  0.04 ,  0.04 ,  0.01 ])
    >>> psd.interp(spec, freq, linear=True).ravel()
    array([ 0.027,  0.04 ,  0.04 ,  0.024])

    Using the second form of the spec input:

    >>> spec = ([   20, 150, 600,  2000],
    ...         [.0053, .04, .04, .0036])
    >>> psd.interp(spec, freq, linear=True)
    array([ 0.027,  0.04 ,  0.04 ,  0.024])
    """
    Freq, PSD, npsds = proc_psd_spec(spec)
    #    spec = np.atleast_2d(spec)
    freq = np.atleast_1d(freq)
    if linear:
        ifunc = interp1d(
            Freq, PSD, axis=0, bounds_error=False, fill_value=0, assume_sorted=True
        )
        psdfull = ifunc(freq)
    else:
        ifunc = interp1d(
            np.log(Freq),
            np.log(PSD),
            axis=0,
            bounds_error=False,
            fill_value=0,
            assume_sorted=True,
        )
        psdfull = ifunc(np.log(freq))
        pv = (freq >= Freq[0]) & (freq <= Freq[-1])
        psdfull[pv] = np.exp(psdfull[pv])
    return psdfull


def rescale(P, F, n_oct=3, freq=None, extendends=True, frange=None):
    """
    Convert PSD from one frequency scale to another.

    Parameters
    ----------
    P : array_like
        Vector or matrix; PSD(s) to convert. Works columnwise if
        matrix.
    F : array_like
        Vector; center frequencies for `P`. If steps are not linear,
        logarithmic spacing is assumed.
    n_oct : scalar; optional
        Specifies the desired octave scale. 1 means full octave, 3
        means 1/3rd octave, 6 means 1/6th octave, etc. The routine
        :func:`get_freq_oct` is used to calculate the frequency vector
        with the default options. To change options, call
        :func:`get_freq_oct` directly and provide that input via
        `freq`.
    freq : array_like or None; optional
        Alternative to `n_oct` and takes precedence. Specifies
        desired output frequencies directly. If steps are not linear,
        logarithmic spacing is assumed for computing the band
        boundaries.
    extendends : bool; optional
        If True and the first and/or last frequency band extends
        beyond the original data, the area of the new band is adjusted
        up by the ratio of the new bandwidth over the portion that is
        covered by the original data. This will cause the
        corresponding PSD value to be higher than it would otherwise
        be, meaning the overall mean-square value will also be a
        little higher.
    frange : 1d array_like or None; optional
        This option can be used to limit the frequency range of the
        output frequencies. If None, this option is ignored for cases
        where `freq` is used, but is set internally to ``(1.0,
        np.inf))`` for cases where `n_oct` is used. Only the first and
        last elements of `frange` are used. Note that the output
        frequencies will be trimmed further if needed by the first and
        last values of `F`.

        .. note::
            When the `n_oct` option is used, if the first value of
            `frange` is <= 0.0, it is internally reset to 1.0.

    Returns
    -------
    Pout : ndarray
        Vector or matrix; converted PSD(s). Rows correspond to the
        new frequency scale; columns correspond with `P`.
    Fctr : ndarray
        Vector; center frequencies of output. Equal to `freq` if that
        was input.
    msv : scalar or 1d ndarray
        Mean square value estimate for each PSD.
    ms : ndarray
        Vector or matrix; converted mean square values directly
        (instead of density). For constant frequency step df,
        ``ms = df*Pout``. Same size as `Pout`.

    Notes
    -----
    The input PSD is assumed to be on a center-band frequency
    scale. For PSD specifications that use constant dB/octave slopes,
    use :func:`interp` instead.

    This algorithm works by interpolating on cummulative area such
    that original contributions to total mean-square per band is
    preserved.

    .. note::

        Note that if the area of the first and/or last band is
        extended (see `extendends` above), the overall mean-square
        value will be higher than the original.

    See :func:`get_freq_oct` for more information on how the octave
    scales are calculated.

    See also
    --------
    :func:`interp`

    Examples
    --------
    Compute a PSD with :func:`scipy.signal.welch` and then rescale it
    to 3rd, 6th, and 12th octave scales starting at 1.0 Hz. Compare
    mean square values from all 4 PSDs.

    .. plot::
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import scipy.signal as signal
        >>> from pyyeti import psd
        >>>
        >>> frange = (1.0, np.inf)
        >>> rng = np.random.default_rng()
        >>> g = rng.normal(size=10000)
        >>> sr = 400
        >>> f, p = signal.welch(g, sr, nperseg=sr)
        >>> p3, f3, msv3, ms3 = psd.rescale(p, f, frange=frange)
        >>> p6, f6, msv6, ms6 = psd.rescale(
        ...     p, f, n_oct=6, frange=frange)
        >>> p12, f12, msv12, ms12 = psd.rescale(
        ...     p, f, n_oct=12, frange=frange)
        >>>
        >>> fig = plt.figure('Example', clear=True,
        ...                  layout='constrained')
        >>> line = plt.semilogx(f, p, label='Linear')
        >>> line = plt.semilogx(f3, p3, label='1/3 Octave')
        >>> line = plt.semilogx(f6, p6, label='1/6 Octave')
        >>> line = plt.semilogx(f12, p12, label='1/12 Octave')
        >>> _ = plt.legend(ncol=2, loc='best')
        >>> _ = plt.xlim([1, 200])
        >>> msv1 = np.sum(p[1:]*(f[1]-f[0]))
        >>> abs(msv1/msv3 - 1) < .12
        True
        >>> abs(msv1/msv6 - 1) < .06
        True
        >>> abs(msv1/msv12 - 1) < .03
        True

    Demonstrate ``extendends=False``. Ends will be less than input
    because there is a region of zero area averaged in:

    >>> in_freq = np.arange(0, 10.1, .25)
    >>> out_freq = np.arange(0, 10.1, 5)
    >>> in_p = np.ones_like(in_freq)
    >>> p, f, ms, mvs = psd.rescale(in_p, in_freq, freq=out_freq)
    >>> p
    array([ 1.,  1.,  1.])
    >>> p, f, ms, mvs = psd.rescale(in_p, in_freq, freq=out_freq,
    ...                             extendends=False)
    >>> p
    array([ 0.525,  1.   ,  0.525])

    The 0.525 value is from:
    ``area/width = 1*(2.5-(-0.125))/5 = 0.525``
    """
    F = np.atleast_1d(F)
    P = np.atleast_1d(P)

    def _get_fl_fu(fcenter):
        Df = np.diff(fcenter)
        # if np.all(Df == Df[0]):
        if (abs(Df / Df[0] - 1.0) < 1e-12).all():
            # linear scale
            Df = Df[0]
            FL = fcenter - Df / 2
            FU = fcenter + Df / 2
        else:
            # output is not linear, assume log
            mid = np.sqrt(fcenter[:-1] * fcenter[1:])
            fact1 = mid[0] / fcenter[1]
            fact2 = fcenter[-1] / mid[-1]
            FL = np.hstack((fact1 * fcenter[0], mid))
            FU = np.hstack((mid, fact2 * fcenter[-1]))
        return FL, FU

    if freq is None:
        if frange is None:
            frange = (1.0, np.inf)
        frange = _set_frange(frange, 1.0, F[-1])
        Wctr, FL, FU = get_freq_oct(n_oct, exact=True, frange=frange)
    else:
        freq = np.atleast_1d(freq)
        if frange is not None:
            freq = freq[(freq >= frange[0]) & (freq <= frange[-1])]
        FL, FU = _get_fl_fu(freq)
        Nmax = np.max(np.nonzero(FL <= F[-1])[0]) + 1
        Nmin = np.min(np.nonzero(FU >= F[0])[0])
        Wctr = freq[Nmin:Nmax]
        FL = FL[Nmin:Nmax]
        FU = FU[Nmin:Nmax]

    oned = False
    if P.ndim == 1 or (P.ndim == 2 and P.shape[0] == 1):
        oned = True
        P = P.reshape(-1, 1)
        cols = 1
    else:
        cols = P.shape[1]

    # calculate cumulative area:
    Df = np.diff(F)
    if np.all(Df == Df[0]):
        # input uses linear frequency scale
        FLin = F - Df[0] / 2
        FUin = F + Df[0] / 2
    else:
        # not linear, assume log
        FLin, FUin = _get_fl_fu(F)

    Df = (FUin - FLin).reshape(-1, 1)
    ca = np.vstack((np.zeros((1, cols)), np.cumsum(Df * P, axis=0)))
    Fa = np.hstack((FLin[0], FUin))

    if extendends:
        fl = FL[0]
        fu = FU[-1]
        if FL[0] < FLin[0]:
            FL[0] = FLin[0]
        if FU[-1] > FUin[-1]:
            FU[-1] = FUin[-1]

    cal = np.zeros((len(FL), cols))
    cau = np.zeros((len(FL), cols))
    for i in range(cols):
        # with np.interp, interpolating cumulative area beyond end
        # points will take the end value -- that's perfect here: 0's
        # on the front, total area on the back
        cal[:, i] = np.interp(FL, Fa, ca[:, i])
        cau[:, i] = np.interp(FU, Fa, ca[:, i])

    # Compute new values
    ms = cau - cal
    psdoct = ms * (1 / (FU - FL).reshape(-1, 1))
    if extendends:
        FL[0] = fl
        FU[-1] = fu
        ms = psdoct * (FU - FL).reshape(-1, 1)
    msv = np.sum(ms, axis=0)
    if oned:
        psdoct = psdoct.ravel()
        ms = ms.ravel()
        msv = msv[0]
    return psdoct, Wctr, msv, ms


def spl(
    x,
    sr,
    nperseg=None,
    overlap=0.5,
    window="hann",
    timeslice=1.0,
    tsoverlap=0.5,
    fs=3,
    pref=2.9e-9,
    frange=(25.0, np.inf),
    extendends=True,
):
    r"""
    Sound pressure level estimation using PSD.

    Parameters
    ----------
    x : 1d array like
        Vector of pressure values.
    sr : scalar
        Sample rate.
    nperseg : int, optional
        Length of each segment for the FFT. Defaults to
        ``int(sr / 5)`` for 5 Hz frequency step in PSD. Note:
        frequency step in Hz = ``sr/nperseg``.
    overlap : scalar; optional
        Amount of overlap in windows, eg 0.5 would be 50% overlap.
    window : str or tuple or array like; optional
        Passed to :func:`scipy.signal.welch`; see that routine for
        more information.
    timeslice : scalar or string-integer
        If scalar, it is the length in seconds for each slice. If
        string, it contains the integer number of points for each
        slice. For example, if `sr` is 1000 samples/second,
        ``timeslice=0.75`` is equivalent to ``timeslice="750"``.
    tsoverlap : scalar in [0, 1) or string-integer
        If scalar, is the fraction of each time-slice to overlap. If
        string, it contains the integer number of points to
        overlap. For example, if `sr` is 1000 samples/second,
        ``tsoverlap=0.5`` and ``tsoverlap="500"`` each specify 50%
        overlap.
    fs : integer; optional
        Specifies output frequency scale. Zero means linear scale,
        anything else is passed to :func:`rescale`. Example:

        ===  ======================================================
          0  linear scale as computed by :func:`scipy.signal.welch`
          1  full octave scale
          3  3rd octave scale
          6  6th octave scale
        ===  ======================================================

    pref : scalar; optional
        Reference pressure. 2.9e-9 psi is considered to be the
        threshhold for human hearing. In Pascals, that value is 2e-5.
    frange : 1d array_like; optional
        Specifies bounds for the frequencies. Only the first and last
        elements are used. If the first value is < 0.0, it is reset
        to 0.0. If the last value is > ``sr/2``, it is reset to
        ``sr/2``. Note that for octave scales, :func:`rescale` is used
        which enforces a minimum of 1.0 Hz.
    extendends : bool; optional
        Passed to :func:`rescale` if an octave scale output is
        requested. See that routine for more information.

    Returns
    -------
    f : ndarray
        The output frequency vector (Hz).
    spl : ndarray
        The sound pressure level vector in dB.
    oaspl : scalar
        The overall sound pressure level.

    Notes
    -----
    This routine ultimately calls :func:`scipy.signal.welch` (via
    :func:`psdmod`) to calculate the PSD. It then converts that to
    mean-square values per band (for linear frequency steps, this is
    just ``PSD * delta_freq``). Loosely, the math is:

    .. math::
        \begin{aligned}
        V &= \frac{PSD \cdot \Delta f}{P_{ref}^2} \\
        SPL &= 10 \; \log_{10} ( V ) \\
        OASPL &= 10 \; \log_{10} \left ( \sum V \right )
        \end{aligned}

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from pyyeti import psd
        >>> rng = np.random.default_rng()
        >>> x = rng.normal(size=100000)
        >>> sr = 4000
        >>> f, spl, oaspl = psd.spl(x, sr, sr, timeslice=len(x)/sr)
        >>> # oaspl should be around 170.75 (since variance = 1):
        >>> shouldbe = 10*np.log10(1/(2.9e-9)**2)
        >>> abs(oaspl/shouldbe - 1) < .01
        True

        Plot the 1/3 octave SPL from above with a full octave SPL. The
        OASPL comes out a little higher for the full octave SPL
        because of the ``extendends=True`` option:

        >>> import matplotlib.pyplot as plt
        >>> full = psd.spl(x, sr, sr, timeslice=len(x)/sr, fs=1)
        >>> lbl = f"1/3 Octave SPL; OASPL={oaspl:.2f}"
        >>> _ = plt.plot(f, spl, "-o", label=lbl)
        >>> lbl = f"Full Octave SPL; OASPL={full[2]:.2f}"
        >>> _ = plt.plot(full[0], full[1], "-o", label=lbl)
        >>> _ = plt.legend()
    """
    if nperseg is None:
        nperseg = int(sr / 5)
    # compute psd
    F, P = psdmod(
        x,
        sr,
        nperseg=nperseg,
        window=window,
        timeslice=timeslice,
        tsoverlap=tsoverlap,
        noverlap=int(overlap * nperseg),
    )
    s, e = _set_frange(frange, 0.0, sr / 2)
    if fs != 0:
        _, F, _, P = rescale(P, F, n_oct=fs, frange=(s, e), extendends=extendends)
    else:
        P = P * F[1]
        pv = (F >= s) & (F <= e)
        F = F[pv]
        P = P[pv]
    v = P / pref**2
    return F, 10 * np.log10(v), 10 * np.log10(np.sum(v))


def psd2time(
    spec,
    fstart,
    fstop,
    *,
    ppc=10,
    df=None,
    winends=None,
    gettime=False,
    expand_method="interp",
    rng=None,
):
    r"""
    Generate a 'random' time domain signal given a PSD specification.

    Parameters
    ----------
    spec : 2d ndarray or 2-element tuple/list
        If ndarray, it has two columns: ``[Freq, PSD]``.
        Otherwise, it must be a 2-element tuple or list, eg:
        ``(Freq, PSD)``.
    fstart : scalar
        Starting frequency in Hz
    fstop : scalar
        Stopping frequency in Hz
    ppc : scalar; optional
        Points per cycle at highest (`fstop`) frequency; if < 2, it is
        internally reset to 2. With `fstop`, determines the sample
        rate: ``sr = ppc * fstop``.
    df : scalar or None; optional
        Serves two purposes: it is the frequency step between
        sinusoids included in the time signal, and it also determines
        the duration of the signal. If None, it is set to ``fstart /
        100`` in order to have 100 cycles at the lowest frequency. If
        input as a scalar value, it is taken as a hint and will be
        internally adjusted lower as needed; see Notes section. If
        routine gives poor results, try refining `df`. If `df` is
        greater than `fstart`, it is reset internally to `fstart` (in
        that case though, you'll only get 1 cycle at the lowest
        frequency). Total duration of the signal: ``T = 1 / df`` (see
        equations below for more details).

        .. note::
            `df` can be used to indirectly specify the number of
            cycles desired at the lowest frequency (`fstart`). For
            example, if ``fstart=5.0`` and you want to have 500 cycles
            of the 5.0 Hz content, then use ``df=0.01``: ``fstart / df
            == 500``.

    winends : None or dictionary; optional
        If None, :func:`pyyeti.dsp.windowends` is not
        called. Otherwise, `winends` must be a dictionary of arguments
        that will be passed to :func:`pyyeti.dsp.windowends` (not
        including `signal`).
    gettime : bool; optional
        If True, a time vector is output.
    expand_method : str; optional
        Either 'interp' or 'rescale', referring to which function in
        this module will be used to expand input `spec` to all needed
        frequencies. Use 'interp' if `spec` is a specification with
        constant dB/octave slopes. Use 'rescale' if `spec` provides
        the PSD levels on a center-band scale. See :func:`interp` and
        :func:`rescale` for more information.
    rng : :class:`numpy.random.Generator` object or None; optional
        Random number generator. If None, a new generator is created
        via :func:`numpy.random.default_rng`. Uniform deviates are
        generated via :func:`rng.random`. Supplying your own `rng` can
        be handy for parallel applications, for example, when you need
        repeatability. For illustration, the following creates a
        PCG-64 DXSM generator and initializes it with a seed of 1::

            from numpy.random import Generator, PCG64DXSM
            rng = Generator(PCG64DXSM(seed=1))

    Returns
    -------
    sig : 1d ndarray
        The time domain signal with properties set to match input PSD
        spectrum. Duration of signal: ``1 / df``.
    sr : scalar
        The sample rate of the signal (``ppc * fstop``)
    time : 1d ndarray; optional
        Time vector for the signal starting at zero with step of ``1.0
        / sr``: ``time = np.arange(len(sig)) / sr``. Only returned if
        `gettime` is True.

    Notes
    -----
    The following outlines the equations used in this routine.

    If :math:`df` is None: :math:`df = fstart / 100`.

    If :math:`df` is greater than :math:`fstart`, it is reset to
    :math:`fstart`:

    .. math::
        df = \min(df, fstart)

    The total time of the signal :math:`T` is determined by the lowest
    frequency cycle (at frequency :math:`df`):

    .. math::
        T = 1 / df

    The number of points needed is determined by the number of cycles
    at the highest frequency multiplied by the points/cycle
    :math:`ppc`:

    .. math::
        N = \lceil {fstop \cdot ppc \cdot T} \rceil

    The frequency step :math:`df` is reset to match :math:`N` (because
    of the possible round-up in the last equation):

    .. math::
        df = fstop \cdot ppc / N

    The final downward adjustment to :math:`df` is to make sure we hit
    the :math:`fstart` frequency exactly:

    .. math::
        df = fstart / \lfloor fstart / df \rfloor

    The frequency vector is defined by:

    .. math::
        f = [fstart, fstart+df, ...,
        \lceil fstop / df \rceil \cdot df]

    A sinusoid is to be defined at each of those frequencies, so we
    need to compute an amplitude and phase for each.

    The amplitude is determined by the magnitude of the PSD. Since a
    "power spectral density" is really mean-square spectral density,
    and since the mean-square value of a sinusoid is the amplitude
    squared over 2, the amplitude of each sinusoid is computed by:

    .. math::
        amp(f) = \sqrt { 2 \cdot PSD(f) \cdot df }

    The phase is determined by using pseudo-random deviates from a
    uniform distribution:

    .. math::
        phase(f) = U(0, 2\pi)

    The amplitude and phase are assembled together in a complex array
    so that an inverse-FFT will give a real time signal matching the
    desired mean-square profile defined by the input PSD.

    Raises
    ------
    ValueError
        If more than one PSD specification is input.
    ValueError
        On invalid setting for `expand_method`.

    See also
    --------
    :func:`interp`, :func:`rescale`, :func:`pyyeti.dsp.windowends`

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> from pyyeti import psd
        >>> spec = np.array([[20,  .0768],
        ...                  [50,  .48],
        ...                  [100, .48]])
        >>> we = dict(portion=0.01)
        >>> seed = 1
        >>> rng = np.random.default_rng(seed)
        >>> sig, sr = psd.psd2time(
        ...     spec, 35, 70, ppc=10, df=.01, winends=we, rng=rng,
        ... )
        >>> sr  # the sample rate should be 70*10 = 700
        700.0

        Compare the resulting psd to the spec from 37 to 68:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.signal as signal
        >>> f, p = signal.welch(sig, sr, nperseg=sr)
        >>> pv = np.logical_and(f >= 37, f <= 68)
        >>> fi = f[pv]
        >>> psdi = p[pv]
        >>> speci = psd.interp(spec, fi).ravel()
        >>> abs(speci - psdi).max() < .05
        True
        >>> abs(np.trapezoid(psdi, fi) - np.trapezoid(speci, fi)) < .25
        True
        >>> fig = plt.figure('Example', clear=True,
        ...                  layout='constrained')
        >>> a = plt.subplot(211)
        >>> line = plt.plot(np.arange(len(sig))/sr, sig)
        >>> plt.grid(True)
        >>> a = plt.subplot(212)
        >>> line = plt.loglog(spec[:, 0], spec[:, 1], label='spec')
        >>> line = plt.loglog(f, p, label='PSD of time signal')
        >>> leg = plt.legend(loc='best')
        >>> x = plt.xlim(20, 100)
        >>> y = plt.ylim(.05, 1)
        >>> plt.grid(True)
        >>> xticks = np.arange(20, 105, 10)
        >>> x = plt.xticks(xticks, xticks)
        >>> yticks = (.05, .1, .2, .5, 1)
        >>> y = plt.yticks(yticks, yticks)
        >>> v = plt.axvline(35, color='black', linestyle='--')
        >>> v = plt.axvline(70, color='black', linestyle='--')
    """
    _freq, _psd, npsds = proc_psd_spec(spec)
    if npsds > 1:
        raise ValueError(
            "only a single PSD specification is currently allowed in "
            f"`psd2time`, but {npsds} were provided"
        )

    if df is None:
        df = fstart / 100

    if df > fstart:
        df = fstart

    if ppc < 2:
        ppc = 2

    # compute parameters
    # 1 cycle of lowest frequency defines length of signal:
    T = 1 / df  # number of seconds for lowest frequency cycle
    N = int(np.ceil(fstop * ppc * T))  # total number of pts
    df = fstop * ppc / N
    # adjust df to make sure fstart is an even multiple of df
    df = fstart / np.floor(fstart / df)
    sr = N * df  # sr = N/T = N/(1/df)
    odd = N & 1

    # define constants
    # freq = np.arange(fstart, fstop + df / 2, df)
    freq = np.arange(fstart, fstop + df, df)  # 4/9/22

    # generate amp(f) vector
    if expand_method == "interp":
        speclevel = interp(spec, freq).ravel()
    elif expand_method == "rescale":
        speclevel, *_ = rescale(_psd.ravel(), _freq, freq=freq)
    else:
        raise ValueError(
            '`expand_method` must be either "interp" or "rescale", '
            f"not {expand_method!r}"
        )

    amp = np.sqrt(2 * speclevel * df)

    m = (N + 1) // 2 if odd else N // 2 + 1

    # build up amp to include zero frequency to fstart and from fstop
    # to fhighest:
    ntop = int(np.floor((fstart - df / 2) / df) + 1)
    nbot = m - ntop - len(amp)
    if nbot > 0:
        amp = np.hstack((np.zeros(ntop), amp, np.zeros(nbot)))
    else:
        amp = np.hstack((np.zeros(ntop), amp))

    # generate F(t)
    # - use uniformly distributed random phase angle:
    if rng is None:
        rng = np.random.default_rng()
    phi = rng.random(m) * np.pi * 2

    # force these terms to be pure cosine
    phi[0] = 0.0
    if not odd:
        phi[m - 1] = 0

    # coefficients:
    a = amp * np.cos(phi)
    b = -amp * np.sin(phi)

    # form matrix ready for ifft:
    if odd:
        a2 = a[1:m] / 2
        b2 = b[1:m] / 2
        r = N * np.hstack((a[0], a2, a2[::-1]))  # real part
        i = N * np.hstack((0, b2, -b2[::-1]))  # imag part
    else:
        a2 = a[1 : m - 1] / 2
        b2 = b[1 : m - 1] / 2
        r = N * np.hstack((a[0], a2, a[m - 1], a2[::-1]))  # real part
        i = N * np.hstack((0, b2, 0, -b2[::-1]))  # imag part

    F_time = np.fft.ifft(r + 1j * i)
    mxi = abs(F_time.imag).max()
    mxr = abs(F_time.real).max()
    if mxi > 1e-7 * mxr:  # pragma: no cover
        # bug in the code if this ever happens
        warn(
            f"method failed accuracy test; (max imag)/(max real) = {mxi / mxr}",
            RuntimeWarning,
        )

    F_time = F_time.real
    if winends is not None:
        F_time = dsp.windowends(F_time, **winends)
    if gettime:
        return F_time, sr, np.arange(N) / sr
    return F_time, sr


def psdmod(sig, sr, nperseg=None, timeslice=1.0, tsoverlap=0.5, getmap=False, **kwargs):
    """
    Modified method for PSD estimation via FFT.

    Parameters
    ----------
    sig : 1d array_like
        Time series of measurement values.
    sr : scalar
        Sample rate.
    nperseg : int, optional
        Length of each segment for the FFT. Defaults to
        ``int(sr / 5)`` for 5 Hz frequency step in PSD. Note:
        frequency step in Hz = ``sr/nperseg``.
    timeslice : scalar or string-integer
        If scalar, it is the length in seconds for each slice. If
        string, it contains the integer number of points for each
        slice. For example, if `sr` is 1000 samples/second,
        ``timeslice=0.75`` is equivalent to ``timeslice="750"``.
    tsoverlap : scalar in [0, 1) or string-integer
        If scalar, is the fraction of each time-slice to overlap. If
        string, it contains the integer number of points to
        overlap. For example, if `sr` is 1000 samples/second,
        ``tsoverlap=0.5`` and ``tsoverlap="500"`` each specify 50%
        overlap.
    getmap : bool, optional
        If True, get the PSD map output (the `Pmap` and `t` variables
        described below).
    *kwargs : optional
        Named arguments to pass to :func:`scipy.signal.welch`.

    Returns
    -------
    f : 1d ndarray
        Array of sample frequencies.
    Pxx : 1d ndarray
        Power spectral density or power spectrum of `sig`.
    Pmap : 2d ndarray; optional
        The PSD map; each column is an output of
        :func:`scipy.signal.welch`. Rows correspond to frequency `f`
        and columns correspond to time `t`. Only output if `getmap` is
        True.
    t : 1d ndarray; optional
        The time vector for the columns of `Pmap`. Only output if
        `getmap` is True.

    Notes
    -----
    This routine calls :func:`pyyeti.dsp.waterfall` for handling the
    timeslices and preparing the output and :func:`scipy.signal.welch`
    to process each time slice. So, the "modified" method is to use
    the PSD averaging (via welch) for each time slice but then take
    the peaks over all these averages.

    For a pure 'maximax' PSD, just set `timeslice` to ``nperseg/sr``
    and `tsoverlap` to 0.5 (assuming 50% overlap is desired).
    Conversely, for a pure Welch periodogram, just set the `timeslice`
    equal to the entire signal (or just use :func:`scipy.signal.welch`
    of course). Usually the desired behavior for :func:`psdmod` is
    somewhere between these two extremes.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyyeti import psd
        >>> from scipy import signal
        >>> TF = 30  # make a 30 second signal
        >>> spec = np.array([[20, 1], [50, 1]])
        >>> sig, sr, t = psd.psd2time(spec, ppc=10, fstart=20,
        ...                           fstop=50, df=1/TF,
        ...                           winends=dict(portion=10),
        ...                           gettime=True)
        >>> f, p = signal.welch(sig, sr, nperseg=sr)
        >>> f2, p2 = psd.psdmod(sig, sr, nperseg=sr, timeslice=4,
        ...                     tsoverlap=0.5)
        >>> f3, p3 = psd.psdmod(sig, sr, nperseg=sr)
        >>> spec = spec.T
        >>> fig = plt.figure('Example', clear=True,
        ...                  layout='constrained')
        >>> _ = plt.subplot(211)
        >>> _ = plt.plot(t, sig)
        >>> _ = plt.title(r'Input Signal - Specification Level = '
        ...               '1.0 $g^{2}$/Hz')
        >>> _ = plt.xlabel('Time (sec)')
        >>> _ = plt.ylabel('Acceleration (g)')
        >>> _ = plt.subplot(212)
        >>> _ = plt.plot(*spec, 'k-', lw=1.5, label='Spec')
        >>> _ = plt.plot(f, p, label='Welch PSD')
        >>> _ = plt.plot(f2, p2, label='PSDmod')
        >>> _ = plt.plot(f3, p3, label='Maximax')
        >>> _ = plt.legend(loc='best')
        >>> _ = plt.xlim(20, 50)
        >>> _ = plt.title('PSD')
        >>> _ = plt.xlabel('Frequency (Hz)')
        >>> _ = plt.ylabel(r'PSD ($g^2$/Hz)')
    """
    if nperseg is None:
        nperseg = int(sr / 5)
    ntimeslice = dsp._proc_timeslice(timeslice, sr, sig.size)[0]
    if nperseg > ntimeslice:
        raise ValueError(
            "`nperseg` too big for current `timeslice` setting;"
            " either decrease `nperseg` or increase `timeslice`"
        )
    welch_inputs = dict(fs=sr, nperseg=nperseg, **kwargs)
    pmap, t, f = dsp.waterfall(
        sig,
        sr,
        timeslice,
        tsoverlap,
        signal.welch,
        which=1,
        freq=0,
        kwargs=welch_inputs,
    )
    p = pmap.max(axis=1)
    if getmap:
        return f, p, pmap, t
    return f, p
