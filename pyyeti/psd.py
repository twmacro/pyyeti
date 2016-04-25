# -*- coding: utf-8 -*-
"""
Power spectral density tools.
"""

import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from pyyeti import dsp
from warnings import warn


def get_freq_oct(n, frange=[1., 10000.], exact=False, trim='band',
                anchor=None):
    r"""
    Get frequency vector on an octave scale.

    Parameters
    ----------
    n : scalar
        Specify octave band:  1 for full octave, 3 for 1/3 octave, 6
        for 1/6, etc.
    frange : 1d array_like
        Specifies bounds for the frequency vector; only the first and
        last elements are used. If the first element <= 0.0, 1. is used
        instead. See also the `trim` input.
    trim : string
        Specify how to trim frequency vector to `frange`. If `trim` is
        "band", trimming is such that the first band includes
        `frange[0]` and the last band include `frange[1]`. If `trim`
        is "center", trimming is such that the `frange` values are just
        outside the range of the returned vector.
    exact : bool
        If False, return an approximate octave scale so that it hits
        the power of 10s, achored at 1 Hz. If True, return an exact
        octave scale, anchored at 1000 Hz.
    anchor : scalar or None
        If scalar, it specifies the anchor. If None, the anchor used
        is specified above (1 or 1000).

    Returns
    -------
    F : 1d ndarray
        Contains the center frequencies on an octave scale.
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
    >>> np.set_printoptions(precision=4)
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
    if frange[0] <= 0.:
        s = 1.
    else:
        s = frange[0]
    e = frange[-1]
    if exact:
        if not anchor:
            anchor = 1000.
        var1 = np.floor(np.log2(s/anchor)*n)
        var2 = np.log2(e/anchor)*n + 1
        bands = np.arange(var1, var2)
        F = anchor * 2**(bands/n)
        factor = 2**(1/(2*n))
    else:
        if not anchor:
            anchor = 1.
        var1 = np.floor(np.log10(s/anchor)*10*n/3)
        var2 = np.log10(e/anchor)*10*n/3 + 1
        bands = np.arange(var1, var2)
        F = anchor * 10**(3*bands/(10*n))
        factor = 10**(3/(20*n))
    FL, FU = F/factor, F*factor
    if trim == 'band':
        Nmax = np.max(np.nonzero(FL <= e)[0]) + 1
        Nmin = np.min(np.nonzero(FU >= s)[0])
    else:
        Nmax = np.max(np.nonzero(F <= e)[0]) + 1
        Nmin = np.min(np.nonzero(F >= s)[0])
    F = F[Nmin:Nmax]
    FL = FL[Nmin:Nmax]
    FU = FU[Nmin:Nmax]
    return F, FL, FU


def interp(spec, freq, linear=False):
    """
    Interpolate values on a PSD specification (or analysis curve).

    Parameters
    ----------
    spec : 2d array
        Matrix containing the PSD specification(s) of the base
        excitation. Columns are: [ Freq PSD1 PSD2 ... PSDn ]. The
        frequency vector must be monotonically increasing.
    freq : 1d array
        Vector of frequencies to interpolate the specification to.
    linear : bool
        If True, use linear interpolation to compute the values;
        otherwise, the interpolation is done using the logs. Using logs
        is appropriate if the `spec` is actually a specification that
        uses constant db/octave slopes. Use ``linear=True`` for
        analysis curves (when not assuming constant db/octave slopes).

    Returns
    -------
    spec2 : 2d array
        Matrix of the interpolated PSD values. Has one fewer columns
        than `spec` because the frequency column is not included.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import psd
    >>> spec = [[20, .0053],
    ...        [150, .04],
    ...        [600, .04],
    ...        [2000, .0036]]
    >>> freq = [100, 200, 600, 1200]
    >>> np.set_printoptions(precision=3)
    >>> psd.interp(spec, freq).flatten()
    array([ 0.027,  0.04 ,  0.04 ,  0.01 ])
    >>> psd.interp(spec, freq, linear=True).flatten()
    array([ 0.027,  0.04 ,  0.04 ,  0.024])
    """
    spec = np.atleast_2d(spec)
    freq = np.atleast_1d(freq)
    if linear:
        ifunc = interp1d(spec[:, 0], spec[:, 1:], axis=0,
                         bounds_error=False, fill_value=0,
                         assume_sorted=True)
        psdfull = ifunc(freq)
    else:
        sp = np.log(spec)
        ifunc = interp1d(sp[:, 0], sp[:, 1:], axis=0,
                         bounds_error=False, fill_value=0,
                         assume_sorted=True)
        psdfull = np.exp(ifunc(np.log(freq)))
    return psdfull


def rescale(P, F, n_oct=3, freq=None, extendends=True):
    """
    Convert PSD from one frequency scale to another.

    Parameters
    ----------
    P : array_like
        Vector or matrix; PSD(s) to convert. Works columnwise if
        matrix.
    F : array_like
        Vector; input frequency scale. If steps are not linear,
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
    This algorithm works by interpolating on cummulative area such
    that original contributions to total mean-square per band is
    preserved.

    .. note::

        Note that if the area of the first and/or last band is
        extended (see `extendends` above), the overall mean-square
        value will be higher than the original.

    See :func:`get_freq_oct` for more information on how the octave scales
    are calculated.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import scipy.signal as signal
        >>> from pyyeti import psd
        >>>
        >>> g = np.random.randn(10000)
        >>> sr = 400
        >>> f, p = signal.welch(g, sr, nperseg=sr)
        >>> p3, f3, msv3, ms3 = psd.rescale(p, f)
        >>> p6, f6, msv6, ms6 = psd.rescale(p, f, n_oct=6)
        >>> p12, f12, msv12, ms12 = psd.rescale(p, f, n_oct=12)
        >>>
        >>> fig = plt.figure('PSD compare')
        >>> line = plt.semilogx(f, p, label='Linear')
        >>> line = plt.semilogx(f3, p3, label='1/3 Octave')
        >>> line = plt.semilogx(f6, p6, label='1/6 Octave')
        >>> line = plt.semilogx(f12, p12, label='1/12 Octave')
        >>> _ = plt.legend(ncol=2, loc='best')
        >>> _ = plt.xlim([1, 200])
        >>> msv1 = np.sum(p*(f[1]-f[0]))
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

    The 0.525 value is from: ``area/width = 1*(2.5-(-.125))/5 = .525``

    """
    F = np.atleast_1d(F)
    P = np.atleast_1d(P)

    def get_fl_fu(fcenter):
        Df = np.diff(fcenter)
        if np.all(Df == Df[0]):
            # linear scale
            Df = Df[0]
            FL = fcenter-Df/2
            FU = fcenter+Df/2
        else:
            # output is not linear, assume log
            mid = np.sqrt(fcenter[:-1] * fcenter[1:])
            fact1 = mid[0] / fcenter[1]
            fact2 = fcenter[-1] / mid[-1]
            FL = np.hstack((fact1*fcenter[0], mid))
            FU = np.hstack((mid, fact2*fcenter[-1]))
        return FL, FU

    if freq is None:
        Wctr, FL, FU = get_freq_oct(n_oct, F, exact=True)
    else:
        freq = np.atleast_1d(freq)
        FL, FU = get_fl_fu(freq)
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
        FLin = F - Df[0]/2
        FUin = F + Df[0]/2
    else:
        # not linear, assume log
        FLin, FUin = get_fl_fu(F)

    Df = (FUin - FLin).reshape(-1, 1)
    ca = np.vstack((np.zeros((1, cols)), np.cumsum(Df*P, axis=0)))
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
    psdoct = ms*(1/(FU-FL).reshape(-1, 1))
    if extendends:
        FL[0] = fl
        FU[-1] = fu
        ms = psdoct * (FU-FL).reshape(-1, 1)
    msv = np.sum(ms, axis=0)
    if oned:
        psdoct = psdoct.flatten()
        ms = ms.flatten()
        msv = msv[0]
    return psdoct, Wctr, msv, ms


def spl(x, sr, nperseg=None, overlap=0.5, window='hanning',
        timeslice=1.0, tsoverlap=0.5, fs=3, pref=2.9e-9):
    r"""
    Sound pressure level estimation using PSD.

    Parameters
    ----------
    x : 1d array like
        Vector of pressure values.
    sr : scalar
        Sample rate.
    nperseg : int, optional
        Length of each segment for the FFT. Defaults to `sr` for 1 Hz
        frequency step in PSD. Note: frequency step in Hz = sr/nperseg.
    overlap : scalar; optional
        Amount of overlap in windows, eg 0.5 would be 50% overlap.
    window : str or tuple or array like; optional
        Passed to :func:`scipy.signal.welch`; see that routine for
        more information.
    timeslice : scalar; optional
        The length in seconds of each time slice.
    tsoverlap : scalar; optional
        Fraction of a time slice for overlapping. 0.5 is 50% overlap.
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
    This routine calls :func:`scipy.signal.welch` to calculate the
    PSD. It then converts that to mean-square values per band (for
    linear frequency steps, this is just PSD * delta_freq). Loosely,
    the math is:

    .. math::
        \begin{aligned}
        V &= \frac{PSD \cdot \Delta f}{P_{ref}^2} \\
        SPL &= 10 \; \log_{10} ( V ) \\
        OASPL &= 10 \; \log_{10} \left ( \sum V \right )
        \end{aligned}

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import psd
    >>> x = np.random.randn(100000)
    >>> sr = 4000
    >>> f, spl, oaspl = psd.spl(x, sr, sr, timeslice=len(x)/sr)
    >>> # oaspl should be around 170.75 (since variance = 1):
    >>> shouldbe = 10*np.log10(1/(2.9e-9)**2)
    >>> abs(oaspl/shouldbe - 1) < .01
    True
    """
    # compute psd
    F, P = psdmod(x, sr, nperseg=nperseg, window=window,
                  timeslice=timeslice, tsoverlap=tsoverlap,
                  noverlap=int(overlap*nperseg))
    # F, P = signal.welch(x, sr, window=window, nperseg=nperseg,
    #                     noverlap=int(overlap*n))
    if fs != 0:
        _, F, _, P = rescale(P, F, n_oct=fs)
    else:
        P = P*F[1]
    v = P/pref**2
    return F, 10*np.log10(v), 10*np.log10(np.sum(v))


def psd2time(fp, ppc, fstart, fstop, df, winends=None, gettime=False):
    """
    Generate a 'random' time domain signal given a PSD specification.

    Parameters
    ----------
    fp : 2d array_like
        Two column matrix of PSD specification:  [freq, spec]. The
        frequency is in Hz and the specification is in units^2/Hz.
    ppc : scalar
        Points per cycle at highest (`fstop`) frequency; if < 2, it is
        internally reset to 2.
    fstart : scalar
        Starting frequency in Hz
    fstop : scalar
        Stopping frequency in Hz
    df : scalar
        Frequency step to be represented in the time signal. If
        routine gives poor results, try refining `df`. If `df` is
        greater than `fstart`, it is reset internally to `fstart`.
    winends : None or dictionary; optional
        If None, :func:`dsp.windowends` is not called. Otherwise,
        `winends` must be a dictionary of arguments that will be
        passed to :func:`dsp.windowends` (not including `signal`).
    gettime : bool; optional
        If True, a time vector is output.

    Returns
    -------
    sig : 1d ndarray
        The time domain signal with properties set to match input PSD
        spectrum.
    sr : scalar
        The sample rate of the signal.
    time : 1d ndarray; optional
        Time vector for the signal starting at zero with step of
        ``1/sr``: ``time = np.arange(len(sig))/sr``

    Notes
    -----
    This routine uses :func:`interp` to expand `fp` to the desired
    frequencies. That routine assumes a constant db/octave slope for
    all segments and values outside of `fp` frequency range are set to
    zero.

    See also
    --------
    :func:`interp`, :func:`dsp.windowends`

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from pyyeti import psd
        >>> spec = [[20,  .0768],
        ...         [50,  .48],
        ...         [100, .48]]
        >>> sig, sr = psd.psd2time(spec, ppc=10, fstart=35,
        ...                        fstop=70, df=.01,
        ...                        winends=dict(portion=.01))
        >>> sr  # the sample rate should be 70*10 = 700
        700.0

        Compare the resulting psd to the spec from 37 to 68:

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import scipy.signal as signal
        >>> f, p = signal.welch(sig, sr, nperseg=sr)
        >>> pv = np.logical_and(f >= 37, f <= 68)
        >>> fi = f[pv]
        >>> psdi = p[pv]
        >>> speci = psd.interp(spec, fi).flatten()
        >>> abs(speci - psdi).max() < .05
        True
        >>> abs(np.trapz(psdi, fi) - np.trapz(speci, fi)) < .25
        True
        >>> fig = plt.figure('psd2time demo')
        >>> a = plt.subplot(211)
        >>> line = plt.plot(np.arange(len(sig))/sr, sig)
        >>> plt.grid()
        >>> a = plt.subplot(212)
        >>> spec = np.array(spec)
        >>> line = plt.loglog(spec[:, 0], spec[:, 1], label='spec')
        >>> line = plt.loglog(f, p, label='PSD of time signal')
        >>> leg = plt.legend(loc='best')
        >>> x = plt.xlim(20, 100)
        >>> y = plt.ylim(.05, 1)
        >>> plt.grid()
        >>> xticks = np.arange(20, 105, 10)
        >>> x = plt.xticks(xticks, xticks)
        >>> yticks = (.05, .1, .2, .5, 1)
        >>> y = plt.yticks(yticks, yticks)
        >>> v = plt.axvline(35, color='black', linestyle='--')
        >>> v = plt.axvline(70, color='black', linestyle='--')
    """
    if df > fstart:
        df = fstart
    if ppc < 2:
        ppc = 2
    # compute parameters
    # 1 cycle of lowest frequency defines length of signal:
    T = 1/df  # number of seconds for lowest frequency cycle
    N = int(np.ceil(fstop*ppc*T))  # total number of pts
    df = fstop*ppc/N
    # adjust df to make sure fstart is an even multiple of df
    df = fstart/np.floor(fstart/df)
    sr = N*df  # sr = N/T = N/(1/df)
    odd = N & 1

    # define constants
    freq = np.arange(fstart, fstop+df/2, df)

    # generate amp(f) vector
    speclevel = interp(fp, freq).flatten()
    amp = np.sqrt(2*speclevel*df)

    m = (N+1)//2 if odd else N//2 + 1

    # build up amp to include zero frequency to fstart and from fstop
    # to fhighest:
    ntop = int(np.floor((fstart-df/2)/df) + 1)
    nbot = m - ntop - len(amp)
    if nbot > 0:
        amp = np.hstack((np.zeros(ntop), amp, np.zeros(nbot)))
    else:
        amp = np.hstack((np.zeros(ntop), amp))

    # generate F(t)
    phi = np.random.rand(m)*np.pi*2  # random phase angle

    # force these terms to be pure cosine
    phi[0] = 0.
    if not odd:
        phi[m-1] = 0

    # coefficients:
    a = amp*np.cos(phi)
    b = -amp*np.sin(phi)

    # form matrix ready for ifft:
    if odd:
        a2 = a[1:m]/2
        b2 = b[1:m]/2
        r = N*np.hstack((a[0], a2, a2[::-1]))  # real part
        i = N*np.hstack((0, b2, -b2[::-1]))    # imag part
    else:
        a2 = a[1:m-1]/2
        b2 = b[1:m-1]/2
        r = N*np.hstack((a[0], a2, a[m-1], a2[::-1]))  # real part
        i = N*np.hstack((0, b2, 0, -b2[::-1]))         # imag part

    F_time = np.fft.ifft(r+1j*i)
    mxi = abs(F_time.imag).max()
    mxr = abs(F_time.real).max()
    if mxi > 1e-7*mxr:
        # bug in the code if this ever happens
        warn('method failed accuracy test; (max imag)/'
             '(max real) = {}'.format(mxi/mxr),
             RuntimeWarning)

    F_time = F_time.real
    if winends is not None:
        F_time = dsp.windowends(F_time, **winends)
    if gettime:
        return F_time, sr, np.arange(N)/sr
    return F_time, sr


def psdmod(sig, sr, nperseg=None, timeslice=1.0, tsoverlap=0.5,
           getmap=False, **kwargs):
    """
    Modified method for PSD estimation via FFT.

    Parameters
    ----------
    sig : 1d array_like
        Time series of measurement values.
    sr : scalar
        Sample rate.
    nperseg : int, optional
        Length of each segment for the FFT. Defaults to `sr` for 1 Hz
        frequency step in PSD. Note: frequency step in Hz = sr/nperseg.
    timeslice : scalar; optional
        The length in seconds of each time slice.
    tsoverlap : scalar; optional
        Fraction of a time slice for overlapping. 0.5 is 50% overlap.
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
    This routine calls :func:`dsp.waterfall` for handling the
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
        >>> spec = [[20, 1], [50, 1]]
        >>> sig, sr, t = psd.psd2time(spec, ppc=10, fstart=20,
        ...                           fstop=50, df=1/TF,
        ...                           winends=dict(portion=10),
        ...                           gettime=True)
        >>> f, p = signal.welch(sig, sr, nperseg=sr)
        >>> f2, p2 = psd.psdmod(sig, sr, nperseg=sr, timeslice=4,
        ...                     tsoverlap=0.5)
        >>> f3, p3 = psd.psdmod(sig, sr, nperseg=sr)
        >>> spec = np.array(spec).T
        >>> fig = plt.figure('psdmod')
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
        >>> _ = plt.tight_layout()

    """
    if nperseg is None:
        nperseg = sr
    welch_inputs = dict(fs=sr, nperseg=nperseg, **kwargs)
    pmap, t, f = dsp.waterfall(sig, sr, timeslice, tsoverlap,
                               signal.welch, which=1, freq=0,
                               kwargs=welch_inputs)
    p = pmap.max(axis=1)
    if getmap:
        return f, p, pmap, t
    return f, p
