# -*- coding: utf-8 -*-
"""
Tools for calculating the shock response spectrum. Adapted and
enhanced from the Yeti version.
"""

import itertools as it
import multiprocessing as mp
import ctypes
import os
from math import sin, cos, exp, sqrt, pi
from warnings import warn
import numpy as np
import scipy.signal as signal
import scipy.interpolate as interp
from pyyeti import dsp, psd


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


#   SRS types:
#    - absolute acceleration
#    - relative acceleration
#    - relative velocity
#    - pseudo velocity
#    - relative displacement
#
#   Recursive relation:
#    yn = b0 xn  +  b1 xn-1  + b2 xn-2  -  a1 yn-1  -  a2 yn-2
#
#     a is always the same
#
#  Steps to derive coefficients follow (note that the coefficients for
#  wn != 0 were taken from reference 1). The following is here for
#  reference.
#
#   1. Write transfer function H(s) for item of interest (minus sign
#      because right-hand-side is -base_acceleration):
#         absacce = -(c s + k) / (s**2 + c s + k)
#         reldisp = -1 / (s**2 + c s + k)
#         relvelo = -s / (s**2 + c s + k)
#         relacce = -s**2 / (s**2 + c s + k)
#
#   2. For:
#        Impulse Invariant Coefs: compute inverse Laplace of H(s)
#        Step Invariant Coefs:    compute inverse Laplace of H(s)/s
#        Ramp Invariant Coefs:    compute inverse Laplace of H(s)/s**2
#
#   3. Convert t --> n dT
#
#   4. Compute z-transform
#
#   5. Multiply by:
#        Impulse Invariant Coefs: dT
#        Step Invariant Coefs:    (1-z**-1)
#        Ramp Invariant Coefs:    (1-z**-1)**2 / (dT*z**-1)
#
#   6. Simplify and extract coefficients
#
#  Note that for the step and ramp invariants, the transfer function
#  H(s) is integrated once or twice, and operated on like the impulse
#  invariant approach. Then, at the end, the result is modified by
#  dividing by the effect of the integration noting that:
#        1/s    --> 1/(1-z**-1)
#        1/s**2 --> dT*z**-1 / (1-z**-1)**2
#
#  For example, the ramp invariant reldisp coefficients for wn == 0
#  were computed as outlined above:
#   1. H(s) = -1/s**2
#   2. inv-Laplace of H(s)/s**2 = invL[-1/s**4] = -t**3 / 6
#   3. Z[t**3 / 6] = -dT**3 / 6 * z**-1 * (1 + 4*z**-1 + z**-2) /
#                          (1-z**-1)**4
#   4. Mult by (1-z**-1)**2 / (dT*z**-1) and simplify:
#        H(z) = -dT**2 / 6 * (1 + 4*z**-1 + z**-2) /
#                            (1 - 2*z**-1 + z**-2)
#
#    References
#    ----------
#    .. [1] “Mechanical vibration and shock – Signal processing – Part
#           4: Shock-response spectrum analysis”, ISO 18431-4.
#
#    .. [2] Morin, A. and Labbé, P., "Derivation of Recursive Digital
#           Filters by the Step-Invariant and the Ramp-Invariant
#           Transformations", DREV R-4325/84, May 1984, UNCLASSIFIED.
#
#    .. [3] David Smallwood, "An Improved Recursive Formula for
#           Calculating Shock Response Spectra", 51st Shock and
#           Vibration Bulletin (1980).
#
#    .. [4] Kjell Ahlin, "Shock Response Spectrum Calculation - An
#           Improvement of the Smallwood Algorithm",
#           http://www.vibrationdata.com/tutorials/Ahlin_SRS.pdf

HIST_ = None
ICVALS_ = None
SIG_ = None
SRSmax_ = None
WN_ = None


def createSharedArray(dimensions, ctype=ctypes.c_double):
    """
    Creates array in shared memory segment and fills with zeros
    """
    shared_arr = mp.RawArray(ctype, int(np.prod(dimensions)))
    return shared_arr


def copyToSharedArray(arr, ctype=ctypes.c_double):
    """
    Create array in shared memory segment
    """
    shared_arr = mp.RawArray(ctype, arr.size)
    # convert to numpy array (shared memory) and reshape:
    a = np.frombuffer(shared_arr).reshape(arr.shape)
    a[:] = arr
    return shared_arr


def absacce(Q, dT, wn):
    """
    Utility routine used by :func:`srs` to get absolute acceleration
    digital filter coefficients. Returns (b, a) for use in
    :func:`scipy.signal.lfilter`.
    """
    zeta = 1 / 2 / Q
    sqz = sqrt(1 - zeta * zeta)
    wd = wn * sqz
    E = exp(-zeta * wn * dT)
    E2 = E * E
    B = dT * wd
    C = E * cos(B)
    if wn == 0:
        b = np.array([0.0, 0.0, 0.0])
    else:
        S = E * sin(B)
        Sb = S / B
        beta0 = 1 - Sb
        beta1 = 2 * (Sb - C)
        beta2 = E2 - Sb
        b = np.array([beta0, beta1, beta2])
    a = np.array([1, -2 * C, E2])
    return b, a


def relacce(Q, dT, wn):
    """
    Utility routine used by :func:`srs` to get relative acceleration
    digital filter coefficients. Returns (b, a) for use in
    :func:`scipy.signal.lfilter`.
    """
    zeta = 1 / 2 / Q
    sqz = sqrt(1 - zeta * zeta)
    wd = wn * sqz
    E = exp(-zeta * wn * dT)
    E2 = E * E
    B = dT * wd
    C = E * cos(B)
    b = np.array([-1.0, 2.0, -1.0])
    if wn != 0.0:
        b *= (E * sin(B)) / B
    a = np.array([1, -2 * C, E2])
    return b, a


def reldisp(Q, dT, wn):
    """
    Utility routine used by :func:`srs` to get relative displacement
    digital filter coefficients. Returns (b, a) for use in
    :func:`scipy.signal.lfilter`.
    """
    zeta = 1 / 2 / Q
    E = exp(-zeta * wn * dT)
    E2 = E * E
    sqz = sqrt(1 - zeta * zeta)
    wd = wn * sqz
    B = dT * wd
    C = E * cos(B)
    if wn == 0:
        # See notes above for the derivation of these coefficients:
        b = np.array([-1.0, -4.0, -1.0]) * dT ** 2 / 6
    else:
        S = E * sin(B)
        f = dT * wn * wn * wn
        q = (2 * zeta * zeta - 1) / sqz
        beta0 = ((1 - C) / Q - q * S - wn * dT) / f
        beta1 = (2 * C * wn * dT - (1 - E2) / Q + 2 * q * S) / f
        beta2 = (-E2 * (wn * dT + 1 / Q) + C / Q - q * S) / f
        b = np.array([beta0, beta1, beta2])
    a = np.array([1, -2 * C, E2])
    return b, a


def pvelo(Q, dT, wn):
    """
    Utility routine used by :func:`srs` to get pseudo-velocity
    (relative displacement * omega) digital filter coefficients.
    Returns (b, a) for use in :func:`scipy.signal.lfilter`.
    """
    zeta = 1 / 2 / Q
    sqz = sqrt(1 - zeta * zeta)
    wd = wn * sqz
    E = exp(-zeta * wn * dT)
    E2 = E * E
    B = dT * wd
    C = E * cos(B)
    if wn == 0:
        b = np.array([0.0, 0.0, 0.0])
    else:
        S = E * sin(B)
        f = dT * wn * wn
        q = (2 * zeta * zeta - 1) / sqz
        beta0 = ((1 - C) / Q - q * S - wn * dT) / f
        beta1 = (2 * C * wn * dT - (1 - E2) / Q + 2 * q * S) / f
        beta2 = (-E2 * (wn * dT + 1 / Q) + C / Q - q * S) / f
        b = np.array([beta0, beta1, beta2])
    a = np.array([1, -2 * C, E2])
    return b, a


def pacce(Q, dT, wn):
    """
    Utility routine used by :func:`srs` to get pseudo-acceleration
    (relative displacement * omega^2) digital filter coefficients.
    Returns (b, a) for use in :func:`scipy.signal.lfilter`.
    """
    zeta = 1 / 2 / Q
    sqz = sqrt(1 - zeta * zeta)
    wd = wn * sqz
    E = exp(-zeta * wn * dT)
    E2 = E * E
    B = dT * wd
    C = E * cos(B)
    if wn == 0:
        b = np.array([0.0, 0.0, 0.0])
    else:
        S = E * sin(B)
        f = dT * wn
        q = (2 * zeta * zeta - 1) / sqz
        beta0 = ((1 - C) / Q - q * S - wn * dT) / f
        beta1 = (2 * C * wn * dT - (1 - E2) / Q + 2 * q * S) / f
        beta2 = (-E2 * (wn * dT + 1 / Q) + C / Q - q * S) / f
        b = np.array([beta0, beta1, beta2])
    a = np.array([1, -2 * C, E2])
    return b, a


def relvelo(Q, dT, wn):
    """
    Utility routine used by :func:`srs` to get relative velocity
    digital filter coefficients. Returns (b, a) for use in
    :func:`scipy.signal.lfilter`.
    """
    if wn == 0.0:
        b = np.array([-1.0, -1.0]) * dT / 2
        a = np.array([1.0, -1.0])
    else:
        zeta = 1 / 2 / Q
        sqz = sqrt(1 - zeta * zeta)
        wd = wn * sqz
        E = exp(-zeta * wn * dT)
        E2 = E * E
        B = dT * wd
        C = E * cos(B)
        S = E * sin(B)
        Sz = S * zeta / sqz
        f = dT * wn * wn
        beta0 = (C + Sz - 1) / f
        beta1 = (1 - E2 - 2 * Sz) / f
        beta2 = (E2 + Sz - C) / f
        a = np.array([1, -2 * C, E2])
        b = np.array([beta0, beta1, beta2])
    return b, a


def _absmeth(resp):
    return abs(resp).max(axis=0)


def _posmeth(resp):
    return abs(resp.max(axis=0))


def _possmeth(resp):
    return resp.max(axis=0)


def _negmeth(resp):
    return abs(resp.min(axis=0))


def _negsmeth(resp):
    return resp.min(axis=0)


def _rmsmeth(resp):
    return np.sqrt((resp ** 2).mean(axis=0))


def fftroll(sig, sr, ppc, frq):
    """
    Increase sample rate using FFT for :func:`srs`.

    Parameters
    ----------
    sig : ndarray
        The signal(s), time-steps x n.
    sr : scalar
        Sample rate.
    ppc : scalar
        Minimum points per cycle.
    frq : scalar
        Highest frequency of the SDOF system.

    Returns
    -------
    signew : ndarray
        The resampled version of `sig` that meets the `ppc`
        requirement.
    srnew : scalar
        The new sample rate.

    Notes
    -----
    Because of the very poor performance of the SciPy FFT for signals
    of unfortunate lengths, if the number of time-steps is odd, the
    last point is truncated. This should be only temporary and
    hopefully not too harmful.

    See also
    --------
    :func:`scipy.signal.resample`
    """
    N = sig.shape[0]
    if N > 1:
        curppc = sr / frq
        factor = int(np.ceil(ppc / curppc))
        if N & 1:
            sig = signal.resample(sig[:-1], factor * (N - 1), axis=0)
        else:
            sig = signal.resample(sig, factor * N, axis=0)
        sr *= factor
    return sig, sr


def lanroll(sig, sr, ppc, frq):
    """
    Increase sample rate using :func:`pyyeti.dsp.resample` for the SRS
    routine.

    Parameters
    ----------
    sig : ndarray
        The signal(s), time-steps x n.
    sr : scalar
        Sample rate.
    ppc : scalar
        Minimum points per cycle.
    frq : scalar
        Highest frequency of the SDOF system.

    Returns
    -------
    signew : ndarray
        The resampled version of `sig` that meets the ppc requirement.
    srnew : scalar
        The new sample rate.

    Notes
    -----
    The `pts` parameter for the :func:`pyyeti.dsp.resample` is set to
    65.  This was determined from trial and error and comparison to
    the FFT method.
    """
    N = sig.shape[0]
    if N > 1:
        curppc = sr / frq
        factor = int(np.ceil(ppc / curppc))
        sig = dsp.resample(sig, factor, 1, pts=65, axis=0)
        sr *= factor
    return sig, sr


def preroll(sig, sr, ppc, frq):
    """
    Apply pre-filter to account for attenuation due to insufficient
    sample rate.

    Parameters
    ----------
    sig : ndarray
        The signal(s), time-steps x n.
    sr : scalar
        Sample rate.
    ppc : scalar
        Minimum points per cycle.
    frq : scalar
        Highest frequency of the SDOF system.

    Returns
    -------
    signew : ndarray
        The filtered version of `sig`.
    srnew : scalar
        The new sample rate (unchanged from input).

    Notes
    -----
    The approach is scale the time-domain signal such that the
    roll-off is compensated for [#srs5]_.

    References
    ----------
    .. [#srs5] Kjell Ahlin, "Shock Response Spectrum Calculation - An
            Improvement of the Smallwood Algorithm",
            http://www.vibrationdata.com/tutorials/Ahlin_SRS.pdf
    """
    b = np.array([0.8767, 1.7533, 0.8767])
    a = np.array([1, 1.6296, 0.8111, 0.0659])
    sig = signal.filtfilt(b, a, sig, axis=0)
    return sig, sr


def linroll(sig, sr, ppc, frq):
    """
    Increase sample rate using linear interpolation for :func:`srs`.

    Parameters
    ----------
    sig : ndarray
        The signal(s), time-steps x n.
    sr : scalar
        Sample rate.
    ppc : scalar
        Minimum points per cycle.
    frq : scalar
        Highest frequency of the SDOF system.

    Returns
    -------
    signew : ndarray
        The resampled version of `sig` that meets the ppc requirement.
    srnew : scalar
        The new sample rate.

    Notes
    -----
    Note that linear interpolation is not recommended for increasing
    sample rate.
    """
    N = sig.shape[0]
    if N > 1:
        curppc = sr / frq
        factor = int(np.ceil(ppc / curppc))
        told = np.arange(N) / sr
        sr *= factor
        tnew = np.linspace(0.0, told[-1], N * factor - 1)
        ifunc = interp.interp1d(told, sig, axis=0)
        sig = ifunc(tnew)
    return sig, sr


def _mk_par_globals(wn, sig, srsmax, hist):
    global WN_, SIG_, SRSmax_, HIST_
    WN_ = np.frombuffer(wn[0]).reshape(wn[1])
    SIG_ = np.frombuffer(sig[0]).reshape(sig[1])
    SRSmax_ = np.frombuffer(srsmax[0]).reshape(srsmax[1])
    if hist[0] is not None:
        HIST_ = np.frombuffer(hist[0]).reshape(hist[1])


def _dosrs_nohist(args):
    """Utility routine for parallel processing for when
    `getresp` is False"""
    (j, (coeffunc, Q, dT, methfunc, S)) = args
    b, a = coeffunc(Q, dT, WN_[j])
    resphist = signal.lfilter(b, a, SIG_, axis=0)
    SRSmax_[j] = methfunc(resphist[S:])


def _dosrs(args):
    """Utility routine for parallel processing for when
    `getresp` is True"""
    (j, (coeffunc, Q, dT, methfunc, S)) = args
    b, a = coeffunc(Q, dT, WN_[j])
    resphist = signal.lfilter(b, a, SIG_, axis=0)
    SRSmax_[j] = methfunc(resphist[S:])
    HIST_[:, :, j] = resphist[S:]


def _mk_par_globals_ic(wn, sig, icvals, srsmax, hist):
    global WN_, SIG_, ICVALS_, SRSmax_, HIST_
    WN_ = np.frombuffer(wn[0]).reshape(wn[1])
    SIG_ = np.frombuffer(sig[0]).reshape(sig[1])
    ICVALS_ = np.frombuffer(icvals[0]).reshape(icvals[1])
    SRSmax_ = np.frombuffer(srsmax[0]).reshape(srsmax[1])
    if hist[0] is not None:
        HIST_ = np.frombuffer(hist[0]).reshape(hist[1])


def _dosrs_nohist_ic(args):
    """Utility routine for parallel processing for when
    `getresp` is False"""
    (j, (coeffunc, Q, dT, methfunc, S, stype)) = args
    b, a = coeffunc(Q, dT, WN_[j])
    resphist = signal.lfilter(b, a, SIG_, axis=0)
    if stype == "reldisp":
        resphist += ICVALS_ / WN_[j] ** 2
    elif stype == "pvelo":
        resphist += ICVALS_ / WN_[j]
    else:
        # stype == 'pacce' or 'absacce'
        resphist += ICVALS_
    SRSmax_[j] = methfunc(resphist[S:])


def _dosrs_ic(args):
    """Utility routine for parallel processing for when
    `getresp` is True"""
    (j, (coeffunc, Q, dT, methfunc, S, stype)) = args
    b, a = coeffunc(Q, dT, WN_[j])
    resphist = signal.lfilter(b, a, SIG_, axis=0)
    if stype == "reldisp":
        resphist += ICVALS_ / WN_[j] ** 2
    elif stype == "pvelo":
        resphist += ICVALS_ / WN_[j]
    else:
        # stype == 'pacce' or 'absacce'
        resphist += ICVALS_
    SRSmax_[j] = methfunc(resphist[S:])
    HIST_[:, :, j] = resphist[S:]


def _process_inputs(stype, peak, rolloff, time):
    """Utility routine for srs"""
    coefs = {
        "absacce": absacce,
        "relacce": relacce,
        "reldisp": reldisp,
        "relvelo": relvelo,
        "pvelo": pvelo,
        "pacce": pacce,
    }
    meth = {
        "pos": _posmeth,
        "neg": _negmeth,
        "abs": _absmeth,
        "rms": _rmsmeth,
        "poss": _possmeth,
        "negs": _negsmeth,
    }
    roll = {
        "fft": fftroll,
        "lanczos": lanroll,
        "prefilter": preroll,
        "linear": linroll,
        "none": None,
    }
    ptr = {"primary": 0, "total": 1, "residual": 2}
    coeffunc = coefs[stype]
    if isinstance(peak, str):
        methfunc = meth[peak]
    else:
        methfunc = peak
    if isinstance(rolloff, str):
        rollfunc = roll[rolloff]
    else:
        rollfunc = rolloff
    ptr = ptr[time]
    return coeffunc, methfunc, rollfunc, ptr


def _process_parallel(parallel, LF, size, maxcpu, getresp):
    """Utility routine for srs"""
    if parallel not in ["auto", "yes", "no"]:
        raise ValueError("invalid parallel option")
    if parallel != "no":
        ncpu = mp.cpu_count()
    if parallel == "auto":
        if (
            LF > 1
            and size > 50000
            and not getresp
            and ncpu > 1
            and not os.sys.platform.startswith("win")
        ):
            parallel = "yes"
        else:
            parallel = "no"
    if parallel == "yes":
        if maxcpu and ncpu > maxcpu:
            ncpu = maxcpu
        elif ncpu > 4:
            ncpu = (ncpu * 4) // 5
    else:
        ncpu = 1
    return parallel, ncpu


def _process_ic(sig, ic, stype):
    """Utility routine for srs"""
    doic = 0
    icvals = None
    s1 = sig[0]
    if ic == "shift":
        sig = sig - s1
    elif ic == "mshift":
        sig = sig - sig.mean(axis=0)
    elif ic == "steady":
        sig = sig - s1
        if stype == "absacce":
            icvals = s1
            doic = 1
        elif stype == "relacce" or stype == "relvelo":
            pass
        else:
            # 'reldisp', 'pvelo' or 'pacce'
            icvals = -s1
            doic = 1
    return sig, s1, doic, icvals


def _add_one_cycle(sig, freq, sr, H, ic, s1):
    """Utility routine for srs"""
    # append zeros to allow for one cycle of lowest non-zero
    # frequency
    pv = (freq > 0).nonzero()[0]
    if pv.size > 0:
        minf = freq[pv].min()
        nzeros = int(np.ceil(sr / minf))
        z = np.zeros((nzeros, H))
        if ic == "steady":
            sig = np.vstack((sig, z - s1))
        else:
            sig = np.vstack((sig, z))
    return sig, sig.shape[0]


def srs(
    sig,
    sr,
    freq,
    Q,
    ic="zero",
    stype="absacce",
    peak="abs",
    ppc=12,
    rolloff="lanczos",
    eqsine=False,
    time="primary",
    getresp=False,
    parallel="auto",
    maxcpu=14,
):
    r"""
    Shock response spectrum - response of single DOF systems to base
    excitation(s).

    Parameters
    ----------
    sig : 1d or 2d array_like
        Base acceleration signal; vector or matrix where column is a
        signal. If size is 1 x n (2d), that means there are n signals,
        each with length 1 (only initial conditions are
        calculated). For length 1 signal(s), you'll probably want to
        set `ic` to 'steady'.
    sr : scalar or None
        Sample rate. Can be None if signal(s) are length 1 and `ic` is
        not 'zero' (see note below).
    freq : 1d array_like
        Frequency vector in Hz. This defines the single DOF systems
        to use.
    Q : scalar > 0.5
        Dynamic amplification factor :math:`Q = 1/(2\zeta)` where
        :math:`\zeta` is the fraction of critical damping.
    ic : string; optional
        Specifies how to handle the initial conditions:

           ========   ===============================================
            `ic`      Initial conditions
           ========   ===============================================
           'zero'     uses zero initial conditions
           'shift'    shifts each signal to start at zero so there
                      are no step inputs and then uses zero initial
                      conditions
           'mshift'   shifts each signal by its mean, then uses zero
                      initial conditions
           'steady'   uses steady-state initial conditions
           ========   ===============================================

    stype : string; optional
        Specifies the type of response to recover:

           =========    =======================================
            `stype`     Response that :func:`srs` calculates
           =========    =======================================
           'absacce'    absolute acceleration
           'relacce'    relative acceleration
           'reldisp'    relative displacement
           'relvelo'    relative velocity
           'pvelo'      pseudo velocity (reldisp * omega)
           'pacce'      pseudo acceleration (reldisp * omega^2)
           =========    =======================================

    peak : string or function; optional
        If a string, it must be one of these values:

           ======    =============================
           `peak`    Value the :func:`srs` returns
           ======    =============================
           'abs'     absolute maximum
           'pos'     maximum, absolute value
           'neg'     minimum, absolute value
           'poss'    maximum, keeping signs
           'negs'    minimum, keeping signs
           'rms'     root-sum-square
           ======    =============================

        If a function, it must accept the 2d response ndarray with
        shape = (len(time), nsignals) and return a 1d array of "peaks"
        with shape = (nsignals,).
        For example, the 'abs' function is::

            def _absmeth(resp):
                return abs(resp).max(axis=0)

    ppc : scalar; optional
        Specifies the minimum points per cycle for SRS calculations.
        See also `rolloff`.

           ======    ==================================
           `ppc`     Maximum error at highest frequency
           ======    ==================================
               3     81.61%
               4     48.23%
               5     31.58%
              10     8.14% (minimum recommended `ppc`)
              12     5.67%
              15     3.64%
              20     2.05%
              25     1.31%
              50     0.33%
           ======    ==================================

    rolloff : string or function or None; optional
        Indicate which method to use to account for the SRS roll off
        when the minimum `ppc` value is not met. Either 'fft' or
        'lanczos' seem the best.  If a string, it must be one of these
        values:

           ===========    ==========================================
           `rolloff`      Notes
           ===========    ==========================================
           'fft'          Use FFT to upsample data as needed.  See
                          :func:`scipy.signal.resample`.
           'lanczos'      Use Lanczos resampling to upsample as
                          needed. See :func:`pyyeti.dsp.resample`.
           'prefilter'    Apply a high freq. gain filter to account
                          for the SRS roll-off. See
                          :func:`preroll` for more information. This
                          option ignores `ppc` [#srs4]_.
           'linear'       Use linear interpolation to increase the
                          points per cycle (this is not recommended;
                          it's only here as a test case).
           'none'         Don't do anything to enforce the minimum
                          `ppc`. Note error bounds listed above.
            None          Same as 'none'.
           ===========    ==========================================

        If a function, the call signature is:
        ``sig_new, sr_new = rollfunc(sig, sr, ppc, frq)``. `sig` is
        ``time x n``. The last three inputs are scalars. For example,
        the 'fft' function is (trimmed of documentation)::

            def fftroll(sig, sr, ppc, frq):
                N = sig.shape[0]
                if N > 1:
                    curppc = sr/frq
                    factor = int(np.ceil(ppc/curppc))
                    sig = signal.resample(sig, factor*N, axis=0)
                    sr *= factor
                return sig, sr

    eqsine : bool; optional
        If true, resulting history is divided by Q before the peak is
        extracted.
    time : string; optional
        Specifies the time-frame for SRS calculation:

           ==========   ============================================
            `time`      Time-frame for SRS calculation
           ==========   ============================================
           'primary'    During the signal(s) as input.
           'residual'   After the signal(s) (zeros are appended to
                        allow one cycle of the lowest frequency).
           'total'      During and after the signal(s) (primary &
                        residual).
           ==========   ============================================

    getresp : bool; optional
        If True, return the response time histories (see `resp` output
        below).
    parallel : string; optional
        Controls the parallelization of the SRS calculations:

           ==========   ============================================
           `parallel`   Notes
           ==========   ============================================
           'auto'       Routine determines whether or not to run
                        parallel.
           'no'         Do not use parallel processing.
           'yes'        Use parallel processing. Beware, depending
                        on the particular problem, using parallel
                        processing can be slower than not using it
                        (especially if `getresp` is True). On
                        Windows, be sure the :func:`srs` call is
                        contained within:
                        ``if __name__ == "__main__":``
           ==========   ============================================

    maxcpu : integer or None; optional
        Specifies maximum number of CPUs to use. If None, it is
        internally set to 4/5 of available CPUs (as determined from
        :func:`multiprocessing.cpu_count`.

    Returns
    -------
    sh : 1d or 2d ndarray
        The SRS results; ``sh.shape = (len(freq), nsignals)``. If
        `sig` is 1d, `sh` will also be 1d:
        ``sh.shape = (len(freq),)``.
    resp : dictionary; optional
        Only returned if `getresp` is True. Members:

        ======   =====================================================
         key     value
        ======   =====================================================
        't'      time vector for responses
        'sr'     sample rate associated with 't' (>= the input `sr`;
                 depends on inputs `sr`, `ppc`, and `rolloff`)
        'hist'   3-D array; shape = ``(len(t), nsignals, len(freq))``
        ======   =====================================================

    Notes
    -----
    The shock response spectrum is the response of single DOF
    system(s) that are excited by an input base acceleration::

                      _____    ^
                     |     |   |
                     |  M  |  ---  SDOF response (x)
                     |_____|
                      /  |
                    K \ |_| C  ^
                      /  |     |
                    [=======] ---  input base acceleration (sig)

    Derivation of the equation of motion follows. First, let:

    .. math::
        \begin{aligned}
        \ddot{z} &= sig \\
        M &= 1 \\
        K &= \omega_n^2 \\
        C &= 2\zeta\omega_n \\
        \end{aligned}

    Note that :math:`\omega_n=2 \pi f_n` where :math:`f_n` is the
    natural frequency in Hz from the input `freq`. The equation of
    motion is:

    .. math::
        \begin{aligned}
        \ddot{x} &= \sum Forces\; on\; M \\
        &= \omega_n^2(z-x)+2\zeta\omega_n(\dot{z}-\dot{x})
        \end{aligned}

    Define a relative coordinate :math:`u = x - z`. Then:

    .. math::
        \begin{aligned}
        \ddot{x}+2\zeta\omega_n\dot{u}+\omega_n^2 u &= 0 \\
        \ddot{u}+2\zeta\omega_n\dot{u}+\omega_n^2 u &= -\ddot{z}
        \end{aligned}

    In general, that equation is solved for each frequency for each
    signal, giving the relative displacement, velocity and
    acceleration.  The absolute acceleration is then calculated from:

    .. math::
        \ddot{x} = \ddot{u} + \ddot{z}

    By assuming the input signal is linear between time points (ramp
    invariant), these equations can be solved in closed form.
    Reference [#srs1]_ below has done this and conveniently put the
    solution in terms of linear digital filter coefficients.
    Furthermore, coefficients are provided to solve for whichever
    response is requested. More information on coefficient derivation
    can be found in references [#srs2]_, and [#srs3]_.  Reference
    [#srs4]_ is a method for accounting for rolloff.

    The maximum errors listed above are a summation of the bias error
    from the ramp invariant solver and the maximum error that can
    occur when selecting peaks (since peaks occur between solution
    points).  The error equations are (noting that
    :math:`sr/f_n = ppc`):

    .. math::
        \begin{aligned}
        &\text{bias error} = 1 - \left[ \sin \left( \frac{\pi f_n}{sr}
        \right) / \frac{\pi f_n}{sr} \right]^2 \\
        &\text{max peak error} = 1 - \cos \left(
        \frac{\pi f_n}{sr} \right)
        \end{aligned}

    Or:

    .. math::
        \begin{aligned}
        &\text{bias error} = 1 - \left[ \sin \left( \frac{\pi}{ppc}
        \right) / \frac{\pi}{ppc} \right]^2 \\
        &\text{max peak error} = 1 - \cos \left(
        \frac{\pi}{ppc} \right)
        \end{aligned}

    .. note::
        The 'zero' and 'mshift' initial conditions may be handled in a
        slightly different manner than one might think: the zero
        initial displacement and velocity conditions occur one time
        step before `sig` begins (where `sig` is also assumed
        zero). This means one time step is analyzed even if the
        signal(s) are length 1; this is why `sr` cannot be None when
        ``ic = 'zero'`` ('mshift' is okay because it behaves like
        'shift' when there is only one time step).

    .. note::
        In addition to the example shown below, this routine is
        demonstrated in the pyYeti :ref:`tutorial`:
        :doc:`/tutorials/srs`. There is also a link to the source
        Jupyter notebook at the top of the tutorial.

    References
    ----------
    .. [#srs1] “Mechanical vibration and shock – Signal processing –
           Part 4: Shock-response spectrum analysis”, ISO 18431-4.

    .. [#srs2] Morin, A. and Labbé, P., "Derivation of Recursive
           Digital Filters by the Step-Invariant and the Ramp-
           Invariant Transformations", DREV R-4325/84, May 1984,
           UNCLASSIFIED.

    .. [#srs3] David Smallwood, "An Improved Recursive Formula for
           Calculating Shock Response Spectra", 51st Shock and
           Vibration Bulletin (1980).

    .. [#srs4] Kjell Ahlin, "Shock Response Spectrum Calculation - An
           Improvement of the Smallwood Algorithm",
           http://www.vibrationdata.com/tutorials/Ahlin_SRS.pdf

    Raises
    ------
    ValueError
        When `sr` is None but number of time steps is greater than 1
        OR `ic` is set to 'zero'.

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from pyyeti import srs
        >>> import numpy as np
        >>> sr = 1000.
        >>> t = np.arange(0, 5, 1/sr)
        >>> sig = np.sin(2*np.pi*15*t)
        >>> Q = 20
        >>> frq = [10, 15, 20]
        >>> sh = srs.srs(sig, sr, frq, Q)
        >>> print(f'{sh[1]:.1f}')
        20.0

        Compare the upsampling/rolloff methods:

        >>> import matplotlib.pyplot as plt
        >>> _ = plt.figure('Example')
        >>> plt.clf()
        >>> sr = 200
        >>> t = np.arange(0, 5, 1/sr)
        >>> sig = np.sin(2*np.pi*15*t) + 3*np.sin(2*np.pi*85*t)
        >>> Q = 50
        >>> frq = np.linspace(5, 100, 476)
        >>> for meth in ['none', 'linear', 'prefilter',
        ...              'lanczos', 'fft']:
        ...    sh = srs.srs(sig, sr, frq, Q, rolloff=meth)
        ...    _ = plt.plot(frq, sh, label=meth)
        >>> _ = plt.legend(loc='best')
        >>> ttl = '85 Hz peak should approach 150'
        >>> _ = plt.title(ttl)
        >>> _ = plt.grid(True)
    """
    (coeffunc, methfunc, rollfunc, ptr) = _process_inputs(stype, peak, rolloff, time)
    freq = np.atleast_1d(freq)
    wn = 2 * pi * freq
    LF = len(freq)
    sig = np.atleast_1d(sig)
    if sig.ndim == 1:
        oneD = True
        sig = sig.reshape(-1, 1)
    else:
        oneD = False
    N = sig.shape[0]  # number of time steps
    H = sig.shape[1]  # number of histories

    if sr is None:
        if N > 1 or ic == "zero":
            raise ValueError(
                "`sr` can only be None if signal(s) are "
                "length 1 AND `ic` is not 'zero'"
            )
        sr = 1.0  # can be anything, just needed for calculations

    parallel, ncpu = _process_parallel(parallel, LF, N * H, maxcpu, getresp)

    if parallel == "yes":
        # global shared vars will be:
        # SRSmax_, WN_, HIST_, SIG_, ICVALS_
        SRSmax = (createSharedArray((LF, H)), (LF, H))
        WN = (copyToSharedArray(wn), wn.shape)
        HIST = (None, None)
    else:
        SRSmax = np.empty((LF, H))

    sig, s1, doic, icvals = _process_ic(sig, ic, stype)

    mf = np.max(freq)
    if rolloff == "prefilter":
        sig, sr = rollfunc(sig, sr, ppc, mf)
        rollfunc = None  # rolloff = 'none'

    if rollfunc and mf != 0 and sr / mf < ppc:
        sig, sr = rollfunc(sig, sr, ppc, mf)
        N = sig.shape[0]
    rollfunc = None

    M = N
    if ptr:
        sig, N = _add_one_cycle(sig, freq, sr, H, ic, s1)

    if getresp:
        resp = {}
        resp["sr"] = sr
        # hist is:  len(time) x nsignals x len(freq)
        if ptr == 2:
            # residual
            resp["t"] = np.arange(M, N) / sr
            if parallel == "yes":
                HIST = (createSharedArray((N - M, H, LF)), (N - M, H, LF))
            else:
                resp["hist"] = np.empty((N - M, H, LF))
        else:
            resp["t"] = np.arange(N) / sr
            if parallel == "yes":
                HIST = (createSharedArray((N, H, LF)), (N, H, LF))
            else:
                resp["hist"] = np.empty((N, H, LF))

    # S is starting time for calcs; only non-zero if residual only:
    S = M if ptr == 2 else 0

    if doic:
        if parallel == "yes":
            SIG = (copyToSharedArray(sig), sig.shape)
            ICVALS = (copyToSharedArray(icvals), icvals.shape)
            args = (coeffunc, Q, 1 / sr, methfunc, S, stype)
            gvars = (WN, SIG, ICVALS, SRSmax, HIST)
            func = _dosrs_ic if getresp else _dosrs_nohist_ic
            with mp.Pool(
                processes=ncpu, initializer=_mk_par_globals_ic, initargs=gvars
            ) as pool:
                for _ in pool.imap_unordered(func, zip(range(LF), it.repeat(args, LF))):
                    pass
            SRSmax = np.frombuffer(SRSmax[0]).reshape(SRSmax[1])
            if getresp:
                HIST = np.frombuffer(HIST[0]).reshape(HIST[1])
                resp["hist"] = HIST
        else:
            dT = 1 / sr
            for j in range(LF):
                b, a = coeffunc(Q, dT, wn[j])
                resphist = signal.lfilter(b, a, sig, axis=0)
                if stype == "reldisp":
                    resphist += icvals / wn[j] ** 2
                elif stype == "pvelo":
                    resphist += icvals / wn[j]
                else:
                    # stype == 'pacce' or 'absacce'
                    resphist += icvals
                SRSmax[j] = methfunc(resphist[S:])
                if getresp:
                    resp["hist"][:, :, j] = resphist[S:]
    else:
        # no initial conditions to worry about:
        if parallel == "yes":
            SIG = (copyToSharedArray(sig), sig.shape)
            args = (coeffunc, Q, 1 / sr, methfunc, S)
            gvars = (WN, SIG, SRSmax, HIST)
            func = _dosrs if getresp else _dosrs_nohist
            with mp.Pool(
                processes=ncpu, initializer=_mk_par_globals, initargs=gvars
            ) as pool:
                for _ in pool.imap_unordered(func, zip(range(LF), it.repeat(args, LF))):
                    pass
            SRSmax = np.frombuffer(SRSmax[0]).reshape(SRSmax[1])
            if getresp:
                HIST = np.frombuffer(HIST[0]).reshape(HIST[1])
                resp["hist"] = HIST
        else:
            dT = 1 / sr
            for j in range(LF):
                b, a = coeffunc(Q, dT, wn[j])
                resphist = signal.lfilter(b, a, sig, axis=0)
                SRSmax[j] = methfunc(resphist[S:])
                if getresp:
                    resp["hist"][:, :, j] = resphist[S:]
    if oneD:
        SRSmax = SRSmax.ravel()
    if getresp:
        if eqsine:
            SRSmax /= Q
            resp["hist"] /= Q
        return SRSmax, resp
    if eqsine:
        SRSmax /= Q
    return SRSmax


def vrs(spec, freq, Q, linear, Fn=None, getmiles=False, getresp=False):
    r"""
    Vibration response specturm - RMS response of single DOF systems
    to base PSD(s).

    Parameters
    ----------
    spec : 2d ndarray or 2-element tuple/list
        If ndarray, its columns are ``[Freq, PSD1, PSD2, ... PSDn]``.
        Otherwise, it must be a 2-element tuple or list, eg:
        ``(Freq, PSD)`` where PSD is: ``[PSD1, PSD2, ... PSDn]``. In
        the second usage, PSD can be 1d in which case the outputs will
        also be 1d; in the first usage, the outputs will be 2d. The
        frequency vector must be monotonically increasing.
    freq : 1d array_like
        Vector of frequencies to define the integration step; see
        usage note 2 below.
    Q : scalar
        Dynamic amplification factor :math:`Q = 1/(2\zeta)` where
        :math:`\zeta` is the fraction of critical damping.
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

        The :func:`pyyeti.psd.interp` routine is called to perform the
        interpolation.
    Fn : 1d array or None
        Defines the frequency(s) at which to compute the response. If
        None, ``Fn = freq``.
    getmiles : bool
        If True, return the Miles' equation estimate for the RMS
        response. Miles' equation assumes the PSD is flat and extends
        forever in both directions from frequency. The estimate is
        typically good if the PSD is flat for at least 2 octaves in
        both directions and the damping is less that 10%.
    getresp : bool
        If True, return the PSD response curves at the frequency(s) in
        `Fn`. Note: internally, this will also set `getmiles` to
        True.

    Returns
    -------
    Zvrs : 1d or 2d ndarray
        The SDOF RMS acceleration response spectrum. For example, the
        response spectrum is in Grms if the spec is given in G^2/Hz.
        ``Zvrs.shape = (len(Fn), n)``, where n is the number of input
        specifications. Will be 1d only if the PSD input in `spec` was
        1d (in the second usage); in that case,
        ``Zvrs.shape = (len(Fn),)``.
    Zmiles : 1d or 2d ndarray; optional
        The Miles' estimate of `Zvrs`. Only returned if `getmiles` or
        `getresp` is True. ``Zmiles(f) = sqrt(pi/2*Fn*Q*psd(f))``.
        Shape will be the same as `Zvrs`.
    PSDresp : dictionary; optional
        Only returned if `getresp` is True. Members:

        ======   =====================================================
         key     value
        ======   =====================================================
        'f'      frequency vector for responses; same as `freq`
        'psd'    3-D array; shape = ``(len(Fn), n, len(f))``, where
                 `n` is the number of input specifications
        ======   =====================================================

    Notes
    -----
    VRS [#srs6]_ computes the acceleration RMS (root-mean-square)
    response of a spectrum of single DOF systems that are excited by
    an input base acceleration PSD(s)::

                      _____    ^
                     |     |   |
                     |  M  |  ---  SDOF acceleration PSD response
                     |_____|
                      /  |
                    K \ |_| C  ^
                      /  |     |
                    [=======] ---  input base acceleration PSD

    The response of each system is computed independently by
    integration across the entire frequency range as specified in
    `freq`.

    The equation for the VRS is:

    .. math::
        Z_{vrs}(f_n) = \sqrt { \sum\limits_{i} {
           \frac{1 + (p_i / Q)^2}
                {(1 - p_i^2)^2 + (p_i / Q)^2}}
           \cdot PSD(freq_i) \cdot \Delta freq_i}\;\;;\;
           p_i = \frac{freq_i}{f_n}

    Miles' equation is:

    .. math::
        Z_{miles}(f_n) = \sqrt{\frac{\pi}{2} \cdot f_n \cdot Q
        \cdot PSD(f_n)}

    Important usage notes:

    1. Responses calculated at the extreme frequency points may not be
       accurate because the transfer function is cut off during
       integration. As a rule of thumb, accuracy can be expected an
       octave away from the end points.

    2. The integration is not accurate until ``delta_f`` as computed
       from `freq` is less than ``f/Q``, i.e. the response at
       frequency ``f`` is not accurate unless ``delta_f < f/Q``, where
       ``Q=1/2/zeta``. The integration should be conservative if this
       condition is not met and ``delta_f`` is not unreasonably large.

    3. Applying a flat PSD spectrum can be used to determine if the
       ``delta_f`` is good since you can compare to the Miles'
       equation results.

    To estimate a peak from the RMS, consider using :math:`\sqrt{2
    \ln(f \cdot T_0)}` for the peak factor instead of the common 3
    (for 3-rms or 3-sigma). This peak factor is derived in the Notes
    section in :func:`pyyeti.fdepsd.fdepsd`.

    See also
    --------
    :func:`srs`, :func:`srs_frf`, :func:`pyyeti.psd.interp`

    References
    ----------
    .. [#srs6] Tom Irvine, "An Introduction to the Vibration Response
           Spectrum - Revision D",
           http://www.vibrationdata.com/tutorials2/vrs.pdf

    Examples
    --------
    Compute response spectra for example shown in reference. The
    results should be::

        vrs                = [6.38, 11.09, 16.06]
        miles              = [6.47, 11.21, 15.04]
        response PSD peaks = [2.69,  4.04,  1.47]

    >>> import numpy as np
    >>> from pyyeti import srs
    >>> spec = np.array([[20, .0053],
    ...                  [150, .04],
    ...                  [600, .04],
    ...                  [2000, .0036]])
    >>> frq = np.arange(20, 2000, 2.)
    >>> Q = 10
    >>> fn = [100, 200, 1000]
    >>> v, m, resp = srs.vrs((spec[:, 0], spec[:, 1]), frq, Q,
    ...                      linear=False, Fn=fn, getresp=True)
    >>> np.set_printoptions(precision=2)
    >>> v
    array([  6.38,  11.09,  16.06])
    >>> m
    array([  6.47,  11.21,  15.04])
    >>> resp['psd'][:, 0].max(axis=1)
    array([ 2.69,  4.04,  1.47])
    """
    if Q <= 0.5:
        raise ValueError("Q must be > 0.5 since VRS assumes underdamped equations.")

    Freq, PSD, npsds = psd.proc_psd_spec(spec)
    freq = np.atleast_1d(freq)

    if Fn is None:
        Fn = freq
        do_interp = False
    else:
        Fn = np.atleast_1d(Fn)
        freq = np.unique(np.hstack((freq, Fn)))
        do_interp = True
    rf = len(freq)

    # expand PSD:
    psdfull = psd.interp((Freq, PSD), freq, linear)
    if PSD.ndim == 1:
        psdfull = psdfull[:, None]

    # check for adequate delta_freq for integration:
    df = np.empty(rf)
    df[:-1] = freq[1:] - freq[:-1]
    df[-1] = freq[-1] - freq[-2]
    if do_interp:
        pv = np.where((freq >= Fn.min()) & (freq <= Fn.max()))[0]
        bad_df = df[pv] > freq[pv] / Q
    else:
        bad_df = df > freq / Q

    if bad_df.any():
        warn(
            "Integration frequency vector may produce inaccurate results;"
            " refine the step. See the documentation for more information.",
            RuntimeWarning,
        )

    # Create delta_f for area calculation:
    df = np.empty(rf)
    df[1:-1] = (freq[2:] - freq[:-2]) / 2
    df[0] = freq[1] - freq[0]
    df[-1] = freq[-1] - freq[-2]

    # Compute Miles' equation
    if getresp or getmiles:
        if do_interp:
            ifunc = interp.interp1d(
                freq,
                psdfull,
                axis=0,
                bounds_error=False,
                fill_value=0,
                assume_sorted=True,
            )
            psdf2 = ifunc(Fn)
            z_miles = np.sqrt((np.pi / 2 * Fn * Q) * psdf2.T).T
        else:
            z_miles = np.sqrt((np.pi / 2 * freq * Q) * psdfull.T).T
        if PSD.ndim == 1:
            z_miles = z_miles.ravel()

    # Compute VRS at each frequency
    z_vrs = np.empty((len(Fn), npsds))
    zeta = 1 / 2 / Q
    if getresp:
        psd_vrs = np.empty((len(Fn), npsds, len(freq)))
        for i, fn in enumerate(Fn):
            p = freq / fn
            p2z2 = (2 * zeta * p) ** 2
            t = ((1 + p2z2) / ((1 - p ** 2) ** 2 + p2z2)) * psdfull.T
            psd_vrs[i] = t  # npsds x len(freq)
            z_vrs[i] = np.sqrt(np.sum(df * t, axis=1))
        resp = {}
        resp["f"] = freq
        resp["psd"] = psd_vrs
        if PSD.ndim == 1:
            z_vrs = z_vrs.ravel()
        return z_vrs, z_miles, resp

    for i, fn in enumerate(Fn):
        p = freq / fn
        p2z2 = (2 * zeta * p) ** 2
        t = ((1 + p2z2) / ((1 - p ** 2) ** 2 + p2z2) * df) * psdfull.T
        z_vrs[i] = np.sqrt(np.sum(t, axis=1))

    if PSD.ndim == 1:
        z_vrs = z_vrs.ravel()
    if getmiles:
        return z_vrs, z_miles
    return z_vrs


def srs_frf(frf, frf_frq, srs_frq, Q, getresp=False, return_srs_frq=None):
    r"""
    Compute SRS from frequency response functions.

    Parameters
    ----------
    frf : 2d array_like
        Columns of FRF data defining base motion in frequency domain:
        [FRF1, FRF2, ... FRFn]. The FRFs can be complex; if so, this
        routine uses the absolute value of each column before
        interpolating to a frequency vector that includes system
        frequencies (note that using the absolute value gives
        equivalent results).  Number of rows must equal
        ``len(frf_frq)``. If it is 1d, it is reshaped into a single
        column: [FRF1].
    frf_frq : 1d array_like
        Frequency vector in Hz for the FRF data.
    srs_frq : 1d array_like or None
        Frequency vector in Hz for the SRS. These are the SDOF
        frequencies for which to compute responses. If input as None,
        `srs_frq` is computed from `frf_frq` such that the maximum
        theoretical response for the input at the FRF frequency is
        obtained. In this case, the computed SDOF frequency will be a
        little higher than the corresponding FRF frequency. How much
        higher depends on the damping: lower damping (higher Q) means
        the SDOF frequency will be closer to the FRF frequency. The
        equations are derived and discussed below.
    Q : scalar
        Dynamic amplification factor :math:`Q = 1/(2\zeta)` where
        :math:`\zeta` is the critical damping ratio.
    getresp : bool; optional
        If True, return the complex response frfs (see `resp` output
        below).
    return_srs_frq : bool or None; optional
        Determines whether or not to return `srs_frq`:

        =======   ====================================================
        Setting   Description
        =======   ====================================================
         None     `return_srs_frq` will be internally reset to True if
                  and only if the input `srs_frq` is None; otherwise,
                  it is set to False
         True     Return `srs_frq` (default if `srs_frq` is None)
         False    Do not return `srs_frq` (default if `srs_frq` is not
                  None)
        =======   ====================================================

    Returns
    -------
    sh : 2d ndarray
        The SRS results: [SRS1, SRS2, .... SRSn];
        ``sh.shape = (len(srs_frq), n)`` where n is the number of
        FRFs.
    srs_frq : 1d ndarray; optional
        The SRS frequency vector that goes with `sh`. See
        `return_srs_frq` description avove.
    resp : dictionary; optional
        Only returned if `getresp` is True. Members:

        =========   ==================================================
         key        value
        =========   ==================================================
        'freq'      frequency vector for responses; this is a superset
                    of `frf_frq` and `srs_frf` with near-duplicates
                    removed
        'frfs'      3-D array; shape = ``(len(freq), n, len(srs_frq))``
        'srs_frq'   SRS frequency vector (here for convenience)
        =========   ==================================================

    Notes
    -----
    The shock response spectrum is the response of single DOF
    system(s) that are excited by an input base acceleration FRF::

                      _____    ^
                     |     |   |
                     |  M  |  ---  SDOF response (x)
                     |_____|
                      /  |
                    K \ |_| C  ^
                      /  |     |
                    [=======] ---  input base acceleration (frf)

    The response of each system is computed independently by
    integration across the entire frequency range as specified in
    ``resp["freq"]`` above.

    Derivation of the equation of motion follows. First, let:

    .. math::
        \begin{aligned}
        \ddot{z} &= sig \\
        M &= 1 \\
        K &= \omega_n^2 \\
        C &= 2\zeta\omega_n \\
        \end{aligned}

    Note that :math:`\omega_n=2 \pi f_n` where :math:`f_n` is the
    natural frequency in Hz from the input `srs_frq`. The equation of
    motion is:

    .. math::
        \begin{aligned}
        \ddot{x} &= \sum Forces\; on\; M \\
        &= \omega_n^2(z-x)+2\zeta\omega_n(\dot{z}-\dot{x})
        \end{aligned}

    Define a relative coordinate :math:`u = x - z`. Then:

    .. math::
        \begin{aligned}
        \ddot{x}+2\zeta\omega_n\dot{u}+\omega_n^2 u &= 0 \\
        \ddot{u}+2\zeta\omega_n\dot{u}+\omega_n^2 u &= -\ddot{z}
        \end{aligned}

    Using the Fourier transform :math:`\mathcal{F}[x(t)] = X(\Omega)`:

    .. math::
        \begin{aligned}
        (-\Omega^2+2\zeta\omega_n\Omega j + \omega_n^2) U(\Omega) &=
        -Z_{acce}(\Omega) \\
        U(\Omega) &= -Z_{acce}(\Omega) /
        (-\Omega^2+2\zeta\omega_n\Omega j + \omega_n^2) \\
        U_{acce}(\Omega) &= \Omega^2 Z_{acce}(\Omega) /
        (-\Omega^2+2\zeta\omega_n\Omega j + \omega_n^2)
        \end{aligned}

    Then:

    .. math::
        \begin{aligned}
        X_{acce}(\Omega) &= U_{acce}(\Omega) + Z_{acce}(\Omega) \\
        &= \left ( \frac{\Omega^2}
        {\omega_n^2-\Omega^2+2\zeta\omega_n\Omega j} + 1 \right)
        Z_{acce}(\Omega)
        \end{aligned}

    The return value `sh` contains the peak of the absolute value of
    :math:`X_{acce}(\Omega)` for all frequencies analyzed.

    **At what frequency is the amplitude of the transfer function
    maximized?**

    Letting :math:`p = \Omega / \omega_n` and collecting terms, the
    transfer function is:

    .. math::
        \begin{aligned}
        H(p) = \frac{X_{acce}(p)}{Z_{acce}(p)}
        &= \left ( \frac{p^2}{1-p^2+2\zeta p j} + 1 \right) \\
        H(p) &= \left ( \frac{1 + 2\zeta p j}{1-p^2+2\zeta p j}\right)
        \end{aligned}

    If we just want the amplitude of the output over the input,
    multiply by the complex conjugate:

    .. math::
        |H(p)|^2 = \left ( \frac{1 + (2\zeta p)^2}{(1-p^2)^2
        +(2\zeta p)^2} \right)

    For small damping values, the peak of :math:`|H(p)|` occurs near
    :math:`p = 1` (this is easy to see in the approximate expression
    below for the maximizing value of :math:`p` (:math:`p_{peak}`)):

    .. math::
        |H(p = 1)| = \sqrt{\frac{1}{(2\zeta)^2} + 1} = \sqrt{Q^2 + 1}
        \approx Q \text{ (for higher Q)}

    To find precisely where :math:`|H(p)|^2` is maximized, set the
    derivative with respect to :math:`p^2` equal to zero and solve for
    :math:`p^2` and then take the square root (:obj:`sympy` can be
    helpful here). The result is:

    .. math::
        p_{peak} = \frac{\sqrt{ \sqrt{1 + 8 \zeta^2} - 1}}{2 \zeta}
                 = Q \sqrt{ \sqrt{1 + 2 / Q^2} - 1}

    A Taylor series expansion of :math:`p^2_{peak}` was done to get
    the following very good approximate expression for
    :math:`p_{peak}`:

    .. math::
        p_{peak} \approx \sqrt{1 - 2 \zeta^2}
                 = \sqrt{1 - \frac{1}{2 Q^2}}

    This routine uses the exact expression above for :math:`p_{peak}`
    to add the maximizing analysis frequency for each SDOF:

    .. math::
        \Omega_{peak} = p_{peak} \cdot \omega_n

    It is important to note however that this will not necessarily
    maximize :math:`X_{acce}(\Omega)` because that also depends on the
    input :math:`Z_{acce}(\Omega)`. If :math:`Z_{acce}(\Omega)` is
    flat, then the above expression will maximize
    :math:`X_{acce}(\Omega)`. In the general case where
    :math:`Z_{acce}(\Omega)` has a peak at a some frequency, to get
    the theoretical maximum SDOF response, the frequency of the SDOF
    (which is the `srs_frq` input to this routine) would need to be
    computed from:

    .. math::
        \omega_n = \frac{\Omega}{p_{peak}}

    Or, in terms of the input variable names:

    .. math::
        srs{\_}frq = \frac{frf{\_}frq}{p_{peak}}

    In this routine, to get the theoretical maximum SDOF response from
    a given :math:`Z_{acce}(\Omega)`, either compute `srs_frq` from
    the above equation before calling this routine or, alternatively,
    input `srs_frq` as ``None`` and let the routine internally perform
    that calculation.

    Examples
    --------
    Make up simple problem to demonstrate a couple of the equations
    derived above.

    >>> import numpy as np
    >>> from pyyeti import srs
    >>> pk_input = 3.0
    >>> pk_frq = 15.0
    >>> frf = np.array([pk_input / 3, pk_input, pk_input / 3])
    >>> frf_frq = np.array([pk_frq - 5, pk_frq, pk_frq + 5])
    >>> srs_frq = np.array([pk_frq])
    >>> Q = 20
    >>> sh = srs.srs_frf(frf, frf_frq, srs_frq, Q)

    Because the input has a peak at 15 hz and that is also the SDOF
    frequency, the peak response will occur at :math:`p = 1` instead of
    :math:`p_{peak}` as derived above. From the equations above, the absolute
    peak response should be: :math:`pk{\_}input \cdot \sqrt{Q^2 + 1}`:

    >>> pk_should_be = pk_input * np.sqrt(Q ** 2 + 1)
    >>> pk_should_be                          # doctest: +SKIP
    60.074953183502359
    >>> sh = srs.srs_frf(frf, frf_frq, srs_frq, Q)
    >>> sh                                    # doctest: +SKIP
    array([[ 60.07495318]])
    >>> abs(pk_should_be - sh[0, 0]) < 1e-10
    True

    If we let the routine compute the SDOF frequencies, we can get the
    theoretical maximum as derived above. Here, we'll check both the
    peak response value and the frequency of the maximizing SDOF:

    >>> p_peak = Q * np.sqrt((np.sqrt(1 + 2 / Q ** 2) - 1))
    >>> frq_should_be = pk_frq / p_peak
    >>> num = 1 + (p_peak / Q) ** 2
    >>> den = (1 - p_peak ** 2) ** 2 + num - 1
    >>> pk_should_be = pk_input * np.sqrt(num / den)
    >>> sh, frq = srs.srs_frf(frf, frf_frq, None, Q)
    >>> i = sh[:, 0].argmax()
    >>> pk_should_be                          # doctest: +SKIP
    60.093641865335883
    >>> sh                                    # doctest: +SKIP
    array([[ 20.03121396],
           [ 60.09364187],
           [ 20.03121396]])
    >>> abs(pk_should_be - sh[i, 0]) < 1e-10
    True
    >>> frq_should_be                         # doctest: +SKIP
    15.009360389892359
    >>> frq                                   # doctest: +SKIP
    array([ 10.00624026,  15.00936039,  20.01248052])
    >>> abs(frq_should_be - frq[i]) < 1e-10
    True

    For the next example, the "equivalent sine" (SRS/Q) will be
    computed for a sawtooth input for several Q values.

    .. note::

        Since this is a pure sinusoidal analysis, one might think that
        the "equivalent sine" result should just be equal to the
        original input. However, because of excitation from nearby
        frequencies (as discussed in detail below), this will not be
        the case in general. Even so, depending on how the input was
        created, it may be valid to consider the input as the
        equivalent sine and not run this routine at all. For example,
        if the input is an envelope over SDOF responses from
        time-domain signals, excitation from nearby frequencies may
        already be accounted for. Running this routine in that
        scenario may just add unneeded conservatism. Think of this
        routine as a simulation of a sine test of a bunch SDOF
        systems.

    .. note::
        It is noted that dividing by :math:`\sqrt{Q^2 + 1}` would make
        it more "equivalent" since that's the gain of the transfer
        function at :math:`p = 1` (see above). However, dividing by
        :math:`Q` is common, and that's what will be done for the
        example below.

    The top plot shows the input :math:`Z_{acce}(\Omega)`.

    The second plot shows the equivalent sine curves for different
    damping values. Each curve has ``len(srs_frq)`` points on it, each
    point being the maximum value of :math:`|X_{acce}(\Omega)|/Q` for
    the corresponding SDOF system.

    The third plot shows the actual FRF response curves (divided by Q)
    for the 44.5 Hz SDOF system with the different damping values. The
    peak of each of these curves, at whatever frequency it occurs at,
    forms the corresponding value on the equivalent sine curve @ 44.5
    Hz.

    The fourth plot shows the transfer functions divided by Q for
    reference.

    Observations:

       1. Equivalent sine curves with lower damping (higher Q), will
          tend to follow the input more closely. This is because the
          high gain of the transfer function near the SDOF natural
          frequency causes the response to hit its maximum peak near
          its natural frequency even if the peak of the input occurs
          at a different frequency. In that scenario, the division by
          Q (nearly) cancels out the gain of the transfer function,
          bringing the response back down to the input level. For
          example, for Q = 50, the FRF peak of the 44.5 Hz SDOF system
          occurs closest to the natural frequency even though the peak
          of the input does not occur there (the nearest input peak is
          at 45.0 Hz). So, the FRF peak is approximately ``Q *
          input``.

       2. Conversely, equivalent sine curves with higher damping
          (lower Q), will tend to smooth over the valleys of the
          input. For these higher damped SDOF systems, the lower gain
          of the transfer function becomes less important, and the
          peak response will occur closer to a peak of the input, even
          if that doesn't match the natural frequency. For example,
          for Q = 5, the FRF peak of the 44.5 Hz SDOF system occurs at
          45.0 Hz because that's where the nearest peak of the input
          is. In that scenario, the division by Q will not bring the
          curve back down to the input level since the FRF peak is the
          product of off-peak transfer function (:math:`\neq Q`)
          multiplied by a higher input at some other frequency. (Note:
          dividing by Q gets these curves closer to the input where
          the input has peaks, but still not as well as the lower
          damped equivalent sine curves ... dividing by
          :math:`\sqrt{Q^2 + 1}` would fix that.)

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyyeti import srs
        >>>
        >>> # saw tooth input:
        >>> sdof = 44.5
        >>> n = 10
        >>> frf = np.ones(n)
        >>> frf[::2] = 0.5
        >>> frf_frq = np.arange(n) * 1.0 + 40.0
        >>> srs_cutoff = frf_frq[-1]
        >>> fstep = 0.01
        >>>
        >>> # add sdof frequency to frf_frq:
        >>> new_frf_frq = np.sort(np.r_[frf_frq, sdof])
        >>> new_frf = np.interp(new_frf_frq, frf_frq, frf)
        >>> frf, frf_frq = new_frf, new_frf_frq
        >>>
        >>> srs_frq = np.arange(frf_frq[0], srs_cutoff, fstep)
        >>>
        >>> fig, ax = plt.subplots(
        ...     4,
        ...     1,
        ...     num="Example",
        ...     clear=True,
        ...     figsize=(9, 12),
        ... )
        >>> _ = ax[0].plot(frf_frq, frf)
        >>>
        >>> for Q in (5, 10, 20, 30, 40, 50):
        ...     sh, resp = srs.srs_frf(
        ...         frf, frf_frq, srs_frq, Q, getresp=True
        ...     )
        ...     _ = ax[1].plot(srs_frq, sh / Q, label=f"Q = {Q}")
        ...     _ = ax[1].legend()
        ...
        ...     i = np.searchsorted(srs_frq, sdof)
        ...     _ = ax[2].plot(
        ...         resp["freq"],
        ...         abs(resp["frfs"][:, 0, i]) / Q,
        ...         label=f"Q = {Q}",
        ...     )
        ...     _ = ax[2].legend()
        ...
        ...     # plot the transfer function (by using unity input)
        ...     n = len(frf)
        ...     _, resp_unity = srs.srs_frf(
        ...         np.ones(n), frf_frq, srs_frq, Q, getresp=True
        ...     )
        ...     _ = ax[3].plot(
        ...         resp_unity["freq"],
        ...         abs(resp_unity["frfs"][:, 0, i]) / Q,
        ...         label=f"Q = {Q}",
        ...     )
        ...     _ = ax[3].legend()
        >>>
        >>> _ = ax[0].set_title("Base Input")
        >>> _ = ax[0].set_ylabel("Acceleration (G)")
        >>> _ = ax[0].set_xlabel(r"$\Omega$ Frequency (Hz)")
        >>> _ = ax[1].set_title("Eq-Sine (Abs-Acce/Q)")
        >>> _ = ax[1].set_ylabel("Abs-Acce Eq-Sine (G)")
        >>> _ = ax[1].set_xlabel(r"$\omega_n$ Frequency (Hz)")
        >>> _ = ax[2].set_title(
        ...         f"(Abs-Acce |FRF| Response of {sdof} Hz SDOF)/Q"
        ...     )
        >>> _ = ax[2].set_ylabel("Abs-Acce |FRF| / Q (G)")
        >>> _ = ax[2].set_xlabel(r"$\Omega$ Frequency (Hz)")
        >>> _ = ax[3].set_title(
        ...         r"Transfer function $|H(\Omega)|/Q$ of "
        ...         f"{sdof} Hz SDOF"
        ...     )
        >>> _ = ax[3].set_ylabel(r"$|H(\Omega)|/Q$")
        >>> _ = ax[3].set_xlabel(r"$\Omega$ Frequency (Hz)")
        >>> for axis in ax:
        ...    _ = axis.set_xlim(39.5, 49.5)
        >>> fig.tight_layout()

    In the previous example, we saw that the equivalent sine of the
    input did not give us the original input back, but it was closer
    for lower damping. Fundamentally, the reason is because only the
    gain at :math:`p = 1` is considered and, furthermore, the division
    by "Q" is an approximation of a more correct division by
    :math:`\sqrt{Q^2 + 1}`. An equivalent sine *could* get back to the
    original input if the original input was flat and the division was
    done using the theoretical maximum gain: :math:`|H(p=p_{peak})|`.
    (That assumes that the routine computing the SDOF responses
    catches the maximum gain frequency.)

    So, how "equivalent" is the equivalent sine for the example shown
    above? In this final example, we'll compute the equivalent sine of
    an equivalent sine for Q = 10. We'll also improve the process a
    bit by dividing by :math:`\sqrt{Q^2 + 1}` instead of Q. That will
    ensure that we get the values correct at the peak input
    frequencies:

    .. plot::
        :context: close-figs

        >>> fig, ax = plt.subplots(
        ...     1,
        ...     1,
        ...     num="Example 2",
        ...     clear=True,
        ...     figsize=(9, 5),
        ... )
        >>> _ = ax.plot(frf_frq, frf, label="Original Input")
        >>>
        >>> Q = 10
        >>> factor = np.sqrt(Q ** 2 + 1)
        >>>
        >>> eqsine = srs.srs_frf(frf, frf_frq, srs_frq, Q) / factor
        >>> lbl = f"Eq-Sine0; Eq-Sine of Input, Q = {Q}"
        >>> _ = ax.plot(srs_frq, eqsine, label=lbl)
        >>>
        >>> for level in range(1):
        ...     eqsine = srs.srs_frf(
        ...         eqsine.ravel(), srs_frq, srs_frq, Q
        ...     ) / factor
        ...     lbl = f"Eq-Sine{level + 1}; Eq-Sine of Eq-Sine{level}"
        ...     _ = ax.plot(srs_frq, eqsine, label=lbl)
        >>>
        >>> _ = ax.legend()
        >>> _ = ax.set_title(f"Eq-Sine (Abs-Acce/Q), Q = {Q}")
        >>> _ = ax.set_ylabel("Abs-Acce Eq-Sine (G)")
        >>> _ = ax.set_xlabel(r"$\omega_n$ Frequency (Hz)")
        >>> _ = ax.set_xlim(39.5, 49.5)
        >>> fig.tight_layout()
    """
    # compute maximizing Omega / omega_n ratio (see math in docstr):
    p_peak = Q * np.sqrt(np.sqrt(1 + 2 / Q ** 2) - 1)
    frf_frq = np.asarray(frf_frq)

    if srs_frq is None:
        srs_frq = frf_frq / p_peak
        if return_srs_frq is None:
            return_srs_frq = True
    else:
        srs_frq = np.asarray(srs_frq)
        if return_srs_frq is None:
            return_srs_frq = False

    ws = 2.0 * np.pi * srs_frq
    n = len(ws)
    ms = np.ones(n, float)
    bs = 1 / Q * ws
    ks = ws ** 2

    frf = np.asarray(frf)
    if frf.ndim == 1:
        frf = frf.reshape(-1, 1)
    nfrf = frf.shape[1]
    frf = np.abs(frf)

    # include transfer function peak frequencies in the forcing
    # function (these are close to the natural frequencies):
    ffreq = np.sort(np.hstack((frf_frq, p_peak * srs_frq)))
    df = np.diff(ffreq)
    pv = np.ones(len(ffreq), bool)
    pv[1:] = df > 1.0e-5
    ffreq = ffreq[pv]
    nf = len(ffreq)

    if len(frf_frq) == 1:
        newfrf = np.zeros((nf, nfrf), float)
        i = np.searchsorted(ffreq, frf_frq)
        if i == nf:
            i -= 1
        newfrf[i] = frf
        frf = newfrf
    else:
        ifunc = interp.interp1d(
            frf_frq, frf, axis=0, bounds_error=False, fill_value=0, assume_sorted=True
        )
        frf = ifunc(ffreq)

    shk = np.empty((n, nfrf), float)
    pvrb = ks < 0.005  # ks/ms < .005 ... since ms == 1
    pvel = np.logical_not(pvrb)
    rb = np.any(pvrb)
    el = np.any(pvel)

    # setup frequency scale for solution:
    freqw = 2 * np.pi * ffreq
    if el:
        fw = freqw.reshape(1, -1)
        H = (
            ks[pvel].reshape(-1, 1)
            - ms[pvel].reshape(-1, 1) @ fw ** 2
            + 1j * (bs[pvel].reshape(-1, 1) @ fw)
        )

    a = np.empty((n, nf), complex)
    if getresp:
        frfs = np.empty((nf, nfrf, n), complex)

    for j in range(nfrf):
        # compute relative response, then absolute (see eqns in srs)
        a[:] = 0.0
        fs = frf[:, j]  # len(frf)
        if rb:
            a[pvrb] = -fs  # / ms ... since ms == 1
        if el:
            a[pvel] = (fs * freqw ** 2) / H
        # from relative to absolute acceleration:
        a += fs
        if getresp:
            frfs[:, j, :] = a.T
        shk[:, j] = abs(a).max(axis=1)

    if getresp:
        resp = {"freq": ffreq, "frfs": frfs, "srs_frq": srs_frq}
        if return_srs_frq:
            return shk, srs_frq, resp
        return shk, resp

    if return_srs_frq:
        return shk, srs_frq
    return shk


def srsmap(timeslice, tsoverlap, sig, sr, freq, Q, wep=0, **srsargs):
    r"""
    Make a shock response spectral map ('waterfall') over time and
    frequency.

    Parameters
    ----------
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
    sig : 1d array_like
        Base acceleration signal; vector.
    sr : scalar
        The sample rate (samples/sec)
    freq : array_like
        Frequency vector in Hz. This defines the single DOF systems
        to use.
    Q : scalar
        Dynamic amplification factor :math:`Q = 1/(2\zeta)` where
        :math:`\zeta` is the fraction of critical damping.
    wep : scalar
        Argument for the :func:`pyyeti.dsp.windowends`; specifies the
        window-ends portion. Each time slice is passed through
        :func:`pyyeti.dsp.windowends` if wep > 0.
    **srsargs : miscellaneous options for :func:`srs`
        Allows the setting of `ic`, `stype`, `peak`, `eqsine`, etc
        options for :func:`srs`.  See :func:`srs` for more
        information.

    Returns
    -------
    mp : 2d ndarray
        The SRS map; columns span time, rows span frequency (so each
        column is an SRS curve). Time increases going across the
        columns and frequency increases going down the rows.
    t : 1d ndarray
        Time vector of center times; corresponds to columns in map.
        Signal is assumed to start at time = 0.
    f : 1d ndarray
        Frequency vector equal to the input `freq`; corresponds to
        rows in map.

    Notes
    -----
    This routine calls :func:`pyyeti.dsp.waterfall` for handling the
    timeslices and preparing the output.  :func:`srs` and
    :func:`pyyeti.dsp.windowends` are passed to that function.

    See also
    --------
    :func:`srs`, :func:`pyyeti.dsp.waterfall`,
    :func:`pyyeti.dsp.windowends`

    Examples
    --------
    Generate a sine sweep signal @ 4 octaves/min; process in 2-second
    windows with 50% overlap; 2% windowends, compute equivalent sine.

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from pyyeti import srs
        >>> from pyyeti import ytools
        >>> from matplotlib import cm, colors
        >>> sig, t, f = ytools.gensweep(10, 1, 50, 4)
        >>> sr = 1/t[1]
        >>> frq = np.arange(1., 50.1)
        >>> Q = 20
        >>> mp, t, f = srs.srsmap(2, .5, sig, sr, frq, Q, .02,
        ...                       eqsine=1)
        >>> _ = plt.figure('Example', clear=True)
        >>> cs = plt.contour(t, f, mp, 40, cmap=cm.plasma)
        >>> # This doesn't work in matplotlib 3.5.0:
        >>> #   cbar = plt.colorbar()
        >>> #   cbar.filled = True
        >>> #   cbar.draw_all()
        >>> # But this does:
        >>> norm = colors.Normalize(
        ...            vmin=cs.cvalues.min(), vmax=cs.cvalues.max()
        ...        )
        >>> sm = plt.cm.ScalarMappable(norm=norm, cmap=cs.cmap)
        >>> cb = plt.colorbar(sm)  # , ticks=cs.levels)
        >>> #
        >>> _ = plt.xlabel('Time (s)')
        >>> _ = plt.ylabel('Frequency (Hz)')
        >>> ttl = 'EQSINE Map of Sine-Sweep @ 4 oct/min, Q = 20'
        >>> _ = plt.title(ttl)

    Also show results on a 3D surface plot:

    .. plot::
        :context: close-figs

        >>> fig = plt.figure("Example 2", clear=True)
        >>> ax = fig.add_subplot(projection="3d")
        >>> x, y = np.meshgrid(t, f)
        >>> surf = ax.plot_surface(x, y, mp, rstride=1, cstride=1,
        ...                        linewidth=0, cmap=cm.plasma)
        >>> _ = fig.colorbar(surf, shrink=0.5, aspect=5)
        >>> ax.view_init(azim=-123, elev=48)
        >>> _ = ax.set_xlabel('Time (s)')
        >>> _ = ax.set_ylabel('Frequency (Hz)')
        >>> _ = ax.set_zlabel('Amplitude')
        >>> _ = plt.title(ttl)
    """
    return dsp.waterfall(
        sig,
        sr,
        timeslice,
        tsoverlap,
        srs,
        which=None,
        freq=freq,
        kwargs=dict(sr=sr, freq=freq, Q=Q, **srsargs),
        slicefunc=dsp.windowends,
        slicekwargs=dict(portion=wep),
    )
