# -*- coding: utf-8 -*-
"""
The Eigensystem Realization Algorithm

Identify mode shapes, frequencies, and damping from free-decay
measurement data.
"""
# Python ERA class written by Natalie Hintz and Tim Widrick.

import sys
import re
import numpy as np
import scipy.linalg as la
from scipy import signal, fft
from pyyeti import ode, dsp, writer
import matplotlib.pyplot as plt
import numbers
from warnings import warn


def _string_to_ints(string):
    """
    Convert "1, 2 5" to np.array([0, 1, 4])
    """
    ints = []
    for substring in re.split("[, ]+", string):
        if "-" in substring:
            a, b = substring.split("-")
            ints += list(range(int(a), int(b) + 1))
        else:
            ints.append(int(substring))
    return np.array(ints) - 1
    # return np.array([int(i) - 1 for i in re.split("[, ]+", string)])


def _string_to_ints2(string):
    """
    Convert Python expression to ndarray or "1, 2 5" to np.array([0,
    1, 4])
    """
    if string[0].isdigit():
        return _string_to_ints(string)
    return np.atleast_1d(eval(string)).astype(int) - 1


class ERA:
    """
    Implements the Eigensystem Realization Algorithm (ERA).

    ERA follows minimum realization theory to identify modal
    parameters (natural frequencies, mode shapes, and damping) from
    impulse response (free-decay) data. In the absense of forces, the
    response is completely determined by the system parameters which
    allows modal parameter identification to work. This code is based
    on Chapter 5 of reference [#era1]_.

    The discrete (z-domain) state-space equations are::

           x[k+1] = A*x[k] + B*u[k]
           y[k]   = C*x[k] + D*u[k]

    where ``u[k]`` are the inputs, ``y[k]`` are the outputs, and
    ``x[k]`` is the state vector at time step ``k``. ERA determines
    ``A``, ``B``, and ``C``. ``D`` is discussed below.

    Let there be ``n_outputs`` outputs, ``n_inputs`` inputs, and
    ``n_tsteps`` time steps. In ERA, impulse response measurements
    (accelerometer data, for example) form a 3-dimensional matrix
    sized ``n_outputs x n_tsteps x n_inputs``. This data can be
    partitioned into ``n_tsteps`` ``n_outputs x n_inputs``
    matrices. These ``n_outputs x n_inputs`` matrices are called
    Markov parameters. ERA computes a set of state-space matrices
    (``A``, ``B``, and ``C``) from these Markov parameters. To begin
    to see a relation between the Markov parameters and the
    state-space matrices, consider the following. Each input ``u[k]``
    is assumed to be unity impulse for one DOF and zero elsewhere. For
    example, if there are two inputs, ``u0[k]`` and ``u1[k]``, they
    are assumed to be (for all ``k``)::

        u0 = [ 1.0, 0.0, 0.0, ... 0.0 ]       # n_inputs x n_tsteps
             [ 0.0, 0.0, 0.0, ... 0.0 ]

        u1 = [ 0.0, 0.0, 0.0, ... 0.0 ]       # n_inputs x n_tsteps
             [ 1.0, 0.0, 0.0, ... 0.0 ]

    Letting ``x[0]`` be zeros (zero initial conditions), iterating
    through the state-space equations yields these definitions for the
    outputs::

        y0[k] = C @ A ** (k - 1) @ B[:, 0]    # n_outputs x 1

        y1[k] = C @ A ** (k - 1) @ B[:, 1]    # n_outputs x 1

    Putting these together::

       Y[k] = [ y0[k], y1[k] ]                # n_outputs x n_inputs

    The ``Y[k]`` matrices are the Markov parameters written as a
    function of the to-be-determined state-space matrices ``A``,
    ``B``, and ``C``. From the above, it can be seen that ``D =
    Y[0]``. By equating the input Markov parameters with the
    expressions for ``Y[k]``, ERA computes a minimum realization for
    ``A``, ``B``, and ``C``.

    .. note::
       This is a special note regarding overdamped modes. This
       algorithm may or may not print the correct frequencies and
       damping for overdamped modes. Overdamped modes have real
       eigenvalues, and there is no way for this routine to know which
       real eigenvalue forms a pair with which other real eigenvalue
       (see :func:`pyyeti.ode.get_freq_damping`). However, the
       state-space matrices and the fit (see the `resp_era` attribute)
       computed by the ERA algorithm will be correct. Note that for
       underdamped modes, the printed frequency and damping values are
       correct because this routine sorts the complex eigenvalues to
       ensure consecutive ordering of these mode pairs.

    .. note::
        This routine is demonstrated in the pyYeti :ref:`tutorial`:
        :doc:`/tutorials/era`. There is also a link to the source
        Jupyter notebook at the top of the tutorial.

    Mode indicators:

        =========  ===================================================
        Indicator  Description
        =========  ===================================================
          MSV      Normalized mode singular value. The MSV value
                   ranges from 0.0 to 1.0 for each mode and is a
                   measure of contribution to the response. Larger
                   values represent larger contribution.
          EMAC     Extended modal amplitude coherence. The EMAC value
                   ranges from 0.0 to 1.0 for each mode and is a
                   temporal measure indicating the consistency of the
                   mode over time to the fit. Higher values indicate
                   higher consistency, meaning that the mode decays as
                   expected. The EMAC is computed based on both
                   observability (outputs) and controllability
                   (inputs). From experience, the EMAC is superior to
                   the MAC value (see below). For more information,
                   see [#era2]_. See also `EMACO`.
          EMACO    EMAC from just observability, as opposed to both
                   observability and controllability. Reference
                   [#era3]_ argues that this is more reliable than
                   EMAC because there are typically more response
                   measurements (observability side) than inputs
                   (controllability side). This means we are relying
                   on the mode shape matrix C, which is typically
                   supported by many measurements, and not relying on
                   the mode participation factors matrix B, which may
                   be more sensitive to the particular excitation. As
                   pointed out in [#era3]_, temporal consistency based
                   on B is highly dependent on the choice of input
                   locations, which may be too few for reliable
                   results.

                   .. note::
                        Note: this algorithm does not use the equation
                        in [#era3]_, which is a simple average of all
                        EMAC values for a mode across all
                        outputs. Instead, it uses a weighted average
                        where the weighting factor for each output is
                        the modeshape coefficient for that output (as
                        done in [#era2]_).

          MPC      Modal phase collinearity. The MPC value ranges from
                   0.0 to 1.0 for each mode and is a spatial measure
                   indicating whether or not the different response
                   locations for a mode are in phase. Higher values
                   indicate more "in-phase" measurements. Only useful
                   if there are multiple outputs (MPC is 1.0 for
                   single output). MPC will be 1.0 for a mode that is
                   100% real (this happens when the real mode
                   diagonalizes the damping). MPC is directly related
                   to the "Modal Complexity Factor" (MCF): ``MPC = 1 -
                   MCF``.
          CMIO     Consistent mode indicator based on observability.
                   ``CMIO = EMACO * MPC`` for each mode.
          MAC      Modal amplitude coherence. The MAC value ranges
                   from 0.0 to 1.0 for each mode and is a temporal
                   measure indicating the importance of the mode over
                   time to the fit. Higher values indicate more
                   importance. The MAC value for a mode is the dot
                   product of two vectors:

                     1. The ideal, reconstructed time history for a
                        mode
                     2. The time history extracted from the input data
                        for the mode after discarding noisy data via
                        singular value decomposition

                   From experience, this indicator is not as useful as
                   the EMAC.
        =========  ===================================================

    References
    ----------
    .. [#era1] Jer-Nan Juang. Applied System Identification. United
               Kingdom: Prentice Hall, 1994.

    .. [#era2] Richard S. Pappa, Kenny B. Elliott, Axel Schenk. "A
               Consistent-Mode Indicator for the Eigensystem
               Realization Algorithm", NASA Technical Memorandum
               107607, April 1992.

    .. [#era3] Yun, G. J., Lee, S. G., & Shang, S. "An improved mode
               accuracy indicator for Eigensystem Realization Analysis
               (ERA) techniques", KSCE Journal of Civil Engineering,
               16, 377-387, March 2012.

    Examples
    --------
    The following example demonstrates ERA by comparing to a known
    system.

    .. plot::
        :context: close-figs

        We have the following mass, damping and stiffness matrices
        (note that the damping has been specially defined to be
        diagonalized by the modes):

        >>> import numpy as np
        >>> import scipy.linalg as la
        >>> from pyyeti import era, ode
        >>> np.set_printoptions(precision=5, suppress=True)
        >>>
        >>> M = np.identity(3)
        >>>
        >>> K = np.array(
        ...     [
        ...         [4185.1498, 576.6947, 3646.8923],
        ...         [576.6947, 2104.9252, -28.0450],
        ...         [3646.8923, -28.0450, 3451.5583],
        ...     ]
        ... )
        >>>
        >>> D = np.array(
        ...     [
        ...         [4.96765646, 0.97182432, 4.0162425],
        ...         [0.97182432, 6.71403672, -0.86138453],
        ...         [4.0162425, -0.86138453, 4.28850828],
        ...     ]
        ... )

        Since we have the system matrices, we can determine the
        frequencies and damping using the eigensolution.

        >>> (w2, phi) = la.eigh(K, M)
        >>> omega = np.sqrt(w2)
        >>> freq_hz = omega / (2 * np.pi)
        >>> freq_hz
        array([  1.33603,   7.39083,  13.79671])
        >>> modal_damping = phi.T @ D @ phi
        >>> modal_damping
        array([[ 0.33578, -0.     , -0.     ],
               [-0.     ,  6.9657 ,  0.     ],
               [-0.     ,  0.     ,  8.66873]])
        >>> zeta = np.diag(modal_damping) / (2 * omega)
        >>> zeta
        array([ 0.02 ,  0.075,  0.05 ])

        What if we only had the response measurements from an impulse
        input and didn't have the system matrices? That's where ERA
        comes in and we'll use it to compute modes and damping. Since
        there is no noise and all modes are excited, ERA will find the
        exact modes and damping.

        First, we need to generate an impulse response for ERA to work
        with. We'll just use an initial velocity condition as the
        source of the impulse:

        >>> dt = 0.01
        >>> t = np.arange(0, 1, dt)
        >>> F = np.zeros((3, len(t)))
        >>> ts = ode.SolveUnc(M, D, K, dt)
        >>> sol = ts.tsolve(force=F, v0=[1, 1, 1])

        We can use the displacement, velocity or acceleration response
        as the input for ERA. Here, we'll use velocity because all
        modes are well represented in the response for the one second
        simulation. Displacement favors the lower frequency modes (so
        ERA might eliminate the highest frequency mode) and
        acceleration favors the higher frequency modes (so ERA might
        eliminate the lowest frequency mode). For those new to ERA, it
        is a very good exercise to experiment with using each of them
        with this problem.

        >>> era_fit = era.ERA(
        ...     sol.v,
        ...     sr=1 / dt,
        ...     auto=True,
        ... )
        <BLANKLINE>
        Current fit includes all modes:
          Mode  Freq (Hz)   % Damp    MSV     EMAC   EMACO    MPC     CMIO    MAC
          ----  ---------  --------  ------  ------  ------  ------  ------  ------
        *    1     1.3360    2.0000  *0.817  *1.000  *1.000  *1.000  *1.000  *1.000
        *    2     7.3908    7.5000  *0.839  *1.000  *1.000  *1.000  *1.000  *1.000
        *    3    13.7967    5.0000  *1.000  *1.000  *1.000  *1.000  *1.000  *1.000
        <BLANKLINE>
        Auto-selected modes fit:
          Mode  Freq (Hz)   % Damp    MSV     EMAC   EMACO    MPC     CMIO    MAC
          ----  ---------  --------  ------  ------  ------  ------  ------  ------
             1     1.3360    2.0000   0.817   1.000   1.000   1.000   1.000   1.000
             2     7.3908    7.5000   0.839   1.000   1.000   1.000   1.000   1.000
             3    13.7967    5.0000   1.000   1.000   1.000   1.000   1.000   1.000

        Compare frequencies, damping, mode shapes:

        >>> np.allclose(era_fit.freqs_hz, freq_hz)
        True
        >>> np.allclose(era_fit.zeta, zeta)
        True
        >>> era_fit.phi / phi  # modes will be scaled differently
        array([[ 1.38989, -1.02337, -1.34712],
               [ 1.38989, -1.02337, -1.34712],
               [ 1.38989, -1.02337, -1.34712]])

    .. plot::
        :context: close-figs

        For demonstration, add some noise to the signal and rerun
        ERA. Note that while this example works out nicely for our
        known system, it will often be necessary to run ERA multiple
        times with different parameters and different response data
        selections to gain confidence in which modes are real.

        For this run, we'll also add some labels and some FFT plots up
        to 30.0 Hz.

        >>> # for repeatability, create a specific generator:
        >>> from numpy.random import Generator, PCG64DXSM
        >>> rng = Generator(PCG64DXSM(seed=1))
        >>> era_fit = era.ERA(
        ...     sol.v + 0.05 * rng.normal(size=sol.v.shape),
        ...     sr=1 / dt,
        ...     auto=True,
        ...     input_labels=["x", "y", "z"],
        ...     FFT=True,
        ...     FFT_range=30.0,
        ... )
        <BLANKLINE>
        Current fit includes all modes:
          Mode  Freq (Hz)   % Damp    MSV     EMAC   EMACO    MPC     CMIO    MAC
          ----  ---------  --------  ------  ------  ------  ------  ------  ------
        *    1     1.3330    6.5080  *0.838  *0.761  *0.779  *0.946  *0.737  *1.000
             2     2.2398   36.5472   0.299   0.010   0.037   0.122   0.005  *0.988
        *    3     7.4181    6.9215  *0.851  *0.768  *0.796  *0.999  *0.795  *1.000
             4     8.6995    6.1563   0.267   0.011   0.025  *0.543   0.014  *0.967
        *    5    13.7967    4.8219  *1.000  *0.753  *0.837  *0.998  *0.836  *1.000
             6    15.1565    2.0682   0.194   0.013   0.037  *0.794   0.030  *0.930
             7    19.2629    4.0293   0.253   0.007   0.021   0.086   0.002  *0.946
             8    22.1264    1.4390   0.276   0.295   0.408  *0.841   0.343  *0.992
             9    23.6593    9.8029   0.255   0.092   0.277   0.217   0.060  *0.937
            10    28.7943    0.6909   0.178  *0.408  *0.613  *0.910  *0.558  *0.940
            11    32.2365    2.5384   0.211   0.020   0.109   0.482   0.053  *0.952
            12    33.8215    1.7932   0.252   0.063   0.103  *0.843   0.087  *0.978
            13    35.6857    2.1774   0.267   0.057   0.131   0.486   0.064  *0.961
            14    38.6010    1.0741   0.181   0.000   0.000   0.373   0.000  *0.940
            15    42.0391    0.2207   0.191   0.306  *0.500   0.170   0.085  *0.951
            16    45.3651   13.9108   0.066   0.000   0.004  *0.819   0.003   0.286
            17    47.1139    0.5886   0.266   0.278   0.412   0.189   0.078  *0.988
        <BLANKLINE>
        Auto-selected modes fit:
          Mode  Freq (Hz)   % Damp    MSV     EMAC   EMACO    MPC     CMIO    MAC
          ----  ---------  --------  ------  ------  ------  ------  ------  ------
             1     1.3330    6.5080   0.838   0.761   0.779   0.946   0.737   1.000
             2     7.4181    6.9215   0.851   0.768   0.796   0.999   0.795   1.000
             3    13.7967    4.8219   1.000   0.753   0.837   0.998   0.836   1.000
    """

    def __init__(
        self,
        resp,
        sr,
        *,
        svd_tol=0.05,
        auto=False,
        alpha=None,
        beta=None,
        MSV_lower_limit=0.35,
        EMAC_lower_limit=0.35,
        EMACO_lower_limit=0.5,
        MPC_lower_limit=0.5,
        CMIO_lower_limit=0.35,
        MAC_lower_limit=0.75,
        all_lower_limits=None,
        damp_range=(0.001, 0.999),
        freq_range=None,
        t0=0.0,
        verbose=True,
        show_plot=True,
        input_labels=None,
        FFT=False,
        FFT_range=None,
        figure_label="ERA Fit",
    ):
        """
        Instantiates a :class:`ERA` solver.

        Parameters
        ----------
        resp : 1d, 2d, or 3d ndarray
            Impulse response measurement data. In the general case,
            this is a 3d array sized ``n_outputs x n_tsteps x
            n_inputs``, where ``n_outputs`` are the number of outputs
            (measurements), ``n_tsteps`` is the number of time steps,
            and ``n_inputs`` is the number of inputs. If there is only
            one input (the typical case), then `resp` can be input as
            a 2d array of size ``n_outputs x n_tsteps``. Further, if
            there is only one output, then `resp` can be input as a 1d
            array of length `n_tsteps`.
        sr : scalar
            Sample rate at which `resp` was sampled.
        svd_tol : scalar; optional
            Determines how many singular values to keep. Can be:

                - Integer greater than 1 to specify number of expected
                  modes (tolerance equal to 2 * expected modes)
                - Float between 0 and 1 to specify required
                  significance of a singular value to be kept

            Consider this the master control on removal of noise. To
            change, you have to rerun. When in doubt, it is
            recommended to try multiple values of `svd_tol`. The
            secondary control is the selection of modes which can be
            interactive (if `auto` is False) or automatic according to
            the parameters below.
        auto : boolean; optional
            Enables automatic selection of true modes. The default is
            False.
        alpha, beta : integers or None; optional
            Number of block rows and block columns to use for the
            Hankel matrix, respectively. If both are None, the Hankel
            matrix is made roughly square in terms of blocks. If both
            are set, `alpha` is given precedence over `beta` if there
            is not enough data to support both settings.

            .. note::
                 The maximum number of modes that can be detected is
                 equal to the minimum dimension of the Hankel matrix
                 divided by 2. So, be careful not to limit either
                 `alpha` or `beta` too low.

        MSV_lower_limit : scalar; optional
            The lower limit for the normalized "mode singular value"
            for modes to be selected when `auto` is True. The MSV
            value ranges from 0.0 to 1.0 for each mode.
        EMAC_lower_limit : scalar; optional
            The lower limit for the "extended modal amplitude
            coherence" value for modes to be selected when `auto` is
            True. The EMAC value ranges from 0.0 to 1.0 for each mode.
        EMACO_lower_limit : scalar; optional
            The lower limit for the "extended modal amplitude
            coherence from observability" value for modes to be
            selected when `auto` is True. The EMACO value ranges from
            0.0 to 1.0 for each mode.
        MPC_lower_limit : scalar; optional
            The lower limit for the "modal phase collinearity" value
            for modes to be selected when `auto` is True. The MPC
            value ranges from 0.0 to 1.0 for each mode.
        CMIO_lower_limit : scalar; optional
            The lower limit for the "consistent mode indicator from
            observability" value for modes to be selected when `auto`
            is True. The CMIO value ranges from 0.0 to 1.0 for each
            mode.
        MAC_lower_limit : scalar; optional
            The lower limit for the "modal amplitude coherence" value
            for modes to be selected when `auto` is True. The MAC
            value ranges from 0.0 to 1.0 for each mode.
        all_lower_limits : scalar or None; optional
            If not None, it sets all the indicator lower limit
            values and any other lower limit settings are quietly
            ignored.
        damp_range : 2-tuple; optional
            Specifies the range (inclusive) of acceptable damping
            values for automatic selection.
        freq_range : 2-tuple or None; optional
            Specifies the range (inclusive) of acceptable frequencies
            (in Hz) values for automatic selection. If None (the
            default), ``freq_range = (0, sr/2)``.
        t0 : scalar; optional
            The initial time value in `resp`. Only used for plotting.
        verbose : bool; optional
            If True, print tables showing frequencies and
            indicators. Ignored (treated as True) if `auto` is False.
        show_plot : boolean; optional
            If True, show plot of ERA fit (with or without FFT). The
            default is True.
        input_labels : list or None; optional
            List of data labels for each input signal to ERA.
        FFT : boolean; optional
            Enables display of FFT plot of input data. The default is
            False.
        FFT_range : scalar or list; optional
            Limits displayed frequency range of FFT plot. A scalar
            value or list with one term will act as a maximum
            cutoff. A pair of values will act as minimum and maximum
            cutoffs, respectively. The default is None.
        figure_label : string; optional
            Label to use for opening figures. Handy for comparing
            different ERA fits side-by-side.

        Notes
        -----
        The class instance is populated with the following members
        (equation numbers are from Chapter 5 of reference [#era1]_):

        ================  ============================================
        Member            Description
        ================  ============================================
        resp              Impulse response data
        resp_era          Final ERA fit to the impulse response data
        n_outputs         Number of outputs
        n_tsteps          Number of time-steps
        n_inputs          Number of inputs
        sr                Sample rate
        t0                Initial time value in `resp` (for plotting)
        time              `resp` time vector used for plotting;
                          created from `sr` and `t0`
        svd_tol           Singular value tolerance
        auto              Automatic or interactive method of
                          identifying true modes
        show_plot         If True, plots are made
        input_labels      Create a legend for approximated fit plot
        lower_limits      Dict of lower limits for each indicator for
                          auto-selecting modes
        damp_range        Range of acceptable damping values for
                          automatic selection
        freq_range        Range of acceptable frequencies (Hz) for
                          automatic selection
        FFT               Produces an FFT plot of input data for
                          comparison to detected modes
        FFT_range         Frequency cutoff(s) for FFT plot
        figure_label      Label for figure windows
        H_0               The H(0) generalized Hankel matrix (eq 5.24)
        H_1               The H(1) generalized Hankel matrix (eq 5.24)
        alpha             Number of block rows (time-steps) in H
        beta              Number of block columns (time-steps) in H
        A_hat             ERA state-space "A" matrix, (eq 5.34)
        B_hat             ERA state-space "B" matrix, (eq 5.34)
        C_hat             ERA state-space "C" matrix, (eq 5.34)
        A_modal           ERA modal state-space "A" matrix (complex
                          eigenvalues on diagonal)
        B_modal           ERA modal state-space "B" matrix
        C_modal           ERA modal state-space "C" matrix (complex
                          mode-shapes)
        eigs              Complex eigenvalues of `A_hat` (discrete)
        eigs_continuous   Continuous version of `eigs`; ln(eigs) * sr
        psi               Complex eigenvectors of `A_hat`
        psi_inv           Inverse of `psi`
        freqs             Real mode frequencies (rad/sec)
        freqs_hz          Real mode frequencies (Hz)
        zeta              Real mode percent critical damping
        phi               Real mode shapes (from `C_modal`); see also
                          the MPC results for how "real" each mode is
        indicators        Dict of all indicator values for each mode
                          pair
        ================  ============================================
        """
        self.resp = np.atleast_3d(resp)
        # 1d --> (1, n_tsteps, 1)
        # 2d --> (n_outputs, n_tsteps, 1)
        # 3d --> (n_outputs, n_tsteps, n_inputs)
        self.n_outputs, self.n_tsteps, self.n_inputs = self.resp.shape

        self.sr = sr
        self.svd_tol = svd_tol
        self.auto = auto
        self.lower_limits = {
            "MSV": MSV_lower_limit,
            "EMAC": EMAC_lower_limit,
            "EMACO": EMACO_lower_limit,
            "MPC": MPC_lower_limit,
            "CMIO": CMIO_lower_limit,
            "MAC": MAC_lower_limit,
        }

        if all_lower_limits is not None:
            for key in self.lower_limits:
                self.lower_limits[key] = all_lower_limits

        self.damp_range = damp_range
        if freq_range is None:
            freq_range = (0.0, sr / 2)
        self.freq_range = freq_range
        self.verbose = verbose
        self.show_plot = show_plot
        self.input_labels = input_labels
        self.FFT = FFT
        self.FFT_range = FFT_range
        self.t0 = t0
        self.figure_label = figure_label

        self._H_generate(alpha, beta)
        self._state_space()
        self._conv_2_modal()
        self._mode_select()

    def _H_generate(self, alpha, beta):
        """
        Given Markov parameters, will generate the system's generalized
        Hankel matrix and time-shifted generalized Hankel matrix.
        """
        # Determining the alpha and beta parameters in order to
        # maximize coverage of data in Hankel matrices
        if alpha is None:
            if beta is None:
                alpha = self.n_tsteps // 2
                beta = self.n_tsteps - alpha
            else:
                beta = min(beta, self.n_tsteps - 2)
                alpha = self.n_tsteps - beta
        else:
            alpha = min(alpha, self.n_tsteps - 2)
            beta_max = self.n_tsteps - alpha
            if beta is None:
                beta = beta_max
            else:
                beta = min(beta, beta_max)

        H_dim = (alpha * self.n_outputs, beta * self.n_inputs)

        # Forming Hankel matrices
        H_0 = np.empty(H_dim)
        H_1 = np.empty(H_dim)

        # Reshaping Markov parameters into form Y = [Y0, Y1, Y2, ..., Yn]
        # Each Y[i] is of shape m x r
        Markov = self.resp.reshape(self.n_outputs, -1)
        rows = np.arange(self.n_outputs)
        cols = np.arange(beta * self.n_inputs)

        for _ in range(alpha):
            # Using block indexing to fill Hankel matrices with Markov
            # parameters
            H_0[rows] = Markov[:, cols]
            H_1[rows] = Markov[:, cols + self.n_inputs]  # Time shift
            # Advancing row and column indices
            rows += self.n_outputs
            cols += self.n_inputs

        self.H_0 = H_0
        self.H_1 = H_1
        self.alpha = alpha
        self.beta = beta

    def _state_space(self, pre_set_n=None):
        """
        This function computes a state-space representation of a system
        given the block Hankel matrix which is decomposed using
        Singular Value Decomposition (SVD) and used to find the
        state-space matrices.

        Notes
        -----
        This function determines the system order from the tolerance
        parameter input to the class. If the tolerance yields an odd
        system order value, it will be altered to become even to
        ensure there are an even number of singular values.
        """

        # Decomposing H_0 using SVD
        R, sigma, ST = np.linalg.svd(self.H_0)  # eq 5.30, #era1

        if pre_set_n is None:
            # svd_tol can be treated as a selection of modes (if greater
            # than 1) or svd_tol can be set as a numeric boundary to sort
            # through non-significant singular values
            if self.svd_tol >= 1:
                n = int(self.svd_tol)
                if n > sigma.size:
                    n = sigma.size
                    warn(
                        f"`svd_tol` ({n}) is greater than number of singular values"
                        f" ({sigma.size}).",
                        RuntimeWarning,
                    )
            else:
                n = np.argwhere(sigma / sigma[0] >= self.svd_tol).size

            # Ensures that number of pairs remains even
            if n % 2 != 0:
                n += 1
        else:
            n = pre_set_n

        # Reshaping R, sigma, ST, accordingly
        R = R[:, :n]
        sigma_sqrt = np.sqrt(sigma[:n])
        ST = ST[:n, :]

        # Recovering P and Q matrices: eq 5.35, #era1
        self.P = R * sigma_sqrt
        self.Q = (sigma_sqrt * ST.T).T

        # Recovering identified state matrices: eq 5.34, #era1
        self.C_hat = self.P[: self.n_outputs, :]
        self.B_hat = self.Q[:, : self.n_inputs]
        self.A_hat = (R / sigma_sqrt).T @ self.H_1 @ (ST.T / sigma_sqrt)

    def _form_modal_realization(self):
        # Compute modal space state-space matrices
        self.A_modal = np.diag(self.eigs)

        # include all time for the EMAC calculation
        self.B_modal_all = self.psi_inv @ self.Q
        self.C_modal_all = self.P @ self.psi

        self.B_modal = self.B_modal_all[:, : self.n_inputs]
        self.C_modal = self.C_modal_all[: self.n_outputs]

        # form estimate of real mode shapes:
        phi = self.C_modal[:, ::2]
        rows = np.abs(phi).argmax(axis=0)
        cols = np.arange(phi.shape[1])
        phi_scaled = phi / phi[rows, cols]
        self.phi = phi_scaled.real

        # Calculate indicators:
        self._MSV()
        self._EMAC()
        self._MPC()
        self._MAC()

        # also calculate the EMACO and CMIO from ref 3:
        # self.EMACO = self.EMACo.mean(axis=1)
        self.CMIO = self.EMACO * self.MPC

        self.indicators = {
            "MSV": self.MSV,
            "EMAC": self.EMAC,
            "EMACO": self.EMACO,
            "MPC": self.MPC,
            "CMIO": self.CMIO,
            "MAC": self.MAC,
        }

    def _conv_2_modal(self):
        """
        This routine converts the system realization problem into
        modal space.
        """
        # Generating eigenvalues and matrix of eigenvectors
        eigs, psi = la.eig(self.A_hat)

        # Retrieving sorted index of eigenvalues based on magnitude of
        # imaginary components
        #  - Using a stable sorter to allow for preservation of
        #    overdamped modes (in case they're consecutive; if not,
        #    see the special note in the documentation for class ERA)
        i = np.argsort(abs(eigs.imag), kind="stable")
        eigs = eigs[i]
        psi = psi[:, i]

        eigs_continuous = np.log(eigs) * self.sr
        freqs, zeta = ode.get_freq_damping(eigs_continuous, suppress_warning=True)

        # Finding the order of increasing frequency
        index = np.argsort(freqs, kind="stable")

        # Locating overdamped modes and putting them first:
        pv = zeta[index] >= 1.0
        index = np.concatenate((index[pv], index[~pv]))

        # Ordering frequencies and damping accordingly
        self.freqs = freqs[index]
        self.zeta = zeta[index]

        # Deriving ordered indexing for mode pairs
        index_pairs = np.empty(index.size * 2, dtype=int)
        for i in range(index.size):
            index_pairs[2 * i] = index[i] * 2
            index_pairs[2 * i + 1] = index[i] * 2 + 1

        # Re-ordering eigenvalues and eigenvectors
        self.eigs = eigs[index_pairs]
        self.eigs_continuous = eigs_continuous[index_pairs]
        self.psi = psi[:, index_pairs]

        # Deriving the frequencies in Hz
        self.freqs_hz = self.freqs / (2 * np.pi)

        # Recovering inverse of eigenvector matrix
        self.psi_inv = np.linalg.inv(self.psi)

        self._form_modal_realization()

    def _remove_noise(self):
        """
        Replace input response history with current ERA fit
        """
        self.resp = self.resp_era
        self._H_generate()
        self._state_space()
        self._conv_2_modal()

    def _remove_fit(self):
        """
        Remove current ERA fit from the input response history
        """
        self.resp = self.resp - self.resp_era
        self._H_generate()
        self._state_space()
        self._conv_2_modal()

    def _remove_fit2(self, n_modes):
        """
        Remove current ERA fit from the input response history
        """
        self.resp = self.resp - self.resp_era
        self._H_generate()
        self._state_space(self.A_hat.shape[0] - 2 * n_modes)
        self._conv_2_modal()

    def _MSV(self):
        """
        This routine calculates the normalized Mode Singular Values (MSV)
        of each detected mode.
        """
        A_m, B_m, C_m = self.A_modal, self.B_modal, self.C_modal
        lamd = np.diag(A_m)

        n = A_m.shape[0]
        MSV = np.empty(n // 2)
        max_lam = self.n_tsteps // self.n_inputs

        # compute MSV for each mode; eq. 5.50, #era1
        for i, j in enumerate(range(0, n - 1, 2)):
            b_vec = B_m[j, :]
            c_vec = C_m[:, j]
            b_abs_sq = np.sqrt(abs(b_vec @ np.conj(b_vec.T)))
            c_abs_sq = np.sqrt(abs(c_vec @ np.conj(c_vec.T)))
            lam_abs = abs(lamd[j]) ** np.arange(0, max_lam - 1)
            MSV[i] = np.sqrt(c_abs_sq * sum(lam_abs) * b_abs_sq)

        # Normalize to max of 1.0:
        self.MSV = MSV / MSV.max()

    def _MAC(self):
        """
        This routine calculates the Modal Amplitude Coherence (MAC) of
        each detected mode.
        """
        # Q_bar: time history sequence for each mode from input data
        # eq. 5.48, #era1
        Q_bar = self.psi_inv @ self.Q

        # Q_hat: reconstructed time history sequence for each mode
        # eq. 5.42, #era1
        Q_hat = np.empty(Q_bar.shape, dtype=complex)
        bi_len = np.arange(self.n_inputs)
        B_modal = self.B_modal
        eigs = self.eigs
        n = B_modal.shape[0]
        for i, j in enumerate(range(0, Q_bar.shape[1], self.n_inputs)):
            Q_hat[:, bi_len + j] = B_modal * (eigs.reshape(n, 1) ** i)

        # Calculating MAC values; eq 5.49, #era1
        MAC = np.empty(n // 2)
        for i, j in enumerate(range(0, n - 1, 2)):
            Q_b_con = abs(Q_bar[j, :] @ np.conj(Q_bar[j, :]).T)
            Q_h_con = abs(Q_hat[j, :] @ np.conj(Q_hat[j, :]).T)
            Q_con = abs(Q_bar[j, :] @ np.conj(Q_hat[j, :]).T)
            MAC[i] = Q_con / np.sqrt(Q_b_con * Q_h_con)

        self.MAC = MAC

    def _calc_emac_in_out(self, modes, z, n_oi, n_tsteps):
        # the output or input EMAC:
        n = modes.shape[1]
        EMAC_oi = np.empty((n // 2, n_oi))
        for i, j in enumerate(range(0, n - 1, 2)):
            # initial mode shape components for all outputs (or mode
            # participation factors for all inputs)
            phi0_j = modes[:n_oi, j]

            # same at decay time:
            k = n_tsteps[j] * n_oi
            phiF_j = modes[k : k + n_oi, j]

            # expected value at decay time:
            phiF_j_expected = phi0_j * z[j] ** n_tsteps[j]

            Rj = abs(phiF_j) / abs(phiF_j_expected)
            pv = Rj > 1
            if pv.any():
                Rj[pv] = 1 / Rj[pv]

            Pj = abs(np.angle(phiF_j / phiF_j_expected))
            Wj = 1 - 4 * Pj / np.pi
            Wj[Wj < 0] = 0.0
            EMAC_oi[i] = Rj * Wj

        return EMAC_oi

    def _EMAC(self):
        """
        This routine calculates the Extended Modal Amplitude Coherence
        (EMAC) of each detected mode.
        """
        # Choose time for decayed value comparisons for both outputs
        # and inputs, while trying to avoid comparing zeros.
        #
        # Outputs: number of time steps is nominally "alpha" (to get
        # to last block row of Hankel matrix ... see eq 5.24)
        #
        # Inputs: number of time steps is nominally "beta" (to get
        # to last block column of Hankel matrix ... see eq 5.24)
        #
        # The alpha and beta values (minus 1) will be used unless the
        # 90% decay value comes first (assuming underdamped). The
        # exponential decay term is part of: e^(s n dt), where "s" is
        # the continuous system eigenvalue.  Expanding s (see
        # pyyeti.ode.get_freq_damping):
        #
        #     s = -omega zeta +/- i omega sqrt(1-zeta^2)
        #
        # For simplicity below, let:
        #
        #     s = a +/- i b
        #
        # Therefore:
        #
        #     e^(s dt) = e^(a dt) e^(+/- i b dt)
        #              = e^(a dt) (cos(b dt) +/- i sin(b dt)
        #
        # The e^(a dt) term is the decay term. We can get that term by
        # using "abs" (because cos^2 + sin^2 = 1):
        #
        #     abs(e^(s dt)) = e^(a dt)
        #
        # We'll find number of time steps 'n' such that e^(a n dt) is
        # approximately 0.1:
        #
        #     e^(a n dt) = 0.1
        #     n = ln(0.1) / (a dt)
        #     n = ln(0.1) / s.real * sr
        #
        # We could also use the discrete eigenvalues (z), since z =
        # e^(s dt):
        #
        #     abs(z)^n = e^(a n dt) = 0.1
        #            n = ln(0.1) / ln(abs(z))
        #
        # So, the maximum number of time steps for each underdamped
        # mode pair is: (these are equivalent as shown above):
        # n_max = (np.log(0.1) / np.log(abs(self.eigs))).astype(int)
        n_max = (np.log(0.1) / self.eigs_continuous.real * self.sr).astype(int)
        n_max[n_max < 2] = 2

        # for outputs, inputs:
        n_output_tsteps = np.fmin(n_max, self.alpha - 1)
        n_input_tsteps = np.fmin(n_max, self.beta - 1)
        z = self.eigs

        EMACo = self._calc_emac_in_out(
            self.C_modal_all, z, self.n_outputs, n_output_tsteps
        )

        # "B" is part of Q ... here we want Q for all mode
        # participation factors
        EMACi = self._calc_emac_in_out(
            self.B_modal_all.T, z, self.n_inputs, n_input_tsteps
        )

        # total EMAC:
        B_modal = self.B_modal
        C_modal = self.C_modal
        n = B_modal.shape[0]
        EMAC = np.empty(n // 2)
        EMACO = np.empty(n // 2)
        for i, j in enumerate(range(0, n - 1, 2)):
            c2 = abs(C_modal[:, j]) ** 2
            b2 = abs(B_modal[j, :]) ** 2

            # just based on observability (outputs)
            num = EMACo[i] @ c2
            den = c2.sum()
            EMACO[i] = num / den

            # include controllability (inputs)
            num *= b2 @ EMACi[i]
            den *= b2.sum()
            EMAC[i] = num / den

        self.EMACo = EMACo
        self.EMACi = EMACi
        self.EMAC = EMAC
        self.EMACO = EMACO

    def _MPC(self):
        """
        This routine calculates the Modal Phase Colinearity for each
        detected mode.
        """
        C = self.C_modal
        n = C.shape[1]
        MPC = np.empty(n // 2)
        for i, j in enumerate(range(0, n - 1, 2)):
            re = C[:, j].real
            im = C[:, j].imag
            Sxx = re @ re
            Syy = im @ im
            Sxy = re @ im
            # Can solve for eigenvalues with `la.eigh`:
            #  om = la.eigh([[Sxx, Sxy], [Sxy, Syy]], eigvals_only=True)
            #  lam1 = om[1]
            #  lam2 = om[0]
            # and then use the equations in the reference ...
            # Or, since this is just 2x2, we can solve by hand (see
            # eqns in pyyeti.ytools._calc_covariance_sine_cosine, then
            # simplify the final eqn by hand)
            MPC[i] = ((Syy - Sxx) ** 2 + 4 * Sxy**2) / (Sxx + Syy) ** 2

        self.MPC = MPC

    def _trim_2_selected_modes(self, selected_modes, saved_model=False):
        """
        This function extracts selected modes and arranges them to be
        presented without excluded modes. It recalculates the modal
        state space matrices as well as the indicators for each
        extracted mode. This function is responsible for updating
        data with the reduced model based on the modes which are kept
        for consideration.
        """
        if len(selected_modes) == 0:
            raise RuntimeError("No modes selected")

        if saved_model:
            freq = self._saved["freqs_hz"]
            zeta = self._saved["zeta"]
            eigs = self._saved["eigs"]
            eigs_continuous = self._saved["eigs_continuous"]
            psi = self._saved["psi"]
            psi_inv = self._saved["psi_inv"]
        else:
            freq = self.freqs_hz
            zeta = self.zeta
            eigs = self.eigs
            eigs_continuous = self.eigs_continuous
            psi = self.psi
            psi_inv = self.psi_inv

        # Identifying position of modal pairs
        self.selected_mode_pairs = []
        for i in selected_modes:
            self.selected_mode_pairs.append(2 * i)
            self.selected_mode_pairs.append(2 * i + 1)

        # Extracting reduced data
        self.freqs_hz = freq[selected_modes]
        self.zeta = zeta[selected_modes]
        self.eigs = eigs[self.selected_mode_pairs]
        self.eigs_continuous = eigs_continuous[self.selected_mode_pairs]
        self.psi = psi[:, self.selected_mode_pairs]
        self.psi_inv = psi_inv[self.selected_mode_pairs, :]

        self._form_modal_realization()

    def _compute_era_fit(self, A, B, C):
        """
        Compute the ERA fit to the response data

        Parameters
        ----------
        A : numpy.ndarray
            Identified state matrix.
        B : numpy.ndarray
            Identified input matrix.
        C : numpy.ndarray
            Identified output matrix.
        """
        # compute the ERA best-fit response:
        n = A.shape[0]
        x = np.zeros((n, self.n_inputs * self.n_tsteps), dtype=A.dtype)
        cols = np.arange(0, self.n_inputs)
        x[:, cols] = B
        for k in range(0, self.n_tsteps - 1):
            x[:, cols + self.n_inputs] = A @ x[:, cols]
            cols += self.n_inputs
        self.resp_era = np.real(C @ x).reshape(
            self.n_outputs, self.n_tsteps, self.n_inputs
        )
        self._plot_era()

    def _plot_era(self):
        """
        Plots input data against reduced model.

        Notes
        -----
        If the parameter `input_labels` is left empty, the approximate
        fit plot will not have a corresponding legend. Otherwise, the
        data and its approximations will appear with a corresponding
        legend. If the number of input labels provided does not match
        the number of input signals, this will return an error.

        If the parameter `FFT` is set to True, this function will
        generate an FFT plot of the input data in the corresponding
        color coding of the approximate fit plot. This will appear
        below the approximate fit plot and serves as a helpful
        comparison of detected modal data. If the scaling of the FFT
        plot is suboptimal, it can be adjusted by changing the values
        input to `FFT_range`, which will serve as a maximum cutoff or
        minimum/maximum cutoff pair, depending on whether one or two
        values are given.
        """
        if not self.show_plot:
            return

        y = self.resp_era
        if not hasattr(self, "time"):
            self.time = np.arange(0, self.n_tsteps) / self.sr + self.t0

        # plot each input in its own window
        for j in range(self.n_inputs):
            fig = plt.figure(
                f"{self.figure_label}, input {j}", clear=True, layout="constrained"
            )

            if self.FFT:
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
            else:
                ax1 = fig.add_subplot(111)

            # Will execute if the user has specified signal labels
            if self.input_labels:
                # Plot original data
                for i in range(self.resp.shape[0]):
                    ax1.plot(
                        self.time,
                        self.resp[i, :, j],
                        label=f"{self.input_labels[i]} (Data)",
                    )

                # reset color cycle so that the colors line up
                ax1.set_prop_cycle(None)

                # Plot ERA fit to data
                for i in range(self.resp.shape[0]):
                    ax1.plot(
                        self.time,
                        y[i, :, j],
                        "--",
                        label=f"{self.input_labels[i]} (ERA Fit)",
                    )
                # Legend will appear next to plot
                ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            else:
                # Plot original data
                ax1.plot(self.time, self.resp[:, :, j].T, label="Data")

                # reset color cycle so that the colors line up
                ax1.set_prop_cycle(None)

                # Plot ERA fit to data
                ax1.plot(self.time, y[:, :, j].T, "--", label="ERA fit")

            # Labeling plot
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Response")
            ax1.set_title("Data (solid) vs. ERA Reduced Model (dashed)")

            # Will execute if the user requests an FFT
            if self.FFT:
                # Recovering magnitude, phase, and frequency FFT data
                mag, phase, frq = dsp.fftcoef(
                    self.resp[:, :, j].T, self.sr, axis=0, maxdf=0.2
                )
                # No defined limits will plot entire FFT
                if not self.FFT_range:
                    ax2.plot(frq, mag)
                # In the case of defined limits, they will be processed
                else:
                    # Converts integer/float input to a list for indexing purposes
                    if isinstance(self.FFT_range, numbers.Real):
                        self.FFT_range = [self.FFT_range]
                    # If only one limit is provided, it is taken as a maximum limit
                    if len(self.FFT_range) == 1:
                        maxlim = max(np.where(frq <= self.FFT_range[0])[0])
                        ax2.plot(frq[:maxlim], mag[:maxlim])
                    # If a pair of limits is provided, it will be used
                    # as minimum/maximum limits
                    elif len(self.FFT_range) == 2:
                        minlim = max(np.where(frq <= self.FFT_range[0])[0])
                        maxlim = max(np.where(frq <= self.FFT_range[1])[0])
                        ax2.plot(frq[minlim:maxlim], mag[minlim:maxlim])

                # Labeling FFT plot
                ax2.set_xlabel("Frequency (Hz)")
                ax2.set_ylabel("Magnitude")
                ax2.set_title("Magnitude of Frequency Responses of Data")

            fig.canvas.draw()
        plt.show()

    @staticmethod
    def _get_head_frm(which):
        headers = [
            " ",
            "Mode",
            "Freq (Hz)",
            "% Damp",
            *which,
        ]
        n_indicators = len(which)
        widths = [1, 4, 9, 8] + [6] * n_indicators
        formats = ["{}", "{:4}", "{:9.4f}", "{:8.4f}"] + n_indicators * ["{}{:5.3f}"]
        head, frm = writer.formheader(headers, widths, formats, sep=(0, 1, 2), just="c")
        # delete 1st underline:
        head = "\n".join([" " + line[1:] for line in head.split("\n")])[:-2]
        return head, frm

    def _get_stars(self, mark, which, dct):
        """
        Return dict of lists to highlight auto-selected modes; one #
        modes length list for each indicator
        """
        stars = {}
        if mark:
            all_star = True
            for indicator in which:
                good_modes = (
                    dct["indicators"][indicator] >= self.lower_limits[indicator]
                )
                stars[indicator] = ["*" if j else " " for j in good_modes]
                all_star = good_modes & all_star
            all_star = all_star & (
                (self.damp_range[0] <= dct["zeta"])
                & (dct["zeta"] <= self.damp_range[1])
            )
            all_star = all_star & (
                (self.freq_range[0] <= dct["freqs_hz"])
                & (dct["freqs_hz"] <= self.freq_range[1])
            )
            rec_keep = all_star.nonzero()[0]
            all_star = ["*" if j else " " for j in all_star]
        else:
            all_star = [" "] * (len(dct["eigs"]) // 2)
            for indicator in which:
                stars[indicator] = all_star
            rec_keep = []
        return stars, all_star, rec_keep

    def _mode_print(self, title, dct, mark):
        """
        This function prints a table of descriptive data for each mode
        of interest.

        Parameters
        ----------
        title : string
            Title for table
        dct : dict
            Contains "freq", "zeta", "indicators", "eigs", etc
        mark : boolean
            If True, mark "good" modes with an asterisk ("*")
        """
        # Prints table of model data
        which = [
            "MSV",
            "EMAC",
            "EMACO",
            "MPC",
            "CMIO",
            "MAC",
        ]

        head, frm = self._get_head_frm(which)
        stars, all_star, self.rec_keep = self._get_stars(mark, which, dct)

        n = len(dct["eigs"]) // 2
        args = [all_star, np.arange(1, n + 1), dct["freqs_hz"], 100 * dct["zeta"]]
        for indicator in which:
            args.append(stars[indicator])
            args.append(dct["indicators"][indicator])

        if not self.auto or self.verbose:
            print(f"\n{title}:")
            print(head)
            writer.vecwrite(sys.stdout, frm, *args)

    def _show_model(self, title, *, mark, saved_model=False):
        dct = self._saved if saved_model else self.__dict__
        self._mode_print(title, dct, mark)
        self._compute_era_fit(dct["A_modal"], dct["B_modal"], dct["C_modal"])

    def _save_model_data(self):
        """
        Saves model data so looping can return to beginning
        """
        names = (
            "freqs_hz",
            "zeta",
            "indicators",
            "A_modal",
            "B_modal",
            "C_modal",
            "eigs",
            "eigs_continuous",
            "psi",
            "psi_inv",
        )
        self._saved = {name: getattr(self, name) for name in names}

    def _mode_select(self):
        """
        This routine isolates chosen modes and recalculates their
        contribution to the overall response after isolation. Can loop
        over the data as many times as needed, or if set to auto will
        function automatically without user input.

        Notes
        -----
        Prompts user to select modes of interest. Recommended modes
        (determined from indicator values) will be marked with an *
        symbol.
        """
        self._show_model("Current fit includes all modes", mark=True)
        self._save_model_data()

        # Runs interactive version of mode selection process
        if not self.auto:
            done = "s"  # "s" for "select different modes"
            while done == "s":
                input_string = (
                    input(
                        "\nSelect modes for ERA fit (to keep or remove) [*]:\n"
                        "\t* = keep marked modes\n"
                        "\ta = keep all modes\n"
                        "\tn = remove noise (via `svd_tol` setting; experimental)\n\n"
                        "\tor: mode #s separated by space and/or comma (eg: 1, 2, 5-8),"
                        " or\n\t    array_like Python expression\n"
                    )
                    .strip(", ")
                    .lower()
                )

                if input_string in ("*", ""):
                    selected_modes = np.array(self.rec_keep)
                elif input_string == "a":
                    selected_modes = np.arange(len(self._saved["freqs_hz"]))
                elif input_string == "n":
                    self._remove_noise()
                    self._show_model(
                        "Fit with all modes after removing noise", mark=True
                    )
                    self._save_model_data()
                    done = "s"
                    continue
                else:
                    selected_modes = _string_to_ints2(input_string)

                # Reducing model using selected modes
                self._trim_2_selected_modes(selected_modes, saved_model=True)
                self._show_model("Reduced model fit", mark=False)

                done = input(
                    "\nDone? (d/s/r) [D]:\n"
                    "\td = Done\n"
                    "\ts = Select different modes\n"
                    "\tr = Remove currently selected modes from input"
                    " data and start over\n"
                ).lower()

                if done == "s":
                    # restart mode selection:
                    self._show_model("Current fit", mark=True, saved_model=True)
                elif done == "r":
                    # remove current fit, then restart mode selection:
                    # self._remove_fit()
                    self._remove_fit2(len(selected_modes))
                    self._show_model("Model data after removing fit", mark=True)
                    self._save_model_data()
                    done = "s"

        else:
            # automatically keep recommended modes
            selected_modes = np.array(self.rec_keep, dtype=int)
            self._trim_2_selected_modes(selected_modes)
            self._show_model("Auto-selected modes fit", mark=False)


def NExT(
    resp,
    sr,
    *,
    ref_channels="all",
    lag_start=1,
    lag_stop,
    domain="time",
    unbiased=True,
    demean=True,
    nperseg=None,
    window="hann",
    noverlap=None,
):
    """
    Use NExT method to prepare measurement data for :class:`ERA`

    **BETA**

    This routine uses the "Natural Excitation Technique" to process
    response measurements into free-decay-like signals using auto and
    cross correlations. This can be done in either the time-domain or
    the frequency-domain. The resulting signals can then be input to
    ERA (the Eigensystem Realization Algorithm) for system
    identification. See references [#nxt1]_ and [#nxt2]_.

    Parameters
    ----------
    resp : 1d or 2d ndarray
        The response signal(s) to process. If 2d, each row is a response
        measurement. This data is expected to represent ambient
        vibration responses over a significant period of time. Longer
        measurements will result in more accurate results.
    sr : scalar
        Sample rate at which `resp` was sampled.
    ref_channels : integer, integer array_like, or "all"; optional
        If integer or integer array_like, specifies row(s) of `resp`
        to use as the reference channel(s) for the cross-correlations
        (starts at 0). Set to "all" to use all rows as reference (the
        default). Note that the auto-correlation of each reference
        channel will be included in the output.
    lag_start : integer; optional
        Cross-correlation lag for start of decay. This routine will
        compute auto-correlations, and for those, any noise present in
        the signal will be included in the 0 lag term. Specifically,
        the noise will be squared and summed, as if it is perfectly
        correlated. Therefore, the default `lag_start` is set to 1 to
        avoid including perfectly correlated noise in the decay
        signal.
    lag_stop : integer
        Cross-correlation lag for end of decay. It is recommended to
        set this to the minimum viable value based on the lowest
        frequency you are interested in for the current run. For
        example, an initial value for `lag_stop` might be computed
        such that the decay will have time for one full cycle of the
        estimated lowest frequency (in Hz)::

            lag_stop = lag_start + sr / lowest_est_freq

    domain : string
        Either "time" or "frequency" to specify how this routine is to
        compute the cross-correlations. If "time",
        :func:`scipy.signal.correlate` is used to compute the
        correlations using the entire time history for each
        cross-correlation. If "frequency", :func:`scipy.fft.ifft` and
        :func:`scipy.signal.csd` are used together to compute the
        correlations, along with looping and windowing and
        averaging. If `domain` is "frequency", see the inputs
        `nperseg`, `window`, and `noverlap`.

        .. note::
           To get the frequency domain method to give identical
           results to the time domain method, set `nperseg` to the
           entire signal and `window` to "boxcar".

    unbiased : bool; optional
        If True, use unbiased estimates for the correlations. This
        should be True for ERA.
    demean : bool; optional
        If True (the default), subtract the mean from each signal
        prior to any other processing. Auto and cross correlations
        computed with demeaned signals are often called auto and cross
        covariances.
    nperseg : integer
        Required if `domain` is "frequency". Specifies the window size
        to pass to :func:`scipy.signal.csd`. Generally, using higher
        values for `nperseg` will result in higher quality
        correlations. In order to get any benefit that might come from
        averaging multiple windows, a reasonable initial setting for
        this value might be to choose the first power-of-2 less than N
        / 2 where N is the number of time-steps. That would result in
        the largest power-of-2 size window size (for fast processing)
        while still giving you 3 windows to average.
    window, noverlap : optional
        These inputs are only used if `domain` is "frequency". The are
        all passed to :func:`scipy.signal.csd`. Note that the ``nfft``
        input is supplied by this routine and is set to ``2 *
        nperseg``.

    Returns
    -------
    3d ndarray
        Decay-like responses shaped as expected by :class:`ERA`:
        ``n_channels x n_decay_tsteps x n_ref_channels``. :class:`ERA`
        will interpret the number of reference channels as the number
        of inputs.

    References
    ----------
    .. [#nxt1] James, G.H., Carne, T.G., and Lauffer, J.P. "The
               Natural Excitation Technique (NExT) for Modal Parameter
               Extraction From Operating Wind Turbines", Sandia
               National Laboratories, Albuquerque, NM and Livermore,
               CA, SAND92-1666 (1993).

    .. [#nxt2] James, G.H., Carne, T.G., and Lauffer, J.P. "The
               Natural Excitation Technique (NExT) for Modal Parameter
               Extraction from Operating Structures", Modal Analysis
               10:260–277 (1995).

    Examples
    --------
    To generate synthetic ambient vibrations of a simple structure,
    the same system that was demoed in :class:`ERA` is excited by
    random forces over 100 seconds. The responses are fed into NExT
    and then into :class:`ERA`.

    .. plot::
        :context: close-figs

        We have the following mass, damping and stiffness matrices
        (note that the damping has been specially defined to be
        diagonalized by the modes):

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import scipy.linalg as la
        >>> from pyyeti import era, ode
        >>> np.set_printoptions(precision=5, suppress=True)
        >>>
        >>> M = np.identity(3)
        >>>
        >>> K = np.array(
        ...     [
        ...         [4185.1498, 576.6947, 3646.8923],
        ...         [576.6947, 2104.9252, -28.0450],
        ...         [3646.8923, -28.0450, 3451.5583],
        ...     ]
        ... )
        >>>
        >>> D = np.array(
        ...     [
        ...         [4.96765646, 0.97182432, 4.0162425],
        ...         [0.97182432, 6.71403672, -0.86138453],
        ...         [4.0162425, -0.86138453, 4.28850828],
        ...     ]
        ... )

        Since we have the system matrices, we can determine the
        frequencies and damping using the eigensolution.

        >>> (w2, phi) = la.eigh(K, M)
        >>> omega = np.sqrt(w2)
        >>> freq_hz = omega / (2 * np.pi)
        >>> freq_hz
        array([  1.33603,   7.39083,  13.79671])
        >>> modal_damping = phi.T @ D @ phi
        >>> modal_damping
        array([[ 0.33578, -0.     , -0.     ],
               [-0.     ,  6.9657 ,  0.     ],
               [-0.     ,  0.     ,  8.66873]])
        >>> zeta = np.diag(modal_damping) / (2 * omega)
        >>> zeta
        array([ 0.02 ,  0.075,  0.05 ])

        Simulate ambient vibrations over 100 seconds:

        >>> dt = 0.01
        >>> t = np.arange(0, 100, dt)
        >>> n = len(t)
        >>> sr = 1 / dt
        >>>
        >>> # for repeatability, create a specific generator:
        >>> from numpy.random import Generator, PCG64DXSM
        >>> rng = Generator(PCG64DXSM(seed=1))
        >>> F = rng.normal(size=(3, len(t)))
        >>> ts = ode.SolveUnc(M, D, K, dt)
        >>> sol = ts.tsolve(force=F)

        Before calling ERA, we'll plot some auto and cross
        correlations directly from NExT. These should have the same
        properties as impulse response functions:

        >>> lag_stop = 200
        >>> t_irf = t[1 : lag_stop + 1]
        >>> irf = era.NExT(sol.a, sr, lag_stop=lag_stop)
        >>>
        >>> fig = plt.figure("corr comp", clear=True, figsize=(6.4, 6.4),
        ...                  layout='constrained')
        >>> ax = fig.subplots(4, 1)
        >>>
        >>> _ = ax[0].plot(t, sol.a.T)
        >>> _ = ax[0].set_title("Original data")
        >>> _ = ax[0].set_ylabel("Acce")
        >>>
        >>> # loop over all 3 ref channels:
        >>> for ref in range(irf.shape[2]):
        ...     # loop over all 3 output channels:
        ...     for out in range(irf.shape[0]):
        ...         _ = ax[ref + 1].plot(t_irf, irf[out, :, ref])
        ...     _ = ax[ref + 1].set_title(f"IRFs with reference channel {ref}")
        ...     _ = ax[ref + 1].set_ylabel("X-corr IRFs")
        >>> _ = ax[3].set_xlabel("Time (s)")

    .. plot::
        :context: close-figs

        Use the time-domain method of NExT and call :class:`ERA` to
        get estimates of frequency, damping, and mode-shapes. We'll
        use a `lag_stop` of 75 (from sr / lowest_freq = 100 / 1.33 ~=
        75).

        >>> era_fit = era.ERA(
        ...     era.NExT(sol.a, sr, lag_stop=75),
        ...     sr=1 / dt,
        ...     auto=True,
        ...     input_labels=["x", "y", "z"],
        ...     figure_label="Time Domain ERA Fit",
        ... )
        <BLANKLINE>
        Current fit includes all modes:
          Mode  Freq (Hz)   % Damp    MSV     EMAC   EMACO    MPC     CMIO    MAC
          ----  ---------  --------  ------  ------  ------  ------  ------  ------
        *    1     1.3307    1.5730  *0.759  *0.996  *0.997  *1.000  *0.997  *1.000
        *    2     7.4842    6.4077  *0.703  *0.830  *0.928  *1.000  *0.928  *1.000
        *    3    13.6264    4.7306  *1.000  *0.834  *0.928  *0.996  *0.925  *1.000
             4    14.6694    3.7619  *0.433   0.330  *0.591  *0.939  *0.555  *0.986
        <BLANKLINE>
        Auto-selected modes fit:
          Mode  Freq (Hz)   % Damp    MSV     EMAC   EMACO    MPC     CMIO    MAC
          ----  ---------  --------  ------  ------  ------  ------  ------  ------
             1     1.3307    1.5730   0.759   0.996   0.997   1.000   0.997   1.000
             2     7.4842    6.4077   0.703   0.830   0.928   1.000   0.928   1.000
             3    13.6264    4.7306   1.000   0.834   0.928   0.996   0.925   1.000

    .. plot::
        :context: close-figs

        That seemed to work out pretty well! Now we'll try the
        frequency-domain method of NExT (the increased `svd_tol` gave
        slightly better results in this case):

        >>> era_fit = era.ERA(
        ...     era.NExT(
        ...         sol.a,
        ...         sr,
        ...         lag_stop=75,
        ...         domain="frequency",
        ...         nperseg=4096,
        ...     ),
        ...     sr=1 / dt,
        ...     svd_tol=0.1,
        ...     auto=True,
        ...     input_labels=["x", "y", "z"],
        ...     figure_label="Frequency Domain ERA Fit",
        ... )
        <BLANKLINE>
        Current fit includes all modes:
          Mode  Freq (Hz)   % Damp    MSV     EMAC   EMACO    MPC     CMIO    MAC
          ----  ---------  --------  ------  ------  ------  ------  ------  ------
        *    1     1.3255    1.4124  *0.709  *0.994  *0.998  *1.000  *0.998  *1.000
        *    2     7.4412    7.3792  *0.595  *0.750  *0.874  *0.998  *0.872  *1.000
        *    3    13.7890    4.9684  *1.000  *0.499  *0.714  *1.000  *0.714  *0.999
        <BLANKLINE>
        Auto-selected modes fit:
          Mode  Freq (Hz)   % Damp    MSV     EMAC   EMACO    MPC     CMIO    MAC
          ----  ---------  --------  ------  ------  ------  ------  ------  ------
             1     1.3255    1.4124   0.709   0.994   0.998   1.000   0.998   1.000
             2     7.4412    7.3792   0.595   0.750   0.874   0.998   0.872   1.000
             3    13.7890    4.9684   1.000   0.499   0.714   1.000   0.714   0.999

    .. plot::
        :context: close-figs

        Both of the above examples used all channels as reference
        channels. The next example uses just the first measurement as
        the reference channel (which also means that the only
        auto-correlation that will be included will be for the first
        measurement):

        >>> era_fit = era.ERA(
        ...     era.NExT(sol.a, sr, lag_stop=75, ref_channels=0),
        ...     sr=1 / dt,
        ...     auto=True,
        ...     input_labels=["x", "y", "z"],
        ...     figure_label="Time Domain ERA Fit",
        ... )
        <BLANKLINE>
        Current fit includes all modes:
          Mode  Freq (Hz)   % Damp    MSV     EMAC   EMACO    MPC     CMIO    MAC
          ----  ---------  --------  ------  ------  ------  ------  ------  ------
        *    1     1.3309    1.6375  *0.951  *0.992  *0.993  *1.000  *0.993  *1.000
        *    2    13.7399    5.1329  *1.000  *0.562  *0.750  *0.998  *0.749  *1.000
        <BLANKLINE>
        Auto-selected modes fit:
          Mode  Freq (Hz)   % Damp    MSV     EMAC   EMACO    MPC     CMIO    MAC
          ----  ---------  --------  ------  ------  ------  ------  ------  ------
             1     1.3309    1.6375   0.951   0.992   0.993   1.000   0.993   1.000
             2    13.7399    5.1329   1.000   0.562   0.750   0.998   0.749   1.000

    In that case, the 7.4 Hz mode was not identified. From the plot, a
    decent fit was found without it since the 7.x Hz content is not
    visibly prominent. To successfully identify that mode, it would be
    best to add more reference channels.
    """
    resp = np.atleast_2d(resp)
    if demean:
        resp = resp - resp.mean(axis=1, keepdims=True)
    n = resp.shape[1]

    if ref_channels == "all":
        ref_channels = np.arange(0, resp.shape[0])
    else:
        ref_channels = np.atleast_1d(ref_channels)

    lags = slice(lag_start, lag_stop + 1)

    # compute cross-correlation functions
    xc = []
    if domain == "time":
        for ref in resp[ref_channels]:
            xc_ref = []
            for sig in resp:
                corr = signal.correlate(sig, ref, mode="full")[n - 1 :]
                xc_ref.append(corr[lags])
            xc.append(xc_ref)

        if unbiased:
            scale = np.arange(n, 0, -1)[lags]
            xc = np.array(xc) / scale

    elif domain == "frequency":
        # csd will scale values down by window.sum() ** 2 ... we don't
        # need or necessarily want that, so scaling up by nperseg ** 2
        # is reasonable (as if window == "boxcar"):
        sc = nperseg**2
        for ref in resp[ref_channels]:
            xc_ref = []
            for sig in resp:
                f, csd = signal.csd(
                    ref,
                    sig,
                    fs=sr,
                    window=window,
                    nperseg=nperseg,
                    nfft=nperseg * 2,
                    noverlap=noverlap,
                    scaling="spectrum",
                    return_onesided=False,
                )
                corr = fft.ifft(csd).real
                xc_ref.append(corr[lags] * sc)
            xc.append(xc_ref)

        if unbiased:
            scale = np.arange(nperseg, 0, -1)[lags]
            xc = np.array(xc) / scale

    else:
        raise ValueError(
            "invalid value for `domain`; must be either 'time' or 'frequency'"
        )

    # put reference channels last instead of first for ERA:
    return np.moveaxis(xc, 0, -1)
