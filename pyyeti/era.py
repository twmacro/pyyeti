# -*- coding: utf-8 -*-
"""
The Eigensystem Realization Algorithm

Identify mode shapes, frequencies, and damping from free-decay
measurement data.
"""
# Python ERA class written by Natalie Hintz and Tim Widrick.

import re
import numpy as np
import scipy.linalg as la
from pyyeti import ode, dsp
import matplotlib.pyplot as plt
import numbers
from warnings import warn


def _string_to_ints(string):
    """
    Convert "1, 2 5" to np.array([0, 1, 4])
    """
    return np.array([int(i) - 1 for i in re.split("[, ]+", string)])


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

    References
    ----------
    .. [#era1] Juang, Jer-Nan. Applied System Identification. United
               Kingdom: Prentice Hall, 1994.

    Examples
    --------
    The following example demonstrates ERA by comparing to a known
    system.

    .. plot::
        :context: close-figs

        Say we had the following mass, damping and stiffness:

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
        frequencies and damping using the eigensolution. We'll see
        that the damping is diagonalized (by design):

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
        >>> ts = ode.SolveExp2(M, D, K, dt)
        >>> sol = ts.tsolve(force=F, v0=[1, 1, 1])

        We can use displacement, velocity or acceleration as the input
        for ERA. Here, we'll use velocity because all modes are well
        represented in the response for the one second simulation.
        Displacement favors the lower frequency modes (so ERA might
        eliminate the highest frequency mode) and acceleration favors
        the higher frequency modes (so ERA might eliminate the lowest
        frequency mode). For those new to ERA, it is a very good
        exercise to experiment with using each of them.

        >>> era_fit = era.ERA(
        ...     sol.v,
        ...     sr=1 / dt,
        ...     auto=True,
        ...     input_labels=["x", "y", "z"],
        ...     FFT=True,
        ...     FFT_range=30.0,
        ... )
        <BLANKLINE>
        Current fit includes all modes:
          Mode   Freq. (Hz)         Zeta             MAC             MSV
          -----------------------------------------------------------------
        *    1     1.33603         0.02000         1.00000         0.81700
        *    2     7.39083         0.07500         1.00000         0.83895
        *    3    13.79671         0.05000         1.00000         1.00000
        <BLANKLINE>
        Auto-selected modes fit:
          Mode   Freq. (Hz)         Zeta             MAC             MSV
          -----------------------------------------------------------------
             1     1.33603         0.02000         1.00000         0.81700
             2     7.39083         0.07500         1.00000         0.83895
             3    13.79671         0.05000         1.00000         1.00000

        Compare frequencies and damping:

        >>> np.allclose(era_fit.freqs_hz, freq_hz)
        True
        >>> np.allclose(era_fit.zeta, zeta)
        True
    """

    def __init__(
        self,
        resp,
        sr,
        svd_tol=0.01,
        auto=False,
        MAC_lower_limit=0.95,
        MSV_lower_limit=0.35,
        damp_range=(0.001, 0.999),
        t0=0.0,
        show_plot=True,
        input_labels=None,
        FFT=False,
        FFT_range=None,
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
            change, you have to rerun. The secondary control is the
            selection of modes which can be interactive (if `auto` is
            False) or automatic according to the parameters below.
        auto : boolean; optional
            Enables automatic selection of true modes. The default is
            False.
        MAC_lower_limit : scalar; optional
            The lower limit for the "modal amplitude coherence" value
            for modes to be selected when `auto` is True. The MAC
            value ranges from 0.0 to 1.0 for each mode and is a
            temporal measure indicating the importance of the mode
            over time to the fit. Higher values indicate more
            importance. The MAC value for a mode is the dot product of
            two vectors:

                1. The ideal, reconstructed time history for a mode
                2. The time history extracted from the input data for
                   the mode after discarding noisy data via singular
                   value decomposition

        MSV_lower_limit : scalar; optional
            The lower limit for the normalized "mode singular value"
            for modes to be selected when `auto` is True. The MSV
            value ranges from 0.0 to 1.0 for each mode and is a
            measure of contribution to the response. Larger values
            represent larger contribution.
        damp_range : 2-tuple; optional
            Specifies the range (inclusive) of acceptable damping
            values for automatic selection.
        t0 : scalar; optional
            The initial time value in `resp`. Only used for plotting.
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

        Notes
        -----
        The class instance is populated with the following members:

        ================  ============================================
        Member            Description
        ================  ============================================
        resp              impulse response data
        resp_era          final ERA fit to the impulse response data
        n_outputs         number of outputs
        n_tsteps          number of time-steps
        n_inputs          number of inputs
        sr                sample rate
        t0                initial time value in `resp` (for plotting)
        time              `resp` time vector used for plotting;
                          created from `sr` and `t0`
        svd_tol           singular value tolerance
        auto              automatic or interactive method of
                          identifying true modes
        show_plot         if True, plots are made
        input_labels      create a legend for approximated fit plot
        MAC_lower_limit   MAC lower limit for auto-selecting modes
        MSV_lower_limit   normalized MSV lower limit for auto-
                          selecting modes
        damp_range        range of acceptable damping values for
                          automatic selection
        FFT               produces an FFT plot of input data for
                          comparison to detected modes
        FFT_range         frequency cutoff(s) for FFT plot
        H_0               the H(0) generalized Hankel matrix (eq 5.24)
        H_1               the H(1) generalized Hankel matrix (eq 5.24)
        A_hat             ERA state-space "A" matrix, (eq 5.34)
        B_hat             ERA state-space "B" matrix, (eq 5.34)
        C_hat             ERA state-space "C" matrix, (eq 5.34)
        A_modal           ERA modal state-space "A" matrix
        B_modal           ERA modal state-space "B" matrix
        C_modal           ERA modal state-space "C" matrix
        eigs              complex eigenvalues of `A_hat` (discrete)
        eigs_continuous   continuous version of `eigs`; ln(eigs) * sr
        psi               complex eigenvectors of `A_hat`
        psi_inv           inverse of `psi`
        freqs             mode pair frequencies (rad/sec)
        freqs_hz          mode pair frequencies (Hz)
        zeta              mode pair percent critical damping
        MAC               MAC values for each mode pair (eq 5.49)
        MSV               MSV values for each mode pair (eq 5.50)
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
        self.MAC_lower_limit = MAC_lower_limit
        self.MSV_lower_limit = MSV_lower_limit
        self.damp_range = damp_range
        self.show_plot = show_plot
        self.input_labels = input_labels
        self.FFT = FFT
        self.FFT_range = FFT_range
        self.t0 = t0

        self._H_generate()
        self._state_space()
        self._conv_2_modal()
        self._mode_select()

    def _H_generate(self):
        """
        Given Markov parameters, will generate the system's generalized
        Hankel matrix and time-shifted generalized Hankel matrix.
        """
        # Determining the alpha and beta parameters in order to
        # maximize coverage of data in Hankel matrices
        alpha = self.n_tsteps // 2
        beta = self.n_tsteps - alpha
        H_dim = (alpha * self.n_outputs, beta * self.n_inputs)

        # Forming Hankel matrices
        self.H_0 = np.empty(H_dim)
        self.H_1 = np.empty(H_dim)

        # Reshaping Markov parameters into form Y = [Y0, Y1, Y2, ..., Yn]
        # Each Y[i] is of shape m x r
        Markov = self.resp.reshape(self.n_outputs, -1)
        rows = np.arange(self.n_outputs)
        cols = np.arange(beta * self.n_inputs)

        for _ in range(alpha):
            # Using block indexing to fill Hankel matrices with Markov
            # parameters
            self.H_0[rows] = Markov[:, cols]
            self.H_1[rows] = Markov[:, cols + self.n_inputs]  # Time shift
            # Advancing row and column indices
            rows += self.n_outputs
            cols += self.n_inputs

    def _state_space(self):
        """
        This function computes a state-space representation of a
        system given its Markov parameters. It transforms the
        parameters into a Hankel matrix which is decomposed using
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

    def _conv_2_modal(self):
        """
        This routine converts the system realization problem into
        modal space.
        """
        # Generating modified state vectors and system order
        A_hat, B_hat, C_hat = self.A_hat, self.B_hat, self.C_hat

        # Generating eigenvalues and matrix of eigenvectors
        self.eigs, self.psi = la.eig(A_hat)

        # Retrieving sorted index of eigenvalues based on magnitude of
        # imaginary components
        #  - Using a stable sorter to allow for preservation of
        #    overdamped modes (in case they're consecutive; if not,
        #    see the special note in the documentation for class ERA)
        i = np.argsort(abs(self.eigs.imag), kind="stable")
        self.eigs = self.eigs[i]
        self.psi = self.psi[:, i]

        self.eigs_continuous = np.log(self.eigs) * self.sr
        self.freqs, self.zeta = ode.get_freq_damping(
            self.eigs_continuous, suppress_warning=True
        )

        # Finding the order of increasing frequency
        index = np.argsort(self.freqs, kind="stable")

        # Locating overdamped modes and putting them first:
        pv = self.zeta[index] >= 1.0
        index = np.concatenate((index[pv], index[~pv]))

        # Ordering frequencies and damping accordingly
        self.freqs = self.freqs[index]
        self.zeta = self.zeta[index]

        # Deriving ordered indexing for mode pairs
        index_pairs = np.empty(index.size * 2, dtype=int)
        for i in range(index.size):
            index_pairs[2 * i] = index[i] * 2
            index_pairs[2 * i + 1] = index[i] * 2 + 1

        # Re-ordering eigenvalues and eigenvectors
        self.eigs = self.eigs[index_pairs]
        self.psi = self.psi[:, index_pairs]

        # Deriving the frequencies in Hz
        self.freqs_hz = self.freqs / (2 * np.pi)

        # Recovering inverse of eigenvector matrix
        self.psi_inv = np.linalg.inv(self.psi)

        # Compute modal space state-space matrices
        self.A_modal = np.diag(self.eigs)
        self.B_modal = self.psi_inv @ B_hat
        self.C_modal = C_hat @ self.psi

        # Calculate indicators:
        self._MSV()
        self._MAC()

    def _remove_fit(self):
        """
        Remove current ERA fit from the input response history
        """
        self.resp = self.resp - self.resp_era
        self._H_generate()
        self._state_space()
        self._conv_2_modal()

    def _MSV(self):
        """
        This routine calculates the Mode Singular Values (MSV) of each
        detected mode.

        Notes
        -----
        MSV serves as an indicator of true modes, but there is no set
        value which a mode's MSV must achieve in order to be
        considered true. It is taken as a relative indicator and
        should be compared against all other detected modes and taken
        into account with other indicators (e.g. MAC).
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

    def _trim_2_selected_modes(self, selected_modes, saved_model=False):
        """
        This function extracts selected modes and arranges them to be
        presented without excluded modes. It recalculates the modal
        state space matrices as well as MAC and MSV values for each
        extracted mode. This function is responsible for updating
        data with the reduced model based on the modes which are kept
        for consideration.
        """
        if saved_model:
            freq = self._saved["freqs_hz"]
            zeta = self._saved["zeta"]
            eigs = self._saved["eigs"]
            psi = self._saved["psi"]
            psi_inv = self._saved["psi_inv"]
        else:
            freq = self.freqs_hz
            zeta = self.zeta
            eigs = self.eigs
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
        self.psi = psi[:, self.selected_mode_pairs]
        self.psi_inv = psi_inv[self.selected_mode_pairs, :]

        # Producing reduced modal matrices
        self.A_modal = np.diag(self.eigs)
        self.B_modal = self.psi_inv @ self.B_hat
        self.C_modal = self.C_hat @ self.psi

        # Recalculate indicators:
        self._MSV()
        self._MAC()

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
            fig = plt.figure(f"ERA Fit, input {j}", clear=True)

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

            fig.tight_layout()
            fig.canvas.draw()
        plt.show()

    def _mode_print(self, title, freq, zeta, mac, msv, mark):
        """
        This function prints a table of descriptive data for each mode
        of interest.

        Parameters
        ----------
        title : string
            Title for table
        freq : 1d ndarray
            Array of modal frequencies, in Hz.
        zeta : 1d ndarray
            Array of modal damping.
        mac : 2d ndarray
            Array of MAC values corresponding with each mode.
        msv : 2d ndarray
            Array of MSV values corresponding with each mode.
        mark : boolean
            If True, mark "good" modes with an asterisk ("*")
        """
        # Prints table of model data
        print(f"\n{title}:")
        print("  Mode   Freq. (Hz)         Zeta             MAC             MSV")
        print("  -----------------------------------------------------------------")

        self.rec_keep = []
        n = mac.shape[0]
        for i in range(n):
            if mark and (
                mac[i] >= self.MAC_lower_limit
                and msv[i] >= self.MSV_lower_limit
                and self.damp_range[0] <= zeta[i] <= self.damp_range[1]
            ):
                ind = "*"
                self.rec_keep.append(i)
            else:
                ind = " "
            print(
                ("{} {:4}  {:{w}.{p}f}" + "      {:{w}.{p}f}" * 3).format(
                    ind, i + 1, freq[i], zeta[i], mac[i], msv[i], w=10, p=5
                )
            )

    def _show_model(self, title, *, mark, saved_model=False):
        if saved_model:
            self._mode_print(
                title,
                self._saved["freqs_hz"],
                self._saved["zeta"],
                self._saved["MAC"],
                self._saved["MSV"],
                mark,
            )
            self._compute_era_fit(
                self._saved["A_modal"], self._saved["B_modal"], self._saved["C_modal"]
            )
        else:
            self._mode_print(title, self.freqs_hz, self.zeta, self.MAC, self.MSV, mark)
            self._compute_era_fit(self.A_modal, self.B_modal, self.C_modal)

    def _save_model_data(self):
        """
        Saves model data so looping can return to beginning
        """
        names = (
            "freqs_hz",
            "zeta",
            "MAC",
            "MSV",
            "A_modal",
            "B_modal",
            "C_modal",
            "eigs",
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
        (determined from MSV and MAC values) will be marked with an *
        symbol.
        """
        self._show_model("Current fit includes all modes", mark=True)

        # Runs interactive version of mode selection process
        if not self.auto:
            self._save_model_data()
            done = "n"
            while done == "n":
                input_string = (
                    input(
                        "\nSelect modes for ERA fit (to keep or remove):\n"
                        "\t- 'Enter' or '*' to keep marked modes\n"
                        "\t- 'a' to keep all modes\n"
                        "\t- mode #s separated by space and/or comma (eg: 1, 2, 5)\n"
                    )
                    .strip(", ")
                    .lower()
                )

                if input_string in ("*", ""):
                    selected_modes = np.array(self.rec_keep)
                elif input_string == "a":
                    selected_modes = np.arange(len(self._saved["freqs_hz"]))
                else:
                    selected_modes = _string_to_ints(input_string)

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

                if done == "n":
                    # restart mode selection:
                    self._show_model("Current fit", mark=True, saved_model=True)
                elif done == "r":
                    # remove current fit, then restart mode selection:
                    self._remove_fit()
                    self._show_model("Model data after removing fit", mark=True)
                    self._save_model_data()
                    done = "n"

        else:
            # automatically keep recommended modes
            selected_modes = np.array(self.rec_keep, dtype=int)
            self._trim_2_selected_modes(selected_modes)
            self._show_model("Auto-selected modes fit", mark=False)
