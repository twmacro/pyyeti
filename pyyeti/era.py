# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 15:20:05 2020

@author: nhintz
"""
import re
import numpy as np
import scipy.linalg as la
from pyyeti import ode, locate, dsp
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
    Implements the Eigensystem Realization Algorithm (ERA). ERA
    follows minimum realization theory to identify modal parameters
    (natural frequencies, mode shapes, and damping) from impulse
    response (free-decay) data. In the absense of forces, the response
    is completely determined by the system parameters which allows
    modal parameter identification to work. This code is based on
    Chapter 5 of reference [#era1]_.

    The discrete (z-domain) state-space equations are::

           x[k+1] = A*x[k] + B*u[k]
           y[k]   = C*x[k] + D*u[k]

    where ``u[k]`` are the inputs, ``y[k]`` are the outputs, and
    ``x[k]`` is the state vector at time step ``k``. ERA determines
    ``A``, ``B``, and ``C``. ``D`` is discussed below.

    Let there be ``m`` outputs, ``r`` inputs, and ``p`` time steps
    (following the convention in [#era1]_). In ERA, impulse response
    measurements (accelerometer data, for example) forms a
    3-dimensional matrix sized ``m x p x r``. This data can be
    partitioned into ``p`` ``m x r`` matrices. These ``m x r``
    matrices are called Markov parameters. ERA computes a set of
    state-space matrices from these Markov parameters. To begin to see
    a relation between the Markov parameters and the state-space
    matrices, consider the following. Each input ``u[k]`` is assumed
    to be unity impulse for one DOF and zero elsewhere. For example,
    if there are two inputs, ``u0[k]`` and ``u1[k]``, they are assumed
    to be::

        u0 = [ 1.0, 0.0, 0.0, ... 0.0 ]       # r x p
             [ 0.0, 0.0, 0.0, ... 0.0 ]

        u1 = [ 0.0, 0.0, 0.0, ... 0.0 ]       # r x p
             [ 1.0, 0.0, 0.0, ... 0.0 ]

    Letting ``x[0]`` be zeros, iterating through the state-space
    equations yields these definitions for the outputs::

        y0[k] = C @ A ** (k - 1) @ B[:, 0]    # m x 1

        y1[k] = C @ A ** (k - 1) @ B[:, 1]    # m x 1

    Putting these together::

       Y[k] = [ y0[k], y1[k] ]                # m x r

    The ``Y[k]`` matrices are the Markov parameters written as a
    function of the to-be-determined state-space matrices ``A``,
    ``B``, and ``C``. From the above, it can be seen that ``D =
    Y[0]``.

    References
    ----------
    .. [#era1] Juang, Jer-Nan. Applied System Identification. United
               Kingdom: Prentice Hall, 1994.
    """

    def __init__(
        self,
        resp,
        h,
        time_vector,
        tol=0.01,
        auto=False,
        overdamped_modes=True,
        input_labels=[],
        FFT=False,
        FFT_limit=None,
    ):
        """
        Instantiates a :class:`ERA` solver.

        Parameters
        ----------
        resp : 1d, 2d, or 3d ndarray
            Impulse response measurement data. In the general case,
            this is a 3d array sized ``m x p x r``, where ``m`` are
            the number of outputs (measurements), ``p`` is the number
            of time steps, and ``r`` is the number of inputs. If there
            is only one input, then `resp` can be input as a 2d array
            of size ``m x p``.  Further, if there is only one output,
            then `resp` can be input as a 1d array of length `p`.
        h : scalar
            Time step at which data was sampled.
        time_vector : 1d ndarray
            Vector of time during which data was collected.
        tol : scalar, optional
            Determines how many singular values to keep. Can be:
                - Integer greater than 1 to specify number of expected modes
                  (tolerance equal to 2 * expected modes)
                - Float between 0 and 1 to specify required significance of a
                  singular value to be kept
        auto : boolean, optional
            Enables automatic selection of true modes. The default is False.
        overdamped_modes : boolean, optional
            Enables inclusion of overdamped modes. The default is True.
        input_labels : list, optional
            List of data labels for each input signal to ERA.
        FFT : boolean, optional
            Enables display of FFT plot of input data. The default is False.
        FFT_limit : scalar or list, optional
            Limits displayed frequency range of FFT plot. A scalar value or
            list with one term will act as a maximum cutoff. A pair of values
            will act as minimum and maximum cutoffs, respectively.
            The default is None.

        Notes
        -----
        The class instance is populated with the following members:

        ===========   =========================================================
        Member        Description
        ===========   =========================================================
        resp          impulse response data
        h             time step
        tol           tolerance of solver
        time_vector   time in which data is collected
        auto          automatic or interactive method of identifying true modes
        overdamped_   will include overdamped modes if True, else will exclude
        modes         overdamped modes
        input_labels  create a legend for approximated fit plot
        FFT           produces an FFT plot of input data for comparison to
                      detected modes
        FFT_limit     enforces frequency cutoff(s) on FFT plot
        ===========   =========================================================

        The class instance calls following internal functions:

        =============   =======================================================
        Function        Description
        =============   =======================================================
        _H_generate     produces generalized Hankel matrix
        _state_vector   identifies state-space realization of the system
        _conv_2_modal   converts system realization from physical domain into
                        the modal domain
        _mode_select    identifies true modes and removes noise modes, can be
                        an automatic or interactive process
        =============   =======================================================
        """
        self.resp = np.atleast_3d(resp)
        self.h = h
        self.tol = tol
        self.time_vector = time_vector
        self.auto = auto
        self.overdamped_modes = overdamped_modes
        self.input_labels = input_labels
        self.FFT = FFT
        self.FFT_limit = FFT_limit
        self._H_generate()
        self._state_vector()
        self._conv_2_modal()
        self._mode_select()

    def _H_generate(self):
        """
        Given Markov parameters, will generate the system's generalized
        Hankel matrix and time-shifted generalized Hankel matrix.
        """
        resp = self.resp
        self.m, self.p, self.r = resp.shape

        m = self.m  # was p
        p = self.p  # was l
        r = self.r  # was m

        # Determining the alpha and beta parameters in order to
        # maximize coverage of data in Hankel matrices
        alpha = p // 2
        beta = p - alpha
        H_dim = (alpha * m, beta * r)

        # Forming Hankel matrices
        self.H_0 = np.empty(H_dim)
        self.H_1 = np.empty(H_dim)

        # Reshaping Markov parameters into form Y = [Y0, Y1, Y2, ..., Yn]
        # Each Y[i] is of shape m x r
        Markov = resp.reshape(m, -1)
        rows = np.arange(m)
        cols = np.arange(beta * r)

        for _ in range(alpha):
            # Using block indexing to fill Hankel matrices with Markov
            # parameters
            self.H_0[rows] = Markov[:, cols]
            self.H_1[rows] = Markov[:, cols + r]  # Time shift
            # Advancing row and column indices
            rows += m
            cols += r

    def _state_vector(self):
        """
        This function produces the modified state vectors of a system
        given its Markov parameters. It transforms the parameters into
        a Hankel matrix which is decomposed using Singular Value
        Decomposition (SVD) and used to find the state vectors.

        Notes
        -----
        This function determines the system order from the tolerance
        parameter input to the class. If the tolerance yields an odd
        system order value, it will be altered to become even to ensure
        there are an even number of singular values.

        """

        # Decomposing H_0 using SVD
        R, sigma, ST = np.linalg.svd(self.H_0)

        # tol can be treated as a selection of modes (if greater than 1)
        # or tol can be set as a numeric boundary to sort through non-
        # significant singular values
        if self.tol >= 1:
            self.n = self.tol
            if self.n > sigma.size:
                self.n = sigma.size
                warn(
                    "Tolerance is greater than number of singular values.",
                    RuntimeWarning,
                )
        else:
            self.n = np.argwhere(sigma / sigma[0] >= self.tol).size

        # Ensures that number of pairs remains even
        if self.n % 2 != 0:
            self.n += 1

        self.n = int(self.n)

        # Reshaping R, sigma, ST, accordingly
        R = R[:, : self.n]
        sigma_sqrt = np.sqrt(sigma[: self.n])
        ST = ST[: self.n, :]

        # Recovering P and Q matrices
        self.P = R * sigma_sqrt
        self.Q = (sigma_sqrt * ST.T).T

        # Recovering identified state matrices
        self.C_hat = self.P[: self.m, :]
        self.B_hat = self.Q[:, : self.r]
        self.A_hat = (R / sigma_sqrt).T @ self.H_1 @ (ST.T / sigma_sqrt)

    def _conv_2_modal(self):
        """
        This routine converts the system realization problem from
        the time domain into modal space.
        """
        h = self.h

        # Generating modified state vectors and system order
        A_hat, B_hat, C_hat = self.A_hat, self.B_hat, self.C_hat

        # Generating eigenvalues and matrix of eigenvectors
        self.eig_val, self.psi = la.eig(A_hat)

        # Retrieving sorted index of eigenvalues based on magnitude of
        # imaginary components
        # Using a stable sorter to allow for preservation of overdamped modes
        i = np.argsort(abs(self.eig_val.imag), kind="stable")
        self.eig_val = self.eig_val[i]
        self.psi = self.psi[:, i]

        S = np.log(self.eig_val) / h
        self.freqs, self.zeta = ode.get_freq_damping(S)

        # Finding the order of increasing frequency
        index = np.argsort(self.freqs, kind="stable")
        zeta_int = self.zeta[index]
        # Locating overdamped modes
        k = np.ravel(np.argwhere(zeta_int >= 1.0))
        od = index[k]
        # Removing overdamped modes from ordered index and
        # re-entering into the beginning of the ordered index
        index_del = np.delete(index, k)
        self.index_new = np.concatenate((od, index_del))

        # Ordering frequencies and damping accordingly
        self.freqs = self.freqs[self.index_new]
        self.zeta = self.zeta[self.index_new]

        # Deriving ordered indexing for mode pairs
        index_pairs = np.empty(self.index_new.size * 2, dtype=int)
        for i in range(self.index_new.size):
            index_pairs[2 * i] = self.index_new[i] * 2
            index_pairs[2 * i + 1] = self.index_new[i] * 2 + 1

        # Re-ordering eigenvalues and eigenvectors
        self.eig_val = self.eig_val[index_pairs]
        self.psi = self.psi[:, index_pairs]

        # Deriving the frequencies in Hz
        self.freqs_Hz = self.freqs / (2 * np.pi)

        # Recovering inverse of eigenvector matrix
        self.psi_inv = np.linalg.inv(self.psi)

        # Converting descriptive matrices into modal space
        self.A_modal = np.diag(self.eig_val)
        self.B_modal = self.psi_inv @ B_hat
        self.C_modal = C_hat @ self.psi

    def _MSV(self, n, A_m, B_m, C_m):
        """
        This routine calculates the Mode Singular Values (MSV) of each
        detected mode.

        Parameters
        ----------
        n : int
            Order of the system.
        A_m : 2d ndarray
            Modal identified state matrix.
        B_m : 2d ndarray
            Modal identified input matrix.
        C_m : 2d ndarray
            Modal identified output matrix.

        Notes
        -----
        MSV serves as an indicator of true modes, but there is no set
        value which a mode's MSV must achieve in order to be
        considered true. It is taken as a relative indicator and
        should be compared against all other detected modes and taken
        into account with other indicators (e.g. MAC).
        """
        lamd = np.diag(A_m)

        # Will only have 3 values but convenient to loop over whole set
        MSV_int = np.empty((n, 1))  # _int stands for intermediate
        max_lam = self.p // self.r

        for j in np.arange(0, n - 1, 2):
            # Extracting even terms to avoid repetition of calculation
            # over the modes of the problem
            b_vec = B_m[j, :]
            c_vec = C_m[:, j]

            # Need to take square root due to self-multiplication
            b_abs_sq = np.sqrt(abs(b_vec @ np.conj(b_vec.T)))
            c_abs_sq = np.sqrt(abs(c_vec @ np.conj(c_vec.T)))
            # Maximum value we raise lambda to is l - 2
            # More flexible by accounting for m
            lam_abs = abs(lamd[j]) ** np.arange(0, max_lam - 1)

            MSV_int[j] = np.sqrt(c_abs_sq * sum(lam_abs) * b_abs_sq)

        # Extracting unique values as calculated for display
        self.MSV_arr = MSV_int[np.arange(0, n - 1, 2)]

    def _MAC(self, n, B_modal, psi_inv, Q, eig_val):
        """
        This routine calculates the Modal Amplitude Coherence (MAC) of
        each detected mode.

        Parameters
        ----------
        n : int
            System order.
        B_modal : 2d ndarray
            Modal identified input matrix.
        psi_inv : 2d ndarray
            Inverse matrix of pis (eigenvector matrix).
        Q : 2d ndarray
            Full matrix from which B_hat is derived.
        eig_val : 1d ndarray
            Eigenvalues of system.

        """
        # Q_bar is equivalent to vector Q but must be reshaped
        Q_bar = psi_inv @ Q
        # Q_hat will be the same shape as Q_bar
        Q_hat = np.empty(Q_bar.shape, dtype=complex)

        # The length of each element array bi will be equal to m
        bi_len = np.arange(self.r)
        # Filling in Q_hat array
        for i in np.arange(1, Q_bar.shape[1] + 1):
            Q_hat[:, bi_len] = B_modal * (eig_val.reshape(n, 1) ** (i - 1))
            bi_len += self.r

        # Convenient to loop over whole data set and extract unique modes
        MAC_int = np.empty((n, 1))

        # Calculating MAC values
        for i in np.arange(0, n - 1, 2):
            Q_b_con = abs(Q_bar[i, :] @ np.conj(Q_bar[i, :]).T)
            Q_h_con = abs(Q_hat[i, :] @ np.conj(Q_hat[i, :]).T)
            Q_con = abs(Q_bar[i, :] @ np.conj(Q_hat[i, :]).T)
            MAC_int[i] = Q_con / np.sqrt(Q_b_con * Q_h_con)

        # Extracting from MAC_int for unique modes
        self.MAC_arr = MAC_int[np.arange(0, n - 1, 2)]

    def _plot_era(self, n, A_hat, B_hat, C_hat):
        """
        Plots input data against reduced model.
        Parameters
        ----------
        n : int
            System order.
        A_hat : numpy.ndarray
            Identified state matrix.
        B_hat : numpy.ndarray
            Identified input matrix.
        C_hat : numpy.ndarray
            Identified output matrix.

        Notes
        -----
        If the parameter `input_labels` is left empty, the approximate fit
        plot will not have a corresponding legend. Otherwise, the data and
        its approximations will appear with a corresponding legend. If the
        number of input labels provided does not match the number of input
        signals, this will return an error.

        If the parameter `FFT` is set to True, this function will generate
        an FFT plot of the input data in the corresponding color coding of
        the approximate fit plot. This will appear below the approximate fit
        plot and serves as a helpful comparison of detected modal data. If the
        scaling of the FFT plot is suboptimal, it can be adjusted by changing
        the values input to `FFT_limit`, which will serve as a maximum cutoff
        or minimum/maximum cutoff pair, depending on whether one or two values
        are given.

        """
        fig = plt.figure("ERA Fit", clear=True)
        fig.subplots_adjust(top=0.8)

        if self.FFT:
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            fig.tight_layout(pad=3.0)
        else:
            ax1 = fig.add_subplot(111)

        x = np.zeros((n, self.r * self.p), dtype=complex)
        cols = np.arange(0, self.r)
        x[:, cols] = B_hat
        for k in np.arange(0, self.p - 1):
            x[:, cols + self.r] = A_hat @ x[:, cols]
            cols += self.r
        y = np.real(C_hat @ x)

        # Will execute if the user has specified signal labels
        if self.input_labels:
            # Plots original data
            ax1.plot(self.time_vector, self.resp.T, label="(Data)")
            # reset color cycle so that the colors line up
            ax1.set_prop_cycle(None)
            # Plots ERA fit data
            ax1.plot(self.time_vector, y.T, "--", label="(ERA fit)")

            legend = []
            for i in range(2 * len(self.input_labels)):
                # First round should be labeled with (Data)
                if i < len(self.input_labels):
                    legend.append(self.input_labels[i] + " (Data)")
                # Second round should be labeled with (ERA Fit)
                else:
                    legend.append(
                        self.input_labels[i - len(self.input_labels)] + " (ERA Fit)"
                    )
            # Legend will appear next to plot
            ax1.legend(legend, bbox_to_anchor=(1.04, 1), loc="upper left")
        else:
            # Plots original data
            ax1.plot(self.time_vector, self.resp.T, label="Data")
            # reset color cycle so that the colors line up
            ax1.set_prop_cycle(None)
            # Plots ERA fit data
            ax1.plot(self.time_vector, y.T, "--", label="ERA fit")

        # Labeling plot
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Response")
        ax1.set_title("Data (solid) vs. ERA Reduced Model (dashed)")

        # Will execute if the user requests an FFT
        if self.FFT:
            # Recovering magnitude, phase, and frequency FFT data
            mag, phase, frq = dsp.fftcoef(self.resp.T, 1 / self.h, axis=0, maxdf=0.2)
            # No defined limits will plot entire FFT
            if not self.FFT_limit:
                ax2.plot(frq, mag)
            # In the case of defined limits, they will be processed
            else:
                # Converts integer/float input to a list for indexing purposes
                if isinstance(self.FFT_limit, numbers.Real):
                    self.FFT_limit = [self.FFT_limit]
                # If only one limit is provided, it is taken as a maximum limit
                if len(self.FFT_limit) == 1:
                    maxlim = max(np.where(frq <= self.FFT_limit[0])[0])
                    ax2.plot(frq[:maxlim], mag[:maxlim])
                # If a pair of limits is provided, it will be used
                # as minimum/maximum limits
                elif len(self.FFT_limit) == 2:
                    minlim = max(np.where(frq <= self.FFT_limit[0])[0])
                    maxlim = max(np.where(frq <= self.FFT_limit[1])[0])
                    ax2.plot(frq[minlim:maxlim], mag[minlim:maxlim])
            # Labeling FFT plot
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Magnitude")
            ax2.set_title("Magnitude of Frequency Responses of Data")

        fig.canvas.draw()
        plt.show()

    def _mode_removal(self, n):
        """
        This function removes modes which the user indicates to be
        the product of noise. This function is only called if the program
        runs interactively.

        Parameters
        ----------
        n : int
            Number of modes.

        """
        # Prompts user for modes which should be removed and removes
        # them from the data

        print(
            "\nSelect the modes you would like to remove (or use '*' to keep "
            "just the marked modes)."
        )
        mode_r = input("Separate your choices with a space and/or comma:\n").strip(", ")

        if mode_r == "*":
            self.kept_modes = np.array(self.rec_keep)
        else:
            delete_modes = _string_to_ints(mode_r)
            self.kept_modes = np.sort(locate.flippv(delete_modes, n // 2))

        # Updates data with new reduced model
        self._mode_sort(
            self.kept_modes,
            self.freqs_Hz,
            self.zeta,
            self.eig_val,
            self.psi,
            self.psi_inv,
        )
        print("\nReduced model data:")
        self._mode_print(self.n, self.freqs_Hz, self.zeta, self.MAC_arr, self.MSV_arr)

        self._plot_era(self.n * 2, self.A_modal, self.B_modal, self.C_modal)

    def _mode_sort(self, selected_modes, freq, zeta, eig_val, psi, psi_inv):
        """
        This function extracts selected modes and arranges them to be
        presented without excluded modes. It recalculates the modal state
        space matrices as well as MAC and MSV values for each extracted mode.
        This function is responsible for updating data with the reduced model
        based on the modes which are kept for consideration.

        Parameters
        ----------
        selected_modes : 1d ndarray
            Array of modes selected for isolation/consideration.
        freq : 1d ndarray
            Array of modal frequencies (in Hz).
        zeta : 1d ndarray
            Array of modal dampings.
        eig_val : 1d ndarray
            Array of system eigenvalues.
        psi : 2d ndarray
            Matrix of eigenvectors.
        psi_inv : 2d ndarray
            Inverse of eigenvector matrix.

        """

        # Identifying number of selected modes
        self.n = selected_modes.size
        # Identifying position of modal pairs
        self.selected_mode_pairs = []
        for i in selected_modes:
            self.selected_mode_pairs.append(2 * i)
            self.selected_mode_pairs.append(2 * i + 1)

        # Extracting reduced data
        self.freqs_Hz = freq[selected_modes]
        self.zeta = zeta[selected_modes]
        self.eig_val = eig_val[self.selected_mode_pairs]
        self.psi = psi[:, self.selected_mode_pairs]
        self.psi_inv = psi_inv[self.selected_mode_pairs, :]

        # Producing reduced modal matrices
        self.A_modal = np.diag(self.eig_val)
        self.B_modal = self.psi_inv @ self.B_hat
        self.C_modal = self.C_hat @ self.psi

        # Recalculating MSV and MAC of new modal selections
        self._MSV(self.n * 2, self.A_modal, self.B_modal, self.C_modal)
        self._MAC(self.n * 2, self.B_modal, self.psi_inv, self.Q, self.eig_val)

    def _mode_print(self, n, freq, zeta, mac, msv):
        """
        This function prints a table of descriptive data for each mode of
        interest.

        Parameters
        ----------
        n : int
            Number of modes.
        freq : 1d ndarray
            Array of modal frequencies, in Hz.
        zeta : 1d ndarray
            Array of modal damping.
        mac : 2d ndarray
            Array of MAC values corresponding with each mode.
        msv : 2d ndarray
            Array of MSV values corresponding with each mode.

        """
        # Prints table of model data
        print(f"  Mode   Freq. (Hz)         Zeta             MAC             MSV")
        print("  -----------------------------------------------------------------")

        self.rec_keep = []
        for i in np.arange(n):
            if mac[i] >= 0.95 and msv[i] >= 2.5:
                ind = "*"
                self.rec_keep.append(i)
            else:
                ind = " "
            print(
                "{} {}\t{:{w}.{p}f}\t{:{w}.{p}f}\t{:{w}.{p}f}\t{:{w}.{p}f}".format(
                    ind, i + 1, freq[i], zeta[i], mac[i][0], msv[i][0], w=10, p=5
                )
            )

    def _mode_select(self):
        """
        This routine isolates chosen modes and recalculates their contribution
        to the overall response after isolation. Can loop over the data as many
        times as needed, or if set to auto will function automatically without
        user input.

        Notes
        -----
        Prompts user to select modes of interest.
        Recommended modes (determined from MSV and MAC values) will
        be marked with an * symbol. If no modes are selected, those
        modes will be auto-selected.

        """
        # If overdamped modes are accepted, they will remain in the data
        if self.overdamped_modes:
            self._MSV(self.n, self.A_modal, self.B_modal, self.C_modal)
            self._MAC(self.n, self.B_modal, self.psi_inv, self.Q, self.eig_val)
            self._plot_era(self.n, self.A_hat, self.B_hat, self.C_hat)
        # If overdamped modes are rejected, they will be removed from the data
        else:
            od_mode_pairs = np.count_nonzero(self.eig_val.imag == 0) // 2
            ud_modes = np.arange(od_mode_pairs, self.freqs_Hz.size)
            self._mode_sort(
                ud_modes, self.freqs_Hz, self.zeta, self.eig_val, self.psi, self.psi_inv
            )
            # Update system order
            self.n *= 2

            self._plot_era(self.n, self.A_modal, self.B_modal, self.C_modal)

        # Plotting full set of detected modes for comparison
        print("Model data:")
        # Printing ERA specs and identified modes
        print(f"Hankel matrix H(0) =\n{self.H_0}\n\n")
        print("System order after singular value decomposition:")
        print(f"{self.n}\n\n")
        print(f"System eigenvalues:\n{self.eig_val}\n\n")

        # Printing table of model data
        self._mode_print(
            self.n // 2, self.freqs_Hz, self.zeta, self.MAC_arr, self.MSV_arr
        )

        # Runs interactive version of mode selection process
        if not self.auto:
            remove = input(
                "Would you like to remove any of the detected modes"
                " before continuing? (y/n) [N]: "
            ).lower()
            if remove == "y":
                self._mode_removal(self.n)
            else:
                self.n = self.n // 2

            self.n_in = self.n
            self.freqs_Hz_in = self.freqs_Hz
            self.zeta_in = self.zeta
            self.MAC_arr_in = self.MAC_arr
            self.MSV_arr_in = self.MSV_arr
            self.A_m_in = self.A_modal
            self.B_m_in = self.B_modal
            self.C_m_in = self.C_modal
            self.eig_val_in = self.eig_val
            self.psi_in = self.psi
            self.psi_inv_in = self.psi_inv

            cont = "y"
            while cont == "y":
                # Prompting for mode selection
                input_string = (
                    input(
                        "\nSelect the modes you would like to include for analysis:\n"
                        "\t- 'Enter' or '*' to selected highlighted modes\n"
                        "\t- 'a' for all modes\n"
                        "\t- mode #s separated by space and/or comma:\n"
                    )
                    .strip(", ")
                    .lower()
                )

                if input_string == "a":
                    selected_modes = np.arange(len(self.freqs_Hz_in))
                elif input_string in ("*", ""):
                    selected_modes = np.array(self.rec_keep)
                else:
                    selected_modes = _string_to_ints(input_string)

                # Reducing model using selected modes
                self._mode_sort(
                    selected_modes,
                    self.freqs_Hz_in,
                    self.zeta_in,
                    self.eig_val_in,
                    self.psi_in,
                    self.psi_inv_in,
                )

                # Printing and plotting reduced model
                print("\nReduced model data:")
                self._mode_print(
                    self.n, self.freqs_Hz, self.zeta, self.MAC_arr, self.MSV_arr
                )

                self._plot_era(self.n * 2, self.A_modal, self.B_modal, self.C_modal)

                # Prompts user whether to continue loop
                cont = input(
                    "Would you like to isolate different modes? (y/n) [N]: "
                ).lower()

                if cont == "y":
                    # Prints initial data as operating menu
                    print("\nInitial model data:")
                    self._mode_print(
                        self.n_in,
                        self.freqs_Hz_in,
                        self.zeta_in,
                        self.MAC_arr_in,
                        self.MSV_arr_in,
                    )

                    self._plot_era(self.n_in * 2, self.A_m_in, self.B_m_in, self.C_m_in)
        # Auto mode select feature
        else:

            self.n_in = self.n // 2
            self.freqs_Hz_in = self.freqs_Hz
            self.zeta_in = self.zeta
            self.MAC_arr_in = self.MAC_arr
            self.MSV_arr_in = self.MSV_arr
            self.A_m_in = self.A_modal
            self.B_m_in = self.B_modal
            self.C_m_in = self.C_modal
            self.eig_val_in = self.eig_val
            self.psi_in = self.psi
            self.psi_inv_in = self.psi_inv

            # Keeps recommended modes
            selected_modes = np.array(self.rec_keep, dtype=int)

            self._mode_sort(
                selected_modes,
                self.freqs_Hz,
                self.zeta,
                self.eig_val,
                self.psi,
                self.psi_inv,
            )

            # Printing and plotting reduced model data
            print("\nReduced model data:")
            self._mode_print(
                self.n, self.freqs_Hz, self.zeta, self.MAC_arr, self.MSV_arr
            )

            self._plot_era(self.n * 2, self.A_modal, self.B_modal, self.C_modal)


if __name__ == "__main__":
    import numpy.random as rand
    from scipy.io import matlab

    rand.seed(40)

    noise = 0.0

    # noise = matlab.loadmat('noise.mat')

    M = np.diag(np.ones(3))
    K = np.array(
        [
            [4185.1498, 576.6947, 3646.8923],
            [576.6947, 2104.9252, -28.0450],
            [3646.8923, -28.0450, 3451.5583],
        ]
    )

    (w2, phi) = la.eigh(K)

    w2 = np.real(abs(w2))

    zetain = np.array([0.02, 0.075, 0.05])
    Z = np.diag(2 * zetain * np.sqrt(w2))
    input_fz = [np.sqrt(w2) / (2 * np.pi)]

    # compute system response to initial velocity input:
    dt = 0.01
    # t = np.arange(0, 0.6, dt)
    t = np.arange(0, 1, dt)
    F = np.zeros((3, len(t)))
    v0 = 5 * rand.rand(3)
    v0 = np.linalg.solve(phi, np.array([-9.1303, -2.2950, -6.3252]))
    sol = ode.SolveExp2(M, Z, w2, dt)
    sol = sol.tsolve(force=F, v0=v0)
    a = sol.a
    v = sol.v
    d = sol.d

    resp_veloc = phi @ v + noise * rand.rand(3, len(t))
    # + noise['no'][:, :-1]

    fit_era = ERA(
        resp_veloc,
        dt,
        t,
        tol=0.1,
        auto=True,
        overdamped_modes=True,
        input_labels=["x", "y", "z"],
        FFT=True,
        FFT_limit=30,
    )
