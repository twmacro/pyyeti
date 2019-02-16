# -*- coding: utf-8 -*-
import numpy as np


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy='1.13')
except TypeError:
    pass


def solvepsd(fs, forcepsd, t_frc, freq, drmlist, incrb=2,
             forcephi=None, rbduf=1.0, elduf=1.0):
    """
    Solve equations of motion in frequency domain with uncorrelated
    PSD forces.

    Parameters
    ----------
    fs : class
        An instance of :class:`SolveUnc` or :class:`FreqDirect` (or
        similar ... must have `.fsolve` method)
    forcepsd : 2d array_like
        The matrix of force psds; each row is a force PSD
    t_frc : 2d array_like
        Transformation matrix from system modal DOF to forced DOF;
        ``rows(t_frc) = rows(forcepsd)``
    freq : 1d array_like
        Frequency vector at which solution will be computed;
        ``len(freq) = cols(forcepsd)``
    drmlist : list_like
        List of lists (or similar) of any number of pairs of data
        recovery matrices: [[atm1, dtm1], [atm2, dtm2], ...]. To not
        use a particular drm, set it to None. For example, to perform
        these 3 types of data recovery::

                acce = atm*a
                disp = dtm*d
                loads = ltma*a + dtmd*d

        `drmlist` would be::

              [[atm, None], [dtm, None], [ltma, ltmd]]

    incrb : 0, 1, or 2; optional
        An input to the :func:`fs.fsolve` method, it specifies how to
        handle rigid-body responses:

        ======  ==============================================
        incrb   description
        ======  ==============================================
           0    no rigid-body is included
           1    acceleration and velocity rigid-body only
           2    all of rigid-body is included (see note below)
        ======  ==============================================

    forcephi : 2d array_like or None; optional
        If not None, it is a force transformation data-recovery matrix
        as in::

             resp = atm*a + dtm*d - forcephi*f

    rbduf : scalar; optional
        Rigid-body uncertainty factor
    elduf : scalar; optional
        Dynamic uncertainty factor

    Returns
    -------
    rms : list
        List of vectors (corresponding to `drmlist`) of rms values of
        all data recovery rows; # of rows of each vector = # rows in
        each drm pair
    psd : list
        List of matrices (corresponding to `drmlist`) of PSD responses
        for all data recovery rows::

               # rows in each PSD = # rows in DRM
               # cols in each PSD = len(freq)

    Notes
    -----
    This routine first calls `fs.fsolve` to solve the modal equations
    of motion. Then, it scales the responses by the corresponding PSD
    input force. All PSD responses are summed together for the overall
    response. For example, for a displacement and acceleration
    dependent response::

        resp_psd = 0
        for i in range(forcepsd.shape[0]):
            # solve for unit frequency response function:
            genforce = t_frc[i:i+1].T @ np.ones((1, len(freq)))
            sol = fs.fsolve(genforce, freq, incrb)
            frf = atm @ sol.a + dtm @ sol.d
            resp_psd += forcepsd[i] * abs(frf)**2

    Examples
    --------
    .. plot::
        :context: close-figs

        >>> from pyyeti import ode
        >>> import numpy as np
        >>> m = np.array([10., 30., 30., 30.])        # diag mass
        >>> k = np.array([0., 6.e5, 6.e5, 6.e5])      # diag stiffness
        >>> zeta = np.array([0., .05, 1., 2.])        # % damping
        >>> b = 2.*zeta*np.sqrt(k/m)*m                # diag damping
        >>> freq = np.arange(.1, 35, .1)              # frequency
        >>> forcepsd = 10000*np.ones((4, freq.size))  # PSD forces
        >>> fs = ode.SolveUnc(m, b, k)
        >>> atm = np.eye(4)    # recover modal accels
        >>> t_frc = np.eye(4)  # assume forces already modal
        >>> drms = [[atm, None]]
        >>> rms, psd = ode.solvepsd(fs, forcepsd, t_frc, freq,
        ...                         drms)

        The rigid-body results should be 100.0 g**2/Hz flat;
        rms = np.sqrt(100*34.8)

        >>> np.allclose(100., psd[0][0])
        True
        >>> np.allclose(np.sqrt(3480.), rms[0][0])
        True

        Plot the four accelerations PSDs:

        >>> import matplotlib.pyplot as plt
        >>> fig = plt.figure('solvepsd demo', figsize=[8, 8])
        >>> labels = ['Rigid-body', 'Underdamped',
        ...           'Critically Damped', 'Overdamped']
        >>> for j, name in zip(range(4), labels):
        ...     _ = plt.subplot(4, 1, j+1)
        ...     _ = plt.plot(freq, psd[0][j])
        ...     _ = plt.title(name)
        ...     _ = plt.ylabel(r'Accel PSD ($g^2$/Hz)')
        ...     _ = plt.xlabel('Frequency (Hz)')
        >>> plt.tight_layout()
    """
    ndrms = len(drmlist)
    forcepsd = np.atleast_2d(forcepsd)
    freq = np.atleast_1d(freq)
    rpsd, cpsd = forcepsd.shape
    unitforce = np.ones((1, cpsd))
    psd = [0.] * ndrms
    rms = [0.] * ndrms

    for i in range(rpsd):
        # solve for unit frequency response function for i'th force:
        genforce = t_frc[i:i + 1].T @ unitforce
        sol = fs.fsolve(genforce, freq, incrb)
        if rbduf != 1.0:
            sol.a[fs.rb] *= rbduf
            sol.v[fs.rb] *= rbduf
            sol.d[fs.rb] *= rbduf
        if elduf != 1.0:
            sol.a[fs.el] *= elduf
            sol.v[fs.el] *= elduf
            sol.d[fs.el] *= elduf
        for j, drmpair in enumerate(drmlist):
            atm = drmpair[0]
            dtm = drmpair[1]
            frf = 0.
            if atm is not None:
                frf += atm @ sol.a
            if dtm is not None:
                frf += dtm @ sol.d
            if forcephi is not None:
                frf -= forcephi[:, i:i + 1] @ unitforce
            psd[j] += forcepsd[i] * abs(frf)**2

    # compute area under curve:
    freqstep = np.diff(freq)
    for j in range(ndrms):
        sumpsd = psd[j][:, :-1] + psd[j][:, 1:]
        rms[j] = np.sqrt(np.sum((freqstep * sumpsd), axis=1) / 2)
    return rms, psd
