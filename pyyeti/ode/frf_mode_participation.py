# -*- coding: utf-8 -*-
import numpy as np


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def getmodepart(
    h_or_frq,
    sols,
    mfreq,
    factor=2 / 3,
    helpmsg=True,
    ylog=False,
    auto=None,
    idlabel="",
    frf_ttl="",
):
    """
    Get modal participation from frequency response plots.

    Parameters
    ----------
    h_or_frq : list/tuple or 1d ndarray
        Plot line handles or frequency vector:

        - If list/tuple, it contains the plot line handles to the FRF
          curves; in this case, the analysis frequency is retrieved
          from the plot.
        - If it is a 1d ndarray, it is the frequency vector; in this
          case, a plot of FRFs (from sols) is generated in figure
          'FRF' (or 'FRF - '+idlabel)

    sols : list/tuple of lists/tuples
        Contains info to determine modal particpation::

            sols = [[Trecover1, accel1, Trecover1_row_labels],
                    [Trecover2, accel2, Trecover2_row_labels],
                     ... ]

        - each Trecover matrix is:  any number  x  modes
        - each accel matrix is: modes x frequencies
        - each row_labels entry is a list/tuple: len = # rows in
          corresponding Trecover (if Trecover only has 1 row, then
          row_labels may be just a string)

        The FRFs are recovered by::

                   FRFs1 = Trecover1*accel1
                   FRFs2 = Trecover2*accel2
                   ...

        accel1, accel2, etc are the complex modal acceleration (or
        displacement or velocity) frequency responses; normally output
        by, for example, :func:`SolveUnc.fsolve`
    mfreq : array_like
        Vector of modal frequencies (Hz)
    factor : scalar; optional
        From 0 to 1 for setting the criteria for choosing secondary
        modes: if mode participation of secondary mode(s) >= `factor`
        * max_participation, include it.
    helpmsg : bool; optional
        If True, print brief message explaining the mouse buttons
        before plotting anything
    ylog : bool; optional
        If True, y-axis will be log
    auto : list/tuple or None; optional

        - If None, operate interactively.
        - If a 2 element vector, it specifies an item in `sols` and a
          Trecover row (0-offset) that this routine will automatically
          (non-interactively) use to select modes. It will select the
          peak of the specified response and, based on mode
          participation, return the results. In other words, it acts
          as if you picked the peak of the specified curve and then
          hit 't'. The 1st element of `auto` selects which `sols`
          entry to use and the 2nd selects the matrix row. For
          example, to choose the 12th row of Trecover3, set `auto` to
          [2, 11].

    idlabel : string; optional If not '', it will be
          used in the figure name. This allows multiple
          getmodepart()'s to be run with the same model, each using
          its own FRF and MP windows. The figure names will be::

                 'FRF - '+idlabel   <-- used only if h_or_frq is freq
                 'MP - '+idlabel

    frf_ttl : string; optional
        Title used for FRF plot

    Returns
    -------
    modes : list
        List of selected mode numbers; 0-offset
    freqs : 1d ndarray
        Vector of frequencies in Hz corresponding to `modes`.

    Notes
    -----
    FRF peaks can only be selected in range of the analysis frequency,
    but modes outside this range may be selected based on modal
    participation.

    This routine will echo modal participation factors to the screen
    and plot them in figure 'MP' (or 'MP - '+idlabel).

    If `auto` is None (or some other non-2-element object), this
    routine works as follows:

        1. If `h` is frequency vector, plot FRFs from `sols` in figure
           'FRF' or 'FRF - '+idlabel
        2. Repeat:

           a. Waits for user to select response. Mouse/key commands::

                  Left  - select response point; valid only in the
                          FRF figure
                  Right - erase last selected mode(s)
                  't'   - done

           b. Plots mode participation bar graph in figure 'MP' (or
              'MP - '+idlabel) showing the frequency(s) of modes
              selected.

    If using `auto`, no plots are generated (see `auto` description
    above).

    See also
    --------
    :class:`SolveUnc`, :class:`FreqDirect`, :func:`modeselect`,
    :func:`pyyeti.datacursor`

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from pyyeti.ode import SolveUnc
    >>> from pyyeti.ode import getmodepart
    >>> import scipy.linalg as la
    >>> K = 40*np.random.randn(5, 5)
    >>> K = K @ K.T  # pos-definite K matrix, M is identity
    >>> M = None
    >>> w2, phi = la.eigh(K)
    >>> mfreq = np.sqrt(w2)/2/np.pi
    >>> zetain = np.array([ .02, .02, .05, .02, .05 ])
    >>> Z = np.diag(2*zetain*np.sqrt(w2))
    >>> freq = np.arange(0.1, 15.05, .1)
    >>> f = np.ones((1, len(freq)))
    >>> Tbot = phi[0:1, :]
    >>> Tmid = phi[2:3, :]
    >>> Ttop = phi[4:5, :]
    >>> fs = SolveUnc(M, Z, w2)
    >>> sol_bot = fs.fsolve(Tbot.T @ f, freq)
    >>> sol_mid = fs.fsolve(Tmid.T @ f, freq)

    Prepare transforms and solutions for :func:`getmodepart`:
    (Note: the top 2 items in sols could be combined since they
    both use the same acceleration)

    >>> sols = [[Tmid, sol_bot.a, 'Bot to Mid'],
    ...         [Ttop, sol_bot.a, 'Bot to Top'],
    ...         [Ttop, sol_mid.a, 'Mid to Top']]

    Approach 1: let :func:`getmodepart` do the FRF plotting:

    >>> lbl = 'getmodepart demo 1'
    >>> mds, frqs = getmodepart(freq, sols,     # doctest: +SKIP
    ...                         mfreq, ylog=1,  # doctest: +SKIP
    ...                         idlabel=lbl)    # doctest: +SKIP
    >>> print('modes =', mds)                   # doctest: +SKIP
    >>> print('freqs =', frqs)                  # doctest: +SKIP

    Approach 2: plot FRFs first, then call :func:`getmodepart`:

    >>> fig = plt.figure('approach 2 FRFs')         # doctest: +SKIP
    >>> fig.clf()                                   # doctest: +SKIP
    >>> for s in sols:                              # doctest: +SKIP
    ...     plt.plot(freq, abs(s[0] @ s[1]).T,      # doctest: +SKIP
    ...              label=s[2])                    # doctest: +SKIP
    >>> _ = plt.xlabel('Frequency (Hz)')            # doctest: +SKIP
    >>> plt.yscale('log')                           # doctest: +SKIP
    >>> _ = plt.legend(loc='best')                  # doctest: +SKIP
    >>> h = plt.gca().lines                         # doctest: +SKIP
    >>> lbl = 'getmodepart demo 2'                  # doctest: +SKIP
    >>> modes, freqs = getmodepart(h, sols,         # doctest: +SKIP
    ...                            mfreq,           # doctest: +SKIP
    ...                            idlabel=lbl)     # doctest: +SKIP
    >>> print('modes =', modes)                     # doctest: +SKIP
    >>> print('freqs =', freqs)                     # doctest: +SKIP
    """
    # check sols:
    if not isinstance(sols, (list, tuple)) or not isinstance(sols[0], (list, tuple)):
        raise ValueError("`sols` must be list/tuple of lists/tuples")

    for j, s in enumerate(sols):
        if len(s) != 3:
            raise ValueError(
                f"sols[{j}] must have 3 elements: [Trecover, accel, labels]"
            )
        Trec = np.atleast_2d(s[0])
        acce = np.atleast_2d(s[1])
        labels = s[2]
        if Trec.shape[0] == 1:
            if isinstance(labels, (list, tuple)) and len(labels) != 1:
                raise ValueError(
                    f"in sols[{j}], Trecover has 1 row, "
                    f"but labels is length {len(labels)}"
                )
        else:
            if not isinstance(labels, (list, tuple)):
                raise ValueError(
                    f"in sols[{j}], labels must be a list/tuple because Trecover"
                    " has more than 1 row"
                )
            if Trec.shape[0] != len(labels):
                raise ValueError(f"in sols[{j}], len(labels) != Trecover.shape[0]")
        if Trec.shape[1] != acce.shape[0]:
            raise ValueError(
                f"in sols[{j}], Trecover is not compatibly sized with accel"
            )

    def _getmds(modepart):
        # find largest contributor mode:
        mode = np.argmax(modepart)
        mx = modepart[mode]
        # find other import participating modes:
        pv = np.nonzero(modepart >= factor * mx)[0]
        # sort, so most significant contributor is first:
        i = np.argsort(modepart[pv])[::-1]
        mds = pv[i]
        for m in mds:
            print(f"\tSelected mode index (0-offset) {m}, frequency {mfreq[m]:.4f}")
        return mds

    mfreq = np.atleast_1d(mfreq)
    if isinstance(h_or_frq, np.ndarray):
        freq = h_or_frq
    else:
        freq = h_or_frq[0].get_xdata()

    if getattr(auto, "__len__", None):
        s = sols[auto[0]]
        r = auto[1]
        Trcv = np.atleast_2d(s[0])[r : r + 1]
        resp = abs(Trcv @ s[1])
        # find which frequency index gave peak response:
        i = np.argmax(resp)
        # compute modal participation at this frequency:
        acce = np.atleast_2d(s[1])[:, i]
        modepart = abs(Trcv.ravel() * acce)
        # pv = np.nonzero((mfreq >= freq[0]) & (mfreq <= freq[-1]))[0]
        # mds = _getmds(modepart[pv])
        mds = _getmds(modepart)
        modes = sorted(mds)
        freqs = mfreq[modes]
        return modes, freqs

    if helpmsg:
        print("Mouse buttons:")
        print("\tLeft   - select response point")
        print("\tRight  - erase last selected mode(s)")
        print("To quit, type 't' inside the axes")

    if idlabel:
        frfname = f"FRF - {idlabel}"
        mpname = f"MP - {idlabel}"
    else:
        frfname = "FRF"
        mpname = "MP"

    import matplotlib.pyplot as plt
    from pyyeti.datacursor import DC

    if isinstance(h_or_frq, np.ndarray):
        freq = h_or_frq
        h = []
        plt.figure(frfname)
        plt.clf()
        for s in sols:
            Trec = np.atleast_2d(s[0])
            acce = np.atleast_2d(s[1])
            curlabels = s[2]
            if isinstance(curlabels, str):
                curlabels = [curlabels]
            for j in range(Trec.shape[0]):
                resp = abs(Trec[j : j + 1] @ acce).ravel()
                h += plt.plot(freq, resp, label=curlabels[j])
        if ylog:
            plt.yscale("log")
        if frf_ttl:
            plt.title(frf_ttl)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("FRF")
        plt.legend(loc="best")
    else:
        h = h_or_frq

    plt.figure(mpname)
    plt.clf()

    modes = []  # list to store modes
    primary = []  # flag to help delete plot objects logically

    def _addpoint(x, y, n, i, axes, ln):
        if ln not in h:
            print("invalid curve ... ignoring")
            return

        # find n'th (zero offset) Trec, acce, label:
        j = 0
        for s in sols:
            T = np.atleast_2d(s[0])
            rows = T.shape[0]
            if j + rows > n:
                row = n - j
                T = T[row]
                acce = np.atleast_2d(s[1])[:, i]
                labels = s[2]
                if isinstance(labels, str):
                    labels = [labels]
                label = labels[row]
                break
            j += rows

        # compute modal participation at this frequency
        modepart = abs(T * acce)
        mds = _getmds(modepart)

        # plot bar chart showing modal participation and label top
        # modes:
        fig = plt.figure(mpname)
        plt.clf()
        # pv = np.nonzero((mfreq >= freq[0]) & (mfreq <= freq[-1]))[0]
        # plt.bar(mfreq[pv], modepart[pv], align='center')
        plt.bar(mfreq, modepart, align="center")
        plt.title(label)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Mode Participation")
        ax = plt.gca()
        for i, m in enumerate(mds):
            barlabel = f"{mfreq[m]:.4f} Hz"
            ax.text(mfreq[m], modepart[m], barlabel, ha="center", va="bottom")
            modes.append(m)
            primary.append(i == 0)
        fig.canvas.draw()

    def _deletepoint(x, y, n, i, ax, line):
        while len(primary) > 0:
            m = modes.pop()
            p = primary.pop()
            print(f"\tMode {m}, {mfreq[m]:.4f} Hz erased")
            if p:
                break

    try:
        DC.off()
        DC.addpt_func(_addpoint)
        DC.delpt_func(_deletepoint)
        DC.getdata()
    finally:
        DC.addpt_func(None)
        DC.delpt_func(None)
        DC.off()

    modes = sorted(set(modes))
    freqs = mfreq[modes]
    return modes, freqs


def modeselect(
    name,
    fs,
    force,
    freq,
    Trcv,
    labelrcv,
    mfreq,
    factor=2 / 3,
    helpmsg=True,
    ylog=False,
    auto=None,
    idlabel="",
):
    """
    Select modes based on mode participation in graphically chosen
    responses.

    Parameters
    ----------
    name : string
        Name of analysis; for example the flight event name; it is
        used for plot title
    fs : class
        An instance of :class:`SolveUnc` or :class:`FreqDirect` (or
        similar ... must have `.fsolve` method)
    force : 2d array_like
        Forcing function in frequency domain; # cols = len(freq)
    freq : 1d array_like
        Frequency vector in Hz where solution is requested
    Trcv : 2d array_like
        Data recovery matrix to the desired DOF
    labelrcv : list/tuple (can be string if `Trcv` has 1 row)
        List/tuple of labels; one for each row in Trcv; used for
        legend. May be a string if `Trcv` has only 1 row.
    mfreq : array_like
        Vector of modal frequencies (Hz)
    factor : scalar; optional
        From 0 to 1 for setting the criteria for choosing secondary
        modes: if mode participation of secondary mode(s) >= `factor`
        * max_participation, include it.
    helpmsg : bool; optional
        If True, print brief message explaining the mouse buttons
        before plotting anything
    ylog : bool; optional
        If True, y-axis will be log
    auto : integer or None; optional

        - If None, operate interactively.
        - If an integer, it specifies a row in `Trcv` row (0-offset)
          that this routine will automatically (non-interactively) use
          to select modes. It will select the peak of the specified
          response and, based on mode participation, return the
          results. In other words, it acts as if you picked the peak
          of the specified curve and then hit 't'. For example, to
          choose the 12th row of `Trcv`, set `auto` to 11.

    idlabel : string; optional
          If not '', it will be used in the figure name. This allows
          multiple getmodepart()'s to be run with the same model, each
          using its own FRF and MP windows. The figure names will be::

                 'FRF - '+idlabel
                 'MP - '+idlabel

    Returns
    -------
    modes : list
        List of selected mode numbers; 0-offset
    freqs : 1d ndarray
        Vector of frequencies in Hz corresponding to `modes`.
    resp : 2d ndarray
        The complex frequency responses of the accelerations recovered
        through `Trcv`; rows(`Trcv`) x len(`freq`)

    Notes
    -----
    This routine is an interface to :func:`getmodepart`. See that
    routine for more information.

    This routine can be very useful in selecting modes for damping or
    just identifying modes. For example, to identify axial modes,
    excite the structure axially and choose axial DOFs for recovery at
    the top, bottom, and somewhere in-between.

    See also
    --------
    :func:`getmodepart`, :class:`SolveUnc`, :class:`FreqDirect`

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from pyyeti.ode import SolveUnc, modeselect
    >>> import scipy.linalg as la
    >>> K = 40*np.random.randn(5, 5)
    >>> K = K @ K.T       # positive definite K matrix, M is identity
    >>> M = None
    >>> w2, phi = la.eigh(K)
    >>> mfreq = np.sqrt(w2)/2/np.pi
    >>> zetain = np.array([ .02, .02, .05, .02, .05 ])
    >>> Z = np.diag(2*zetain*np.sqrt(w2))
    >>> freq = np.arange(0.1, 15.05, .1)
    >>> f = phi[4:].T @ np.ones((1, len(freq)))  # force of DOF 5
    >>> Trcv = phi[[0, 2, 4]]                    # recover DOF 1, 3, 5
    >>> labels = ['5 to 1', '5 to 3', '5 to 5']
    >>> fs = SolveUnc(M, Z, w2)                     # doctest: +SKIP
    >>> mfr = modeselect('Demo', fs, f, freq,       # doctest: +SKIP
    ...                  Trcv, labels, mfreq)       # doctest: +SKIP
    >>> print('modes =', mfr[0])                    # doctest: +SKIP
    >>> print('freqs =', mfr[1])                    # doctest: +SKIP
    """
    sol = fs.fsolve(force, freq)
    sols = [[Trcv, sol.a, labelrcv]]
    if isinstance(auto, int):
        auto = [0, auto]
    modes, freqs = getmodepart(
        freq, sols, mfreq, factor, helpmsg, ylog, auto, idlabel, frf_ttl=name
    )
    return modes, freqs, Trcv @ sol.a
