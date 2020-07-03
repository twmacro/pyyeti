# -*- coding: utf-8 -*-
"""
Statistics tools for tolerance bounds/intervals and order statistics.
"""

import warnings
import numpy as np
from scipy.stats import norm, nct, chi2, binom
from scipy.special import betainc
from scipy.optimize import brentq


def ksingle(p, c, n):
    """
    Compute statistical k-factor for a single-sided tolerance limit.

    Parameters
    ----------
    p : scalar or array_like; real
        Portion of population to bound; 0 < p < 1
    c : scalar or array_like; real
        Probability level (confidence); 0 < c < 1
    n : scalar or array_like; integer
        Number of observations in sample; n > 1

    Returns
    -------
    k : scalar or ndarray; real
        The statistical k-factor for a single-sided tolerance
        limit.

    Notes
    -----
    The inputs `p`, `c`, and `n` must be broadcast-compatible.

    The k-factor allows the computation of a tolerance bound that has
    the probability `c` of bounding at least the proportion `p` of the
    population.  The statistics are based on having a sample of a
    normally distributed population; `n` is the number of observations
    in the sample.

    The tolerance bound is computed by::

        bound = m + k std  (or  bound = m - k std)

    where `m` is the sample mean and `std` is the sample standard
    deviation.

    .. note::
        The math behind this routine is covered in the pyYeti
        :ref:`tutorial`: :doc:`/tutorials/flight_data_statistics`.
        There is also a link to the source Jupyter notebook at the top
        of the tutorial.

    See also
    --------
    :func:`kdouble`

    Examples
    --------
    Assume we have 21 samples. Determine the k-factor to have a 90%
    probability of bounding 99% of the population. In other words, we
    need the 'P99/90' single-sided k-factor for N = 21.  (From table:
    N = 21, k = 3.028)

    >>> from pyyeti.stats import ksingle
    >>> ksingle(.99, .90, 21)                # doctest: +ELLIPSIS
    3.0282301090342...

    Make a table of single-sided k-factors using 50% confidence. The
    probabilities will be: 95%, 97.725%, 99% and 99.865%. Number of
    samples will be: 2-10, 1000000. Have `n` define the rows and `p`
    define the columns:

    >>> import numpy as np
    >>> from pandas import DataFrame
    >>> n = [[i] for i in range(2, 11)]  # create list of lists
    >>> n.append([1000000])
    >>> p = [.95, .97725, .99, .99865]
    >>> table = ksingle(p, .50, n)
    >>> DataFrame(table, index=[i[0] for i in n], columns=p)
              0.95000   0.97725   0.99000   0.99865
    2        2.338727  2.880624  3.375968  4.391208
    3        1.938416  2.369068  2.764477  3.579188
    4        1.829514  2.231482  2.600817  3.362580
    5        1.779283  2.168283  2.525770  3.263359
    6        1.750462  2.132099  2.482840  3.206631
    7        1.731792  2.108690  2.455081  3.169958
    8        1.718720  2.092314  2.435669  3.144318
    9        1.709060  2.080220  2.421337  3.125390
    10       1.701632  2.070925  2.410323  3.110845
    1000000  1.644854  2.000003  2.326348  2.999978
    """
    n = np.asarray(n)
    sn = np.sqrt(n)
    pnonc = sn * norm.ppf(p)
    return nct.ppf(c, n - 1, pnonc) / sn


def _getr(n, prob, tol):
    """
    Get R needed by :func:`kdouble` routine.

    This routine calculates R defined by the integral::

                       1/sqrt(n) + R
                            /
       prob = 1/sqrt(2*pi)  |  exp(-t^2/2) dt
                           /
                       1/sqrt(n) - R

    The inputs `n` and `prob` must be broadcast-compatible. `tol` is
    the error tolerance.

    Uses Newton's method (with derivative from Leibniz's rule) for
    root finding.

    See also :func:`kdouble`.
    """
    sn = 1 / np.sqrt(n)
    spi = 1 / np.sqrt(2 * np.pi)

    # initial guess at r = r_inf * (1+1/(2*n)
    r = norm.ppf(prob + (1 - prob) / 2) * (1 + 1 / (2 * n))
    rold = r + 10
    loops = 0
    MAXLOOPS = 100
    while np.any(abs(r - rold) > tol) and loops < MAXLOOPS:
        rold = r
        lhi = sn + rold
        llo = sn - rold
        num = norm.cdf(lhi) - norm.cdf(llo) - prob
        den = spi * (np.exp(-(lhi ** 2) / 2) + np.exp(-(llo ** 2) / 2))
        r = rold - num / den
        loops += 1
    if loops == MAXLOOPS:  # pragma: no cover
        warnings.warn(
            "maximum number of loops exceeded. Solution will likely be inaccurate.",
            RuntimeWarning,
        )
    return r


def kdouble(p, c, n, tol=1e-12):
    """
    Compute statistical k-factor for a double-sided tolerance interval.

    Parameters
    ----------
    p : scalar or array_like; real
        Portion of population to bound; 0 < p < 1
    c : scalar or array_like; real
        Probability level (confidence); 0 < c < 1
    n : scalar or array_like; integer
        Number of observations in sample; n > 1
    tol : scalar; optional
        Error tolerance to pass to the :func:`_getr` routine

    Returns
    -------
    k : scalar or ndarray; real
        The statistical k-factor for a double-sided tolerance
        interval.

    Notes
    -----
    The inputs `p`, `c`, and `n` must be broadcast-compatible.

    The k-factor allows the computation of a tolerance interval that
    has the probability `c` of containing at least the proportion `p`
    of the population.  The statistics are based on having a sample of
    a normally distributed population; `n` is the number of
    observations in the sample.

    The bounds of the tolerance interval are calculated by::

        lower = m - k std
        upper = m + k std

    where `m` is the sample mean and `std` is the sample standard
    deviation.

    See references [#stat1]_, [#stat2]_, and [#stat3]_ for the
    mathematical background on these tolerance limits.

    References
    ----------
    .. [#stat1] Churchill Eisenhart; Millard Hastay; and W. Allen
           Wallis, 'Selected Techniques of Statistical Analysis for
           Scientific and Industrial Research and Production and
           Management Engineering,' by the Statistical Research Group,
           Columbia University, First Edition, New York and London,
           McGraw-Hill Book Company, Inc, 1947.

    .. [#stat2] Albert H. Bowker, 'Computation of Factors for
           Tolerance Limits on a Normal Distribution when the Sample
           Size is Large,' Annals of Mathematical Statistics, Vol. 17,
           1946, pp 238-240.

    .. [#stat3] A. Wald and J. Wolfowitz, 'Tolerance Limits for a
           Normal Distribution,' Annals of Mathematical Statistics,
           Vol. 17, 1946, pp 208-215.

    See also
    --------
    :func:`ksingle`

    Examples
    --------
    Assume we have 21 samples. Determine the k-factor to have a 90%
    probability of the interval containing 99% of the population. In
    other words, we need the 'P99/90' double-sided k-factor for N =
    21.  (From table: N = 21, k = 3.340)

    >>> from pyyeti.stats import kdouble
    >>> kdouble(.99, .90, 21)                # doctest: +ELLIPSIS
    3.3404115111514...

    Make a table of double-sided k-factors using 50% confidence. The
    probabilities `p` will be: 95%, 97.725%, 99% and 99.865%. Number
    of samples `n` will be: 2-10, 1000000. Have `n` define the rows
    and `p` define the columns:

    >>> import numpy as np
    >>> from pandas import DataFrame
    >>> n = [[i] for i in range(2, 11)]  # create list of lists
    >>> n.append([1000000])
    >>> p = [.95, .9545, .99, .9973]
    >>> table = kdouble(p, .50, n)
    >>> DataFrame(table, index=[i[0] for i in n], columns=p)
               0.9500    0.9545    0.9900    0.9973
    2        3.502564  3.568593  4.502465  5.175585
    3        2.697379  2.750060  3.498689  4.041051
    4        2.456441  2.505211  3.200478  3.706120
    5        2.336939  2.383750  3.052519  3.540237
    6        2.264675  2.310279  2.962781  3.439587
    7        2.215998  2.260775  2.902111  3.371448
    8        2.180884  2.225054  2.858183  3.322030
    9        2.154316  2.198021  2.824834  3.284445
    10       2.133494  2.176829  2.798616  3.254847
    1000000  1.959966  2.000004  2.575831  2.999979

    """
    p = np.asarray(p)
    c = np.asarray(c)
    n = np.asarray(n)
    chi = chi2.ppf(1 - c, n - 1)
    r = _getr(n, p, tol)
    return np.sqrt((n - 1) / chi) * r


def order_stats(which, *, p=None, c=None, n=None, r=None):
    """
    Compute a parameter from order statistics.

    Parameters
    ----------
    which : str
        Either 'p', 'c', 'n', or 'r' to specify which of the following
        arguments is to be computed from the others.
    p : scalar or array_like; real, (0, 1)
        Proportion of population
    c : scalar or array_like; real, (0, 1)
        Probability of bounding proportion `p` of the population
        (confidence level).
    n : scalar or array_like; integer
        Sample size
    r : scalar or ndarray; integer
        Largest-value order statistic. Note::

            number of failures = r - 1

    Returns
    -------
    One of `p`, `c`, `n`, or `r`; according to `which`.

    Notes
    -----
    One of the inputs of  `p`, `c`, `n`, and `r` can be left as None;
    the remaining inputs must be broadcast-compatible and must be
    named.

    The binomial distribution forms the mathematical foundation of
    this routine; see reference [#stat4]_. [#stat5]_ has a good
    definition of the order statistic. See also "Bernoulli Trials",
    reference [#stat6]_, which ties some of these ideas together in
    the analysis of success/failure probabilities.

    References
    ----------
    .. [#stat4] Wikipedia, "Binomial distribution",
            https://en.wikipedia.org/wiki/Binomial_distribution

    .. [#stat5] Wikipedia, "Order statistic",
            https://en.wikipedia.org/wiki/Order_statistic

    .. [#stat6] Wikipedia, "Bernoulli trial",
            https://en.wikipedia.org/wiki/Bernoulli_trial

    Examples
    --------
    Start with 700 samples of unknown distribution. After sorting,
    which of the samples represents at least a P99/90 level? From
    published tables, `r` should be 4, meaning the 4-th highest value
    of the 700 is an estimate of the P99/90 level (or higher). Another
    way to look at that result is that 3 failures (or fewer) out of
    700 trials demonstrates at least a P99/90 level.

    >>> from pyyeti.stats import order_stats
    >>> order_stats('r', p=.99, c=.90, n=700)
    4

    Holding the probability constant at 90%, the portion of the
    population bounded has to be at least 99%. But, what did it turn
    out to be?

    >>> order_stats('p', c=.90, n=700, r=4)     # doctest: +ELLIPSIS
    0.99048109...

    Instead, hold the portion constant. What is the probability of
    covering 99% percent of the population by selecting the 4th highest
    of 700?

    >>> order_stats('c', p=.99, n=700, r=4)     # doctest: +ELLIPSIS
    0.91927834...

    How many samples did we really need to reach at least the P99/90
    level by selecting the 4th highest?

    >>> order_stats('n', p=.99, c=.90, r=4)
    667

    Generate a 90% confidence table showing the number of trials
    needed for: `r` will go from 1 to 12 (defining the rows), and
    population coverage will be: [.95, .9772, .99, .9973, .99865]
    (defining the columns). Display using a pandas DataFrame:

    >>> from pandas import DataFrame
    >>> r = np.arange(1, 13).reshape(-1, 1)
    >>> p = [.95, .97725, .99, .9973, .99865]
    >>> table = order_stats('n', c=.90, r=r, p=p)
    >>> DataFrame(table, index=r.ravel(), columns=p)
        0.95000  0.97725  0.99000  0.99730  0.99865
    1        45      101      230      852     1705
    2        77      170      388     1440     2880
    3       105      233      531     1970     3941
    4       132      292      667     2473     4947
    5       158      350      798     2959     5920
    6       184      406      926     3433     6868
    7       209      461     1051     3899     7800
    8       234      516     1175     4358     8717
    9       258      569     1297     4811     9624
    10      282      622     1418     5259    10521
    11      306      675     1538     5704    11410
    12      330      727     1658     6145    12293
    """
    if which == "c":
        r = np.asarray(r)
        p = np.asarray(p)
        return binom.sf(r - 1, n, 1 - p)
    elif which == "r":
        # c = np.asarray(c)
        # p = np.asarray(p)
        # return binom.ppf(1-c, n, 1-p)  # gets 'value too deep error'
        b = np.broadcast(c, n, p)
        r = np.empty(b.shape)
        r.flat = [binom.ppf(1 - c, n, 1 - p) for (c, n, p) in b]
        if r.ndim == 0:
            return int(r[()])
        return r.astype(int)
    elif which == "n":

        def _func(n, p, s, pr):
            return p - (1 - betainc(s + 1, n - s, pr))

        def _run_brentq(c, r, p):
            # find [a, b] interval by brute force:
            a = r
            b = 2 * a
            loops = 0
            while _func(b, 1 - c, r - 1, 1 - p) < 0 and loops < 30:
                a = b
                b = 2 * a
                loops += 1
            return brentq(_func, a, b, args=(1 - c, r - 1, 1 - p))

        b = np.broadcast(c, r, p)
        n = np.empty(b.shape)
        n.flat = [_run_brentq(c, r, p) for (c, r, p) in b]
        return np.ceil(n).astype(int)
    elif which == "p":

        def _func(pr, p, s, n):
            return p - binom.cdf(s, n, pr)

        b = np.broadcast(c, r, n)
        n = np.empty(b.shape)
        n.flat = [1 - brentq(_func, 0, 1, args=(1 - c, r - 1, n)) for (c, r, n) in b]
        if n.ndim == 0:
            return n[()]
        return n
    raise ValueError("invalid `which` setting")
