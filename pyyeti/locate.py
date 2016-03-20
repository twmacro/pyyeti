# -*- coding: utf-8 -*-
"""
Tools for locating data or subarrays inside other arrays.
"""
import numpy as np


def find_vals(m, v):
    """
    Get partition vector for all occurrences of all values in `v` in
    `m`.

    Parameters
    ----------
    m : array_like
        Array to be searched.
    v : array_like
        Array of values to find in m.

    Returns
    -------
    pv : 1d ndarray
        Values are indexes into `m` of any value in `v`.  Will be
        empty if `m` has none of the values in `v`.

    Notes
    -----
    `m` is flattened to 1d before searching (using column-major
    ordering 'F').  The values in `pv` correspond to::

          [  0      r  ...
             1    r+1
           ...    ...
           r-1   2r-1  ... r*c-1 ]  where m is r x c

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import locate
    >>> m = np.array([[10, 20], [30, 20]])
    >>> locate.find_vals(m, 20)  # doctest: +ELLIPSIS
    array([2, 3]...)
    >>> locate.find_vals(m, 30)  # doctest: +ELLIPSIS
    array([1]...)
    >>> locate.find_vals(m, 100)
    array([], dtype=int64)
    """
    m = np.atleast_1d(m)
    v = np.atleast_1d(v)
    m = m.flatten(order='F')
    v = v.flatten()
    pv = np.zeros(len(m), dtype=bool)
    for i in range(len(v)):
        pv |= m == v[i]
    return pv.nonzero()[0]


def find_subseq(seq, subseq):
    """
    Returns indices of where subseq occurs in seq.  Both are 1d numpy
    arrays.

    Parameters
    ----------
    seq : array_like
        Array to search in. It is flattened before searching.
    subseq : array_like
        Array to search for. It is flattened before searching.

    Returns
    -------
    pv : 1d ndarray
        Vector of indices:

            - length will be equal to the number of occurrences of
              subseq
            - the indices are to the start of each subseq in seq

        Will be empty if subseq is not found in seq.

    Examples
    --------
    >>> from pyyeti import locate
    >>> a = [1, 2, 3, 4, 5, 6, 2, 3]
    >>> sub = [2, 3]
    >>> locate.find_subseq(a, sub)  # doctest: +ELLIPSIS
    array([1, 6]...)
    >>> locate.find_subseq(a, [6, 5])
    array([], dtype=int64)
    """
    seq = np.asarray(seq).reshape(-1)
    subseq = np.asarray(subseq).reshape(-1)
    target = np.dot(subseq, subseq)
    candidates = np.where(np.correlate(seq, subseq,
                                       mode='valid') == target)[0]
    # some of the candidates entries may be false positives; check:
    check = candidates[:, np.newaxis] + np.arange(len(subseq))
    mask = np.all((np.take(seq, check) == subseq), axis=-1)
    return candidates[mask]


def find_rows(matrix, row):
    """
    Returns indices of where row occurs in matrix.

    Parameters
    ----------
    matrix : array
        2d numpy array.
    row : array
        1d numpy array.

    Returns
    -------
    pv : array
        A 1d numpy array of row indices.  Will be empty if row is not
        found or if length(row) != cols(matrix).

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import locate
    >>> mat = np.array([[7, 3], [6, 8], [4, 0],
    ...                 [9, 2], [1, 5], [6, 8]])
    >>> locate.find_rows(mat,np.array([1, 2]))
    array([], dtype=int64)
    >>> pv = locate.find_rows(mat,np.array([6, 8]))
    >>> pv          # doctest: +ELLIPSIS
    array([1, 5]...)
    >>> mat[pv, :]
    array([[6, 8],
           [6, 8]])
    """
    (r1, c1) = np.shape(matrix)
    c2 = len(row)
    if c1 != c2:
        return np.array([], dtype=int)
    return np.nonzero(abs(matrix - row).sum(axis=1) == 0)[0]


def mat_intersect(D1, D2, keep=0):
    """
    Get row intersection partition vectors between two matrices or
    vectors.

    Parameters
    ----------
    D1 : array_like
        1d or 2d array.
    D2 : array_like
        1d or 2d array.
    keep : integer
        0, 1 or 2:

           - if 0, loop over smaller matrix, finding where the rows
             occur in the larger
           - if 1, loop over D1, finding where the rows occur in D2
           - if 2, loop over D2, finding where the rows occur in D1

    Returns
    -------
    pv1 : 1d ndarray
        Row index vector into D1.
    pv2 : 1d ndarray
        Row index vector into D2.

    Notes
    -----
    `pv1` and `pv2` are found such that::

        D1[pv1] == D2[pv2]
        (Note for matrices:  M[i] == M[i, :])

    For matrices, the number of columns in D1 and D2 must be equal to
    get non-empty results.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import locate
    >>> mat1 = np.array([[7, 3], [6, 8], [4, 0], [9, 2], [1, 5]])
    >>> mat2 = np.array([[9, 2], [1, 5], [7, 3]])
    >>> pv1, pv2 = locate.mat_intersect(mat1, mat2)
    >>> pv1  # doctest: +ELLIPSIS
    array([3, 4, 0]...)
    >>> pv2  # doctest: +ELLIPSIS
    array([0, 1, 2]...)
    >>> np.all(mat1[pv1] == mat2[pv2])
    True
    >>> locate.mat_intersect(mat1, mat2, 1)  # doctest: +ELLIPSIS
    (array([0, 3, 4]...), array([2, 0, 1]...))
    >>> locate.mat_intersect(mat2, mat1, 2)  # doctest: +ELLIPSIS
    (array([2, 0, 1]...), array([0, 3, 4]...))
    >>> locate.mat_intersect(mat2, mat1)     # doctest: +ELLIPSIS
    (array([0, 1, 2]...), array([3, 4, 0]...))
    >>> mat3 = np.array([[1, 2, 3]])
    >>> locate.mat_intersect(mat1, mat3)     # doctest: +ELLIPSIS
    (array([], dtype=int...), array([], dtype=int...)
    """
    D1 = np.array(D1)
    D2 = np.array(D2)
    if D1.ndim == D2.ndim == 1:
        c1 = c2 = 1
        r1 = len(D1)
        r2 = len(D2)
        D1 = np.atleast_2d(D1)
        D2 = np.atleast_2d(D2)
        D1 = D1.T
        D2 = D2.T
    else:
        D1 = np.atleast_2d(D1)
        D2 = np.atleast_2d(D2)
        (r1, c1) = np.shape(D1)
        (r2, c2) = np.shape(D2)
    if c1 != c2:
        return np.array([], dtype=int), np.array([], dtype=int)
    # loop over the smaller one if keep == 0:
    if (keep == 0 and r1 <= r2) or keep == 1:
        r = r1
        d1 = D1
        d2 = D2
        switch = False
    else:
        r = r2
        d1 = D2
        d2 = D1
        switch = True
    pv1 = np.zeros(r, dtype=np.int64)
    pv2 = np.zeros(r, dtype=np.int64)
    j = 0
    for i in range(r):
        l = find_rows(d2, d1[i])
        if l.size > 0:
            pv1[j] = i
            pv2[j] = l[0]
            j += 1
    if j == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if switch:
        t = pv1[:j]
        pv1 = pv2[:j]
        pv2 = t
    else:
        pv1 = pv1[:j]
        pv2 = pv2[:j]
#    if switch and keep == 1:
#        si = pv1.argsort()
#        return pv1.take(si), pv2.take(si)
#    elif not switch and keep == 2:
#        si = pv2.argsort()
#        return pv1.take(si), pv2.take(si)
    return pv1, pv2


def index2bool(pv, n):
    """
    Return a True/False vector of length `n` where the True values are
    located according to `pv`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import locate
    >>> pv = np.array([0, 3, 5])
    >>> locate.index2bool(pv, 8)
    array([ True, False, False,  True, False,  True, False, False], dtype=bool)
    """
    tf = np.zeros(n, dtype=bool)
    tf[pv] = True
    return tf


def flippv(pv, n):
    """Flips the meaning of an index partition vector.

    Parameters
    ----------
    pv : 1d ndarray
        The index partition to flip.
    n : integer
        The length of the dimension to partition.

    Returns
    -------
    notpv : ndarray
        Index vector; the complement of pv.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import locate
    >>> pv = np.array([0, 3, 5])
    >>> locate.flippv(pv, 8)  # doctest: +ELLIPSIS
    array([1, 2, 4, 6, 7]...)
    """
    tf = np.ones(n, dtype=bool)
    tf[pv] = False
    return tf.nonzero()[0]


def find_unique(y, tol=1e-6):
    """
    Find values in a vector that differ from previous adjacent value.

    Parameters
    ----------
    y : 1d array_like
        y-axis data vector
    tol : scalar; optional
        A value is considered the same as the previous if the
        difference is less than tol*max(abs(all differences)).

    Returns
    -------
    pv : ndarray
        True/False vector with True for the unique values.

    Examples
    --------
    >>> from pyyeti import locate
    >>> locate.find_unique([1, 1, 1, 1])
    array([ True, False, False, False], dtype=bool)
    >>> locate.find_unique([4, 4, -2, -2, 0, -2])
    array([ True, False,  True, False,  True,  True], dtype=bool)
    """
    y = np.atleast_1d(y)
    m = np.diff(y)
    stol = abs(tol * abs(m).max())
    pv = np.hstack((True, abs(m) > stol))
    return pv


def list_intersect(L1, L2):
    """
    Get list intersection partition vectors between two lists

    Parameters
    ----------
    L1 : list
       List 1; the output vectors maintain the order of `L1`.
    L1 : list
       List 2.

    Returns
    -------
    pv1 : 1d ndarray
        Index vector into L1.
    pv2 : 1d ndarray
        Index vector into L2.

    Notes
    -----
    `pv1` and `pv2` are found such that::

         [L1[i] for i in pv1] == [L2[i] for i in pv2]

    Examples
    --------
    >>> from pyyeti import locate
    >>> pv1, pv2 = locate.list_intersect(['a', 3, 'z', 0],
    ...                                     [0, 'z', 1, 'a'])
    >>> pv1  # doctest: +ELLIPSIS
    array([0, 2, 3]...)
    >>> pv2  # doctest: +ELLIPSIS
    array([3, 1, 0]...)
    >>> locate.list_intersect(['a', 'b'],
    ...                          [1, 2])       # doctest: +ELLIPSIS
    (array([], dtype=int...), array([], dtype=int...))
    """
    inters = set(L1) & set(L2)
    r = len(inters)
    if r == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    pv1 = np.zeros(r, dtype=np.int64)
    pv2 = np.zeros(r, dtype=np.int64)
    j = 0
    for j, item in enumerate(inters):
        pv1[j] = L1.index(item)
        pv2[j] = L2.index(item)
    si = pv1.argsort()
    return pv1[si], pv2[si]
