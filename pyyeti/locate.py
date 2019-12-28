# -*- coding: utf-8 -*-
"""
Tools for locating data or subarrays inside other arrays.
"""
import numpy as np


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def find_vals(m, v):
    """
    Find all occurrences of all values in `v` in `m`

    Parameters
    ----------
    m : array_like
        Array to be searched.
    v : array_like
        Array of values to find in `m`.

    Returns
    -------
    ndarray
        True/False vector with True indicating position of any value
        in `v` within `m`. Will be all False if no values in `v` are
        in `m`.

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
    >>> locate.find_vals(m, 20)
    array([False, False,  True,  True], dtype=bool)
    >>> locate.find_vals(m, 30)
    array([False,  True, False, False], dtype=bool)
    >>> locate.find_vals(m, 100)
    array([False, False, False, False], dtype=bool)
    """
    m = np.atleast_1d(m)
    v = np.atleast_1d(v)
    m = m.ravel(order="F")
    v = v.ravel()
    pv = np.zeros(len(m), dtype=bool)
    for i in v:
        pv |= m == i
    return pv


def find_duplicates(v, tol=0.0):
    """
    Find duplicate values in a vector (or within a tolerance).

    Parameters
    ----------
    v : 1d array_like
        Vector to find duplicates in.
    tol : scalar; optional
        Tolerance for checking for duplicates. Values are considered
        duplicates if the absolute value of the difference is <=
        `tol`.

    Returns
    -------
    dups : 1d ndarray
        Bool partition vector for repeated values. `dups` will have
        True for any value that is repeated anywhere else in the
        vector. It will be all False if there are no repeated values.

    Examples
    --------
    >>> from pyyeti import locate
    >>> locate.find_duplicates([0, 10, 2, 2, 6, 10, 10])
    array([False,  True,  True,  True, False,  True,  True], dtype=bool)
    """
    v = np.atleast_1d(v)
    i = np.argsort(v)
    dif = np.diff(v[i])
    dups = np.zeros(v.size, bool)
    tf = abs(dif) <= tol
    dups[i[1:-1]] = np.logical_or(tf[1:], tf[:-1])
    dups[i[0]] = tf[0]
    dups[i[-1]] = tf[-1]
    return dups


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
              `subseq`
            - the indices are to the start of each `subseq` in `seq`

        Will be empty if `subseq` is not found in `seq`.

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
    candidates = np.where(np.correlate(seq, subseq, mode="valid") == target)[0]
    # some of the candidates entries may be false positives; check:
    check = candidates[:, np.newaxis] + np.arange(len(subseq))
    mask = np.all((np.take(seq, check) == subseq), axis=-1)
    return candidates[mask]


def find_rows(matrix, row):
    """
    Get True/False vector indicating where `row` occurs in `matrix`

    Parameters
    ----------
    matrix : array
        2d numpy array.
    row : array
        1d numpy array.

    Returns
    -------
    1d ndarray
        True/False vector with True indicating where `row` occurs in
        `matrix`. Will be all False if `row` does not occur in
        `matrix` or if ``len(row) != cols(matrix)``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import locate
    >>> mat = np.array([[7, 3], [6, 8], [4, 0],
    ...                 [9, 2], [1, 5], [6, 8]])
    >>> locate.find_rows(mat,np.array([1, 2]))
    array([False, False, False, False, False, False], dtype=bool)
    >>> pv = locate.find_rows(mat,np.array([6, 8]))
    >>> pv
    array([False,  True, False, False, False,  True], dtype=bool)
    >>> mat[pv, :]
    array([[6, 8],
           [6, 8]])
    """
    c1 = np.shape(matrix)[1]
    c2 = len(row)
    if c1 != c2:
        return np.array([], dtype=int)
    # return np.nonzero(abs(matrix - row).sum(axis=1) == 0)[0]
    return abs(matrix - row).sum(axis=1) == 0


def _bytes_view(arr, dtype):
    # view columns as byte-string -- this will be sortable and make
    # the matrix look like a 1d vector; that means np.searchsorted
    # will work on it.

    # first, need to ensure c-contiguous:
    arr = np.ascontiguousarray(arr, dtype)

    # special precaution for floats:
    if np.issubdtype(arr.dtype, np.floating):
        # add zero to get rid of the minus sign on 0. ... the byte
        # string for 0. and -0. are different:

        # In [2]: a = np.array([-0.])
        # In [3]: a.view(np.dtype((np.void, 8)))
        # array([b'\x00\x00\x00\x00\x00\x00\x00\x80'], dtype='|V8')
        #
        # In [4]: a = np.array([0.])
        # In [5]: a.view(np.dtype((np.void, 8)))
        # array([b'\x00\x00\x00\x00\x00\x00\x00\x00'], dtype='|V8')

        # also:
        # In [6]: np.array([-0.]).tostring()
        # Out[6]: b'\x00\x00\x00\x00\x00\x00\x00\x80'
        # In [7]: (0. + np.array([-0.])).tostring()
        # Out[7]: b'\x00\x00\x00\x00\x00\x00\x00\x00'
        arr += 0.0
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[-1])))


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
          - if 1, loop over `D1`, finding where the rows occur in `D2`
          - if 2, loop over `D2`, finding where the rows occur in `D1`

    Returns
    -------
    pv1 : 1d ndarray
        Row index vector into `D1`.
    pv2 : 1d ndarray
        Row index vector into `D2`.

    Notes
    -----
    `pv1` and `pv2` are found such that::

        D1[pv1] == D2[pv2]
        (Note for matrices:  M[i] == M[i, :])

    For matrices, the number of columns in `D1` and `D2` must be equal
    to get non-empty results.

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
    # (we'll be searching for needles in the haystack)
    if (keep == 0 and r1 <= r2) or keep == 1:
        needles = D1
        haystack = D2
        switch = False
    else:
        needles = D2
        haystack = D1
        switch = True

    # to use the byte-string view, types must be the same:
    out_dtype = np.find_common_type([haystack.dtype, needles.dtype], [])

    # view entire rows as a single values:
    haystack = _bytes_view(haystack, out_dtype).ravel()
    needles = _bytes_view(needles, out_dtype).ravel()

    i = haystack.argsort()
    pvi = np.searchsorted(haystack, needles, sorter=i)

    # since searchsorted can return length as index:
    pvi[pvi == i.size] -= 1
    pv2 = i[pvi]

    # trim pv2 down to exact matches and create pv1:
    pv1 = np.where(haystack[pv2] == needles)[0]
    pv2 = pv2[pv1]

    if switch:
        pv1, pv2 = pv2, pv1

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
        Index vector; the complement of `pv`.

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
        difference is less than ``tol*max(abs(all differences))``.

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
    L2 : list
       List 2.

    Returns
    -------
    pv1 : 1d ndarray
        Index vector into `L1`.
    pv2 : 1d ndarray
        Index vector into `L2`.

    Notes
    -----
    `pv1` and `pv2` are found such that::

         [L1[i] for i in pv1] == [L2[i] for i in pv2]

    Examples
    --------
    >>> from pyyeti import locate
    >>> pv1, pv2 = locate.list_intersect(['a', 3, 'z', 0],
    ...                                  [0, 'z', 1, 'a'])
    >>> pv1  # doctest: +ELLIPSIS
    array([0, 2, 3]...)
    >>> pv2  # doctest: +ELLIPSIS
    array([3, 1, 0]...)
    >>> locate.list_intersect(['a', 'b'],
    ...                       [1, 2])       # doctest: +ELLIPSIS
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


def merge_lists(list1, list2):
    """
    Merge two lists, trying to maintain order of both

    Parameters
    ----------
    list1 : list
        The first list to merge.
    list2 : list
        The second list to merge.

    Returns
    -------
    mlist : list
        The merged list; guaranteed to be a new list.
    pv1 : list
        List of indices specifying where the elements of `list1` are
        in `mlist`; eg: ``list1 = [mlist[i] for i in pv1]``
    pv2 : list
        List of indices specifying where the elements of `list2` are
        in `mlist`; eg: ``list2 = [mlist[i] for i in pv2]``

    Notes
    -----
    The order of `list1` is maintained. The order of `list2` is
    maintained unless it conflicts with `list1`. When a merge is
    ambiguous, the `list1` elements are merged first.

    Examples
    --------
    >>> from pyyeti import locate
    >>> l1 = ['one', 'four', 'ten']
    >>> l2 = ['zero', 'one', 'two', 'four', 'five']
    >>> l3, i, j = locate.merge_lists(l1, l2)
    >>> l3
    ['zero', 'one', 'two', 'four', 'ten', 'five']
    >>> i, j
    ([1, 3, 4], [0, 1, 2, 3, 5])
    >>> locate.merge_lists(l1, [])
    (['one', 'four', 'ten'], [0, 1, 2], [])
    """
    merged = list1[:]
    elements = []
    for e in list2:
        try:
            i = merged.index(e)
        except ValueError:
            # e is not in merged ... save it so it can be
            # inserted later, just in front of the next
            # common element
            elements.append(e)
        else:
            # e is in merged, so insert currenly accumulated
            # elements that weren't (these get inserted in
            # order, just in front of the common element e)
            for j, x in enumerate(elements):
                merged.insert(i + j, x)
            del elements[:]
    merged.extend(elements)
    # form pv1, knowing that list1 elements are in order:
    pv1 = [0] * len(list1)
    prev = 0
    for i, e in enumerate(list1):
        prev = pv1[i] = merged.index(e, prev)
    return merged, pv1, [merged.index(e) for e in list2]
