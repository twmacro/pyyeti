# -*- coding: utf-8 -*-
"""
Tools for working with data that originated in Nastran. Typically,
a Nastran modes run is executed with the "nas2cam" DMAP (CAM is now
replaced by Python but the DMAP retains the old name). This creates an
op2/op4 file pair which is read by :func:`pyyeti.nastran.rdnas2cam`.
After that, the tools in this module can be used to create rigid-body
modes, form data recovery matrices, make partition vectors based on
sets, form RBE3-like interpolation matrices, etc.

The functions provided by this module can be accessed by just
importing the "nastran" package. For example, you can access the
:func:`rbgeom` function in these two ways:

>>> from pyyeti import nastran
>>> from pyyeti.nastran import n2p
>>> n2p.rbgeom is nastran.rbgeom
True
"""

import math
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
import scipy.linalg as linalg
from pyyeti import locate

__all__ = ['addgrid', 'addulvs', 'build_coords', 'coordcardinfo',
           'expanddof', 'formdrm', 'formrbe3', 'formtran', 'formulvs',
           'get_coordinfo', 'getcoords', 'make_uset', 'mkdofpv',
           'mksetpv', 'mkusetmask', 'rbcoords', 'rbgeom',
           'rbgeom_uset', 'rbmove', 'upasetpv', 'upqsetpv', 'usetprt']


def rbgeom(grids, refpoint=np.array([[0, 0, 0]])):
    """
    Compute 6 rigid-body modes from geometry.

    Parameters
    ----------
    grids : 3-column matrix
        Coordinates in basic; columns are [x, y, z].
    refpoint : integer or 3-element vector
        Defines location that rb modes will be relative to: either row
        index into `grids` (starting at 0) or the [x, y, z] location.

    Returns
    -------
    rb : ndarray
        Rigid-body modes; rows(grids)*6 x 6.

    Notes
    -----
    All grids are assumed to be in the same rectangular coordinate
    system. For a much more sophisticated routine, see
    :func:`rbgeom_uset`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> grids = np.array([[0., 0., 0.], [30., 10., 20.]])
    >>> nastran.rbgeom(grids)
    array([[  1.,   0.,   0.,   0.,   0.,  -0.],
           [  0.,   1.,   0.,  -0.,   0.,   0.],
           [  0.,   0.,   1.,   0.,  -0.,   0.],
           [  0.,   0.,   0.,   1.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   1.,   0.],
           [  0.,   0.,   0.,   0.,   0.,   1.],
           [  1.,   0.,   0.,   0.,  20., -10.],
           [  0.,   1.,   0., -20.,   0.,  30.],
           [  0.,   0.,   1.,  10., -30.,   0.],
           [  0.,   0.,   0.,   1.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   1.,   0.],
           [  0.,   0.,   0.,   0.,   0.,   1.]])
    """
    grids = np.reshape(grids, (-1, 3))
    r = np.shape(grids)[0]
    if np.size(refpoint) == 1:
        grids = grids - grids[refpoint]
    elif np.any(refpoint != [0, 0, 0]):
        grids = grids - refpoint
    rbmodes = np.zeros((r * 6, 6))
    rbmodes[1::6, 3] = -grids[:, 2]
    rbmodes[2::6, 3] = grids[:, 1]
    rbmodes[::6, 4] = grids[:, 2]
    rbmodes[2::6, 4] = -grids[:, 0]
    rbmodes[::6, 5] = -grids[:, 1]
    rbmodes[1::6, 5] = grids[:, 0]
    for i in range(6):
        rbmodes[i::6, i] = 1.
    return rbmodes


def rbgeom_uset(uset, refpoint=np.array([[0, 0, 0]])):
    """
    Compute 6 rigid-body modes from geometry using a USET table.

    Parameters
    ----------
    uset : ndarray
        A 6-column matrix as output by
        :func:`pyyeti.op2.OP2.rdn2cop2`.
    refpoint : integer or vector
        Defines location that rb modes will be relative to. Either an
        integer specifying the node ID (which is in the uset table),
        or it is a coordinates vector [x, y, z] in basic.

    Returns
    -------
    rb : 6-column array
        Rigid-body modes in "global" coordinates.

    Notes
    -----
    The return `rb` is analogous to the output of Nastran VECPLOT
    option 4. Here, "global" means the combination of all local
    coordinate systems. In other words, the rigid-body modes are in
    all the local coordinates of the grids. The refpoint is given
    unit translations and rotations in the basic coordinate system.

    All SPOINTs, all EPOINTs, and GRIDS in the Q-set or in the "left
    over" C-set will have 0's.

    This routine will handle grids in rectangular, cylindrical, and
    spherical coordinates.

    See also
    --------
    :func:`pyyeti.nastran.bulk2uset`, :func:`rbgeom`,
    :func:`pyyeti.op2.rdnas2cam`, :func:`pyyeti.op2.OP2.rdn2cop2`,
    :func:`usetprt`.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> #  first, make a uset table:
    >>> #   node 100 in basic is @ [5, 10, 15]
    >>> #   node 200 in cylindrical coordinate system is @
    >>> #   [r, th, z] = [32, 90, 10]
    >>> cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0],
    ...                     [0, 1, 0]])
    >>> uset = nastran.addgrid(
    ...     None, [100, 200], 'b', [0, cylcoord],
    ...     [[5, 10, 15], [32, 90, 10]], [0, cylcoord])
    >>> np.set_printoptions(precision=2, suppress=True)
    >>> nastran.rbgeom_uset(uset)   # rb modes relative to [0, 0, 0]
    array([[  1.,   0.,   0.,   0.,  15., -10.],
           [  0.,   1.,   0., -15.,   0.,   5.],
           [  0.,   0.,   1.,  10.,  -5.,   0.],
           [  0.,   0.,   0.,   1.,   0.,   0.],
           [  0.,   0.,   0.,   0.,   1.,   0.],
           [  0.,   0.,   0.,   0.,   0.,   1.],
           [  0.,   0.,   1.,   0., -10.,   0.],
           [  0.,  -1.,   0.,  32.,  -0., -10.],
           [  1.,   0.,   0.,   0.,  32.,  -0.],
           [  0.,   0.,   0.,   0.,   0.,   1.],
           [  0.,   0.,   0.,   0.,  -1.,   0.],
           [  0.,   0.,   0.,   1.,   0.,   0.]])
    """
    # find the grids (ignore spoints and epoints)
    r = uset.shape[0]
    grids = uset.index.get_level_values('id')
    dof = uset.index.get_level_values('dof')
    grid_rows = dof != 0
    # get the q-set and the left-over c-set:
    qset = mksetpv(uset, "p", "q")
    if qset.any():
        qdof1 = dof[qset] == 1
        qgrids = grids[qset][qdof1]
        if qgrids.any():
            pvq = mkdofpv(uset, 'p', qgrids)[0]
            grid_rows[pvq] = False

    rbmodes = np.zeros((r, 6))
    if not any(grid_rows):
        return rbmodes
    uset = uset.iloc[grid_rows]
    ngrids = uset.shape[0] // 6

    # rigid-body modes in basic coordinate system:
    uset_dof1 = uset.iloc[::6, :]
    if np.size(refpoint) == 1:
        refpoint = uset_dof1.index.get_loc((refpoint, 1))
    xyz = uset_dof1.loc[:, 'x':'z'].values
    rb = rbgeom(xyz, refpoint)

    # treat as rectangular here; fix cylindrical & spherical below
    rb2 = np.zeros((np.shape(rb)))
    for j in range(ngrids):
        i = 6 * j
        t = uset.iloc[i + 3:i + 6, 1:].values.T
        rb2[i:i + 3] = t @ rb[i:i + 3]
        rb2[i + 3:i + 6] = t @ rb[i + 3:i + 6]

    # fix up cylindrical:
    grid_loc = np.arange(0, uset.shape[0], 6)
    cyl = uset.loc[(slice(None), 2), 'y'] == 2
    if cyl.any():
        grid_loc_cyl = grid_loc[cyl]
        for i in grid_loc_cyl:
            t = uset.iloc[i + 3:i + 6, 1:].values.T
            loc = uset.iloc[i, 1:]
            loc2 = t @ (loc - uset.iloc[i + 2, 1:]).values
            if abs(loc2[1]) + abs(loc2[0]) > 1e-8:
                th = math.atan2(loc2[1], loc2[0])
                c = math.cos(th)
                s = math.sin(th)
                t = np.array([[c, s], [-s, c]])
                rb2[i:i + 2] = t @ rb2[i:i + 2]
                rb2[i + 3:i + 5] = t @ rb2[i + 3:i + 5]

    # fix up spherical:
    sph = uset.loc[(slice(None), 2), 'y'] == 3
    if sph.any():
        grid_loc_sph = grid_loc[sph]
        for i in grid_loc_sph:
            t = uset.iloc[i + 3:i + 6, 1:].values.T
            loc = uset.iloc[i, 1:]
            loc2 = t @ (loc - uset.iloc[i + 2, 1:]).values
            if abs(loc2[1]) + abs(loc2[0]) > 1e-8:
                phi = math.atan2(loc2[1], loc2[0])
                c = math.cos(phi)
                s = math.sin(phi)
                t = np.array([[c, s], [-s, c]])
                rb2[i:i + 2] = t @ rb2[i:i + 2]
                rb2[i + 3:i + 5] = t @ rb2[i + 3:i + 5]
                loc2[:2] = t @ loc2[:2]
            if abs(loc2[2]) + abs(loc2[0]) > 1e-8:
                th = math.atan2(loc2[0], loc2[2])
            else:
                th = 0
            c = math.cos(th)
            s = math.sin(th)
            t = np.array([[s, 0, c], [c, 0, -s], [0, 1, 0]])
            rb2[i:i + 3] = t @ rb2[i:i + 3]
            rb2[i + 3:i + 6] = t @ rb2[i + 3:i + 6]

    # prepare final output:
    rbmodes[grid_rows] = rb2
    return rbmodes


def rbmove(rb, oldref, newref):
    """
    Returns rigid-body modes relative to new reference point.

    Parameters
    ----------
    rb : 6 column ndarray
        Original rigid-body modes; assumed to be n x 6.
    oldref : 3 element array
        Original [x, y, z] reference location in basic coordinates.
    newref : 3 element array
        New [x, y, z] reference location in basic coordinates.

    Returns
    -------
    rbnew : 6 column ndarray
        New rigid-body modes from:  ``rb * rbgeom(oldref, newref)``.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> grids = np.array([[0., 0., 0.], [30., 10., 20.]])
    >>> rb0 = nastran.rbgeom(grids)
    >>> rb1 = nastran.rbgeom(grids, [2., 4., -5.])
    >>> rb1_b = nastran.rbmove(rb0, [0., 0., 0.], [2., 4., -5.])
    >>> np.all(rb1_b == rb1)
    True
    """
    return rb @ rbgeom(oldref, newref)


def rbcoords(rb, verbose=2):
    """
    Return coordinates of each node given rigid-body modes.

    Parameters
    ----------
    rb : 6 column ndarray
       Rigid-body modes. Nodes can be in any mixture of coordinate
       systems. Number of rows is assumed to be (6 x nodes) ... other
       DOF (like SPOINTs) must be partitioned out before calling this
       routine.
    verbose : integer
       If 1, print 1 summary line; and if > 1, print warnings for
       nodes as well.

    Returns
    -------
    coords : ndarray
        A 3-column matrix of [x, y, z] locations of each node,
        relative to same location as `rb`.
    maxdev : float
        Maximum absolute error of any deviation from the expected
        pattern.
    maxerr : float
        Maximum percent deviation; this is the maximum deviation
        for a node divided by the maximum x, y or z coordinate
        location for the node.

    Notes
    -----
    The expected pattern for each node in the rigid-body modes is::

        [ 1 0 0    0   Z  -Y
          0 1 0   -Z   0   X
          0 0 1    Y  -X   0
          0 0 0    1   0   0
          0 0 0    0   1   0
          0 0 0    0   0   1 ]

    That pattern shown assumes the node is in the same coordinate
    system as the reference node. If this is not the case, the 3x3
    coordinate transformation matrix (from reference to local) will
    show up in place of the the 3x3 identity matrix shown above. This
    routine will use that 3x3 matrix to convert coordinates to that of
    the reference before checking for the expected pattern. The
    matrix inversion is done in a least squares sense. This all means
    is that the use of local coordinate systems is acceptable for this
    routine. Zero rows (like what could happen for q-set dof) get
    zero coordinates.

    Raises
    ------
    ValueError
        When `rb` is not 6*n x 6.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> # generate perfect rigid-body modes to test this routine
    >>> coords = np.array([[0, 0, 0],
    ...                    [1, 2, 3],
    ...                    [4, -5, 25]])
    >>> rb = nastran.rbgeom(coords)
    >>> coords_out, mxdev, mxerr = nastran.rbcoords(rb)
    >>> np.allclose(coords_out, coords)
    True
    >>> np.allclose(0., mxdev)
    True
    >>> np.allclose(0., mxerr)
    True

    Now show example when non-rb modes are passed in:

    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> not_rb = np.dot(np.arange(12).reshape(12, 1),
    ...                 np.arange(6).reshape(1, 6))
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> nastran.rbcoords(not_rb)
    Warning:  deviation from standard pattern, node #1 starting at index 0:
      Max deviation = 2.6 units.
      Max % error   = 217%.
      Rigid-Body Rotations:
                 0.0000     0.0000     0.0000
                 0.6000     0.8000     1.0000
                 1.2000     1.6000     2.0000
    <BLANKLINE>
    Warning:  deviation from standard pattern, node #2 starting at index 6:
      Max deviation = 2.6 units.
      Max % error   = 217%.
      Rigid-Body Rotations:
                 0.0000     0.0000     0.0000
                 0.6000     0.8000     1.0000
                 1.2000     1.6000     2.0000
    <BLANKLINE>
    Maximum absolute coordinate location error: 2.6 units
    Maximum % error: 217%.
    (array([[ 1. ,  1.2,  0. ],
           [ 1. ,  1.2,  0. ]]), 2.6000000000000005, 216.66666666666674)
    """
    r, c = np.shape(rb)
    if c != 6:
        raise ValueError("`rb` must have 6 columns")
    if (r // 6) * 6 != r:
        raise ValueError("`rb` must have a multiple of 6 rows")
    n = r // 6
    coords = np.zeros((n, 3))
    maxerr = 0
    maxdev = 0
    haderr = 0
    for j in range(n):
        row = j * 6
        T = rb[row:row + 3, :3]
        R = linalg.lstsq(T, rb[row:row + 3, 3:])[0]
        deltax = R[1, 2]
        deltay = R[2, 0]
        deltaz = R[0, 1]

        deltax2 = -R[2, 1]
        deltay2 = -R[0, 2]
        deltaz2 = -R[1, 0]
        dev = np.max(np.vstack((np.max(np.abs(np.diag(R))),
                                np.abs(deltax - deltax2),
                                np.abs(deltay - deltay2),
                                np.abs(deltaz - deltaz2))))
        coords[j] = [deltax, deltay, deltaz]
        mc = np.max(np.abs(coords[j]))
        if mc > np.finfo(float).eps:
            err = dev / mc * 100.
        else:
            err = dev / np.finfo(float).eps * 100.
        maxdev = max([maxdev, dev])
        maxerr = max([maxerr, err])
        if verbose > 0 and (dev > mc * 1.e-6 or math.isnan(dev)):
            if verbose > 1:
                print("Warning:  deviation from standard pattern, "
                      "node #{} starting at index {}:".
                      format(j + 1, row))
                print("  Max deviation = {:.3g} units.".format(dev))
                print("  Max % error   = {:.3g}%.".format(err))
                print("  Rigid-Body Rotations:")
                for k in range(3):
                    print("         {:10.4f} {:10.4f} {:10.4f}"
                          .format(R[k, 0], R[k, 1], R[k, 2]))
                print("")
            haderr = 1
    if verbose > 0 and haderr:
        print("Maximum absolute coordinate location error: "
              "{:.3g} units".format(maxdev))
        print("Maximum % error: {:.3g}%.".format(maxerr))
    return coords, maxdev, maxerr


def expanddof(dof):
    """
    Expands DOF specification

    Parameters
    ----------
    dof : 1d or 2d array
        `dof` can be input in 2 different ways:

         1. 1d array. Each element is assumed to be a GRID ID. All 6
            DOF associated with the ID will be included in the output.
         2. 2d 2-column DOF array. Each row is: [ID, DOF]. Here, DOF
            specifies which degrees-of-freedom of the ID to find.
            The DOF can be input in the same way as Nastran accepts
            it: 0 or any combo of digits 1-6; eg, 123456 for all 6.

    Returns
    -------
    outdof : 2d ndarray
        The expanded version of the dof input.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> nastran.expanddof([1, 2])             # doctest: +ELLIPSIS
    array([[1, 1],
           [1, 2],
           [1, 3],
           [1, 4],
           [1, 5],
           [1, 6],
           [2, 1],
           [2, 2],
           [2, 3],
           [2, 4],
           [2, 5],
           [2, 6]]...)
    >>> nastran.expanddof([[1, 34], [2, 156]])   # doctest: +ELLIPSIS
    array([[1, 3],
           [1, 4],
           [2, 1],
           [2, 5],
           [2, 6]]...)
    """
    dof = np.atleast_1d(dof).astype(np.int64)
    if dof.ndim < 2 or dof.shape[1] == 1:
        return np.array([[n, i]
                         for n in dof.ravel()
                         for i in range(1, 7)])
    elif dof[:, 1].max() <= 6:
        return dof
    return np.array([[node, int(i)]
                     for node, arg in dof
                     for i in str(arg)])


def mkusetmask(nasset=None):
    r"""
    Get bit-masks for use with the Nastran USET table.

    Parameters
    ----------
    nasset : None or string; optional
        Specifies Nastran set or sets. If a string, can be a single
        set (eg, 'a') or multiple sets combined with the '+' (eg,
        'a+o+m').

    Returns
    -------
    mask : integer or dict
        If `nasset` is None, returns a dictionary of bit-masks that is
        indexed by the lowercase set letter(s). Otherwise, `mask` is
        the bit mask for the specific set(s).

    Notes
    -----
    Note that the analyst rarely needs to use this function directly;
    other routines will call this routine automatically and use the
    resulting vector or mask internally.

    The sets (and supersets) currently accounted for are::

        Sets              Supersets

         M  -------------------------------------\
         S  ------------------------------\       > G --\
         O  -----------------------\       > N --/       \
         Q  ----------------\       > F --/       \       \
         R  ---------\       > A --/       \       \       > P
         C  --\       > T --/       \       > FE    > NE  /
         B  ---> L --/               > D   /       /     /
         E  ------------------------/-----/-------/-----/

    User-defined sets: U1, U2, U3, U4, U5, and U6.

    Note: MSC.Nastran apparently changes the B-set bitmask not only
    between different versions but also between different machines.
    Sometimes the 2nd bit goes to the B-set and sometimes it goes to
    the S-set. However, so far, the S-set always has other bits set
    that can be (and are) checked. Therefore, to work around this
    difficulty, the :func:`pyyeti.op2.OP2.rdn2cop2` routine clears the
    2nd bit for all S-set DOF. Because of that, this routine can
    safely assume that the 2nd bit belongs to the B-set and no manual
    changes are required.

    See also
    --------
    :func:`mksetpv`, :func:`pyyeti.op2.rdnas2cam`,
    :func:`pyyeti.op2.OP2.rdn2cop2`, :func:`usetprt`

    Examples
    --------
    >>> from pyyeti import nastran
    >>> nastran.mkusetmask('q')
    4194304
    >>> nastran.mkusetmask('b')
    2097154
    >>> nastran.mkusetmask('q+b')
    6291458
    """
    m = 1
    b = 2 | (1 << 21)
    o = 4
    r = 8
    s = 1024 | 512
    q = 1 << 22
    c = 1 << 20
    e = 1 << 11
    a = q | r | b | c
    l = c | b
    t = l | r
    f = a | o
    n = f | s
    g = n | m
    p = g | e
    usetmask = {'m': m,
                'b': b,
                'o': o,
                'r': r,
                's': s,
                'q': q,
                'c': c,
                'e': e,
                'a': a,
                'l': l,
                't': t,
                'f': f,
                'n': n,
                'g': g,
                'p': p,
                'fe': f | e,
                'd': e | a,
                'ne': n | e,
                'u1': 1 << 31,
                'u2': 1 << 30,
                'u3': 1 << 29,
                'u4': 1 << 28,
                'u5': 1 << 27,
                'u6': 1 << 26}
    if isinstance(nasset, str):
        sets = nasset.split('+')
        usetmask1 = 0
        for set_ in sets:
            usetmask1 = usetmask1 | usetmask[set_]
        return usetmask1
    return usetmask


def usetprt(file, uset, printsets="M,S,O,Q,R,C,B,E,L,T,A,F,N,G"):
    r"""
    Print Nastran DOF set membership information from USET table.

    Parameters
    ----------
    file : string or file handle
        Either a name of a file or a file handle as returned by
        :func:`open`. Use 1 to write to the screen, 0 to write nothing
        -- just get output.
    uset : ndarray
        A 6-column matrix as output by
        :func:`pyyeti.op2.OP2.rdn2cop2`.
    printsets : string; optional
        A comma delimited string specifying which sets to print, see
        description below.

    Returns
    -------
    table : pandas DataFrame
        DataFrame showing set membership. The index has 'id', 'dof',
        'dof#'; the columns are the sets requested in `printsets` in
        the order given below. The rows will be truncated to non-zero
        rows.

    Notes
    -----
    `printsets` is a comma delimited strings that specifies which sets
    to print. It can be input in lower or upper case. Sets that are
    identical are printed together (as G and P often are). The value
    of "*" is equivalent to specifying all sets::

        "M,S,O,Q,R,C,B,E,L,T,A,D,F,FE,N,NE,G,P,U1,U2,U3,U4,U5,U6"

    For example, `printsets` = "R, C, B, A" will print only those sets
    (but not necessarily in that order).

    The sets (and supersets) currently accounted for are::

       Sets              Supersets

        M  -------------------------------------\
        S  ------------------------------\       > G --\
        O  -----------------------\       > N --/       \
        Q  ----------------\       > F --/       \       \
        R  ---------\       > A --/       \       \       > P
        C  --\       > T --/       \       > FE    > NE  /
        B  ---> L --/               > D   /       /     /
        E  ------------------------/-----/-------/-----/

    User-defined sets: U1, U2, U3, U4, U5, and U6.

    See also
    --------
    :func:`pyyeti.op2.OP2.rdn2cop2`, :func:`mksetpv`,
    :func:`rbgeom_uset`, :func:`pyyeti.op2.rdnas2cam`

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> #  first, make a uset table:
    >>> #   node 100 in basic is @ [5, 10, 15]
    >>> #   node 200 in cylindrical coordinate system is @
    >>> #   [r, th, z] = [32, 90, 10]
    >>> cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0],
    ...                     [0, 1, 0]])
    >>> uset = nastran.addgrid(
    ...     None, [100, 200], ['b', 'c'], [0, cylcoord],
    ...     [[5, 10, 15], [32, 90, 10]], [0, cylcoord])
    >>> table = nastran.usetprt(
    ...      1, uset, printsets='r, c')  # doctest: +ELLIPSIS
    R-set
          -None-
    <BLANKLINE>
    C-set
                 -1-        -2-    ...   -6-     ...  -10-
         1=      200-1      200-2  ...   200-6
    <BLANKLINE>
    >>> table = nastran.usetprt(
    ...      1, uset, printsets='*')  # doctest: +ELLIPSIS
    M-set, S-set, O-set, Q-set, R-set, E-set, U1-set, ... U6-set
          -None-
    <BLANKLINE>
    C-set
                 -1-        -2-    ...   -6-     ...  -10-
         1=      200-1      200-2  ...   200-6
    <BLANKLINE>
    B-set
                 -1-        -2-    ...   -6-     ...  -10-
         1=      100-1      100-2  ...   100-6
    <BLANKLINE>
    L-set, T-set, A-set, D-set, F-set, FE-set, ..., G-set, P-set
                 -1-        -2-    ...   -10-
         1=      100-1      100-2  ...    200-4 =    10
        11=      200-5      200-6
    <BLANKLINE>
    >>> table = nastran.usetprt(1, uset)  # doctest: +ELLIPSIS
    M-set, S-set, O-set, Q-set, R-set, E-set
          -None-
    <BLANKLINE>
    C-set
                 -1-        -2-    ...   -6-     ...  -10-
         1=      200-1      200-2  ...   200-6
    <BLANKLINE>
    B-set
                 -1-        -2-    ...   -6-     ...  -10-
         1=      100-1      100-2  ...   100-6
    <BLANKLINE>
    L-set, T-set, A-set, F-set, N-set, G-set
                 -1-        -2-    ...   -10-
         1=      100-1      100-2  ...    200-4 =    10
        11=      200-5      200-6
    <BLANKLINE>
    >>> table   # doctest: +ELLIPSIS
                  M  S  O  Q  R  C  B  E   L   T   A   F   N   G
    id  dof dof#...
    100 1   1     0  0  0  0  0  0  1  0   1   1   1   1   1   1
        2   2     0  0  0  0  0  0  2  0   2   2   2   2   2   2
        3   3     0  0  0  0  0  0  3  0   3   3   3   3   3   3
        4   4     0  0  0  0  0  0  4  0   4   4   4   4   4   4
        5   5     0  0  0  0  0  0  5  0   5   5   5   5   5   5
        6   6     0  0  0  0  0  0  6  0   6   6   6   6   6   6
    200 1   7     0  0  0  0  0  1  0  0   7   7   7   7   7   7
        2   8     0  0  0  0  0  2  0  0   8   8   8   8   8   8
        3   9     0  0  0  0  0  3  0  0   9   9   9   9   9   9
        4   10    0  0  0  0  0  4  0  0  10  10  10  10  10  10
        5   11    0  0  0  0  0  5  0  0  11  11  11  11  11  11
        6   12    0  0  0  0  0  6  0  0  12  12  12  12  12  12
    """
    usetmask = mkusetmask()
    nasset = uset.iloc[:, 0].values
    allsets = (list('MSOQRCBELTADF') +
               ['FE', 'N', 'NE', 'G', 'P', 'U1',
                'U2', 'U3', 'U4', 'U5', 'U6'])
    table = []
    for _set in allsets:
        table.append((nasset & usetmask[_set.lower()]) != 0)
    table = np.column_stack(table)

    # replace True's with set membership number: 1 to ?
    table = table.astype(np.int64)
    r, c = table.shape
    n = np.count_nonzero(table, axis=0)
    for i in range(c):
        pv = table[:, i].astype(bool)
        table[pv, i] = 1 + np.arange(n[i])
    if printsets == '*':
        printsets = allsets
    else:
        printsets = (''.join(printsets.split())).upper().split(',')
    # keep only columns in table that are printed:
    printpv, pv2 = locate.list_intersect(allsets, printsets)
    # make sure printsets is in order of table:
    printsets = [printsets[i] for i in pv2]
    table = table[:, printpv]

    pv = table.any(axis=1)
    if pv.any():
        # make a dataframe:
        uset_index = uset.index
        ind = [uset_index.get_level_values(i)
               for i in range(uset_index.nlevels)]
        ind.append(1 + np.arange(r))
        ind = pd.MultiIndex.from_arrays(
            ind, names=[*uset_index.names, 'dof#'])
        return_table = pd.DataFrame(
            table, index=ind, columns=printsets)
        return_table = return_table.loc[pv]
    else:
        return_table = None

    if file == 0:
        return return_table

    if isinstance(file, str):
        f = open(file, "w")
    elif file == 1:
        f = sys.stdout
    else:
        f = file

    nsets = len(printsets)
    colheader = ("     "
                 "        -1-        -2-        -3-        -4-"
                 "        -5-        -6-        -7-        -8-"
                 "        -9-       -10-")
    printed = np.zeros((nsets), dtype=np.int64)

    s = 0
    while s < nsets:  # loop over printing-sets:
        header = printsets[s] + "-set"
        printed[s] = 1
        S = s + 1
        while S < nsets:
            if np.all(table[:, S] == table[:, s]):
                header += ", " + printsets[S] + "-set"
                printed[S] = 1
            S += 1

        # form a modified version of USET for printing this set
        pv = table[:, s].nonzero()[0]

        # set s for next loop:
        s = (printed == 0).nonzero()[0]
        if s.size > 0:
            s = s[0]
        else:
            s = nsets

        if pv.any():
            f.write("{}\n{}\n".format(header, colheader))
            # uset_mod = uset[pv, :2].astype(np.int64)
            uset_mod = uset.iloc[pv, :0].reset_index().values
            full_rows = pv.size // 10
            rem = pv.size - 10 * full_rows
            count = 1
            if full_rows:
                usetfr = uset_mod[:full_rows * 10]
                for j in range(full_rows):
                    f.write('{:6d}='.format(count))
                    for k in range(10):
                        r = j * 10 + k
                        f.write(
                            ' {:8d}-{:1d}'.format(usetfr[r, 0],
                                                  usetfr[r, 1]))
                    f.write(' ={:6d}\n'.format(count + 9))
                    count += 10
            if rem:
                uset_rem = uset_mod[-rem:].astype(np.int64)
                f.write('{:6d}='.format(count))
                for j in range(rem):
                    f.write(' {:8d}-{:1d}'.format(uset_rem[j, 0],
                                                  uset_rem[j, 1]))
                f.write("\n")
            f.write("\n")
        else:
            f.write("{}\n      -None-\n\n".format(header))
    if isinstance(file, str):
        f.close()
    return return_table


def mksetpv(uset, major, minor):
    r"""
    Make a set partition vector from a Nastran USET table.

    Parameters
    ----------
    uset : ndarray
        A 6-column matrix as output by
        :func:`pyyeti.op2.OP2.rdn2cop2`.
    majorset : integer or string
        An integer bitmask or a set letter or letters (see below).
    minorset : integer or string
        An integer bitmask or a set letter or letters.

    Returns
    -------
    pv : 1d ndarray
        A True/False vector for partitioning `minorset` from
        `majorset`. Length = number of DOF in `majorset`.

    Notes
    -----
    The inputs majorset and minorset can be specified as a combination
    of sets by using the '+' sign. See help in :func:`mkusetmask` for
    more information on how to specify the sets.

    The sets (and supersets) currently accounted for are::

        Sets              Supersets

         M  -------------------------------------\
         S  ------------------------------\       > G --\
         O  -----------------------\       > N --/       \
         Q  ----------------\       > F --/       \       \
         R  ---------\       > A --/       \       \       > P
         C  --\       > T --/       \       > FE    > NE  /
         B  ---> L --/               > D   /       /     /
         E  ------------------------/-----/-------/-----/

    User-defined sets: U1, U2, U3, U4, U5, and U6.

    See also
    --------
    :func:`mkdofpv`, :func:`pyyeti.op2.rdnas2cam`,
    :func:`formulvs`, :func:`usetprt`, :func:`rbgeom_uset`,
    :func:`mkusetmask`, :func:`pyyeti.op2.OP2.rdn2cop2`

    Raises
    ------
    ValueError
        When `minorset` is not completely contained in `majorset`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> # First, make a uset table
    >>> #  node 100 in basic is @ [5, 10, 15]
    >>> #  node 200 in cylindrical is @ [r, th, z] = [32, 90, 10]
    >>> #  z_cyl = x-basic; r_cyl = y-basic
    >>> #  Also, put 100 in b-set and 200 in m-set.
    >>> cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0],
    ...                     [0, 1, 0]])
    >>> uset = nastran.addgrid(
    ...     None, [100, 200], ['b', 'm'], [0, cylcoord],
    ...     [[5, 10, 15], [32, 90, 10]], [0, cylcoord])
    >>> bset = nastran.mksetpv(uset, 'p', 'b')        # 1:6 are true
    >>> np.set_printoptions(linewidth=75)
    >>> bset
    array([ True,  True,  True,  True,  True,  True, False, False, False,
           False, False, False], dtype=bool)
    >>> mset = nastran.mksetpv(uset, 'p', 'm')        # 7:12 are true
    >>> mset
    array([False, False, False, False, False, False,  True,  True,  True,
            True,  True,  True], dtype=bool)
    >>> rcqset = nastran.mksetpv(uset, 'p', 'r+c+q')  # all false
    >>> rcqset
    array([False, False, False, False, False, False, False, False, False,
           False, False, False], dtype=bool)
    """
    if isinstance(major, str):
        major = mkusetmask(major)
    if isinstance(minor, str):
        minor = mkusetmask(minor)
    uset_set = uset['uset'].values
    pvmajor = (uset_set & major) != 0
    pvminor = (uset_set & minor) != 0
    if np.any(~pvmajor & pvminor):
        raise ValueError("`minorset` is not completely contained"
                         "in `majorset`")
    pv = pvminor[pvmajor]
    return pv


def mkdofpv(uset, nasset, dof):
    """
    Make a DOF partition vector for a particular set from a Nastran
    USET table.

    Parameters
    ----------
    uset : ndarray
        A 6-column matrix as output by
        :func:`pyyeti.op2.OP2.rdn2cop2`. Allowed to have only the
        first two columns if ``nasset == 'p'``.
    nasset : string or integer
        The set(s) to partition the dof out of (eg, 'p' or 'b+q').
        May also be an integer bitmask (see :func:`mkusetmask` for
        more information).
    dof : 1d or 2d array
        `dof` can be input in 2 different ways:

         1. 1 column, each row is an ID (grid, spoint, etc). All
            DOF associated with the ID that are in the set will be
            included. An error will be generated if any ID is
            missing.
         2. 2 column DOF array, each row is: [ID DOF]. Here, DOF
            specifies which degrees-of-freedom of the ID to find.
            The DOF can be input in the same way as Nastran accepts
            it: 0 or any combo of digits 1-6; eg, 123456 for all 6.
            An error is generated if any DOF are missing. See
            examples.

    Returns
    -------
    pv : vector
        Index vector for partitioning dof out of set; this
        maintains the order of DOF as specified.
    outdof : vector
        The expanded version of the dof input, in order of output.

    Raises
    ------
    ValueError
        When requested `dof` are not found in the `nasset`.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> # Want an A-set partition vector for all available a-set dof
    >>> # of grids 100 and 200:
    >>> ids = np.array([[100], [200]])
    >>> uset = nastran.addgrid(
    ...     None, [100, 200], 'b', 0,
    ...     [[5, 10, 15], [32, 90, 10]], 0)
    >>> nastran.mkdofpv(uset, "a", ids)         # doctest: +ELLIPSIS
    (array([ 0,  1,  2,  3,  4,  5... 10, 11]...), array([[100,   1],
           [100,   2],
           [100,   3],
           [100,   4],
           [100,   5],
           [100,   6],
           [200,   1],
           [200,   2],
           [200,   3],
           [200,   4],
           [200,   5],
           [200,   6]]...))
    >>> # add an spoint for testing:
    >>> uset = uset.append(nastran.make_uset(991, 0, 4194304))
    >>> # request spoint 991 and dof 123 for grid 100 (in that order):
    >>> ids2 = [[991, 0], [100, 123]]
    >>> nastran.mkdofpv(uset, "a", ids2)        # doctest: +ELLIPSIS
    (array([12,  0,  1,  2]...), array([[991,   0],
           [100,   1],
           [100,   2],
           [100,   3]]...))
    """
    if isinstance(uset, pd.DataFrame):
        if nasset != 'p':
            setpv = mksetpv(uset, "p", nasset)
            uset = uset.loc[setpv]
        uset_set = (uset.index.get_level_values('id') * 10 +
                    uset.index.get_level_values('dof'))
    else:
        if nasset == 'p':
            uset_set = (uset[:, 0] * 10 + uset[:, 1]).astype(np.int64)
        else:
            setpv = mksetpv(uset, "p", nasset)
            uset_set = (uset[setpv, 0] * 10 +
                        uset[setpv, 1]).astype(np.int64)

    dof = expanddof(dof)
    dof = dof[:, 0] * 10 + dof[:, 1]

    i = np.argsort(uset_set)
    pvi = np.searchsorted(uset_set, dof, sorter=i)
    # since searchsorted can return length as index:
    pvi[pvi == i.size] -= 1
    pv = i[pvi]
    chk = uset_set[pv] != dof
    if np.any(chk):
        ids = (dof[chk] // 10)
        dof = dof[chk] - 10 * ids
        missing_dof = np.column_stack((ids, dof))
        msg = ("set '{}' does not contain all of the dof in `dof`."
               " These are missing:\n{!s}"
               .format(nasset, missing_dof))
        raise ValueError(msg)
    ids = dof // 10
    dof = dof - 10 * ids
    outdof = np.column_stack((ids, dof))
    return pv, outdof


def coordcardinfo(uset, cid=None):
    """
    Returns 'coordinate card' data from information in USET table

    Parameters
    ----------
    uset : ndarray
        A 6-column matrix as output by
        :func:`pyyeti.op2.OP2.rdn2cop2`.
    cid : None or integer
        If integer, it is the id of the coordinate system to get data
        for. If None, all coordinate system information is returned.

    Returns
    -------
    ci : list or dictionary
        If `cid` was an integer, the return is a list::

            [name, [[4x3 matrix as shown below]] ]

        The 4x3 matrix is (as described in :func:`addgrid`)::

                    [ cid type reference_id ]
                    [ Ax   Ay   Az          ]
                    [ Bx   By   Bz          ]
                    [ Cx   Cy   Cz          ]

        If cid was None, the return is a dictionary of lists for all
        coordinate systems in `uset` (not including 0)::

            {cid1 : [name, ...], cid2 : [...]}.

        `name` is either 'CORD2R' (`type` == 1), 'CORD2C' (`type` ==
        2), or 'CORD2S' (`type` ==3). `ref` is always 0, regardless
        what the original reference coordinate system was. `A`, `B`,
        `C` are the 3-element vectors defining the origin (`A`), the
        Z-axis direction (`B`), and the X-axis direction (`C`).

    Notes
    -----
    The only way to get the basic system (cid = 0) is to request it
    specifically (and `uset` could be anything in this case)::

        c0 = coordcardinfo(uset, 0)

    The return dictionary will be empty if `cid` is None and there are
    no coordinate systems other than 0 in the `uset` table.

    Raises
    ------
    ValueError
        When requested `cid` is not found.

    Examples
    --------
    >>> import numpy as np
    >>> from pyyeti import nastran
    >>> sccoord = np.array([[501, 1, 0], [2345.766, 0, 0],
    ...                     [2345.766, 10, 0], [3000, 0, 0]])
    >>> uset = nastran.addgrid(None, 1001, 'b', sccoord, [0, 0, 0],
    ...                        sccoord)
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> nastran.coordcardinfo(uset)
    {501: ['CORD2R', array([[  501.   ,     1.   ,     0.   ],
           [ 2345.766,     0.   ,     0.   ],
           [ 2345.766,     1.   ,     0.   ],
           [ 2346.766,     0.   ,     0.   ]])]}
    >>> nastran.coordcardinfo(uset, 0)
    ['CORD2R', array([[ 0.,  1.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]])]
    >>> nastran.coordcardinfo(uset, 501)
    ['CORD2R', array([[  501.   ,     1.   ,     0.   ],
           [ 2345.766,     0.   ,     0.   ],
           [ 2345.766,     1.   ,     0.   ],
           [ 2346.766,     0.   ,     0.   ]])]
    >>> # add random-ish cylindrical and spherical systems to test:
    >>> cylcoord = np.array([[601, 2, 501], [10, 20, 30],
    ...                      [100, 20, 30], [10, 1, 1]])
    >>> sphcoord = np.array([[701, 3, 601], [35, 15, -10],
    ...                      [55, 15, -10], [45, 30, 1]])
    >>> uset = nastran.addgrid(uset, 1002, 'b', cylcoord, [2, 90, 5],
    ...                        cylcoord)
    >>> uset = nastran.addgrid(
    ...     uset, 1003, 'b', sphcoord, [12, 40, 45], sphcoord)
    >>> cyl601 = nastran.coordcardinfo(uset, 601)
    >>> sph701 = nastran.coordcardinfo(uset, 701)
    >>> uset = nastran.addgrid(uset, 2002, 'b', cyl601[1], [2, 90, 5],
    ...                        cyl601[1])
    >>> uset = nastran.addgrid(
    ...     uset, 2003, 'b', sph701[1], [12, 40, 45], sph701[1])
    >>> np.allclose(uset.loc[(1002, 1), 'x':'z'],
    ...             uset.loc[(2002, 1), 'x':'z'])
    True
    >>> np.allclose(uset.loc[(1003, 1), 'x':'z'],
    ...             uset.loc[(2003, 1), 'x':'z'])
    True
    """
    if cid == 0:
        return ['CORD2R', np.array([[0, 1, 0],
                                    [0., 0., 0.],
                                    [0., 0., 1.],
                                    [1., 0., 0.]])]

    dof = uset.index.get_level_values('dof')
    pv = (dof == 2).nonzero()[0]
    if pv.size == 0:
        if cid is not None:
            raise ValueError('{} not found ... USET table '
                             'has no grids?'.format(cid))
        return {}

    def _getlist(coordinfo):
        coordinfo = coordinfo.values
        A = coordinfo[1]
        # transpose so T transforms from basic to local:
        T = coordinfo[2:].T
        B = A + T[2]
        C = A + T[0]
        typ = int(coordinfo[0, 1])
        name = ['CORD2R', 'CORD2C', 'CORD2S'][typ - 1]
        return [name, np.vstack(([cid, typ, 0], A, B, C))]

    if cid is not None:
        pv2 = (uset.iloc[pv, 1] == cid).nonzero()[0]
        if pv2.size == 0:
            raise ValueError('{} not found in USET table.'.
                             format(cid))
        pv2 = pv2[0]
        r = pv[pv2]
        coordinfo = uset.iloc[r:r + 5, 1:]
        return _getlist(coordinfo)

    CI = {}
    pv2 = (uset.iloc[pv, 1] > 0).nonzero()[0]
    if pv2.size == 0:
        return CI
    pv = pv[pv2]
    ids = set(uset.iloc[pv, 1].astype(np.int64))
    for cid in ids:
        pv2 = (uset.iloc[pv, 1] == cid).nonzero()[0][0]
        r = pv[pv2]
        coordinfo = uset.iloc[r:r + 5, 1:]
        CI[cid] = _getlist(coordinfo)
    return CI


def _get_coordinfo_byid(refid, uset):
    """
    Returns 5x3 coordinate system information for the reference
    coordinate system.

    Parameters
    ----------
    refid : integer
        Coordinate system id.
    uset : ndarray
        A 6-column matrix as output by
        :func:`pyyeti.op2.OP2.rdn2cop2`.

    Returns
    -------
    cordinfo : 2d ndarray
        5x3 coordinate system information for `refid`.

    See :func:`get_coordinfo` for more information.
    """
    if refid == 0:
        return np.vstack((np.array([[0, 1, 0], [0., 0., 0.]]),
                          np.eye(3)))
    try:
        dof = uset.index.get_level_values('dof')
        pv = (dof == 2).nonzero()[0]
        pos = (uset.iloc[pv, 1] == refid).nonzero()[0][0]
        if pos.size > 0:
            i = pv[pos]
            return uset.iloc[i:i + 5, 1:].values
    except:
        raise ValueError('reference coordinate id {} not '
                         'found in `uset`.'.format(refid))


def get_coordinfo(cord, uset, coordref):
    """
    Function for getting coordinfo as needed by the USET table.
    Called by addgrid.

    Parameters
    ----------
    cord : scalar or 4x3 array_like
        If scalar, it is a coordinate system id (must be 0 or appear
        in either `uset` or `coordref`). If 4x3 matrix, format is as
        on a Nastran CORD2* card::

            [ id type reference_id ]
            [ Ax   Ay   Az         ]
            [ Bx   By   Bz         ]
            [ Cx   Cy   Cz         ]

        where type is 0 (rectangular), 1 (cylindrical), or 2
        (spherical).
    uset : ndarray
        A 6-column matrix as output by
        :func:`pyyeti.op2.OP2.rdn2cop2`. Not used unless needed.
    coordref : dictionary
        Read/write dictionary with the keys being the coordinate
        system id and the values being the 5x3 matrix returned below.
        For speed reasons, this routine will look in `coordref` before
        `uset` for a coordinate system. Can be empty.

    Returns
    -------
    cordout : 5x3 ndarray
        Coordinate information in a 5x3 matrix:

        .. code-block:: none

            [id  type 0]  # output coord. sys. id and type
            [xo  yo  zo]  # origin of coord. system
            [    T     ]  # 3x3 transformation to basic
            Note that T is for the coordinate system, not a grid
            (unless type = 0 which means rectangular)

    Notes
    -----
    If neither `uset` nor `coordref` have a needed coordinate system,
    this routine will error out.

    See also
    --------
    :func:`addgrid`, :func:`pyyeti.op2.OP2.rdn2cop2`
    """
    if np.size(cord) == 1:
        try:
            return coordref[cord]
        except KeyError:
            ci = _get_coordinfo_byid(cord, uset)
            coordref[cord] = ci
            return ci
    cord = np.atleast_2d(cord)
    cid_type = cord[0, :2].astype(np.int64)
    try:
        refinfo = coordref[cord[0, 2]]
    except KeyError:
        refinfo = _get_coordinfo_byid(cord[0, 2], uset)
        coordref[cord[0, 2]] = refinfo
    a = cord[1]
    b = cord[2]
    c = cord[3]
    a2r = math.pi / 180.
    if refinfo[0, 1] == 2:   # cylindrical
        a = np.hstack((a[0] * math.cos(a[1] * a2r),
                       a[0] * math.sin(a[1] * a2r),
                       a[2]))
        b = np.hstack((b[0] * math.cos(b[1] * a2r),
                       b[0] * math.sin(b[1] * a2r),
                       b[2]))
        c = np.hstack((c[0] * math.cos(c[1] * a2r),
                       c[0] * math.sin(c[1] * a2r),
                       c[2]))
    if refinfo[0, 1] == 3:   # spherical
        s = math.sin(a[1] * a2r)
        a = a[0] * np.hstack((s * math.cos(a[2] * a2r),
                              s * math.sin(a[2] * a2r),
                              math.cos(a[1] * a2r)))
        s = math.sin(b[1] * a2r)
        b = b[0] * np.hstack((s * math.cos(b[2] * a2r),
                              s * math.sin(b[2] * a2r),
                              math.cos(b[1] * a2r)))
        s = math.sin(c[1] * a2r)
        c = c[0] * np.hstack((s * math.cos(c[2] * a2r),
                              s * math.sin(c[2] * a2r),
                              math.cos(c[1] * a2r)))
    ab = b - a
    ac = c - a
    z = ab / linalg.norm(ab)
    y = np.cross(z, ac)
    y = y / linalg.norm(y)
    x = np.cross(y, z)
    x = x / linalg.norm(x)
    Tg = refinfo[2:]
    location = refinfo[1] + Tg @ a
    T = Tg @ np.vstack((x, y, z)).T
    row1 = np.hstack((cid_type, 0))
    coordinfo = np.vstack((row1, location, T))
    coordref[cid_type[0]] = coordinfo
    return coordinfo


def build_coords(cords):
    """
    Builds the coordinate system dictionary from array of coordinate
    card information.

    Parameters
    ----------
    cords : 2d array_like
        2d array, n x 12:

        .. code-block:: none

            [cid, ctype, refcid, a1, a2, a3, b1, b2, b3, c1, c2, c3]
            where:
                ctype = 1 for rectangular
                ctype = 2 for cylindrical
                ctype = 3 for spherical

    Returns
    -------
    coordref : dictionary
        Dictionary with the keys being the coordinate system id
        (`cid`) and the values being the 5x3 matrix::

            [cid ctype 0]  # output coord. sys. id and type
            [xo   yo  zo]  # origin of coord. system
            [     T     ]  # 3x3 transformation to basic
            Note that T is for the coordinate system, not a grid
            (unless type = 0 which means rectangular)

    Notes
    -----
    This routine loops over the coordinate systems according to
    reference cid order.

    Raises
    ------
    RuntimeError
        When a reference cid is not found.
    RuntimeError
        When non-equal duplicate coordinate systems are found. (Equal
        duplicates are quietly ignored).
    """
    # resolve coordinate systems, and store them in a dictionary:
    cords = np.atleast_2d(cords)
    coordref = {}
    if np.size(cords, 0) > 0:
        j = np.argsort(cords[:, 0])
        cords = cords[j, :]
        cids = cords[:, 0]
        duprows = np.nonzero(np.diff(cids) == 0)[0]
        if duprows.size > 0:
            delrows = []
            for i in duprows:
                if np.all(cords[i] == cords[i + 1]):
                    delrows.append(i + 1)
                else:
                    raise RuntimeError('duplicate but unequal '
                                       'coordinate systems detected.'
                                       ' cid = {}'.format(cids[i]))
            cords = np.delete(cords, delrows, axis=0)
            cids = cords[:, 0]
        # make a uset table for the cord cards ...
        #  but have to do it in order:
        ref_ids = 0
        n = np.size(cords, 0)
        selected = np.zeros(n, dtype=np.int64)
        loop = 1
        while np.any(selected == 0):
            pv = locate.find_vals(cords[:, 2], ref_ids)
            if pv.size == 0:
                msg = ('Could not resolve coordinate systems. Need '
                       'these coordinate cards:\n{!s}'
                       .format(ref_ids))
                raise RuntimeError(msg)
            selected[pv] = loop
            loop += 1
            ref_ids = cords[pv, 0]
        J = np.argsort(selected)
        for j in range(n):
            cs = np.reshape(cords[J[j], :], (4, 3))   # , order='C')
            addgrid(None, j + 1, 'b', 0, [0, 0, 0], cs, coordref)
    return coordref


def getcoords(uset, gid, csys, coordref=None):
    r"""
    Get coordinates of a grid or location in a specified coordinate
    system.

    Parameters
    ----------
    uset : ndarray
        A 6-column matrix as output by
        :func:`pyyeti.op2.OP2.rdn2cop2`.
    gid : integer or 3 element vector
        If integer, it is a grid id in `uset`. Otherwise, it is a 3
        element vector:  [x, y, z] specifiy location in basic.
    csys : integer or 4x3 matrix
        Specifies coordinate system to get coordinates of `gid` in.
        If integer, it is the id of the coordinate system which must
        be defined in either `uset` or `coordref` (unless it is 0).
        If a 4x3 matrix, it completely defines the coordinate system::

              [ cid type reference_id ]
              [ Ax   Ay   Az          ]
              [ Bx   By   Bz          ]
              [ Cx   Cy   Cz          ]

        See help on :func:`addgrid` for more information on the 4x3.
    coordref : dictionary or None; optional
        If None, this input is ignored. Otherwise, it is a read/write
        dictionary with the keys being the coordinate system id and
        the values being the 5x3 matrix returned below. For speed
        reasons, this routine will look in `coordref` before `uset`
        for a coordinate system. Can be empty.

    Returns
    -------
    coords : ndarray
        3-element ndarray of location in `csys`::

            - Rectangular: [x, y, z]
            - Cylindrical: [R, theta, z]    (theta is in deg)
            - Spherical:   [R, theta, phi]  (theta and phi are in deg)

    Notes
    -----
    Coordinate conversions from global to basic are (where
    [xo; yo; zo] is the coordinate system location in basic and T is
    the coordinate transform to basic):

    Rectangular (type = 1)::

        [xb; yb; zb] = T*[x; y; z] + [xo; yo; zo]

    .. math::
        \left\{
          \begin{array}{c} x_b \\ y_b \\ z_b \end{array}
        \right\}
        = \textbf{T}
        \left\{
          \begin{array}{c} x \\ y \\ z \end{array}
        \right\}
        +
        \left\{
          \begin{array}{c} x_o \\ y_o \\ z_o \end{array}
        \right\}

    Cylindrical (type = 2)::

        # c = cos(theta); s = sin(theta)
        [xb; yb; zb] = T*[R c; R s; z] + [xo; yo; zo]

    .. math::
        \left\{
          \begin{array}{c} x_b \\ y_b \\ z_b \end{array}
        \right\}
        = \textbf{T}
        \left\{
          \begin{array}{c} R \cos \theta \\ R \sin \theta \\ z
          \end{array}
        \right\}
        +
        \left\{
          \begin{array}{c} x_o \\ y_o \\ z_o \end{array}
        \right\}

    Spherical (type = 3)::

        # s1 = sin(theta); s2 = sin(phi)
        [xb; yb; zb] = T*[R s1 c2; R s1 s2; R c1] + [xo; yo; zo]

    .. math::
        \left\{
          \begin{array}{c} x_b \\ y_b \\ z_b \end{array}
        \right\}
        = \textbf{T}
        \left\{
          \begin{array}{c}
          R \sin \theta \cos \phi \\
          R \sin \theta \sin \phi \\
          R \cos \theta
          \end{array}
        \right\}
        +
        \left\{
          \begin{array}{c} x_o \\ y_o \\ z_o \end{array}
        \right\}

    This routine does the inverse of those equations, as follows:

    Rectangular (type = 1)::

        [x; y; z] = T'*([xb; yb; zb] - [xo; yo; zo])

    .. math::
        \left\{
          \begin{array}{c} x \\ y \\ z \end{array}
        \right\}
        = \textbf{T}^{\rm T}
        \left\{
          \begin{array}{c} x_b - x_o \\ y_b - y_o \\ z_b - z_o
          \end{array}
        \right\}

    Cylindrical (type = 2)::

        [x; y; z] = T'*([xb; yb; zb] - [xo; yo; zo])
        R = rss(x, y)
        theta = atan2(y, x)

    .. math::
        \left\{
          \begin{array}{c} x \\ y \\ z \end{array}
        \right\}
        = \textbf{T}^{\rm T}
        \left\{
          \begin{array}{c} x_b - x_o \\ y_b - y_o \\ z_b - z_o
          \end{array}
        \right\}

        R = \sqrt{x^2 + y^2}

        \theta = \mathrm{atan2}(y, x)

    Spherical (type = 3)::

        [x; y; z] = T'*([xb; yb; zb] - [xo; yo; zo])
        R = rss(x, y, z)
        phi = atan2(y, x)
        if abs(sin(phi)) > abs(cos(phi)):
            theta = atan2(y/sin(phi), z)
        else:
            theta = atan2(x/cos(phi), z)

    .. math::
        \left\{
          \begin{array}{c} x \\ y \\ z \end{array}
        \right\}
        = \textbf{T}^{\rm T}
        \left\{
          \begin{array}{c} x_b - x_o \\ y_b - y_o \\ z_b - z_o
          \end{array}
        \right\}

        R = \sqrt{x^2 + y^2 + z^2}

        \phi = \mathrm{atan2}(y, x)

        \theta =
        \begin{cases}
        \mathrm{atan2}(y/(\sin \phi), z),
        &\text{if }
        \left|{\sin \phi}\right| > \left|{\cos \phi}\right| \\
        \mathrm{atan2}(x/(\cos \phi), z),
        &\text{otherwise}
        \end{cases}

    See also
    --------
    :func:`pyyeti.op2.OP2.rdn2cop2`, :func:`pyyeti.nastran.bulk2uset`,
    :func:`coordcardinfo`, :func:`pyyeti.nastran.wtcoordcards`,
    :func:`rbgeom_uset`.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> # node 100 in basic is @ [5, 10, 15]
    >>> # node 200 in cylindrical coordinate system is @
    >>> # [r, th, z] = [32, 90, 10]
    >>> cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0],
    ...                     [0, 1, 0]])
    >>> sphcoord = np.array([[2, 3, 0], [0, 0, 0], [0, 1, 0],
    ...                      [0, 0, 1]])
    >>> uset = None
    >>> uset = nastran.addgrid(uset, 100, 'b', 0, [5, 10, 15], 0)
    >>> uset = nastran.addgrid(uset, 200, 'b', cylcoord,
    ...                        [32, 90, 10], cylcoord)
    >>> uset = nastran.addgrid(uset, 300, 'b', sphcoord,
    ...                        [50, 90, 90], sphcoord)
    >>> np.set_printoptions(precision=2, suppress=True)
    >>> # get coordinates of node 200 in basic:
    >>> nastran.getcoords(uset, 200, 0)
    array([ 10.,   0.,  32.])
    >>> # get coordinates of node 200 in cylindrical (cid 1):
    >>> nastran.getcoords(uset, 200, 1)
    array([ 32.,  90.,  10.])
    >>> # get coordinates of node 200 in spherical (cid 2):
    >>> r = np.hypot(10., 32.)
    >>> th = 90.
    >>> phi = math.atan2(10., 32.)*180/math.pi
    >>> nastran.getcoords(uset, 200, 2) - np.array([r, th, phi])
    array([ 0.,  0.,  0.])
    """
    if np.size(gid) == 1:
        xyz_basic = uset.loc[(gid, 1), 'x':'z'].values
    else:
        xyz_basic = np.asarray(gid).ravel()
    if np.size(csys) == 1 and csys == 0:
        return xyz_basic
    # get input "coordinfo" [ cid type 0; location(1x3); T(3x3) ]:
    if coordref is None:
        coordref = {}
    coordinfo = get_coordinfo(csys, uset, coordref)
    xyz_coord = coordinfo[1]
    T = coordinfo[2:]   # transform to basic for coordinate system
    g = T.T @ (xyz_basic - xyz_coord)
    ctype = coordinfo[0, 1].astype(np.int64)
    if ctype == 1:
        return g
    if ctype == 2:
        R = math.hypot(g[0], g[1])
        theta = math.atan2(g[1], g[0])
        return np.array([R, theta * 180 / math.pi, g[2]])
    R = linalg.norm(g)
    phi = math.atan2(g[1], g[0])
    s = math.sin(phi)
    c = math.cos(phi)
    if abs(s) > abs(c):
        theta = math.atan2(g[1] / s, g[2])
    else:
        theta = math.atan2(g[0] / c, g[2])
    return np.array([R, theta * 180 / math.pi, phi * 180 / math.pi])


def _get_loc_a_basic(coordinfo, a):
    """
    Function for getting location of point "a" in basic; called by
    :func:`addgrid`.

    `coordinfo` is 5x3 and `a` is [x, y, z]
    """
    # tranformation from global to basic:
    Tg = coordinfo[2:]
    coordloc = coordinfo[1]
    if coordinfo[0, 1] == 1:
        location = coordloc + Tg @ a
    else:
        a2r = math.pi / 180.
        if coordinfo[0, 1] == 2:   # cylindrical
            vec = np.array([a[0] * math.cos(a[1] * a2r),
                            a[0] * math.sin(a[1] * a2r),
                            a[2]])
        else:                     # spherical
            s = math.sin(a[1] * a2r)
            vec = a[0] * np.array([s * math.cos(a[2] * a2r),
                                   s * math.sin(a[2] * a2r),
                                   math.cos(a[1] * a2r)])
        location = coordloc + Tg @ vec
    return location


def _ensure_iter(obj):
    try:
        iter(obj)
    except TypeError:
        obj = (obj,)
    return obj


def make_uset(idlist, doflist, uset=0, x=np.nan, y=np.nan, z=np.nan,
              use_product=True):
    """
    Make a uset DataFrame

    Parameters
    ----------
    idlist : integer or list_like of integers
        Grid or SPOINT id(s).
    doflist : integer or list_like of integers
        DOF to be used for all id(s) in `idlist` if `use_product` is
        True. If `use_product` is False, `doflist` should be
        compatibly-sized with `idlist`.
    uset : integer or list_like of integers; optional
        Specifies the Nastran set membership. The :func:`mkusetmask`
        can return a suitable value; eg: ``mkusetmask('a')``
    x : scalar float or list_like of floats; optional
        The x coordinate(s) of node(s) in `idlist`
    y : scalar float or list_like of floats; optional
        The y coordinate(s) of node(s) in `idlist`
    z : scalar float or list_like of floats; optional
        The z coordinate(s) of node(s) in `idlist`
    use_product : bool; optional
        Use :func:`pandas.MultiIndex.from_product` to form MultiIndex;
        otherwise, :func:`pandas.MultiIndex.from_arrays` is used (and
        `idlist` and `doflist` must be compatibly-sized).

    Returns
    -------
    pandas DataFrame
        A DataFrame similar to what is output by
        :func:`pyyeti.op2.OP2.rdn2cop2`.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> nastran.make_uset(991, 0, uset=4194304)  # doctest: +ELLIPSIS
                uset   x   y   z
    id  dof...
    991 0    4194304 NaN NaN NaN
    """
    idlist = _ensure_iter(idlist)
    doflist = _ensure_iter(doflist)
    if use_product:
        ind = pd.MultiIndex.from_product([idlist, doflist],
                                         names=['id', 'dof'])
    else:
        ind = pd.MultiIndex.from_arrays([idlist, doflist],
                                        names=['id', 'dof'])
    return pd.DataFrame(dict(uset=uset, x=x, y=y, z=z),
                        index=ind,
                        columns=['uset', *'xyz'])


def _addgrid_get_ci(coord, uset, coordref, cmap):
    cid = id(coord)
    try:
        ci = cmap[cid]
    except KeyError:
        ci = get_coordinfo(coord, uset, coordref)
        cmap[cid] = ci
    return ci


def _addgrid_proc_ci(coord, uset, coordref, cmap):
    """
    :func:`addgrid` utility to ensure "coord" is iterable
    """
    try:
        len(coord)
    except TypeError:
        # assume integer id, just make it iterable
        coord = itertools.cycle((coord,))
    else:
        # is iterable ... but is it just one coord matrix?
        try:
            ci = _addgrid_get_ci(coord, uset, coordref, cmap)
        except (TypeError, ValueError, IndexError):
            # assume it's a good iterable already
            pass
        else:
            # wasn't iterable of coords, save result & make it
            # iterable:
            cmap[id(coord)] = ci
            coord = itertools.cycle((coord,))
    return coord


def _addgrid_get_uset(nasset, mask, smap):
    sid = id(nasset)
    try:
        uset = smap[sid]
    except KeyError:
        if len(nasset) == 6:
            uset = [mask[nasset[0]],
                    mask[nasset[1]],
                    mask[nasset[2]],
                    mask[nasset[3]],
                    mask[nasset[4]],
                    mask[nasset[5]]]
        else:
            uset = mask[nasset]
        smap[sid] = uset
    return uset


def addgrid(uset, gid, nasset, coordin, xyz, coordout, coordref=None):
    """
    Add a grid or grids to a USET table.

    Parameters
    ----------
    uset : pandas DataFrame or None
        A DataFrame as output by :func:`pyyeti.op2.OP2.rdn2cop2`; can
        be None. If not None, this routine will use
        :func:`pandas.concat` to return a new, expanded DataFrame.
    gid : integer or list_like of integers
        Grid id(s), all must be unique.
    nasset : string or list_like of strings
        The set(s) to put the grid in (eg "m"); each string must
        either be one of these letters: m, s, o, q, r, c, b, e, or it
        can be a 6-character string of set letters, one for each dof.
    coordin : integer or 4x3 matrix; or list_like of those items
        If integer(s), specifies id(s) of the input coordinate system
        which is defined in uset (or coordref). If a 4x3 matrix(es),
        defines the input coordinate system (see below). Note the id 0
        is the basic coordinate system and is always available.
    xyz : 1d or 2d array_like
        Each row defines grid location(s) in `coordin` coordinates::

                 rectangular:  [X, Y, Z]
                 cylindrical:  [R, Theta, Z]
                 spherical:    [R, Theta, Phi]
                 - angles are specified in degrees

    coordout: integer or 4x3 matrix
        Same format as `coordin`. Defines the output coordinate
        system of the grid (see description below for more
        information).
    coordref : dictionary or None; optional
        If None, this input is ignored. Otherwise, it is a read/write
        dictionary (which can be empty) with the keys being the
        coordinate system id and the values being the 5x3 matrix::

             [cid  type 0]  # output coord. sys. id and type
             [xo   yo  zo]  # origin of coord. system
             [     T     ]  # 3x3 transformation to basic
             Note that T is for the coordinate system, not a grid
             (unless type = 0 which means rectangular)

        For example, to create a `coordref` with coordinate system
        104, you can do this::

            coordref = {}
            addgrid(None, 1, "b", 0, [0, 0, 0], crd104, coordref)

    Returns
    -------
    uset : pandas DataFrame
        Updated version of the input `uset`. Order of grids is as
        input.

    Notes
    -----
    When defining multiple grid entries, it is most efficient to use
    the list_like inputs as opposed to calling :func:`addgrid` for
    each grid. When using the list_like inputs, each entry must be
    compatible (singleton items are compatible with list_like items).

    This routine updates the `coordref` dictionary (if it is a
    dictionary) for future reference.

    To define a coordinate system, `coordin` or `coordout` must be 4x3
    size matrices containing the same information that would be on a
    CORD2R, CORD2C, or CORD2S entry::

        [ cid type reference_id ]
        [ Ax   Ay   Az          ]
        [ Bx   By   Bz          ]
        [ Cx   Cy   Cz          ]

    where 'cid' is the id of the new coordinate system (must be
    unique), 'type' is defined as::

        1 - rectangular
        2 - cylindrical
        3 - spherical

    and the locations of A, B, and C are given in the coordinate
    system indicated by 'reference_id'.

    In the demo below, a single call to this routine is used both for
    simplicity and efficiency. The other way to do this is to call
    this routine for each node; in that case, the uset DataFrame is
    expanded each call (meaning a new DataFrame is created every for
    every successive node). For example, the less efficient method is
    this:

        uset = None
        uset = nastran.addgrid(uset, 100, 'b', 0, [5, 10, 15], 0)
        uset = nastran.addgrid(uset, 200, 'b', cylcoord,
                               [32, 90, 10], cylcoord)

    And the more efficient method (as done below) is this:

        uset = nastran.addgrid(
            None, [100, 200], 'b', [0, cylcoord],
            [[5, 10, 15], [32, 90, 10]], [0, cylcoord])

    See also
    --------
    :func:`pyyeti.nastran.bulk2uset`, :func:`rbgeom_uset`,
    :func:`formrbe3`, :func:`pyyeti.op2.rdnas2cam`,
    :func:`pyyeti.op2.OP2.rdn2cop2`, :func:`usetprt`

    Raises
    ------
    ValueError
        If the grid id `gid` is already in `uset` or if a referenced
        coordinate system is not found in `uset` or `coordref`.

    Examples
    --------
    >>> from pyyeti import nastran
    >>> import numpy as np
    >>> # node 100 in basic is @ [5, 10, 15]
    >>> # node 200 in cylindrical coordinate system is @
    >>> # [r, th, z] = [32, 90, 10]
    >>> cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0],
    ...                     [0, 1, 0]])
    >>> uset = nastran.addgrid(
    ...     None, [100, 200], 'b', [0, cylcoord],
    ...     [[5, 10, 15], [32, 90, 10]], [0, cylcoord])
    >>> pd.options.display.float_format = lambda x: '{:.1f}'.format(x)
    >>> uset     # doctest: +ELLIPSIS
                uset    x    y    z
    id  dof...
    100 1    2097154  5.0 10.0 15.0
        2    2097154  0.0  1.0  0.0
        3    2097154  0.0  0.0  0.0
        4    2097154  1.0  0.0  0.0
        5    2097154  0.0  1.0  0.0
        6    2097154  0.0  0.0  1.0
    200 1    2097154 10.0  0.0 32.0
        2    2097154  1.0  2.0  0.0
        3    2097154  0.0  0.0  0.0
        4    2097154  0.0  0.0  1.0
        5    2097154  1.0  0.0  0.0
        6    2097154  0.0  1.0  0.0
    >>> pd.options.display.float_format = None
    """
    # if uset is not None and np.any(uset[:, 0] == gid):
    gid = _ensure_iter(gid)
    if uset is not None:
        curgids = uset.index.levels[0]
        pv = curgids.isin(gid)
        if pv.any():
            raise ValueError(
                "these grid ids are already in `uset` table: {}".
                format([*curgids[pv]]))

    # ensure basic coordinate system is present:
    if coordref is None:
        coordref = {0: np.vstack((np.array([[0, 1, 0], [0., 0., 0.]]),
                                  np.eye(3)))}
    else:
        try:
            coordref[0]
        except KeyError:
            coordref[0] = np.vstack((np.array([[0, 1, 0],
                                               [0., 0., 0.]]),
                                     np.eye(3)))

    mask = mkusetmask()

    # allocate dataframe:
    dof = np.arange(1, 7)
    usetid = make_uset(gid, dof)

    # ensure nasset is iterable:
    smap = {}
    if isinstance(nasset, str):
        nasset = itertools.cycle((nasset,))

    # make sure coordin and coordout are iterables:
    cmap = {}
    cin = _addgrid_proc_ci(coordin, uset, coordref, cmap)
    cout = _addgrid_proc_ci(coordout, uset, coordref, cmap)

    # make sure xyz is 2d ndarray:
    xyz = np.atleast_2d(xyz)

    # cols = ['uset', 'x', 'y', 'z']
    for g, u, _cin, _xyz, _cout in zip(gid, nasset, cin, xyz, cout):
        _uset = _addgrid_get_uset(u, mask, smap)

        # get location of point in basic:
        _cin = _addgrid_get_ci(_cin, uset, coordref, cmap)
        loc = _get_loc_a_basic(_cin, _xyz)

        # form x, y, z columns of dataframe:
        _cout = _addgrid_get_ci(_cout, uset, coordref, cmap)
        _xyz = np.vstack((loc, _cout))

        # put in dataframe:
        usetid.loc[g, 'uset'] = _uset
        usetid.loc[g, 'x':'z'] = _xyz

    if uset is not None:
        # concatenate to maintain order as input:
        usetid = pd.concat([uset, usetid], axis=0)
    return usetid


def _solve(a, b):
    """This is :func:`scipy.linalg.solve` but with a matrix condition
    check on `a`. Call by :func:`formrbe3`."""
    c = np.linalg.cond(a)
    if c > 1 / np.finfo(float).eps:
        warnings.warn('matrix is poorly conditioned (cond={:.3e}). '
                      'Solution will likely be inaccurate.'.format(c),
                      RuntimeWarning)
    return linalg.solve(a, b)


def formrbe3(uset, GRID_dep, DOF_dep, Ind_List, UM_List=None):
    """
    Form a least squares interpolation matrix, like RBE3 in Nastran.

    Parameters
    ----------
    uset : ndarray
        A 6-column matrix as output by
        :func:`pyyeti.op2.OP2.rdn2cop2`.
    GRID_dep : integer
        Id of dependent grid.
    DOF_dep : integer
        Contains all or a subset of the digits 123456 giving the
        dependent component DOF.
    Ind_List : list
        [DOF_Ind1, GRIDS_Ind1, DOF_Ind2, GRIDS_Ind2, ...], where::

            DOF_Ind1   : 1 or 2 element 1d array_like containing the
                         component DOF (ie, 123456) of the nodes in
                         GRIDS_Ind1 and, optionally, the weighting
                         factor for these DOF. If not input, the
                         weighting factor defaults to 1.0.
            GRIDS_Ind1 : 1d array_like of node ids corresponding to
                         DOF_Ind1
            ...
            eg:  [[123, 1.2], [95, 195, 1000], 123456, 95]

    UM_List : None or array_like; optional
        [GRID_MSET1, DOF_MSET1, GRID_MSET2, DOF_MSET2, ...] where::

              GRID_MSET1 : first grid in the M-set
              DOF_MSET1  : DOF of first grid in M-set (integer subset
                           of 123456). No weighting factors are
                           allowed here.
              GRID_MSET2 : second grid in the M-set
              DOF_MSET2  : DOF of second grid in M-set
              ...

        The `UM_List` option changes what is dependent and what is
        independent. The M-set DOF will become the dependent DOF
        instead of `GRID_dep`, `DOF_dep` (though it can include these
        DOF). The total number of M-set DOF must equal the original
        amount defined in `GRID_dep`, `DOF_dep` (max of 6). All M-set
        DOF must be within the the set of previously entered DOF
        (either dependent or independent).

    Returns
    -------
    rbe3 : ndarray
        The interpolation matrix. Size is # dependent DOF rows by #
        independent DOF columns. The order of rows and columns
        corresponds to the order the DOF occur in the USET table
        `uset`.

    Notes
    -----
    The approach used is:
        - Use :func:`rbgeom_uset` to form the rigid-body modes based
          on geometry and relative to `GRID_dep`.
        - Partition the rows of the rigid-body modes down to the
          independent DOF.
        - Use least squares approach to 'invert' the rigid-body modes.
        - Transform the result to global coordinates.
        - Partition the rows to the desired dependent DOF.

    If the UM_List option is used, these additional steps are done:

        - Partition current rbe3 matrix into four parts to separate:

            - dependent DOF into M-set and non-M-set
            - independent DOF into M-set and non-M-set

        - Solve for both parts of the M-set and merge.
        - Reorder if necessary.

    Simplifications are made if the M-set is completely contained
    within either the dependent or independent set.

    When the `UM_List` option is used, unless the M-set is equal to
    the dependent set (which would be the same as not including the
    `UM_List` input), there will be a matrix inversion. This matrix
    must be non-singular. If it is close to singular (or singular),
    this routine will print a warning message and, if singular,
    scipy.linalg will raise the LinAlgError exception. In this case,
    choose a different, non-singular set for the M-set. This is
    similar to choosing DOF for the SUPORT card in Nastran.

    Raises
    ------
    ValueError
        When the `UM_List` input is used but the size does not match
        the `GRID_dep` and `DOF_dep` size.

    See also
    --------
    :func:`addgrid`, :func:`rbgeom_uset`

    Examples
    --------
    >>> from pyyeti import nastran
    >>> # First, make a uset table using all basic coords to simplify
    >>> # visual inspection:
    >>> locs = [[ 1,  0, 0],   #  node 100 in basic
    ...         [ 0,  1, 0],   #  node 200 in basic
    ...         [-1,  0, 0],   #  node 300 in basic
    ...         [ 0, -1, 0],   #  node 400 in basic
    ...         [ 0,  0, 0]]   #  node 500 in basic
    >>> uset = nastran.addgrid(None, np.arange(100, 600, 100), 'b', 0,
    ...                        locs, 0)
    >>> #
    >>> # Define the motion of grid 500 to be average of translational
    >>> # motion of grids:  100, 200, 300, and 400.
    >>> rbe3 = nastran.formrbe3(
    ...     uset, 500, 123456, [123, [100, 200, 300, 400]])
    >>> np.set_printoptions(linewidth=75)
    >>> print(rbe3+0)
    [[ 0.25  0.    0.    0.25  0.    0.    0.25  0.    0.    0.25  0.    0.  ]
     [ 0.    0.25  0.    0.    0.25  0.    0.    0.25  0.    0.    0.25  0.  ]
     [ 0.    0.    0.25  0.    0.    0.25  0.    0.    0.25  0.    0.    0.25]
     [ 0.    0.    0.    0.    0.    0.5   0.    0.    0.    0.    0.   -0.5 ]
     [ 0.    0.   -0.5   0.    0.    0.    0.    0.    0.5   0.    0.    0.  ]
     [ 0.    0.25  0.   -0.25  0.    0.    0.   -0.25  0.    0.25  0.    0.  ]]
    >>> #
    >>> # Example showing UM_List option:
    >>> rbe3um = nastran.formrbe3(
    ...     uset, 500, 123456, [123, [100, 200, 300, 400]],
    ...     [100, 12, 200, 3, 300, 23, 400, 3])
    >>> print(rbe3um+0)
    [[ 0.  -1.   0.  -1.  -1.   0.   4.   0.   0.   0.   0.   0. ]
     [ 0.   0.5 -0.5  0.  -0.5 -0.5  0.   2.   0.   0.   0.   2. ]
     [-1.   0.   0.   0.   0.   0.   0.   0.   2.   1.  -1.   0. ]
     [ 0.  -0.5 -0.5  0.   0.5 -0.5  0.   2.   0.   0.   0.  -2. ]
     [ 1.   0.   0.   0.   0.   0.   0.   0.   0.   0.   2.   0. ]
     [-1.   0.   0.   0.   0.   0.   0.   0.   2.  -1.  -1.   0. ]]
    >>> #
    >>> # Example showing UM_List option including some dependent dof:
    >>> rbe3um2 = nastran.formrbe3(
    ...     uset, 500, 123456, [123, [100, 200, 300, 400]],
    ...     [100, 12, 200, 3, 300, 3, 500, 23])
    >>> print(rbe3um2+0)
    [[ 0.   -1.    0.   -1.    0.   -1.    0.    0.    4.    0.    0.    0.  ]
     [ 0.    1.    0.    0.    1.   -1.    0.    0.    0.    0.    0.    4.  ]
     [ 0.    0.    0.    0.    0.    0.    0.    1.    0.    2.    0.    0.  ]
     [ 1.    0.    0.    0.    0.    0.    0.    0.    0.    0.    2.    0.  ]
     [ 0.    0.25  0.25  0.    0.5  -0.25  0.25  0.    0.    0.    0.    1.  ]
     [ 0.5   0.    0.    0.    0.    0.    0.    0.5   0.    0.5   0.5   0.  ]]
    """
    # form dependent DOF table:
    ddof = expanddof([[GRID_dep, DOF_dep]])

    # form independent DOF table:
    usetdof = uset.iloc[:, :0].reset_index().values
    idof = []
    wtdof = []
    for j in range(0, len(Ind_List), 2):
        # eg:  [[123, 1.2], [95, 195, 1000], 123456, 95]
        DOF_ind = np.atleast_1d(Ind_List[j])
        GRIDS_ind = np.atleast_1d(Ind_List[j + 1])
        if len(DOF_ind) == 2:
            wtcur = DOF_ind[1]
            DOF_ind = DOF_ind[0]
        else:
            wtcur = 1.0
        newdof = expanddof([[n, DOF_ind] for n in GRIDS_ind])
        idof.extend(newdof)
        wtdof.extend([wtcur for i in range(len(newdof))])

    idof = np.array(idof)
    wtdof = np.array(wtdof)

    # Sort idof according to uset:
    pv = locate.mat_intersect(idof, usetdof, 2)[0]
    idof = idof[pv]
    wtdof = wtdof[pv]

    if UM_List is not None:
        mdof = expanddof([[UM_List[j], UM_List[j + 1]]
                          for j in range(0, len(UM_List), 2)])
        if np.size(mdof, 0) != np.size(ddof, 0):
            raise ValueError("incorrect size of M-set DOF ({}): "
                             "must equal size of Dep DOF ({})."
                             .format(np.size(mdof, 0),
                                     np.size(ddof, 0)))
        # The rest of the code uses 'mdof' to sort rows of the output
        # matrix. We could leave it as input, or sort it according to
        # the uset table. For now, sort it according to uset:
        pv = locate.mat_intersect(mdof, usetdof, 2)[0]
        mdof = mdof[pv]

    # partition uset table down to needed dof only:
    npids = np.vstack((ddof[0, :1], idof[:, :1]))
    ids = sorted(list(set(npids[:, 0])))
    alldof = mkdofpv(uset, "p", ids)[0]
    uset = uset.iloc[alldof]

    # form partition vectors:
    ipv = mkdofpv(uset, "p", idof)[0]

    # need transformation from basic to global for dependent grid
    pv = mkdofpv(uset, "p", GRID_dep)[0]

    # need to scale rotation weights by characteristic length:
    rot = idof[:, 1] > 3
    if np.any(rot):
        # get characterstic length of rbe3:
        deploc = uset.iloc[pv[0], 1:]
        n = uset.shape[0] // 6
        delta = uset.iloc[::6, 1:] - deploc
        Lc = np.sum(np.sqrt(np.sum(delta * delta, axis=1))) / (n - 1)
        if Lc > 1.e-12:
            wtdof[rot] = wtdof[rot] * (Lc * Lc)

    # form rigid-body modes relative to GRID_dep
    rbb = rbgeom_uset(uset, GRID_dep)
    T = rbb[pv]
    rb = rbb[ipv]
    rbw = rb.T * wtdof
    rbe3 = _solve(rbw @ rb, rbw)
    rbe3 = (T @ rbe3)[ddof[:, 1] - 1]
    if UM_List is None:
        return rbe3

    # find m-set dof that belong to current dependent set:
    dpv_m = locate.mat_intersect(ddof, mdof, 2)[0]
    # this works when the m-set is a subset of the independent set:
    if not dpv_m.any():
        mpv = mkdofpv(uset.iloc[ipv], "p", mdof)[0]
        rbe3_um = rbe3[:, mpv]
        notmpv = locate.flippv(mpv, len(ipv))
        rhs = np.hstack((np.eye(len(mpv)), -rbe3[:, notmpv]))
        rbe3 = _solve(rbe3_um, rhs)
        # rearrange columns to uset order:
        curdof = np.vstack((ddof, idof[notmpv]))
        pv = locate.mat_intersect(curdof, usetdof, 2)[0]
        return rbe3[:, pv]

    # some dependent retained, so find m-set dof that belong to
    # current independent set:
    ipv_m = locate.mat_intersect(idof, mdof, 2)[0]

    if not np.any(ipv_m):
        # already done, except reordering:
        rbe3 = rbe3[dpv_m]
        # rearrange columns to uset order:
        pv = locate.mat_intersect(idof, usetdof, 2)[0]
        return rbe3[:, pv]

    # To include UM option:
    #   Note:  if the M-set is entirely within either the dependent
    #   set or the independent set, simplifications are made (see
    #   above). The steps here assume the M-set is composed of DOF
    #   from both the dependent and independent sets.
    #
    #   1. partition rbe3 to four parts, A, B, C & D:
    #      - before rearranging to have m-set on left:
    #
    #          Dm      | A,  B |  Im
    #          Dn   =  | C,  D |  In
    #
    #   2. solve for Im and Dm in terms of Dn and In:
    #
    #          Dm      | A inv(C),   -A inv(C) D + B |  Dn
    #          Im   =  | inv(C),     -inv(C) D       |  In
    #
    #   Matrix C must be square and non-singular. The resulting
    #   matrix is reordered as described in the help section.

    # partition rbe3 -- taking care NOT to rearrange columns
    nd, ni = np.shape(rbe3)
    dpv_mzo = locate.index2bool(dpv_m, nd)
    notdpv_m = np.logical_not(dpv_mzo)

    ipv_mzo = locate.index2bool(ipv_m, ni)
    notipv_m = np.logical_not(ipv_mzo)

    A = rbe3[np.ix_(dpv_mzo, ipv_mzo)]
    B = rbe3[np.ix_(dpv_mzo, notipv_m)]
    C = rbe3[np.ix_(notdpv_m, ipv_mzo)]      # must be square
    D = rbe3[np.ix_(notdpv_m, notipv_m)]

    n = np.size(C, 0)
    # m-set rows from independent part
    E = _solve(C, np.hstack((np.eye(n), -D)))
    r = np.size(A, 0)
    c = np.size(C, 1)
    # m-set rows from dependent part
    F = A @ E + np.hstack((np.zeros((r, c)), B))
    rbe3a = np.vstack((F, E))

    didof = np.vstack((ddof[dpv_mzo], idof[ipv_mzo]))
    pv = locate.mat_intersect(didof, mdof, 2)[0]
    rbe3 = rbe3a[pv]

    # rearrange columns to uset order:
    curdof = np.vstack((ddof[notdpv_m], idof[notipv_m]))
    pv = locate.mat_intersect(curdof, usetdof, 2)[0]
    return rbe3[:, pv]


def _findse(nas, se):
    """
    Find row in nas['selist'] the superelement `se`.

    Parameters
    ----------
    nas : dictionary
        This is the nas2cam dictionary:  ``nas = op2.rdnas2cam()``
    se : integer
        The id of the superelement.

    Returns
    -------
    r : index
        Rows index to where `se` is.
    """
    r = np.nonzero(nas['selist'][:, 0] == se)[0]
    if r.size == 0:
        msg = ("superelement {} not found. Current `selist` is:\n{!s}"
               .format(se, nas['selist']))
        raise ValueError(msg)
    return r[0]


def _get_node_ids(uset):
    ids = uset.index.get_level_values('id')
    dof = uset.index.get_level_values('dof')
    return ids[dof <= 1]


def upasetpv(nas, seup):
    """
    Form upstream A-set partition vector for a downstream SE

    Parameters
    ----------
    nas : dictionary
        This is the nas2cam dictionary: ``nas = op2.rdnas2cam()``
    seup : integer
        The id of the upstream superelement.

    Returns
    -------
    pv : 1d ndarray
        An index partition vector for partitioning the upstream A-set
        degrees of freedom of superelement SEUP from the P-set of the
        downstream superelement. This partition vector is not a
        True/False type because the A-set DOF order may be different
        downstream than from upstream (from reordering done on a
        CSUPER entry).

    Notes
    -----
    Example usage::

        # External superelement 100 is upstream of the residual. On
        # the CSUPER entry, the A-set of 100 were assigned new ids and
        # the order was changed. Form the ULVS matrix:
        from pyyeti import nastran
        nas = nastran.rdnas2cam('nas2cam')
        pv = nastran.upasetpv(nas, 100)
        ulvs100 = nas['phg'][0][pv]  # this will reorder as needed

    See also
    --------
    :func:`mksetpv`, :func:`pyyeti.op2.rdnas2cam`, :func:`formulvs`,
    :func:`upqsetpv`.
    """
    r = _findse(nas, seup)
    sedn = nas['selist'][r, 1]
    usetdn = nas['uset'][sedn]
    dnids = nas['dnids'][seup]
    maps = nas['maps'][seup]

    # number of rows in pv should equal size of upstream a-set
    pv = usetdn.index.isin(dnids, level=0).nonzero()[0]
    if len(pv) < len(dnids):
        # must be an external se, but non-csuper type (the extseout,
        # seconct, etc, type)
        upids = nas['upids'][sedn]
        pv = locate.find_vals(upids, dnids)
        ids = _get_node_ids(usetdn)

        # number of rows should equal size of upstream a-set
        pv = usetdn.index.isin(ids[pv], level='id').nonzero()[0]
        if len(pv) < len(dnids):   # pragma: no cover
            raise ValueError('not all upstream DOF could'
                             ' be found in downstream')
    if len(maps) > 0:
        if not np.all(maps[:, 1] == 1):   # pragma: no cover
            raise ValueError('column 2 of MAPS for {} is not all 1.'
                             '  Stopping.'.format(seup))
        # definition of maps:  dn = up(maps) ... want up = dn(maps2)
        # example:
        # maps = [ 2, 0, 1 ]
        # maps2 = [ pos_of_1st pos_of_2nd pos_of_3rd ] = [ 1, 2, 0 ]
        maps2 = np.argsort(maps[:, 0])
        pv = pv[maps2]
    return pv


def upqsetpv(nas, sedn=0):
    """
    Form upstream Q-set partition vector for a downstream SE

    Parameters
    ----------
    nas : dictionary
        This is the nas2cam dictionary: ``nas = op2.rdnas2cam()``
    sedn : integer; optional
        The id of the downstream superelement.

    Returns
    -------
    pv : 1d ndarray
        A True/False vector for partitioning upstream Q-set DOF from
        downstream P-set.

    Notes
    -----
    If necessary, this routine will call itself recursively up the
    superelement tree to account for all upstream Q-set. This will
    take care of the Benfield-Hruda case where upstream Q-set go to
    the B-set of another superelement before going to its ultimate
    superelement.

    Note: if any upstream SEe does not have any DOF assigned to the
    Q-set (as is quite typical for partitioned SE's), this routine
    will then assume that all SPOINTs are Q-set DOF.

    Example usage::

        # Form rigid-body modes for SE 0, and then zero out all DOF
        # that correspond to upstream Q-set:
        from pyyeti import nastran
        nas = nastran.rdnas2cam('nas2cam')
        rb = nastran.rbgeom_uset(nas['uset'][0])
        pv = nastran.upqsetpv(nas, 0)
        rb[pv] = 0.0

    See also
    --------
    :func:`mksetpv`, :func:`pyyeti.op2.rdnas2cam`, :func:`formulvs`,
    :func:`upasetpv`.
    """
    selist = nas['selist']
    rows = (selist[:, 1] == sedn).nonzero()[0]
    if rows.size == 0:
        msg = ('downstream superelement {} not found in 2nd '
               'column of `selist`. Current `selist` is:\n{!s}'
               .format(sedn, nas['selist']))
        raise ValueError(msg)

    usetdn = nas['uset'][sedn]
    pv = np.zeros(usetdn.shape[0], bool)

    for r in rows:
        seup = selist[r, 0]
        if seup == sedn:
            continue
        usetup = nas['uset'][seup]
        dnids = nas['dnids'][seup]
        maps = nas['maps'][seup]

        qup = mksetpv(usetup, 'a', 'q')
        if not qup.any():
            # assume any a-set spoints are q-set
            dof = usetup.index.get_level_values('dof')
            qup = dof[mksetpv(usetup, 'p', 'a')] == 0
            # qup = usetup[mksetpv(usetup, 'p', 'a'), 1] == 0

        # check to see if the upstream se has upstreams;
        # if so, include its qup:
        if (selist[:, 1] == seup).any():
            qup2 = upqsetpv(nas, seup)
            qup = qup | qup2[mksetpv(usetup, 'p', 'a')]

        if qup.any():
            # expand downstream ids to include all dof:
            # number of rows in pv1 should equal size of
            # upstream a-set
            pv1 = usetdn.index.isin(dnids, level='id')

            if pv1.size < dnids.size:
                # must be an external se, but non-csuper type (the
                # extseout, seconct, etc, type)

                # The downstream "upids" contains all upstream
                # internally generated ids. It also has zeros for the
                # downstream DOF (in the uset table) that are not from
                # upstream SEs. Each upstream "dnids" is a subset of
                # the downstream "upids". By finding where "dnids"
                # occurs in "upids", we're actually finding where
                # "dnids" occurs in the p-set of the downstream ...
                # just what we want.
                upids = nas['upids'][sedn]
                pv1 = locate.find_vals(upids, dnids)
                ids = _get_node_ids(usetdn)

                # length of pv1 should equal size of upstream a-set
                pv1 = usetdn.index.isin(ids[pv1], level='id')
                if pv1.size < dnids.size:   # pragma: no cover
                    raise ValueError('not all upstream DOF could'
                                     ' be found in downstream')

            if len(maps) > 0:
                if not np.all(maps[:, 1] == 1):   # pragma: no cover
                    raise ValueError(
                        'column 2 of MAPS for {} is not all 1.'
                        ' Stopping.'.format(seup))
                pv[pv1] = qup[maps[:, 0].astype(np.int64)]
            else:
                pv[pv1] = qup
    return pv


def _proc_mset(nas, se, dof):
    """
    Private utility routine to get m-set information for
    :func:`formtran`.

    Returns: (hasm, m, pvdofm, gm)
    """
    # see if any of the DOF are in the m-set
    hasm = 0
    uset = nas['uset'][se]
    m = np.nonzero(mksetpv(uset, "g", "m"))[0]
    pvdofm = gm = None
    if m.size > 0:
        iddof = uset.iloc[m, :0].reset_index().values
        pvdofm = locate.mat_intersect(iddof, dof)[0]

        if pvdofm.size > 0:
            hasm = 1
            m = m[pvdofm]
            # need gm
            gm = nas['gm'][se]
            gm = gm[pvdofm]
    return hasm, m, pvdofm, gm


def _formtran_0(nas, dof, gset):
    """
    Utility routine called by :func:`formtran` when se == 0. See that
    routine for more information.
    """
    uset = nas['uset'][0]
    pvdof, dof = mkdofpv(uset, "g", dof)

    if gset:
        ngset = np.count_nonzero(mksetpv(uset, 'p', 'g'))
        tran = np.zeros((len(pvdof), ngset))
        tran[:, pvdof] = np.eye(len(pvdof))
        return tran, dof

    if 'phg' in nas and 0 in nas['phg']:
        return nas['phg'][0][pvdof], dof

    if 'pha' not in nas or 0 not in nas['pha']:
        raise RuntimeError("neither nas['phg'][0] nor "
                           "nas['pha'][0] are available.")

    o = np.nonzero(mksetpv(uset, "g", "o"))[0]
    iddof = uset.iloc[:, :0].reset_index().values
    if o.size > 0:   # pragma: no cover
        v = locate.mat_intersect(iddof[o], dof)[0]
        if v.size > 0:
            raise RuntimeError("some of the DOF of SE 0 go to the"
                               " O-set. Routine not set up for"
                               " this.")

    a = np.nonzero(mksetpv(uset, "g", "a"))[0]
    pvdofa = locate.mat_intersect(iddof[a], dof)[0]
    if pvdofa.size > 0:
        a = a[pvdofa]
        sets = a
    else:
        a = []
        sets = np.zeros(0, np.int64)

    hasm, m, pvdofm, gm = _proc_mset(nas, 0, dof)

    if hasm:
        o_n = mksetpv(uset, "n", "o")
        if np.any(o_n):
            if np.any(gm[:, o_n]):
                raise RuntimeError('M-set for residual is dependent'
                                   ' on O-set (through GM). '
                                   'Routine not set up for this.')
        sets = np.hstack((sets, m))

    # see if any of the DOF are in the s-set
    hass = 0
    s = np.nonzero(mksetpv(uset, "g", "s"))[0]
    if s.size > 0:
        pvdofs = locate.mat_intersect(iddof[s], dof)[0]
        if pvdofs.size > 0:
            hass = 1
            s = s[pvdofs]
            sets = np.hstack((sets, s))

    fulldof = iddof[sets]
    pv, pv2 = locate.mat_intersect(fulldof, dof, 2)

    if len(pv2) != len(pvdof):
        notpv2 = locate.flippv(pv2, len(pvdof))
        msg = ("bug in :func:`_formtran_0` since dof in "
               "recovery set does not contain all of the "
               "dof in `dof`. These dof are missing:\n{!s}"
               .format(dof[notpv2]))
        raise RuntimeError(msg)
    # sets = [ a, m, s ]
    cols = nas['pha'][0].shape[1]
    tran = np.zeros((len(pv), cols))
    R = len(a)
    tran[:R] = nas['pha'][0][pvdofa]

    if hasm:
        a_n = np.nonzero(mksetpv(uset, "n", "a"))[0]
        tran[R:R + len(m)] = gm[:, a_n] @ nas['pha'][0]
        R += len(m)

    if hass:
        cu = tran.shape[1]
        tran[R:R + len(s)] = np.zeros((len(s), cu))

    # order DOF as requested:
    tran = tran[pv]
    return tran, dof


def formtran(nas, se, dof, gset=False):
    """
    Make a transformation matrix from A-set DOF to specified DOF
    within the same SE.

    Parameters
    ----------
    nas : dictionary
        This is the nas2cam dictionary:  ``nas = op2.rdnas2cam()``
    se : integer
        The id of the superelement.
    dof : 1d or 2d array
        One or two column matrix: [ids] or [ids, dofs]; if one column,
        the second column is internally set to 123456 for each id
    gset : bool; optional
        If true, and `sedn` == 0, transform from g-set instead of
        modal DOF. See below.

    Returns
    -------
    Tran : ndarray
        Transformation from the A-set DOF of superelement `se` to
        the specified DOF (`dof`) of the same superelement. The
        transformation is as follows::

            if sedn > 0:
              {DOF} = Tran * {T & Q}

            if sedn == 0:
              {DOF} = Tran * {modal}       (for gset == False)
              {DOF} = Tran * {G-Set}       (for gset == True)

    outdof : ndarray
        The expanded version of the `dof` input as returned by
        :func:`mkdofpv`::

             [id1, dof1; id1, dof2; ... id2, dof1; ...]

    Notes
    -----
    This routine is the workhorse of other routines such as
    :func:`formdrm` and :func:`formulvs`.

    Example usage::

        # Want data recovery matrix from t & q dof to grids 3001 and
        # 3002 of se 300:
        from pyyeti import nastran
        import op2
        nas = op2.rdnas2cam('nas2cam')
        drm = nastran.formtran(nas, 300, [3001, 3002])
        # or, equivalently:
        drm = nastran.formdrm(nas, 300, 300, [3001, 3002])

    See also
    --------
    :func:`formdrm`, :func:`formulvs`, :func:`mkdofpv`
    """
    if se == 0:
        return _formtran_0(nas, dof, gset)

    uset = nas['uset'][se]
    pvdof, dof = mkdofpv(uset, "g", dof)
    t_a = np.nonzero(mksetpv(uset, "a", "t"))[0]
    q_a = np.nonzero(mksetpv(uset, "a", "q"))[0]

    # if all dof are in a-set we can quit quickly:
    a = mksetpv(uset, "g", "a")
    if np.all(a[pvdof]):
        pvdofa = mkdofpv(uset, "a", dof)[0]
        tran = np.eye(sum(a))
        tran = tran[pvdofa]
        return tran, dof

    sets = np.zeros(0, np.int64)
    t = np.nonzero(mksetpv(uset, "g", "t"))[0]
    iddof = uset.iloc[:, :0].reset_index().values
    pvdoft = locate.mat_intersect(iddof[t], dof)[0]
    hast = 0
    if pvdoft.size > 0:
        hast = 1
        t = t[pvdoft]
        sets = np.hstack((sets, t))

    o = np.nonzero(mksetpv(uset, "g", "o"))[0]
    pvdofo = locate.mat_intersect(iddof[0], dof)[0]
    haso = 0
    if pvdofo.size > 0:
        haso = 1
        o = o[pvdofo]
        sets = np.hstack((sets, o))

    if 'goq' in nas and se in nas['goq']:
        goq = nas['goq'][se]
    else:
        q1 = sum(mksetpv(uset, "g", "q"))
        if q1 > 0:
            warnings.warn("nas['goq'][{}] not found, but q-set do"
                          " exist. Assuming it is all zeros. "
                          "This can happen when q-set DOF are "
                          "defined but modes are not calculated.".
                          format(se), RuntimeWarning)
            o1 = sum(mksetpv(uset, "g", "o"))
            goq = np.zeros((o1, q1))
        else:
            goq = np.array([[]])

    if 'got' in nas and se in nas['got']:
        got = nas['got'][se]
    else:
        warnings.warn("nas['got'][{}] not found. Assuming it is "
                      "all zeros. This can happen for a Benfield"
                      "-Hruda collector superelement since all "
                      "b-set (really upstream q-set) are not "
                      "connected to other DOF in the stiffness.".
                      format(se), RuntimeWarning)
        o1 = sum(mksetpv(uset, "g", "o"))
        t1 = sum(mksetpv(uset, "g", "t"))
        got = np.zeros((o1, t1))

    ct = got.shape[1]
    cq = goq.shape[1]
    hasm, m, pvdofm, gm = _proc_mset(nas, se, dof)
    if hasm:
        t_n = np.nonzero(mksetpv(uset, "n", "t"))[0]
        o_n = np.nonzero(mksetpv(uset, "n", "o"))[0]
        q_n = np.nonzero(mksetpv(uset, "n", "q"))[0]
        sets = np.hstack((sets, m))

    # see if any of the DOF are in the q-set
    q = np.nonzero(mksetpv(uset, "g", "q"))[0]
    hasq = 0
    if q.size > 0:
        pvdofq = locate.mat_intersect(iddof[q], dof)[0]
        if pvdofq.size > 0:
            hasq = 1
            q = q[pvdofq]
            sets = np.hstack((sets, q))

    # see if any of the DOF are in the s-set
    hass = 0
    s = np.nonzero(mksetpv(uset, "g", "s"))[0]
    if s.size > 0:
        pvdofs = locate.mat_intersect(iddof[s], dof)[0]
        if pvdofs.size > 0:
            hass = 1
            s = s[pvdofs]
            sets = np.hstack((sets, s))

    fulldof = iddof[sets]
    pv, pv2 = locate.mat_intersect(fulldof, dof, 2)

    if len(pv2) != len(pvdof):
        notpv2 = locate.flippv(pv2, len(pvdof))
        msg = ("bug in :func:`formtran` since dof in "
               "recovery set does not contain all of the "
               "dof in `dof`. These dof are missing:\n{!s}"
               .format(dof[notpv2]))
        raise RuntimeError(msg)

    # sets = [ t, o, m, q, s ]
    tran = np.zeros((len(pv), ct + cq))
    R = 0
    if hast:
        I = np.eye(ct)
        R = len(t)
        tran[:R, t_a] = I[pvdoft]

    if haso:
        tran[R:R + len(o), t_a] = got[pvdofo]
        if cq:
            tran[R:R + len(o), q_a] = goq[pvdofo]
        R += len(o)

    if hasm:
        ulvsm = np.zeros((gm.shape[0], ct + cq))
        gmo = gm[:, o_n]
        v = np.nonzero(np.any(gmo, 0))[0]
        if v.size > 0:
            gmo = gmo[:, v]
            ulvsm[:, t_a] = gm[:, t_n] + gmo @ got[v]
            if cq:
                ulvsm[:, q_a] = gmo @ goq[v]
        else:
            ulvsm[:, t_a] = gm[:, t_n]
        if cq:
            # m-set dependent on q-set (via MPC maybe)
            ulvsm[:, q_a] += gm[:, q_n]
        tran[R:R + len(m)] = ulvsm
        R += len(m)

    if hasq:
        I = np.eye(cq)
        tran[R:R + len(q), q_a] = I[pvdofq]
        R += len(q)

    if hass:
        cu = tran.shape[1]
        tran[R:R + len(s)] = np.zeros((len(s), cu))

    # order DOF as requested:
    tran = tran[pv]
    return tran, dof


def formulvs(nas, seup, sedn=0, keepcset=True, shortcut=True,
             gset=False):
    """
    Form ULVS for an upstream SE relative to a given downstream SE.

    Parameters
    ----------
    nas : dictionary
        This is the nas2cam dictionary:  ``nas = op2.rdnas2cam()``
    seup : integer
        The id of the upstream superelement.
    sedn : integer; optional
        The id of the downstream superelement.
    keepcset : bool; optional
        If true, keeps any C-set rows/columns in the result. This is
        useful when the C-set are real (that is, NOT used for 'left-
        over' DOF after defining the Q-set). Set `keepcset=False` to
        delete C-set.
    shortcut : bool; optional
        If true, use the ULVS already in `nas` if it's there.
    gset : bool; optional
        If true, and `sedn` == 0, transform from g-set instead of
        modal DOF. See below.

    Returns
    -------
    ULVS : 2d numpy ndarray or 1.0
        Transformation from either the modal or physical DOF of the
        downstream superelement sedn to the T and Q-set DOF of the
        upstream superelement seup. The transformation (called ULVS
        here) is as follows::

          if sedn > 0:
            {upstream T & Q} = ULVS * {downstream T & Q}

          if sedn == 0:
            {upstream T & Q} = ULVS * {modal}      (for gset == False)
            {upstream T & Q} = ULVS * {G-Set}      (for gset == True)

        Returns 1 if seup == sedn.

    Notes
    -----
    This routine starts from seup and works down to sedn, forming the
    appropriate ULVS at each level (by calling formtran()) and
    multiplying them together to form the total ULVS from sedn DOF to
    seup T & Q-set DOF.

    The routine :func:`addulvs` is an interface routine to this
    routine that simplifies the creation of the standard ULVS matrices
    (`sedn` = 0) for inclusion in the `nas` data structure.

    Example usage::

        # From recovery matrix from se 0 q-set to t & q set of se 500:
        from pyyeti import nastran
        import op2
        nas = op2.rdnas2cam('nas2cam')
        ulvs = nastran.formulvs(nas, 500)

    See also
    --------
    :func:`addulvs`, :func:`formdrm`, :func:`formtran`
    """
    # work from up to down:
    r = _findse(nas, seup)
    sedown = nas['selist'][r, 1]
    if sedown == seup or sedn == seup:
        return 1.0
    if (shortcut and sedn == 0 and not gset and
            'ulvs' in nas and seup in nas['ulvs']):
        return nas['ulvs'][seup]
    ulvs = 1.0
    while True:
        usetup = nas['uset'][seup]
        usetdn = nas['uset'][sedown]
        tqup = upasetpv(nas, seup)
        ulvs1 = formtran(nas, sedown, usetdn[tqup, :2], gset)[0]
        # get rid of c-set if required
        if not keepcset:
            noncrows = np.logical_not(mksetpv(usetup, "a", "c"))
            if sedown != 0:
                nonccols = np.logical_not(mksetpv(usetdn, "a", "c"))
                ulvs1 = ulvs1[np.ix_(noncrows, nonccols)]
            else:
                ulvs1 = ulvs1[noncrows]
        ulvs = np.dot(ulvs, ulvs1)
        if sedown == sedn:
            return ulvs
        seup = sedown
        r = _findse(nas, seup)
        sedown = nas['selist'][r, 1]


def formdrm(nas, seup, dof, sedn=0, gset=False):
    """
    Form a displacement data recovery matrix for specified dof.

    Parameters
    ----------
    nas : dictionary
        This is the nas2cam dictionary:  ``nas = op2.rdnas2cam()``
    seup : integer
        The id of the upstream superelement.
    dof : 1d or 2d integer array
        One or two column matrix: [ids] or [[ids, dofs]]; if 1d, the
        DOF for each node is internally set to 123456. The DOF can be
        entered in Nastran style, eg: 123 for the three
        translations. For example, all of these are equivalent::

            dof = 99
            dof = [99]
            dof = [[99, 123456]]
            dof = [[99, 1], [99, 2], [99, 3],
                   [99, 4], [99, 5], [99, 6]]

    sedn : integer
        The id of the downstream superelement.
    gset : bool; optional
        If true, and `sedn` == 0, transform from g-set instead of
        modal DOF. See below.

    Returns
    -------
    DRM : ndarray
        Transformation from downstream DOF to the specified
        upstream DOF (`dof`). The transformation is as follows::

          if sedn > 0:
            {upstream DOF} = DRM * {downstream T & Q}

          if sedn == 0:
            {upstream DOF} = DRM * {modal}     (for gset == False)
            {upstream DOF} = DRM * {G-Set}     (for gset == True)

    outdof : ndarray
        The expanded version of the `dof` input as returned by
        :func:`mkdofpv`::

             [id1, dof1; id1, dof2; ... id2, dof1; ...]

    Notes
    -----
    This routine uses :func:`formulvs` and :func:`formtran` as
    necessary.

    Example usage::

        # Want data recovery matrix from se 0 to grids 3001 and
        # 3002 of se 300:
        from pyyeti import nastran
        import op2
        nas = op2.rdnas2cam()
        drm, dof = nastran.formdrm(nas, 300, [3001, 3002])

        # for only the translations:
        drm, dof = nastran.formdrm(nas, 300, [[3001, 123],
                                              [3002, 123])

    See also
    --------
    :func:`formulvs`, :func:`formtran`
    """
    t, outdof = formtran(nas, seup, dof, gset=gset)
    u = formulvs(nas, seup, sedn, keepcset=True,
                 shortcut=True, gset=gset)
    if np.size(u) > 1:
        # check for null c-sets (extra cols in t):
        c = t.shape[1]
        r = u.shape[0]
        if r < c and not np.any(t[:, r:]):
            t = t[:, :r]
    return np.dot(t, u), outdof


def addulvs(nas, *ses, **kwargs):
    """
    Add ULVS matrices to the nas (nas2cam) record.

    Parameters
    ----------
    nas : dictionary
        This is the nas2cam dictionary: ``nas = op2.rdnas2cam()``
    *ses : list
        Remaining args are the superelement ids for which to compute a
        ULVS via :func:`formulvs`.
    **kwargs : dict; optional
        Named arguments to pass to :func:`formulvs`

    Notes
    -----
    Example usage::

        addulvs(nas, 100, 200, 300)
    """
    if 'ulvs' not in nas:
        nas['ulvs'] = {}

    for se in ses:
        nas['ulvs'][se] = formulvs(nas, se, **kwargs)
