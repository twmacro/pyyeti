# -*- coding: utf-8 -*-
"""
Setting up data recovery for relative displacements
"""
import numpy as np
import scipy.linalg as la
from pyyeti.nastran import n2p


# FIXME: We need the str/repr formatting used in Numpy < 1.14.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


def relative_displacement_dtm(nas, node_pairs):
    """
    Form relative displacements data recovery matrix

    Parameters
    ----------
    nas : dictionary
        This is the nas2cam dictionary: ``nas = op2.rdnas2cam()``. For
        a Craig-Bampton component, `nas` will also need to have `mug1`
        and `tug1` entries as shown in the example usage below. The
        matrix `mug1` is a data recovery matrix (to recover the
        displacements) and `tug1` is the DOF map to `mug1`: ``[node,
        dof]``. These get created during an "EXTSEOUT" Nastran run.
        The naming convention is::

            tug1 --> nas['tug1'][se]
            mug1 --> nas['extse'][se]['mug1']

    node_pairs : 2d array_like
        Four column matrix where each row contains the superelement ID
        and node ID for two non-coincident nodes: ``[SE1, Node1, SE2,
        Node2]``. The relative displacement between each node pair is
        computed along a vector from Node1 to Node2 with positive
        meaning increased distance.

    Returns
    -------
    reldtm : 2d ndarray
        This is the displacement-dependent relative displacement
        recovery matrix. There is one row per node pair and the order
        given in `node_pairs` is preserved.
    dist : 1d ndarray
        Vector of distances between node pairs.
    labels : list
        List of labels briefly describing each row in `reldtm`. For
        example, if one row in `node_pairs` is: ``[101, 3, 102, 30]``,
        the label for that row would be: 'SE102,30 - SE101,3'.

    Notes
    -----
    The algorithm works as follows:

      1. Forms two data recovery matrices for the X, Y and Z DOF of
         each node listed in `node_pairs`. One (`DTMG`) recovers from
         residual g-set DOF and the other (`DTMQ`) recovers from the
         residual q-set.

      2. Multiplies `DTMG` by the g-set residual rigid-body modes. The
         resulting 6-column matrix is passed to
         :func:`pyyeti.nastran.n2p.find_xyz_triples` which calculates
         the location of each node and applies coordinated transforms
         to `DTMQ` such that it recovers in the basic coordinate
         system for all nodes.

      3. For each pair of nodes in `node_pairs`:

         a. Form a new rectangular coordinate system based at Node1
            with the z-axis pointing to Node2.

         b. Transform the Node1 and Node2 rows of `DTMQ` to output in
            the new coordinate system.

         c. Form a single row of the final `reldtm` by subtracting the
            Node1 Z recovery from the Node2 Z recovery: ``dtm2[2] -
            dtm1[2]``. This means that a positive relative
            displacement corresponds to an increased distance between
            the two nodes.

    As a final check, the magnitude of the rigid-body part of
    `reldtm` is examined. If the largest value is greater than 1e-6 a
    warning message is printed.

    Example usage::

        # load nastran data:
        nas = op2.rdnas2cam('nas2cam')

        SC = 101
        n2p.addulvs(nas, SC)

        # read in more data for SC since it is a Craig-Bampton model:
        if 'tug1' not in nas:
            nas['tug1'] = {}
        nas['tug1'][SC] = nastran.rddtipch('outboard.pch')

        if 'extse' not in nas:
            nas['extse'] = {}
        nas['extse'][SC] = nastran.op4.read('outboard.op4')

        node_pairs = [
            [SC,  3,   SC, 10],
            [ 0, 11,   SC, 18],
        ]

        reldtm, dist, lbls = relative_displacement_dtm(
            nas, node_pairs)

        # add the above items to the data recovery:
        drdefs = cla.DR_Def({'se': 0})

        @cla.DR_Def.addcat
        def _():
            name = 'reldisp'
            desc = 'Relative Displacements'
            units = 'in'
            labels = lbls
            drms = {name: reldtm}
            drfunc = f"Vars[se]['{name}'] @ sol.d"
            histpv = 'all'
            drdefs.add(**locals())

        # prepare spacecraft data recovery matrices
        DR = cla.DR_Event()
        DR.add(nas, drdefs)

        # initialize results (ext, mnc, mxc for all drms)
        results = DR.prepare_results(mission, event)

        # solve equations of motion:
        ts = ode.SolveUnc(*mbk, h)
        sol = ts.tsolve(genforce, static_ic=1)
        sol.t = t
        sol = DR.apply_uf(sol, *mbk, nas['nrb'], rfmodes)
        results.time_data_recovery(sol, nas['nrb'],
                                   caseid, DR, LC, j)

        # write report of results:
        results.rpttab()

    Raises
    ------
    ValueError
        When Node1 and Node2 of any node pair are coincident.
    """
    # to get locations in basic, need rb modes relative to origin:
    rbg = n2p.rbgeom_uset(nas["uset"][0])

    # call formdrm only once per superelement:
    node_pairs = np.atleast_2d(node_pairs)
    nrel = node_pairs.shape[0]
    dtmq = np.empty((nrel * 6, nas["lambda"][0].shape[0]))
    dtmg = np.empty((nrel * 6, 6))

    senodes = np.vstack((node_pairs[:, :2], node_pairs[:, 2:]))
    senodesdof = np.array([[se, n, i] for se, n in senodes for i in range(1, 4)])

    for se in set(senodes[:, 0]):
        pvd = senodesdof[:, 0] == se
        dof = senodesdof[pvd, 1:]
        try:
            tq = n2p.formdrm(nas, se, dof)[0]
            tx = n2p.formdrm(nas, se, dof, gset=True)[0]
        except ValueError:
            u1x = n2p.formulvs(nas, se, gset=True)
            pv = n2p.mkdofpv(nas["tug1"][se], "p", dof)[0]
            tq = nas["extse"][se]["mug1"][pv] @ nas["ulvs"][se]
            tx = nas["extse"][se]["mug1"][pv] @ u1x
        dtmq[pvd] = tq
        dtmg[pvd] = tx @ rbg

    mats = {"dtmq": dtmq}
    xyz = n2p.find_xyz_triples(dtmg, mats=mats, inplace=True)

    reldtm = np.empty((nrel, dtmq.shape[1]))
    dist = np.empty(nrel)

    ext_coords = xyz.coords.max(axis=0)
    for i in range(nrel):
        n1 = slice(i * 3, i * 3 + 3)
        n2 = slice((nrel + i) * 3, (nrel + i) * 3 + 3)

        # make transform from basic to C:
        a = xyz.coords[i * 3]
        b = xyz.coords[(i + nrel) * 3]

        # check for coincident nodes:
        if la.norm((b - a) / ext_coords) < 1e-5:
            raise ValueError(
                f"coincident nodes detected at index {i} of "
                f"`node_pairs`: {node_pairs[i]}"
            )

        c = a + (b - a)[[1, 2, 0]]
        C = np.array([[1, 1, 0], a, b, c])
        To_local = n2p.mkusetcoordinfo(C, None, {})[2:].T
        dtm1 = To_local @ dtmq[n1]
        dtm2 = To_local @ dtmq[n2]

        # positive for moving apart (b - a > 0):
        reldtm[i] = dtm2[2] - dtm1[2]
        dist[i] = la.norm(b - a)

    labels = [f"SE{se2},{n2} - SE{se1},{n1}" for se1, n1, se2, n2 in node_pairs]
    return reldtm, dist, labels
