import numpy as np
import scipy.linalg as la
from types import SimpleNamespace
import pandas as pd
from pyyeti import ode, frclim


class FEM_2D:
    """
    Documentation
    """

    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements = elements

        #  convert the nodes and elements to pandas DataFrames:
        self.nodes_df = (
            pd.DataFrame(nodes, columns=("node", "x", "y"))
            .astype({"node": np.int64})
            .set_index("node")
        )
        cols = ("node1", "node2", "area", "E", "I", "rho")
        self.elements_df = pd.DataFrame(elements, columns=cols).astype(
            {"node1": np.int64, "node2": np.int64}
        )

        self.model = self._form_mk()

    # Put mass and stiffness in global coordinates:
    @staticmethod
    def _rot(mat, th):
        """
        Coordinate transformation for an element

        Parameters
        ----------
        mat : 2d array_like
            Mass or stiffness matrix for an element; 6x6
        th : scalar
            Angle of element in radians from x-axis

        Returns
        -------
        rot_mat : 2d ndarray
            Transformed matrix:  ``T.T @ mat @ T``
        """
        c = np.cos(th)
        s = np.sin(th)
        trans = np.array(
            [
                [c, s, 0, 0, 0, 0],
                [-s, c, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, c, s, 0],
                [0, 0, 0, -s, c, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        return trans.T @ mat @ trans

    def _form_mk(self):
        """
        Form the stiffness matrix, and optionally the mass matrix.

        Parameters
        ----------
        nodes : 2d array_like
            3-column matrix defining the nodes: [id, x, y]
        elements : 2d array_like
            5 or 6-column column matrix of element properties::

                [node1, node2, A, E, I [, rho]]
                A   = cross-section area
                E   = Young's modulus
                I   = area moment of inertia
                rho = material density, optional

        Returns
        -------
        A SimpleNamespace with the members:

        K : 2d ndarray
            Stiffness matrix
        M : 2d ndarray
            Mass matrix; will be None if `rho` was not part of
            `elements`
        elen : 1d ndarray
            Length of each element (ordered like elements)
        etheta : 1d ndarray
            Angle from x-axis of each element (rad)
        nodes : 2d ndarray
            Sorted version of the input
        elements : 2d ndarray
            Copy of the input
        """
        nodes, elements = np.atleast_2d(self.nodes, self.elements)
        if nodes.shape[1] != 3:  # Checks how many columns there are
            raise ValueError(
                "'nodes' is incorrectly sized (must have 3 columns)"
            )  # Raises your own error
        re, ce = elements.shape
        if ce != 5 and ce != 6:
            raise ValueError(
                "'elements' is incorrectly sized (must have 5 or 6 columns)"
            )
        rn = nodes.shape[0]  # Number of rows in nodes array.
        i = np.argsort(nodes[:, 0])  # Sorts the nodes
        nodes = nodes[i]  # Create the sorted nodes array
        ids = nodes[:, 0].astype(int)  # ids are first column of nodes

        DOF = 3 * rn  # DOF = total degrees of freedom.
        K = np.zeros((DOF, DOF))  # K is a square matrix DOFxDOF

        if ce == 6:
            rho = elements[:, 5]  # get the density of each element
            M = K.copy()  # M is a square matrix of shape DOFxDOF
        else:  # Set densities to 0 if not given.
            rho = 0.0 * elements[:, 0]
            M = None  # No mass matrix

        elen = np.empty(re)
        etheta = np.empty(re)

        N1 = elements[:, 0].astype(int)  # Node 1 IDs
        N2 = elements[:, 1].astype(int)  # Node 2 IDs

        for n, (n1, n2, (a, e, I), rho) in enumerate(
            zip(N1, N2, elements[:, 2:5], rho)
        ):

            p1 = (ids == n1).nonzero()[0]
            p2 = (ids == n2).nonzero()[0]

            if p1.size == 0:  # node1 is undefined
                raise ValueError(
                    ("elements[{}] references undefined node: {}").format(n, n1)
                )
            else:
                p1 = p1[0]  # node1 is defined
            if p2.size == 0:
                raise ValueError(
                    ("elements[{}] references undefined node: {}").format(n, n2)
                )
            else:
                p2 = p2[0]

            xy1 = nodes[p1, 1:]

            xy2 = nodes[p2, 1:]
            l1 = xy2 - xy1  # element length (array)
            L = elen[n] = np.hypot(*l1)  # length
            angle = etheta[n] = np.arctan2(l1[1], l1[0])

            # Element (local) Bernoulli-Euler Mass Matrix
            bk = a * e / L
            fk = e * I / (L ** 3)
            k_t = np.array(
                [
                    [bk, 0, 0, -bk, 0, 0],
                    [0, 12 * fk, 6 * L * fk, 0, -12 * fk, 6 * L * fk],
                    [0, 6 * L * fk, 4 * L ** 2 * fk, 0, -6 * L * fk, 2 * L ** 2 * fk],
                    [-bk, 0, 0, bk, 0, 0],
                    [0, -12 * fk, -6 * L * fk, 0, 12 * fk, -6 * L * fk],
                    [0, 6 * L * fk, 2 * L ** 2 * fk, 0, -6 * L * fk, 4 * L ** 2 * fk],
                ]
            )

            # Element (local) Bernoulli-Euler Mass Matrix
            if ce == 6:
                m_e = rho * a * L
                m_t = (
                    m_e
                    / 420
                    * np.array(
                        [
                            [140, 0, 0, 70, 0, 0],
                            [0, 156, 22 * L, 0, 54, -13 * L],
                            [0, 22 * L, 4 * L ** 2, 0, 13 * L, -3 * L ** 2],
                            [70, 0, 0, 140, 0, 0],
                            [0, 54, 13 * L, 0, 156, -22 * L],
                            [0, -13 * L, -3 * L ** 2, 0, -22 * L, 4 * L ** 2],
                        ]
                    )
                )

            if angle != 0.0:
                k_t = self._rot(k_t, angle)
                if ce == 6:
                    m_t = self._rot(m_t, angle)

            # Put element mass and stiffness in full mass and stiffness
            p1 *= 3  # 3 DOF
            p2 *= 3
            v = [p1, p1 + 1, p1 + 2, p2, p2 + 1, p2 + 2]  # 6 DOF
            v = np.ix_(v, v)  # 6x6
            K[v] += k_t
            if ce == 6:  # if given a density, calculate mass
                M[v] += m_t

        return SimpleNamespace(
            K=K, M=M, elen=elen, etheta=etheta, nodes=nodes, elements=elements
        )

    def apply_constraints(self, bcs):
        """
        Apply constraints to model

        Parameters
        ----------
        consts : 2d array_like
            Constraint matrix: [id, c1, c2, c3]. c1, c2, c3 are each 0
            or 1 specifying whether or a DOF is fixed. The order of
            DOF are: x, y, rz

        Returns
        -------
        None

        Notes
        -----
        The input `model` is updated to contain to additional
        members::

            fixed = index vector indicating which DOF are constrained
            free  = index vector indicating which DOF are free

        For example, to restrain node 10 in the x-direction only::

            model = form_mk(nodes, elements)
            apply_constraints(model, [10, 1, 0, 0])
        """
        bcs = np.atleast_2d(bcs)  # Views inputs as array at least 2d
        r, c = bcs.shape  # r = Num of rows, c = Num of columns
        if c > 0 and c != 4:  # Checks that there are 4 columns
            raise ValueError(
                "'consts' is incorrectly sized (must have 4 columns)"
            )  # Prints/raises this error if !=4
        fixed = np.zeros(self.model.K.shape[0], bool)
        # Returns an array of bool values that is the size/length of
        # the num of rows of model.K. Since, np.zeroes returns all 0s,
        # the array will be all 'False'.

        ids = self.model.nodes[:, 0].astype(int)
        # Returns array of only node ids and turns them into integers
        # instead of floats
        if c > 0:
            for n, bc in enumerate(bcs):
                p1 = (ids == bc[0]).nonzero()[0]
                # ids==const[0] returns bool (T/F) if id = const[0]
                # which is the loc of the id data in the constraint
                # array .nonzero finds the nonzero or nonFalse
                # elements and saves them in a 2d array
                if p1.size == 0:
                    raise ValueError(
                        ("consts[{}] references undefined node: {}").format(n, bc[0])
                    )
                else:
                    p1 = p1[0]  # Takes the integer in the array
                p1 *= 3
                v = slice(p1, p1 + 3)
                fixed[v] = bc[1:].astype(bool)
        self.model.free = (~fixed).nonzero()[0]
        # bitwise inversion; opposite or inverse value for each
        # element in fixed; then store nonzero values
        self.model.fixed = fixed.nonzero()[0]  # stores nonzero values

    def _ensure_bcs(self):
        """
        Ensures that :func:`apply_constraints` has been called (so that
        `model.free` and `model.fixed` exist.
        """
        if getattr(self.model, "free", None) is None:
            self.apply_constraints([])

    def solveig(self):
        """
        Solves eigenvalue problem

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        On return, `model` will have 3 additional members:

            lam = vector of eigenvalues
            fhz = sqrt(lambda)/(2*pi); natural frequencies in Hz
            phi = mode shape matrix:
                  ``lambda = phi.T @ K[free, free] @ phi``

        This routine will use ``model.free`` if present (see
        :func:`apply_constraints`); otherwise, it will solve the
        free-free eigenvalue problem.
        """
        self._ensure_bcs()
        model = self.model
        vv = np.ix_(model.free, model.free)
        w, phi = la.eigh(model.K[vv], model.M[vv])
        model.lam = w  # natural frequencies**2 (eigenvalues)
        model.phi = phi  # the mode shapes (eigenvectors)
        r = model.K.shape[0]
        c = model.phi.shape[1]
        model.phi_full = np.zeros((r, c))
        model.phi_full[model.free] = model.phi

        ind = pd.MultiIndex.from_product(
            [self.nodes_df.index, ["x", "y", "rz"]], names=["node", "dof"]
        )
        model.phi_full_df = pd.DataFrame(model.phi_full, index=ind)
        model.fhz = np.sqrt(abs(w)) / (2 * np.pi)  # the nat freqs in Hz


def test_ntfl_rbdamp():
    # PLOT_DIFF = True
    # PLOT_DIFF = False

    INC_RB_DAMPING = True
    # INC_RB_DAMPING = False

    TRY_SU_SOLVER = True  # only used if INC_RB_DAMPING is True
    # TRY_SU_SOLVER = False   # only used if INC_RB_DAMPING is True

    USE_PRE_EIG = False  # only used if SolveUnc is used
    # USE_PRE_EIG = True  # only used if SolveUnc is used

    if not INC_RB_DAMPING:
        Solver = ode.SolveUnc
        if USE_PRE_EIG:
            FrcLim = frclim
            opts = dict(pre_eig=True)
        else:
            FrcLim = frclim  # _solveunc_norb
            opts = dict()
        factor = 1.0  # 5e-3
        # make up some random damping matrices for source and load:
        # - the system version will be assembled from these two for
        #   consistency

        def insert_damping(model, factor):
            # define random modal damping with no damping on rb-modes:
            b = np.random.randn(*model.K.shape)
            b = b.T @ b * factor
            nrb = (abs(model.lam) < 5e-3).sum()
            b[:nrb] = 0.0
            b[:, :nrb] = 0.0

            # want B (physical damping)
            P = model.phi_full
            # Bmodal = P.T  B  P
            iP = la.inv(P)
            model.B = iP.T @ b @ iP

    else:
        if TRY_SU_SOLVER:
            Solver = ode.SolveUnc
            if USE_PRE_EIG:
                # pre_eig = True doesn't work ... not sure why at the
                # moment (maybe conflict with rb modes setting)
                FrcLim = frclim
                opts = dict(pre_eig=True)
            else:
                FrcLim = frclim  # _solveunc_norb
                opts = dict()
        else:
            Solver = ode.FreqDirect
            FrcLim = frclim
            opts = dict()
        factor = 5.0e-3

        def insert_damping(model, factor):
            b = np.random.randn(*model.K.shape)
            model.B = b.T @ b * factor

    # full system:
    #     ^
    #     |
    #   Fe|-->
    #     O 1
    #      \
    #       \2        3        4    each node has 3 dof
    #        O========O========O
    #     |--Source---|- Load -|

    nodes = np.array([[1, 0.0, 3.0], [2, 3.0, 0.0], [3, 8.0, 0.0], [4, 13.0, 0.0]])

    # node1 node2 area E I rho
    elements = np.array(
        [
            [1, 2, 0.005, 1e5, 0.01, 1.3],
            [2, 3, 0.005, 1e5, 0.01, 1.3],
            [3, 4, 0.005, 1e5, 0.01, 1.3],
        ]
    )
    sys = FEM_2D(nodes, elements)
    sys.solveig()

    # Source:
    source_nodes = np.array([[1, 0.0, 3.0], [2, 3.0, 0.0], [3, 8.0, 0.0]])

    # node1 node2 area E I rho
    source_elements = np.array(
        [[1, 2, 0.005, 1e5, 0.01, 1.3], [2, 3, 0.005, 1e5, 0.01, 1.3]]
    )
    source = FEM_2D(source_nodes, source_elements)
    source.solveig()

    # Load:
    load_nodes = np.array([[3, 8.0, 0.0], [4, 13.0, 0.0]])

    # node1 node2 area E I rho
    load_elements = np.array([[3, 4, 0.005, 1e5, 0.01, 1.3]])
    load = FEM_2D(load_nodes, load_elements)
    load.solveig()

    insert_damping(source.model, factor)
    insert_damping(load.model, factor)

    # Assemble source & load manually to compare with "sys" above ...
    # this is to test our assembling methodology so we can compare
    # a random damping for NT vs the full system
    #
    # To do this, write the DOF of each upstream in terms of the
    # independent downstream DOF:
    #   source_dof = Ss @ sys_dof
    #   load_dof = Sl @ sys_dof
    #
    # where:
    #   source_dof = [1(x,y,rz), 2(x,y,rz), 3_source(x,y,rz)]
    #   load_dof = [3_load(x,y,rz), 4(x,y,rz)]
    #   sys_dof = [1(x,y,rz), 2(x,y,rz), 3(x,y,rz), 4(x,y,rz)]
    #
    # By inspection (each I is 3x3 identity):
    #   Ss = [[I 0 0 0]
    #         [0 I 0 0]
    #         [0 0 I 0]]
    #   Sl = [[0 0 I 0]
    #         [0 0 0 I]]

    Ss = np.eye(3 * 3, 4 * 3)
    Sl = np.eye(2 * 3, 4 * 3, k=2 * 3)

    msys = Ss.T @ source.model.M @ Ss + Sl.T @ load.model.M @ Sl
    ksys = Ss.T @ source.model.K @ Ss + Sl.T @ load.model.K @ Sl
    bsys = Ss.T @ source.model.B @ Ss + Sl.T @ load.model.B @ Sl

    # store damping in 'sys':
    sys.model.B = bsys

    if not (np.allclose(msys, sys.model.M) or np.allclose(ksys, sys.model.K)):
        raise RuntimeError("bad system assembly")

    # run an analysis on 'sys', then use NT to duplicate it
    # (hopefully)
    freq = np.arange(1.0, 250, 0.1)
    Fsys = np.zeros((sys.model.K.shape[0], freq.shape[0]))
    Fsys[:3] = 1.0

    su_sys = Solver(sys.model.M, sys.model.B, sys.model.K, **opts)
    sol_sys = su_sys.fsolve(Fsys, freq)
    # compute i/f force on load:
    iff_sys = (
        load.model.M[:3] @ sol_sys.a[2 * 3 :]
        + load.model.B[:3] @ sol_sys.v[2 * 3 :]
        + load.model.K[:3] @ sol_sys.d[2 * 3 :]
    )

    # extract i/f accel of load:
    ifa_sys = sol_sys.a[2 * 3 : 3 * 3]

    # Norton-Thevenin run:
    # 1. compute free accel
    # 2. call ntlf; that will:
    #    - compute source & load apparent mass
    #    - apply N-T equations
    #    - return i/f force & accel

    # 1. compute free accel:
    su_source = Solver(source.model.M, source.model.B, source.model.K, **opts)
    sol_source = su_source.fsolve(Fsys[: source.model.M.shape[0]], freq)
    free_accel = sol_source.a[2 * 3 :]

    # setup source and load for ntfl:
    # - need recovery matrix for bdof of source and load:
    Sbdof = np.eye(3, source.model.M.shape[0], 2 * 3)
    Lbdof = np.eye(3, load.model.M.shape[0], 0)

    Source = [source.model.M, source.model.B, source.model.K, Sbdof]
    Load = [load.model.M, load.model.B, load.model.K, Lbdof]
    NT = FrcLim.ntfl(Source, Load, free_accel, freq)

    assert np.allclose(abs(iff_sys), abs(NT.F), rtol=1e-04, atol=1e-04)
    assert np.allclose(abs(ifa_sys), abs(NT.A), rtol=1e-04, atol=1e-04)
