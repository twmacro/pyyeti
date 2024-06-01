import inspect
from pathlib import Path
import numpy as np
from scipy import linalg as la
from pyyeti import frclim, ode
from pyyeti.nastran import op2, n2p, bulk, op4
import pytest


def test_calcAM():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    maa = nas["maa"][101]
    kaa = nas["kaa"][101]
    uset = nas["uset"][101]
    b = n2p.mksetpv(uset, "a", "b")
    q = ~b
    b = np.nonzero(b)[0]

    freq = np.arange(1.0, 80.0, 1.0)

    pv = np.any(maa, axis=0)
    q = q[pv]
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]
    baa = np.zeros_like(maa)
    baa[q, q] = 2 * 0.05 * np.sqrt(kaa[q, q])

    nb = len(b)
    bdrm = np.zeros((nb, maa.shape[0]))
    bdrm[:nb, :nb] = np.eye(nb)
    AM1 = frclim.calcAM((maa, baa, kaa, b), freq)
    AM2 = frclim.calcAM((maa, baa, kaa, bdrm), freq)
    assert np.allclose(AM1, AM2)

    fs = ode.SolveUnc(maa, baa, kaa, pre_eig=True)
    AM2 = frclim.calcAM((maa, baa, kaa, b), freq, fs)
    assert np.allclose(AM1, AM2)


def test_ntfl():
    freq = np.arange(0.0, 25.1, 0.1)
    M1 = 10.0
    M2 = 30.0
    M3 = 3.0
    M4 = 2.0
    c1 = 15.0
    c2 = 15.0
    c3 = 15.0
    k1 = 45000.0
    k2 = 25000.0
    k3 = 10000.0

    # 2. Solve coupled system:

    MASS = np.array([[M1, 0, 0, 0], [0, M2, 0, 0], [0, 0, M3, 0], [0, 0, 0, M4]])
    DAMP = np.array(
        [
            [c1, -c1, 0, 0],
            [-c1, c1 + c2, -c2, 0],
            [0, -c2, c2 + c3, -c3],
            [0, 0, -c3, c3],
        ]
    )
    STIF = np.array(
        [
            [k1, -k1, 0, 0],
            [-k1, k1 + k2, -k2, 0],
            [0, -k2, k2 + k3, -k3],
            [0, 0, -k3, k3],
        ]
    )
    F = np.vstack((np.ones((1, len(freq))), np.zeros((3, len(freq)))))
    fs = ode.SolveUnc(MASS, DAMP, STIF, pre_eig=True)
    fullsol = fs.fsolve(F, freq)
    A_coupled = fullsol.a[1]
    F_coupled = (
        M2 / 2 * A_coupled
        - k2 * (fullsol.d[2] - fullsol.d[1])
        - c2 * (fullsol.v[2] - fullsol.v[1])
    )

    # 3. Solve for free acceleration; SOURCE setup: [m, b, k, bdof]:

    ms = np.array([[M1, 0], [0, M2 / 2]])
    cs = np.array([[c1, -c1], [-c1, c1]])
    ks = np.array([[k1, -k1], [-k1, k1]])
    source = [ms, cs, ks, [[0, 1]]]
    fs_source = ode.SolveUnc(ms, cs, ks, pre_eig=True)
    sourcesol = fs_source.fsolve(F[:2], freq)
    As = sourcesol.a[1:2]  # free acceleration

    # LOAD setup: [m, b, k, bdof]:

    ml = np.array([[M2 / 2, 0, 0], [0, M3, 0], [0, 0, M4]])
    cl = np.array([[c2, -c2, 0], [-c2, c2 + c3, -c3], [0, -c3, c3]])
    kl = np.array([[k2, -k2, 0], [-k2, k2 + k3, -k3], [0, -k3, k3]])
    load = [ml, cl, kl, [[1, 0, 0]]]

    # 4. Use NT to couple equations. First value (rigid-body motion)
    # should equal ``Source Mass / Total Mass = 25/45 = 0.55555...``
    # Results should match the coupled method.

    r = frclim.ntfl(source, load, As, freq)
    assert np.allclose(25 / 45, abs(r.R[0, 0]))
    assert np.allclose(A_coupled, r.A)
    assert np.allclose(F_coupled, r.F)
    with pytest.raises(ValueError):
        frclim.ntfl(source, load, As, freq[:-1])

    r2 = frclim.ntfl(r.SAM, r.LAM, As, freq)
    assert r.SAM is r2.SAM
    assert r.LAM is r2.LAM
    assert np.all(r.TAM == r2.TAM)
    assert np.allclose(r.R, r2.R)
    assert np.allclose(r.A, r2.A)
    assert np.allclose(r.F, r2.F)


def test_ntfl_indeterminate():
    srcdir = Path(inspect.getfile(frclim)).parent / "tests" / "nas2cam_extseout"

    # modal damping zeta:
    zeta = 0.02

    ids = ("out", "in")
    uset, cords, b = {}, {}, {}
    mats = {}
    for id in ids:
        uset[id], cords[id], b[id] = bulk.asm2uset(srcdir / f"{id}board.asm")
        mats[id] = op4.read(srcdir / f"{id}board.op4")

        # add damping:
        bxx = 0 * mats[id]["kxx"]
        q = ~b[id]
        lam = np.diag(mats[id]["kxx"])[q]
        damp = 2 * np.sqrt(lam) * zeta
        bxx[q, q] = damp
        mats[id]["bxx"] = bxx

    maa = {
        "in": mats["in"]["mxx"],
        "out": mats["out"]["mxx"],
    }
    kaa = {
        "in": mats["in"]["kxx"],
        "out": mats["out"]["kxx"],
    }
    baa = {
        "in": mats["in"]["bxx"],
        "out": mats["out"]["bxx"],
    }

    # couple models together to compute coupled system response:
    # dep = S indep
    # dep = {in b; in q; out b; out q}
    # indep = {in/out b; in q; out q}
    m = maa["in"].shape[0]
    n = maa["out"].shape[0]
    nb = np.count_nonzero(b["in"])
    S = {}
    S["in"] = np.block(
        [
            [np.eye(nb), np.zeros((nb, m + n - 2 * nb))],
            [np.zeros((m - nb, nb)), np.eye(m - nb), np.zeros((m - nb, n - nb))],
        ]
    )
    S["out"] = np.block(
        [
            [np.eye(nb), np.zeros((nb, m + n - 2 * nb))],
            [np.zeros((n - nb, m)), np.eye(n - nb)],
        ]
    )
    S["tot"] = np.vstack((S["in"], S["out"]))

    mc = S["tot"].T @ la.block_diag(maa["in"], maa["out"]) @ S["tot"]
    kc = S["tot"].T @ la.block_diag(kaa["in"], kaa["out"]) @ S["tot"]
    bc = S["tot"].T @ la.block_diag(baa["in"], baa["out"]) @ S["tot"]

    # check coupling against Nastran:
    lam, phi = la.eigh(kc, mc)
    freqsys = np.sqrt(abs(lam)) / 2 / np.pi
    eigen = bulk.rdeigen(srcdir / "assemble.out")
    assert np.allclose(freqsys[6:], eigen[0]["cycles"][6:])

    # use first tload vector of "inboard" to apply a force to the system
    pa_in = mats["in"]["px"][:, 0]
    pc = S["in"].T @ pa_in

    # define freq vector
    freq = np.geomspace(0.01, 100.0, 4 * 167 + 1)
    # freq = np.arange(0.1, 100.0, 0.1)
    # ts = ode.SolveUnc(mc, bc, kc, pre_eig=True)
    ts = ode.FreqDirect(mc, bc, kc)

    # keep magnitude of force as-is across frequency domain:
    force = pc[:, None] @ np.ones((1, len(freq)))
    sol = ts.fsolve(force, freq)

    # recover displacements, velocities, and accelerations for both
    # components:
    d, v, a = {}, {}, {}
    ifforce = {}
    for id in ids:
        d[id] = S[id] @ sol.d
        v[id] = S[id] @ sol.v
        a[id] = S[id] @ sol.a

        ifforce[id] = (maa[id] @ a[id] + baa[id] @ v[id] + kaa[id] @ d[id])[:nb]

    # some sanity checks:
    assert abs(ifforce["out"] + (ifforce["in"] - force[:nb])).max() < 1e-3
    assert np.allclose(a["in"][:nb], a["out"][:nb])
    assert np.allclose(v["in"][:nb], v["out"][:nb])
    assert np.allclose(d["in"][:nb], d["out"][:nb])

    # couple via NT:

    # need free-acceleration:
    ts_in = ode.FreqDirect(maa["in"], baa["in"], kaa["in"])
    force_in = pa_in[:, None] @ np.ones((1, len(freq)))
    As = ts_in.fsolve(force_in, freq).a[:nb]

    AM = {}
    NT = {}
    with pytest.warns(
        RuntimeWarning, match="Switching from `SolveUnc` to `FreqDirect`"
    ):
        for method in ("cb", "drm"):
            AM[method] = {}
            for id in ids:
                if method == "cb":
                    drm = np.arange(nb)
                else:
                    drm = np.zeros((nb, maa[id].shape[0]))
                    drm[:, b[id]] = np.eye(nb)
                AM[method][id] = frclim.calcAM([maa[id], baa[id], kaa[id], drm], freq)

            NT[method] = frclim.ntfl(AM[method]["in"], AM[method]["out"], As, freq)

            assert abs(ifforce["out"] - NT[method].F).max() < 1e-3
            assert abs(a["out"][:nb] - NT[method].A).max() < 1e-4


def test_sefl():
    assert np.allclose(1.5, frclim.sefl(1.5, 40, 80))
    assert np.allclose(1.5 / 2, frclim.sefl(1.5, 80, 40))
    assert np.allclose(1.5 * (1 / 2) ** 2, frclim.sefl(1.5, 80, 40, 2))


def test_stdfs():
    m1 = 710  # modal mass + residual mass of lv
    m2 = 3060  # modal mass + residual mass of s/c
    Q = 10
    spec = 1.75
    fl = frclim.stdfs(m2 / m1, Q) * m2 * 1.75
    assert abs(6393.1622 - fl) < 1e-3
    fl = frclim.stdfs(m2 / m1, [Q, Q]) * m2 * 1.75
    assert abs(6393.1622 - fl) < 1e-3


def test_ctdfs():
    m1 = 30  # lv modal mass 75-90 Hz
    M1 = 622  # lv residual mass above 90 Hz
    m2 = 972  # sc modal mass 75-90 Hz
    M2 = 954  # sc residual mass above 90 Hz
    msc = 6961  # total sc mass
    faf = 40  # fundamental axial frequency of s/c
    Q = 10
    spec = 1.75
    fl = frclim._ctdfs_old(m1 / M1, m2 / M2, M2 / M1, Q)[0] * M2 * spec
    assert abs(8686.1 / fl - 1) < 1e-4

    fl = frclim._ctdfs_old(m1 / M1, m2 / M2, M2 / M1, [Q, Q])[0] * M2 * spec
    assert abs(8686.1 / fl - 1) < 1e-4

    fl = frclim.ctdfs(m1 / M1, m2 / M2, M2 / M1, Q)[0] * M2 * spec
    assert abs(8686.1 / fl - 1) < 1e-4

    fl = frclim.ctdfs(m1 / M1, m2 / M2, M2 / M1, [Q, Q])[0] * M2 * spec
    assert abs(8686.1 / fl - 1) < 1e-4

    fl1 = frclim._ctdfs_old(1e-5, m2 / M2, M2 / M1, Q)
    fl2 = frclim._ctdfs_old(0, m2 / M2, M2 / M1, [Q, Q])
    assert np.allclose(fl1, fl2)

    fl1 = frclim.ctdfs(1e-5, m2 / M2, M2 / M1, Q)
    fl2 = frclim.ctdfs(0, m2 / M2, M2 / M1, [Q, Q])
    assert np.allclose(fl1, fl2)

    assert np.all((1, 1) == frclim._ctdfs_old(m1 / M1, 0, M2 / M1, Q))
    assert np.all((1, 1) == frclim.ctdfs(m1 / M1, 0, M2 / M1, Q))
