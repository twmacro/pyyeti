import numpy as np
from scipy.integrate import odeint
from pyyeti import srs
import scipy.stats as stats
import scipy.signal as signal
import pytest


def get_params():
    Q = 10
    zeta = 1 / 2 / Q
    freq = np.arange(1, 15, 0.4)
    nfreq = len(freq)
    j = 2 * nfreq // 3
    k = j - 3
    sr = 200
    t = np.arange(0, 0.2, 1 / sr)
    w = 2 * np.pi * freq
    B = 2 * zeta * w
    K = w**2
    zdd = np.random.randn(len(t)) + np.sin(2 * np.pi * freq[j] * t) + 1.15
    b, a = signal.butter(3, 30 / (sr / 2))
    zdd = signal.filtfilt(b, a, zdd)
    return Q, freq, nfreq, j, k, sr, t, w, B, K, zdd


def get_params_2():
    Q = 15
    zeta = 1 / 2 / Q
    freq = np.arange(3, 15, 0.4)
    nfreq = len(freq)
    j = 2 * nfreq // 3
    k = j - 3
    sr = 800
    t = np.arange(0, 0.5, 1 / sr)
    w = 2 * np.pi * freq
    B = 2 * zeta * w
    K = w**2
    n = len(t)
    zdd = np.zeros(n, float)
    cut = 80
    zdd[10:cut] = 1.0
    return Q, freq, nfreq, j, k, sr, t, w, B, K, cut, zdd


# base drive for an sdof:
#
#                      _____    ^
#                     |     |   |
#                     |     |  ---  SDOF response (x)
#                     |_____|
#                      /  |
#                      \ |_|    ^
#                      /  |     |
#                    [======]  ---  input base acceleration (zdd)
#
#   xdd = sum forces on m
#       = w^2 (z - x) + 2 zeta w (zd - xd)
#   xdd + 2 zeta w (xd - zd) + w^2 (x - z) = 0
#
#   let u = x - z
#       xdd = udd + zdd
#
#   udd + zdd + 2 zeta w ud + w^2 u = 0
#   udd + 2 zeta w ud + w^2 u = -zdd


def deriv(y, t, F, B, K):
    """Function to return derivative for ODE solver.

    Equations::
        M xdd + B xd + K x = f
          z = xd
          zd = xdd
          M zd + B z + K x = f

          Therefore:
          xd = z
          zd = inv(M) (f - K x - B z)

          y = [x, z]

          M = I for our problem
    """
    d = np.zeros(y.shape, dtype=float)
    n = len(y) // 2
    d[:n] = y[n:]
    d[n:] = F(t) - K * y[:n] - B * y[n:]
    return d


def do_odeint_sol(t, zdd, ic, w, B, K, nfreq):
    def F(x):
        return np.interp(x, t, -zdd)

    zdd = zdd.copy()
    if ic == "steady":
        # udd = ud = 0;  u = -zdd/w^2
        u0 = -zdd[0] / K
        if w[0] == 0:
            u0[0] = 0
    elif ic == "shift":
        u0 = np.zeros(nfreq)
        zdd -= zdd[0]
    elif ic == "mshift":
        u0 = np.zeros(nfreq)
        zdd -= np.mean(zdd)
        t = np.hstack((-t[1], t))
        zdd = np.hstack((0, zdd))
    else:
        u0 = np.zeros(nfreq)
        t = np.hstack((-t[1], t))
        zdd = np.hstack((0, zdd))
    ud0 = np.zeros(nfreq)
    y0 = np.hstack((u0, ud0))
    u_ud = odeint(deriv, y0, t, args=(F, B, K))
    if ic == "mshift" or ic == "zero":
        u_ud = u_ud[1:]
        zdd = zdd[1:]
        t = t[1:]
    u = u_ud[:, :nfreq]
    ud = u_ud[:, nfreq:]
    pacce = K * u
    pvelo = w * u
    xdd = -B * ud - pacce
    udd = xdd - zdd.reshape(-1, 1)
    return xdd, udd, ud, u, pacce, pvelo


def do_srs_sol(
    zdd,
    sr,
    freq,
    Q,
    ic="steady",
    stype="absacce",
    ppc=10,
    rolloff="fft",
    eqsine=False,
    time="primary",
    parallel="auto",
):
    sh, resp = srs.srs(
        zdd,
        sr,
        freq,
        Q,
        getresp=True,
        ic=ic,
        stype=stype,
        rolloff=rolloff,
        eqsine=eqsine,
        time=time,
        parallel=parallel,
    )
    sh1 = srs.srs(
        zdd,
        sr,
        freq,
        Q,
        ic=ic,
        peak="abs",
        stype=stype,
        rolloff=rolloff,
        eqsine=eqsine,
        time=time,
        parallel=parallel,
    )
    shpos = srs.srs(
        zdd,
        sr,
        freq,
        Q,
        ic=ic,
        peak="pos",
        stype=stype,
        rolloff=rolloff,
        eqsine=eqsine,
        time=time,
        parallel=parallel,
    )
    shneg = srs.srs(
        zdd,
        sr,
        freq,
        Q,
        ic=ic,
        peak="neg",
        stype=stype,
        rolloff=rolloff,
        eqsine=eqsine,
        time=time,
        parallel=parallel,
    )
    shposs = srs.srs(
        zdd,
        sr,
        freq,
        Q,
        ic=ic,
        peak="poss",
        stype=stype,
        rolloff=rolloff,
        eqsine=eqsine,
        time=time,
        parallel=parallel,
    )
    shnegs = srs.srs(
        zdd,
        sr,
        freq,
        Q,
        ic=ic,
        peak="negs",
        stype=stype,
        rolloff=rolloff,
        eqsine=eqsine,
        time=time,
        parallel=parallel,
    )
    shrms = srs.srs(
        zdd,
        sr,
        freq,
        Q,
        ic=ic,
        peak="rms",
        stype=stype,
        rolloff=rolloff,
        eqsine=eqsine,
        time=time,
        parallel=parallel,
    )
    sh0, resp0 = srs.srs(
        zdd[0],
        sr,
        freq,
        Q,
        getresp=True,
        ic=ic,
        stype=stype,
        rolloff=rolloff,
        eqsine=eqsine,
        time=time,
        parallel=parallel,
    )
    sh01 = srs.srs(
        zdd[0],
        sr,
        freq,
        Q,
        ic=ic,
        stype=stype,
        rolloff=rolloff,
        eqsine=eqsine,
        time=time,
        parallel=parallel,
    )
    return (sh, resp, sh1, shpos, shneg, shposs, shnegs, shrms, sh0, resp0, sh01)


def do_comp(
    stype,
    ic,
    sh,
    resp,
    sh1,
    shpos,
    shneg,
    shposs,
    shnegs,
    shrms,
    sh0,
    resp0,
    sh01,
    sol,
    comp_time_hist=True,
):
    rtol = 1e-3
    atol = 1e-3
    # resp['hist'] is time x nsignals x nfreq
    # sol is time x nfreq
    if comp_time_hist:
        lcc = np.array(
            [stats.pearsonr(v1, v2)[0] for v1, v2 in zip(resp["hist"][:, 0].T, sol.T)]
        )
        assert np.allclose(lcc, 1)
        assert np.allclose(resp["hist"][:, 0], sol, rtol=rtol, atol=atol)

    mx = np.abs(np.max(resp["hist"][:, 0], axis=0))
    mn = np.abs(np.min(resp["hist"][:, 0], axis=0))
    mxs = np.max(resp["hist"][:, 0], axis=0)
    mns = np.min(resp["hist"][:, 0], axis=0)
    av = np.mean(resp["hist"][:, 0], axis=0)
    amx = np.max(abs(resp["hist"][:, 0]), axis=0)
    rms = np.sqrt(np.mean(resp["hist"][:, 0] ** 2, axis=0))
    omx = np.max(sol, axis=0)
    omn = np.min(sol, axis=0)
    oav = np.mean(sol, axis=0)
    assert np.allclose(mxs, omx, rtol=rtol, atol=atol)
    assert np.allclose(mns, omn, rtol=rtol, atol=atol)
    if comp_time_hist:
        assert np.allclose(av, oav, rtol=rtol, atol=atol)
        if ic == "mshift":
            assert np.all(resp0["hist"][0, 0, :] == 0)
        else:
            assert np.all(resp["hist"][0, 0, :] == resp0["hist"][0, 0, :])
    assert np.all(np.abs(resp0["hist"][0, 0, :]) == sh0)
    assert np.allclose(sh, sh1)
    assert np.allclose(sh01, sh0)
    assert np.allclose(amx, sh)
    assert np.allclose(mx, shpos)
    assert np.allclose(mn, shneg)
    assert np.allclose(mxs, shposs)
    assert np.allclose(mns, shnegs)
    assert np.allclose(rms, shrms)


def test_srs_absacce():
    Q, freq, nfreq, j, k, sr, t, w, B, K, zdd = get_params()
    stype = "absacce"
    for parallel in ["yes", "no"]:
        for ic in ["steady", "zero"]:
            # get solution via srs.srs:
            (
                sh,
                resp,
                sh1,
                shpos,
                shneg,
                shposs,
                shnegs,
                shrms,
                sh0,
                resp0,
                sh01,
            ) = do_srs_sol(zdd, sr, freq, Q, ic=ic, stype=stype, parallel=parallel)
            # get solution via odeint:
            xdd, udd, ud, u, pacce, pvelo = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)
            do_comp(
                stype,
                ic,
                sh,
                resp,
                sh1,
                shpos,
                shneg,
                shposs,
                shnegs,
                shrms,
                sh0,
                resp0,
                sh01,
                xdd,
            )
    for ic in ["shift", "mshift"]:
        # get solution via srs.srs:
        (
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
        ) = do_srs_sol(zdd, sr, freq, Q, ic=ic, stype=stype)
        # get solution via odeint:
        xdd, udd, ud, u, pacce, pvelo = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)
        do_comp(
            stype,
            ic,
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
            xdd,
        )


def test_srs_relacce():
    Q, freq, nfreq, j, k, sr, t, w, B, K, zdd = get_params()
    stype = "relacce"
    for parallel in ["yes", "no"]:
        for ic in ["steady", "zero"]:
            # get solution via srs.srs:
            (
                sh,
                resp,
                sh1,
                shpos,
                shneg,
                shposs,
                shnegs,
                shrms,
                sh0,
                resp0,
                sh01,
            ) = do_srs_sol(zdd, sr, freq, Q, ic=ic, stype=stype, parallel=parallel)
            # get solution via odeint:
            xdd, udd, ud, u, pacce, pvelo = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)
            do_comp(
                stype,
                ic,
                sh,
                resp,
                sh1,
                shpos,
                shneg,
                shposs,
                shnegs,
                shrms,
                sh0,
                resp0,
                sh01,
                udd,
            )
    for ic in ["shift", "mshift"]:
        # get solution via srs.srs:
        (
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
        ) = do_srs_sol(zdd, sr, freq, Q, ic=ic, stype=stype)
        # get solution via odeint:
        xdd, udd, ud, u, pacce, pvelo = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)
        do_comp(
            stype,
            ic,
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
            udd,
        )


def test_srs_relvelo():
    Q, freq, nfreq, j, k, sr, t, w, B, K, zdd = get_params()
    stype = "relvelo"
    for parallel in ["yes", "no"]:
        for ic in ["steady", "zero"]:
            # get solution via srs.srs:
            (
                sh,
                resp,
                sh1,
                shpos,
                shneg,
                shposs,
                shnegs,
                shrms,
                sh0,
                resp0,
                sh01,
            ) = do_srs_sol(zdd, sr, freq, Q, ic=ic, stype=stype, parallel=parallel)
            # get solution via odeint:
            xdd, udd, ud, u, pacce, pvelo = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)
        do_comp(
            stype,
            ic,
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
            ud,
        )
    for ic in ["shift", "mshift"]:
        # get solution via srs.srs:
        (
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
        ) = do_srs_sol(zdd, sr, freq, Q, ic=ic, stype=stype)
        # get solution via odeint:
        xdd, udd, ud, u, pacce, pvelo = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)
        do_comp(
            stype,
            ic,
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
            ud,
        )


def test_srs_reldisp():
    Q, freq, nfreq, j, k, sr, t, w, B, K, zdd = get_params()
    stype = "reldisp"
    for parallel in ["yes", "no"]:
        for ic in ["steady", "zero"]:
            # get solution via srs.srs:
            (
                sh,
                resp,
                sh1,
                shpos,
                shneg,
                shposs,
                shnegs,
                shrms,
                sh0,
                resp0,
                sh01,
            ) = do_srs_sol(zdd, sr, freq, Q, ic=ic, stype=stype, parallel=parallel)
            # get solution via odeint:
            xdd, udd, ud, u, pacce, pvelo = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)
            do_comp(
                stype,
                ic,
                sh,
                resp,
                sh1,
                shpos,
                shneg,
                shposs,
                shnegs,
                shrms,
                sh0,
                resp0,
                sh01,
                u,
            )
    for ic in ["shift", "mshift"]:
        # get solution via srs.srs:
        (
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
        ) = do_srs_sol(zdd, sr, freq, Q, ic=ic, stype=stype)
        # get solution via odeint:
        xdd, udd, ud, u, pacce, pvelo = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)
        do_comp(
            stype,
            ic,
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
            u,
        )


def test_srs_pacce():
    Q, freq, nfreq, j, k, sr, t, w, B, K, zdd = get_params()
    stype = "pacce"
    for parallel in ["yes", "no"]:
        for ic in ["steady", "zero"]:
            # get solution via srs.srs:
            (
                sh,
                resp,
                sh1,
                shpos,
                shneg,
                shposs,
                shnegs,
                shrms,
                sh0,
                resp0,
                sh01,
            ) = do_srs_sol(zdd, sr, freq, Q, ic=ic, stype=stype, parallel=parallel)
            # get solution via odeint:
            xdd, udd, ud, u, pacce, pvelo = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)
            do_comp(
                stype,
                ic,
                sh,
                resp,
                sh1,
                shpos,
                shneg,
                shposs,
                shnegs,
                shrms,
                sh0,
                resp0,
                sh01,
                pacce,
            )
    for ic in ["shift", "mshift"]:
        # get solution via srs.srs:
        (
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
        ) = do_srs_sol(zdd, sr, freq, Q, ic=ic, stype=stype)
        # get solution via odeint:
        xdd, udd, ud, u, pacce, pvelo = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)
        do_comp(
            stype,
            ic,
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
            pacce,
        )


def test_srs_pvelo():
    Q, freq, nfreq, j, k, sr, t, w, B, K, zdd = get_params()
    stype = "pvelo"
    for parallel in ["yes", "no"]:
        for ic in ["steady", "zero"]:
            # get solution via srs.srs:
            (
                sh,
                resp,
                sh1,
                shpos,
                shneg,
                shposs,
                shnegs,
                shrms,
                sh0,
                resp0,
                sh01,
            ) = do_srs_sol(zdd, sr, freq, Q, ic=ic, stype=stype, parallel=parallel)
            # get solution via odeint:
            xdd, udd, ud, u, pacce, pvelo = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)
            do_comp(
                stype,
                ic,
                sh,
                resp,
                sh1,
                shpos,
                shneg,
                shposs,
                shnegs,
                shrms,
                sh0,
                resp0,
                sh01,
                pvelo,
            )
    for ic in ["shift", "mshift"]:
        # get solution via srs.srs:
        (
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
        ) = do_srs_sol(zdd, sr, freq, Q, ic=ic, stype=stype)
        # get solution via odeint:
        xdd, udd, ud, u, pacce, pvelo = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)
        do_comp(
            stype,
            ic,
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
            pvelo,
        )


def test_srs_multiple_signals():
    Q, freq, nfreq, j, k, sr, t, w, B, K, cut, zdd = get_params_2()
    ic = "zero"
    zdd = zdd[:cut]
    zddm = np.vstack((zdd, zdd, zdd)).T
    for stype in ["absacce", "relacce", "relvelo", "reldisp", "pacce", "pvelo"]:
        for time in ["primary", "total", "residual"]:
            for ic in ["steady", "shift", "zero", "mshift"]:
                sh = srs.srs(zdd, sr, freq, Q, ic=ic, stype=stype, time=time)
                shr, resp = srs.srs(
                    zdd, sr, freq, Q, ic=ic, stype=stype, time=time, getresp=True
                )
                shm = srs.srs(zddm, sr, freq, Q, ic=ic, stype=stype, time=time)
                shmr, respm = srs.srs(
                    zddm, sr, freq, Q, ic=ic, stype=stype, time=time, getresp=True
                )
                assert np.all(sh.shape == (nfreq,))
                assert np.all(sh.shape == shr.shape)
                assert np.all(shm.shape == (nfreq, 3))
                assert np.all(shm.shape == shmr.shape)
                assert np.all(resp["t"] == respm["t"])
                assert resp["sr"] == respm["sr"]
                resp_sz = len(resp["t"]), 1, nfreq
                respm_sz = len(resp["t"]), 3, nfreq
                assert np.all(resp["hist"].shape == resp_sz)
                assert np.all(respm["hist"].shape == respm_sz)
                for i in range(3):
                    assert np.all(shm[:, i] == sh)
                    assert np.all(shmr[:, i] == shr)
                    assert np.all(resp["hist"][:, 0] == respm["hist"][:, i])


def test_srs_total():
    Q, freq, nfreq, j, k, sr, t, w, B, K, cut, zdd = get_params_2()
    ic = "zero"
    for i, stype in enumerate(
        ["absacce", "relacce", "relvelo", "reldisp", "pacce", "pvelo"]
    ):
        # get solution via srs.srs:
        (
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
        ) = do_srs_sol(zdd[:cut], sr, freq, Q, ic=ic, stype=stype, time="total")
        # length of time history should be original + 1 cycle of
        # lowest frequency:
        assert resp["hist"].shape[0] == int(np.ceil(cut + sr / freq[0]))
        # get solution via odeint:
        sol = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)[i]
        do_comp(
            stype,
            ic,
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
            sol,
            comp_time_hist=False,
        )


def test_srs_residual():
    Q, freq, nfreq, j, k, sr, t, w, B, K, cut, zdd = get_params_2()
    ic = "zero"
    for i, stype in enumerate(["absacce"]):
        # get solution via srs.srs:
        (
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
        ) = do_srs_sol(
            zdd[:cut], sr, freq, Q, ic=ic, stype=stype, time="residual", parallel="yes"
        )
        # length of time history should be 1 cycle of lowest frequency:
        assert resp["hist"].shape[0] == int(np.ceil(sr / freq[0]))
        # get solution via odeint:
        sol = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)[i][cut:]
        do_comp(
            stype,
            ic,
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
            sol,
            comp_time_hist=False,
        )
    for i, stype in enumerate(
        ["absacce", "relacce", "relvelo", "reldisp", "pacce", "pvelo"]
    ):
        # get solution via srs.srs:
        (
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
        ) = do_srs_sol(zdd[:cut], sr, freq, Q, ic=ic, stype=stype, time="residual")
        # length of time history should be 1 cycle of lowest frequency:
        assert resp["hist"].shape[0] == int(np.ceil(sr / freq[0]))
        # get solution via odeint:
        sol = do_odeint_sol(t, zdd, ic, w, B, K, nfreq)[i][cut:]
        do_comp(
            stype,
            ic,
            sh,
            resp,
            sh1,
            shpos,
            shneg,
            shposs,
            shnegs,
            shrms,
            sh0,
            resp0,
            sh01,
            sol,
            comp_time_hist=False,
        )


def test_srs_rolloff():
    sr = 200
    t = np.arange(0, 5, 1 / sr)
    offset = 12
    sig = np.sin(2 * np.pi * 15 * t) + 3 * np.sin(2 * np.pi * 35 * t) + offset
    Q = 50
    frq = [5, 10, 35, 38]
    sh1 = srs.srs(sig, sr, frq, Q, rolloff="fft", ppc=5)
    sh2 = srs.srs(sig, sr, frq, Q, rolloff=None, ppc=5)
    sh3 = srs.srs(sig, sr, frq, Q, rolloff="none", ppc=5)
    sh4 = srs.srs(sig, sr, frq, Q, rolloff="lanczos", ppc=5)
    assert np.all(sh1 == sh2)
    assert np.all(sh1 == sh3)
    assert np.all(sh1 == sh4)
    sh1 = srs.srs(sig, sr, frq, Q, rolloff="fft", ppc=10)
    assert np.all(sh1 >= sh2)
    sh2 = srs.srs(sig, sr, frq, Q, rolloff="fft", ppc=15)
    sh1_fft = sh1
    sh2_fft = sh2

    # compare sh1, sh2; first two should nearly match:
    assert abs(sh1[:2] - sh2[:2]).max() < 0.005 * abs(sh1[:2]).max()
    # sh2 should be higher on other 2:
    assert np.all(sh2[2:] > sh1[2:])

    assert np.max(sh2) > 140 + offset and np.max(sh2) < 160 + offset
    sh1, resp = srs.srs(sig, sr, frq, Q, rolloff="fft", ppc=15, getresp=True)
    assert sh1[0] >= sh2[0]
    assert sh1[-1] == sh2[-1]
    sh1l = srs.srs(sig, sr, frq, Q, rolloff="linear", ppc=15)
    assert sh1l[0] < sh2[0]
    assert sh1l[-1] < sh2[-1]
    # prefilter doesn't pay attention to the ppc; if input, it's
    # just used:
    sh1 = srs.srs(sig, sr, frq, Q, rolloff="prefilter", ppc=5)
    sh2, resp = srs.srs(sig, sr, frq, Q, rolloff="prefilter", ppc=15, getresp=True)
    assert np.all(sh1 == sh2)

    sh1 = srs.srs(sig, sr, frq, Q, rolloff="lanczos", ppc=10)

    # compare sh1, sh4; first two should nearly match:
    assert abs(sh1[:2] - sh4[:2]).max() < 0.005 * abs(sh1[:2]).max()
    # sh1 should be higher on other 2:
    assert np.all(sh1[2:] > sh4[2:])

    sh2 = srs.srs(sig, sr, frq, Q, rolloff="lanczos", ppc=15)

    # compare sh1, sh2; first two should nearly match:
    assert abs(sh1[:2] - sh2[:2]).max() < 0.005 * abs(sh1[:2]).max()
    # sh2 should be higher on other 2:
    assert np.all(sh2[2:] > sh1[2:])

    assert np.max(sh2) > 140 + offset and np.max(sh2) < 160 + offset

    # compare lanczos to fft ... should be pretty close:
    assert abs(sh1_fft - sh1).max() < 0.001 * abs(sh1).max()
    assert abs(sh2_fft - sh2).max() < 0.001 * abs(sh2).max()
    # print('10: ', sh1, sh1_fft_10, sh1-sh1_fft_10, sh1l)
    # print('15: ', sh2, sh2_fft_15, sh2-sh2_fft_15)

    # test passing in functions
    def rmsmeth(resp):
        return np.sqrt(np.mean(resp**2, axis=0))

    sh1 = srs.srs(sig, sr, frq, Q, rolloff=srs.fftroll, peak=rmsmeth)
    sh2 = srs.srs(sig, sr, frq, Q, rolloff="fft", peak="rms")
    assert np.all(sh1 == sh2)

    sh1, resp1 = srs.srs(
        sig, sr, frq, Q, rolloff=srs.fftroll, peak=rmsmeth, getresp=True
    )
    sh2, resp2 = srs.srs(sig, sr, frq, Q, rolloff="fft", peak="rms", getresp=True)
    assert np.all(sh1 == sh2)
    assert np.all(resp1["hist"] == resp2["hist"])


def test_eqsine():
    sr = 200
    t = np.arange(0, 5, 1 / sr)
    sig = np.ones((11, 1)).dot(np.sin(2 * np.pi * 15 * t)[None, :])
    Q = 35
    frq = np.linspace(5, 10, 5)
    sh = srs.srs(sig, sr, frq, Q, rolloff="none", eqsine=1)
    sh1 = srs.srs(sig, sr, frq, Q, rolloff="none") / Q
    assert np.allclose(sh, sh1)
    sh, resp = srs.srs(sig, sr, frq, Q, rolloff="none", eqsine=1, getresp=1)
    sh1, resp1 = srs.srs(sig, sr, frq, Q, rolloff="none", getresp=1)
    assert np.allclose(sh, sh1 / Q)
    assert np.all(resp["t"] == resp1["t"])
    assert np.allclose(resp["hist"], resp1["hist"] / Q)


def test_auto_parallel():
    sr = 1000
    t = np.arange(0, 5, 1 / sr)
    sig = np.ones((11, 1)).dot(np.sin(2 * np.pi * 15 * t)[None, :]).T
    # 1000 * 5 * 11 = 75000
    Q = 35
    frq = np.linspace(5, 50, 5)
    sh = srs.srs(sig, sr, frq, Q)
    sh1 = srs.srs(sig, sr, frq, Q, parallel="no", maxcpu=None)
    assert np.allclose(sh, sh1)
    sh1 = srs.srs(sig, sr, frq, Q, parallel="yes", maxcpu=2)
    assert np.allclose(sh, sh1)

    # Note that the following test can fail on systems with a high
    # number of cpus. Increasing the allowable number of file handles
    # solves the issue. For example, in one case with a 720 cpus,
    # resetting ulimit as shown solved the issue:
    #
    #     ulimit -n 2048   # ulimit was 1024
    sh1 = srs.srs(sig, sr, frq, Q, parallel="auto", maxcpu=None)
    assert np.allclose(sh, sh1)


def test_odd_fft_srs():
    t = np.linspace(0, 3, 101)
    sr = 1 / t[1]  # 1/.03 = 33
    sig = np.sin(2 * np.pi * 2 * t)
    Q = 35
    frq = 5.0
    sh1 = srs.srs(sig, sr, frq, Q, rolloff="fft")
    sh2 = srs.srs(sig[:-1], sr, frq, Q, rolloff="fft")
    assert np.allclose(sh1, sh2)


def test_srs_bad_parallel():
    sr = 20
    t = np.arange(0, 1, 1 / sr)
    sig = np.sin(2 * np.pi * 15 * t)
    Q = 50
    frq = 5
    with pytest.raises(ValueError):
        srs.srs(sig, sr, frq, Q, parallel=12)


def test_srs_bad_Q():
    sr = 20
    t = np.arange(0, 1, 1 / sr)
    sig = np.sin(2 * np.pi * 15 * t)
    Q = 0.1
    frq = 5
    with pytest.raises(ValueError):
        srs.srs(sig, sr, frq, Q)


def test_vrs():
    import numpy as np
    from pyyeti import srs

    spec = np.array([[20, 0.0053], [150, 0.04], [600, 0.04], [2000, 0.0036]])
    frq = np.arange(20, 2000, 2.0)
    Q = 10
    fn = [100, 200, 1000]
    v, m, resp = srs.vrs(spec, frq, Q, linear=False, Fn=fn, getresp=True)
    assert np.all(resp["f"] == frq)
    v_sbe = np.array([6.38, 11.09, 16.06])
    m_sbe = np.array([6.47, 11.21, 15.04])
    assert np.all(np.abs(v - v_sbe[:, None]) < 0.01)
    assert np.all(np.abs(m - m_sbe[:, None]) < 0.01)
    assert np.all(
        np.abs(np.max(resp["psd"][:, 0], axis=1) - np.array([2.69, 4.04, 1.47])) < 0.01
    )

    v2 = srs.vrs(spec, frq, Q, linear=False, Fn=fn)
    assert np.all(v2 == v)

    v3, m3 = srs.vrs(spec, frq, Q, linear=False, Fn=fn, getmiles=True)
    assert np.all(v3 == v)
    assert np.all(m3 == m)

    v3b, m3b = srs.vrs(spec, frq, Q, linear=False, getmiles=True)
    v3c, m3c = srs.vrs(spec, frq, Q, linear=False, Fn=frq, getmiles=True)
    assert np.all(v3b == v3c)
    assert np.all(m3b == m3c)

    v1 = srs.vrs(spec, frq, Q, linear=False)
    v2 = srs.vrs(spec, frq, Q, linear=False, Fn=frq)
    assert np.all(v1 == v2)

    spec = np.hstack((spec, spec[:, 1:]))
    v4, m4, resp4 = srs.vrs(spec, frq, Q, linear=False, Fn=fn, getresp=True)
    rtol = 1e-13
    atol = 1e-13
    assert np.allclose(v4[:, 0:1], v, rtol, atol)
    assert np.allclose(v4[:, 1:2], v, rtol, atol)
    assert np.allclose(m4[:, 0:1], m, rtol, atol)
    assert np.allclose(m4[:, 1:2], m, rtol, atol)
    assert np.allclose(resp4["psd"][:, :1], resp["psd"], rtol, atol)
    assert np.allclose(resp4["psd"][:, 1:], resp["psd"], rtol, atol)
    assert resp4["psd"].shape == (3, 2, len(frq))

    v5, m5 = srs.vrs(
        (spec[:, 0], spec[:, 1]), frq, Q, linear=True, Fn=fn, getmiles=True
    )
    assert np.all(np.abs(v5 - [6.38, 11.09, 21.78]) < 0.01)
    assert np.all(np.abs(m5 - [6.47, 11.21, 21.56]) < 0.01)
    with pytest.raises(ValueError):
        srs.vrs(spec[:, :2], frq, Q=0.1, linear=True)


def test_srs_frf():
    # scalar frequency test, single frf
    srs_frq = [3]
    frf_frq = np.array(srs_frq)
    n = len(frf_frq)
    frf = np.random.randn(n) + 1j * np.random.randn(n)
    Q = 20
    srs_frq = frf_frq
    sh = srs.srs_frf(frf, frf_frq, srs_frq, Q)
    pks_should_be = np.abs(frf) * np.sqrt(Q**2 + 1)
    assert np.abs(sh - pks_should_be) < 1e-12

    # scalar frequency test, multi frf
    srs_frq = [3]
    frf_frq = np.array(srs_frq)
    n = len(frf_frq)
    frf = np.random.randn(n, 3) + 1j * np.random.randn(n, 3)
    Q = 20
    srs_frq = frf_frq
    sh = srs.srs_frf(frf, frf_frq, srs_frq, Q)
    pks_should_be = np.abs(frf) * np.sqrt(Q**2 + 1)
    assert np.all(np.abs(sh.max(axis=0) - pks_should_be) < 1e-12)

    # multiple frequency test, with zeros
    frf_frq = np.arange(0, 50, 0.1)
    n = len(frf_frq)
    while True:
        frf = np.random.randn(n, 3) + 1j * np.random.randn(n, 3)
        # the check below will fail if the 0 frequency system has
        # maximum frf input:
        i = abs(frf).argmax(axis=0)
        if (i > 0).all():
            break
    # to ensure that peak input is peaky enough that the peak actually
    # occurs @ the natural frequency:
    i = abs(frf).argmax(axis=0)
    frf[i, [0, 1, 2]] *= 2.0
    Q = 20
    srs_frq = frf_frq
    sh = srs.srs_frf(frf, frf_frq, srs_frq, Q)
    pks_should_be = np.abs(frf).max(axis=0) * np.sqrt(Q**2 + 1)
    assert np.all(np.abs(sh.max(axis=0) - pks_should_be) < 1e-12)

    # multiple frequency test, without zeros
    frf_frq = np.arange(0.1, 50, 0.1)
    n = len(frf_frq)
    frf = np.random.randn(n, 3) + 1j * np.random.randn(n, 3)
    # to ensure that peak input is peaky enough that the peak actually
    # occurs @ the natural frequency:
    i = abs(frf).argmax(axis=0)
    frf[i, [0, 1, 2]] *= 2.0
    Q = 20
    srs_frq = frf_frq
    sh = srs.srs_frf(frf, frf_frq, srs_frq, Q)
    pks_should_be = np.abs(frf).max(axis=0) * np.sqrt(Q**2 + 1)
    assert np.all(np.abs(sh.max(axis=0) - pks_should_be) < 1e-12)

    # test getresp:
    # frfs are integration frequency x nfrfs x srs_frq
    sh, resp = srs.srs_frf(frf, frf_frq, srs_frq[::2], Q, getresp=True)
    assert len(resp["freq"]) == resp["frfs"].shape[0]
    assert resp["frfs"].shape[1] == frf.shape[1]
    assert resp["frfs"].shape[2] == srs_frq[::2].shape[0]
    sh2 = abs(resp["frfs"]).max(axis=0)
    assert np.allclose(sh2.T, sh)


def test_srs_frf_2():
    pk_input = 3.0
    pk_frq = 15.0
    frf = np.array([pk_input / 3, pk_input, pk_input / 3])
    frf_frq = np.array([pk_frq - 5, pk_frq, pk_frq + 5])
    srs_frq = np.array([pk_frq])
    Q = 20
    sh = srs.srs_frf(frf, frf_frq, srs_frq, Q)

    pk_should_be = pk_input * np.sqrt(Q**2 + 1)
    sh = srs.srs_frf(frf, frf_frq, srs_frq, Q)
    assert abs(pk_should_be - sh[0, 0]) < 1e-10

    p_peak = Q * np.sqrt((np.sqrt(1 + 2 / Q**2) - 1))
    frq_should_be = pk_frq / p_peak
    num = 1 + (p_peak / Q) ** 2
    den = (1 - p_peak**2) ** 2 + num - 1
    pk_should_be = pk_input * np.sqrt(num / den)
    sh, frq = srs.srs_frf(frf, frf_frq, None, Q)
    i = sh[:, 0].argmax()
    assert abs(pk_should_be - sh[i, 0]) < 1e-10
    assert abs(frq_should_be - frq[i]) < 1e-10


def test_srs_frf_3():
    pk_input = 3.0
    pk_frq = 15.0
    frf = np.array([pk_input / 3, pk_input, pk_input / 3])
    frf_frq = np.array([pk_frq - 5, pk_frq, pk_frq + 5])
    srs_frq = np.array([pk_frq])
    Q = 20
    sh = srs.srs_frf(frf, frf_frq, srs_frq, Q, scale_by_Q_only=True)
    assert abs(60.0 - sh[0, 0]) < 1e-10

    sh, frq = srs.srs_frf(frf, frf_frq, None, Q, scale_by_Q_only=True)
    assert np.allclose(frq, frf_frq)
    assert np.allclose(sh.ravel(), frf * Q)

    with pytest.raises(ValueError):
        srs.srs_frf(
            frf,
            frf_frq,
            None,
            Q,
            scale_by_Q_only=True,
            getresp=True,
        )


def test_srsmap():
    from pyyeti import ytools, dsp

    sig, ts, fs = ytools.gensweep(10, 1, 50, 4)
    sr = 1 / ts[1]
    frq = np.arange(1.0, 50.1)
    Q = 20
    mp, t, f = srs.srsmap(2, 0.5, sig, sr, frq, Q, 0.02, eqsine=1)
    assert np.all(f == frq)
    dt = 1.0
    segments = int(np.floor(ts[-1] / dt)) - 1
    assert len(t) == segments
    t_should_be = np.arange(segments) * dt + dt
    assert np.allclose(t, t_should_be)
    # test segment 63:
    seg = 63
    pv = np.logical_and(ts >= t[seg] - dt, ts < t[seg] + dt)
    sh = srs.srs(dsp.windowends(sig[pv], 0.02), sr, frq, Q, eqsine=1)
    assert np.allclose(sh, mp[:, seg])


def test_zerofreq():
    dt = 0.01
    n = 700
    f = np.zeros(n, float)
    f[10:110] = 1.0  # area = 1
    f[110:210] = -1.0  # area = -1
    f[300:400] = -1.0
    f[400:500] = 1.0
    frq = 0.0
    sh, resp = srs.srs(f, 1 / dt, frq, 25, stype="relacce", getresp=True)
    np.allclose(resp["hist"][:, 0, 0], -f)

    sh, resp = srs.srs(f, 1 / dt, frq, 25, stype="relvelo", getresp=True)
    np.allclose(resp["hist"][110, 0, 0], -1.0)
    np.allclose(resp["hist"][210:400, 0, 0], 0.0)
    np.allclose(resp["hist"][400, 0, 0], 1.0)
    np.allclose(resp["hist"][500:, 0, 0], 0.0)

    sh, resp = srs.srs(f, 1 / dt, frq, 25, stype="reldisp", getresp=True)
    # area under velo triangle = -1 @ 210
    np.allclose(resp["hist"][210:300, 0, 0], -1.0)
    # comes back to initial position after second velo triangle:
    np.allclose(resp["hist"][500:, 0, 0], 0.0)

    for stype in ["absacce", "pvelo", "pacce"]:
        sh, resp = srs.srs(f, 1 / dt, frq, 25, stype=stype, getresp=True)
        np.allclose(sh, 0.0)
        np.allclose(resp["hist"], 0.0)
