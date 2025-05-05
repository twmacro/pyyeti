import re
import numpy as np
import scipy.linalg as la
from pyyeti import era, ode
import pytest
import warnings


def get_model_data():
    M = np.identity(3)
    K = np.array(
        [
            [4185.1498, 576.6947, 3646.8923],
            [576.6947, 2104.9252, -28.0450],
            [3646.8923, -28.0450, 3451.5583],
        ]
    )
    D = np.array(
        [
            [4.96765646, 0.97182432, 4.0162425],
            [0.97182432, 6.71403672, -0.86138453],
            [4.0162425, -0.86138453, 4.28850828],
        ]
    )

    (w2, phi) = la.eigh(K, M)
    omega = np.sqrt(w2)
    freq_hz = omega / (2 * np.pi)
    modal_damping = phi.T @ D @ phi
    zeta = np.diag(modal_damping) / (2 * omega)

    return M, K, D, freq_hz, zeta, phi


def test_era():
    M, K, D, freq_hz, zeta, phi = get_model_data()

    dt = 0.01
    t = np.arange(0, 1, dt)
    F = np.zeros((3, len(t)))
    ts = ode.SolveExp2(M, D, K, dt)
    sol = ts.tsolve(force=F, v0=[1, 1, 1])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        era_fit = era.ERA(
            sol.v,
            sr=1 / dt,
            auto=True,
            input_labels=["x", "y", "z"],
            all_lower_limits=0.1,
            FFT=True,
            # FFT_range=(0.0, 25.0),
            # show_plot=False,
            verbose=False,
        )

    assert np.allclose(era_fit.freqs_hz, freq_hz)
    assert np.allclose(era_fit.zeta, zeta)
    phi_rat = phi / era_fit.phi
    phi_rat = phi_rat / phi_rat[0]
    assert np.allclose(np.ones((3, 3)), phi_rat)

    for val in era_fit.lower_limits.values():
        assert np.isclose(0.1, val)

    # different size H:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        era_fit = era.ERA(
            sol.v,
            sr=1 / dt,
            auto=True,
            alpha=65,
            input_labels=["x", "y", "z"],
            all_lower_limits=0.1,
            FFT=True,
            FFT_range=(0.0, 25.0),
            # show_plot=False,
            verbose=False,
        )

    assert np.allclose(era_fit.freqs_hz, freq_hz)
    assert np.allclose(era_fit.zeta, zeta)
    phi_rat = phi / era_fit.phi
    phi_rat = phi_rat / phi_rat[0]
    assert np.allclose(np.ones((3, 3)), phi_rat)

    # different size H 2:
    era_fit = era.ERA(
        sol.v,
        sr=1 / dt,
        auto=True,
        beta=65,
        input_labels=["x", "y", "z"],
        all_lower_limits=0.1,
        show_plot=False,
        verbose=False,
    )

    assert np.allclose(era_fit.freqs_hz, freq_hz)
    assert np.allclose(era_fit.zeta, zeta)
    phi_rat = phi / era_fit.phi
    phi_rat = phi_rat / phi_rat[0]
    assert np.allclose(np.ones((3, 3)), phi_rat)

    # different size H 3:
    era_fit = era.ERA(
        sol.v,
        sr=1 / dt,
        auto=True,
        alpha=55,
        beta=40,
        input_labels=["x", "y", "z"],
        all_lower_limits=0.1,
        show_plot=False,
        verbose=False,
    )

    assert np.allclose(era_fit.freqs_hz, freq_hz)
    assert np.allclose(era_fit.zeta, zeta)
    phi_rat = phi / era_fit.phi
    phi_rat = phi_rat / phi_rat[0]
    assert np.allclose(np.ones((3, 3)), phi_rat)

    # different usage of svd_tol:
    era_fit = era.ERA(
        sol.v,
        sr=1 / dt,
        auto=True,
        svd_tol=6,
        input_labels=["x", "y", "z"],
        show_plot=False,
        verbose=False,
    )

    assert np.allclose(era_fit.freqs_hz, freq_hz)
    assert np.allclose(era_fit.zeta, zeta)
    phi_rat = phi / era_fit.phi
    phi_rat = phi_rat / phi_rat[0]
    assert np.allclose(np.ones((3, 3)), phi_rat)

    with pytest.raises(RuntimeError, match="No modes selected"):
        era_fit = era.ERA(
            sol.v,
            sr=1 / dt,
            auto=True,
            svd_tol=6,
            input_labels=["x", "y", "z"],
            all_lower_limits=1.1,  # force failure
            show_plot=False,
            verbose=False,
        )


def test_NExT():
    with pytest.raises(ValueError, match="invalid value for `domain`"):
        era.NExT([1, 2, 3], 100, lag_stop=5, domain="bad value")

    M, K, D, freq_hz, zeta, phi = get_model_data()

    dt = 0.01
    t = np.arange(0, 100, dt)
    sr = 1 / dt
    np.random.seed(1)  # for repeatability
    F = np.random.randn(3, len(t))
    ts = ode.SolveUnc(M, D, K, dt)
    sol = ts.tsolve(force=F)

    era_fit1 = era.ERA(
        era.NExT(sol.a, sr, lag_stop=75),
        sr=sr,
        auto=True,
        show_plot=False,
        verbose=False,
    )

    # check for equivalence:
    era_fit2 = era.ERA(
        era.NExT(
            sol.a,
            sr,
            lag_stop=75,
            domain="frequency",
            nperseg=10_000,
            window="boxcar",
        ),
        sr=1 / dt,
        auto=True,
        show_plot=False,
        verbose=False,
    )
    assert np.allclose(era_fit1.freqs_hz, era_fit2.freqs_hz)

    # show that unbiased is doing something:
    era_fit2 = era.ERA(
        era.NExT(sol.a, sr, lag_stop=75, unbiased=False),
        sr=1 / dt,
        auto=True,
        show_plot=False,
        verbose=False,
    )
    assert not np.allclose(era_fit1.freqs_hz, era_fit2.freqs_hz)

    # show that demeaning is working:
    era_fit2 = era.ERA(
        era.NExT(sol.a + 10, sr, lag_stop=75),
        sr=1 / dt,
        auto=True,
        show_plot=False,
        verbose=False,
    )
    assert np.allclose(era_fit1.freqs_hz, era_fit2.freqs_hz)

    era_fit2 = era.ERA(
        era.NExT(sol.a + 0.1, sr, lag_stop=75, demean=False),
        sr=1 / dt,
        auto=True,
        show_plot=False,
        verbose=False,
    )
    assert not np.allclose(era_fit1.freqs_hz, era_fit2.freqs_hz)
