import numpy as np
import scipy.linalg as la
from pyyeti import era, ode
import pytest

# pyyeti/era.py 412 78 81% 25-32, 41-43, 576-577, 607-615, 666-669,
# 679-681, 788-791, 797-800, 806-809, 1023, 1026-1031, 1105, 1146-1152,
# 1167, 1179-1182, 1319-1370, 1650

# 25-32, 41-43, 673-674, 684-686, 793-796, 802-805, 811-814,
# 1031-1036, 1151-1157, 1172, 1184-1187, 1325-1376


def test_era():
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

    dt = 0.01
    t = np.arange(0, 1, dt)
    F = np.zeros((3, len(t)))
    ts = ode.SolveExp2(M, D, K, dt)
    sol = ts.tsolve(force=F, v0=[1, 1, 1])

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
