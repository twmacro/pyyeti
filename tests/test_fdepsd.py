import numpy as np
from pyyeti import psd, dsp
from pyyeti.fdepsd import fdepsd
import scipy.signal as signal
import pytest


def compare(fde1, fde2):
    assert np.allclose(fde1.freq, fde2.freq)
    assert np.allclose(fde1.psd, fde2.psd)
    assert np.allclose(fde1.peakamp, fde2.peakamp)
    assert np.allclose(fde1.binamps, fde2.binamps)
    assert np.allclose(fde1.count, fde2.count)
    assert np.allclose(fde1.srs, fde2.srs)
    assert np.allclose(fde1.var, fde2.var)
    assert fde1.parallel in ("no", "yes")
    assert fde2.parallel in ("no", "yes")
    assert fde1.resp in ("absacce", "pvelo")
    assert fde2.resp in ("absacce", "pvelo")
    assert fde1.ncpu >= 1
    assert fde2.ncpu >= 1
    assert np.allclose(fde1.sig, fde2.sig)


def test_fdepsd_absacce():
    np.random.seed(1)
    TF = 60  # make a 60 second signal
    sp = 1.0
    spec = np.array([[20, sp], [50, sp]])
    sig, sr, t = psd.psd2time(
        spec,
        ppc=10,
        fstart=20,
        fstop=50,
        df=1 / TF,
        winends=dict(portion=10),
        gettime=True,
    )
    freq = np.arange(30.0, 50.1)
    q = 25
    fde_auto = fdepsd(sig, sr, freq, q)
    fde_no = fdepsd(sig, sr, freq, q, parallel="no")
    fde_yes = fdepsd(sig, sr, freq, q, parallel="yes")

    compare(fde_auto, fde_no)
    compare(fde_auto, fde_yes)
    pv = np.logical_and(freq > 32, freq < 45)
    assert abs(np.mean(fde_auto.psd.iloc[pv, :2], axis=0) - sp).max() < 0.22
    assert abs(np.mean(fde_auto.psd.iloc[pv, 2:], axis=0) - sp).max() < 0.12
    assert np.all(fde_auto.freq == freq)

    # check the damage indicators:
    flight_over_test = fde_auto.di_sig.loc[freq[0]] / fde_auto.di_test.loc[freq[0]]
    var = fde_auto.var[freq[0]]
    assert abs(1 - flight_over_test["b=4"] ** (1 / 2) / var) < 0.1
    assert abs(1 - flight_over_test["b=8"] ** (1 / 4) / var) < 0.2
    assert abs(1 - flight_over_test["b=12"] ** (1 / 6) / var) < 0.3

    fde_none = fdepsd(sig, sr, freq, q, rolloff=None)
    fde_pre = fdepsd(sig, sr, freq, q, rolloff="prefilter")
    assert np.all((fde_none.psd <= fde_pre.psd).values)

    T0 = 120
    fde_T0 = fdepsd(sig, sr, freq, q, T0=T0)
    factor = np.log(freq * 60) / np.log(freq * T0)
    assert np.allclose(
        fde_T0.psd.iloc[:, :2], fde_auto.psd.iloc[:, :2].multiply(factor, axis=0)
    )
    assert np.all((fde_T0.psd.iloc[:, 2:] < fde_auto.psd.iloc[:, 2:]).values)


def test_fdepsd_pvelo():
    TF = 60  # make a 60 second signal
    sp = 1.0
    spec = np.array([[20, sp], [50, sp]])
    sig, sr, t = psd.psd2time(
        spec,
        ppc=10,
        fstart=20,
        fstop=50,
        df=1 / TF,
        winends=dict(portion=10),
        gettime=True,
    )
    freq = np.arange(30.0, 50.1)
    q = 25
    fde_auto = fdepsd(sig, sr, freq, q, resp="pvelo")
    fde_no = fdepsd(sig, sr, freq, q, resp="pvelo", parallel="no", verbose=True)

    # test some features:
    sig5 = signal.detrend(sig)
    sig5 = dsp.windowends(sig5, portion=min(int(0.25 * sr), 50, len(sig5)))
    b, a = signal.butter(3, 5 / (sr / 2), "high")
    sig5 = signal.lfilter(b, a, sig5)
    fde_yes = fdepsd(
        sig5,
        sr,
        freq,
        q,
        resp="pvelo",
        parallel="yes",
        hpfilter=None,
        detrend=False,
        winends=None,
    )

    compare(fde_auto, fde_no)
    compare(fde_auto, fde_yes)
    pv = np.logical_and(freq > 32, freq < 45)
    assert abs(np.mean(fde_auto.psd.iloc[pv, :2], axis=0) - sp).max() < 0.22
    assert abs(np.mean(fde_auto.psd.iloc[pv, 2:], axis=0) - sp).max() < 0.22
    assert np.all(fde_auto.freq == freq)

    # check the damage indicators:
    flight_over_test = fde_auto.di_sig.loc[freq[0]] / fde_auto.di_test.loc[freq[0]]
    var = fde_auto.var[freq[0]]
    assert abs(1 - flight_over_test["b=4"] ** (1 / 2) / var) < 0.1
    assert abs(1 - flight_over_test["b=8"] ** (1 / 4) / var) < 0.2
    assert abs(1 - flight_over_test["b=12"] ** (1 / 6) / var) < 0.3

    fde_none = fdepsd(sig, sr, freq, q, resp="pvelo", rolloff=None)
    fde_pre = fdepsd(sig, sr, freq, q, resp="pvelo", rolloff="prefilter")
    assert np.all((fde_none.psd <= fde_pre.psd).values)

    T0 = 120
    fde_T0 = fdepsd(sig, sr, freq, q, T0=T0, resp="pvelo")
    factor = np.log(freq * 60) / np.log(freq * T0)
    assert np.allclose(
        fde_T0.psd.iloc[:, :2], fde_auto.psd.iloc[:, :2].multiply(factor, axis=0)
    )
    assert np.all((fde_T0.psd.iloc[:, 2:] < fde_auto.psd.iloc[:, 2:]).values)


def test_fdepsd_winends():
    TF = 60  # make a 60 second signal
    sp = 1.0
    spec = np.array([[20, sp], [50, sp]])
    sig, sr, t = psd.psd2time(
        spec,
        ppc=10,
        fstart=20,
        fstop=50,
        df=1 / TF,
        winends=dict(portion=40),
        gettime=True,
    )
    sig[0] = 20
    freq = np.arange(1.0, 3.1)
    q = 25
    fde = fdepsd(sig, sr, freq, q, hpfilter=None)
    fde2 = fdepsd(sig, sr, freq, q, hpfilter=None, winends=None)
    fde3 = fdepsd(sig, sr, freq, q, hpfilter=None, winends=dict(portion=20))
    assert np.all((fde2.psd.iloc[:, 0] > 4 * fde.psd.iloc[:, 0]).values)
    assert np.all((fde2.psd.iloc[:, 0] > 4 * fde3.psd.iloc[:, 0]).values)


def test_fdepsd_error():
    sig = [[1, 2, 3], [4, 5, 6]]
    sr = 100
    freq = [1, 2, 3]
    q = 20
    with pytest.raises(ValueError):
        fdepsd(sig, sr, freq, q)
    with pytest.raises(ValueError):
        fdepsd(freq, sr, freq, q, resp="badresp")


def test_ski_slope():
    # np.random.seed(1)
    spec = np.array([[20.0, 1.0], [100.0, 1.0], [150.0, 10.0], [1000.0, 10.0]])
    sig, sr = psd.psd2time(spec, 20, 1000)

    sig[0] = sig.max()

    freq = np.arange(20, 1000.0, 10.0)
    Q = 10

    d = fdepsd(sig, sr, freq, Q)
    assert d.psd["G1"].iat[0] < 1.0
