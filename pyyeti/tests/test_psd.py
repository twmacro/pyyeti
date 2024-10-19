import numpy as np
from numpy.random import Generator, MT19937
from pyyeti import psd
import scipy.signal as signal
import pytest


# temporary patch for numpy < 2.0
try:
    np.trapezoid
except AttributeError:
    np.trapezoid = np.trapz


def test_get_freq_oct():
    assert np.allclose(
        np.array(psd.get_freq_oct(1, [1, 5])),
        np.array(
            [
                [1.0, 1.99526231, 3.98107171],
                [0.70794578, 1.41253754, 2.81838293],
                [1.41253754, 2.81838293, 5.62341325],
            ]
        ),
    )

    assert np.allclose(
        np.array(psd.get_freq_oct(1, [1, 5], trim="outside")),
        np.array(
            [
                [1.0, 1.99526231, 3.98107171],
                [0.70794578, 1.41253754, 2.81838293],
                [1.41253754, 2.81838293, 5.62341325],
            ]
        ),
    )

    assert np.allclose(
        np.array(psd.get_freq_oct(1, [1, 5], trim="band")),
        np.array(
            [
                [1.0, 1.99526231, 3.98107171],
                [0.70794578, 1.41253754, 2.81838293],
                [1.41253754, 2.81838293, 5.62341325],
            ]
        ),
    )

    assert np.allclose(
        np.array(psd.get_freq_oct(1, [1, 5], trim="inside")),
        np.array([[1.99526231], [1.41253754], [2.81838293]]),
    )

    e = np.array(psd.get_freq_oct(1, [1, 2.5], trim="inside"))
    assert np.all(e.shape == (3, 0))

    assert np.allclose(
        np.array(psd.get_freq_oct(1, [1, 2.5], trim="center")),
        np.array(
            [[1.0, 1.99526231], [0.70794578, 1.41253754], [1.41253754, 2.81838293]]
        ),
    )

    with pytest.raises(ValueError):
        psd.get_freq_oct(1, [1, 2.5], trim="badstring")


def test_proc_psd_spec():
    with pytest.raises(ValueError):
        psd.proc_psd_spec(np.array([1]))
    with pytest.raises(ValueError):
        psd.proc_psd_spec((1, 2, 3))
    with pytest.raises(ValueError):
        psd.proc_psd_spec((np.arange(10), np.random.randn(11)))
    with pytest.raises(ValueError):
        psd.proc_psd_spec((np.arange(10), np.random.randn(10, 2, 2)))


def test_area():
    spec = np.array(
        [
            [20, 0.0053, 0.01],
            [150, 0.04, 0.04],
            [600, 0.04, 0.04],
            [2000, 0.0036, 0.04 * 600 / 2000],
        ]
    )
    # the last value is to trigger the s == -1 logic in area

    spec2 = spec[:, 1:]
    f2 = spec[:, 0]
    areas = psd.area(spec)
    areas2 = psd.area((f2, spec2))

    fi = np.arange(200, 20001) * 0.1
    pi = psd.interp(spec, fi, linear=False)
    areas3 = np.trapezoid(pi, fi, axis=0)

    assert np.allclose(areas, areas2)
    assert np.allclose(areas, areas3)

    s1 = spec[:, :2]
    assert np.all(psd.area(s1) == psd.area((*s1.T,)))


def test_interp():
    spec = np.array([[20, 0.5], [50, 1.0]])
    freq = [15, 35, 60]
    psdlinear = psd.interp(spec, freq, linear=True).ravel()
    psdlog = psd.interp(spec, freq, linear=False).ravel()
    assert np.allclose(psdlinear, [0, 0.75, 0])
    assert np.allclose(psdlog, [0, 0.76352135927358911, 0])


def test_interp_nans():
    Freq = np.arange(1, 10, 2)
    freq = np.arange(1, 10)
    PSD = (Freq**2) / 100

    nan = [np.nan, np.nan]
    Freq = np.hstack((Freq, nan))
    PSD = np.hstack((PSD, nan))

    plog = (freq**2) / 100
    plog2 = psd.interp((Freq, PSD), freq, linear=False)

    assert np.allclose(plog, plog2)


def test_rescale():
    g = np.random.randn(10000)
    sr = 400
    f, p = signal.welch(g, sr, nperseg=sr)
    p3, f3, msv3, ms3 = psd.rescale(p, f)
    p6, f6, msv6, ms6 = psd.rescale(p, f, n_oct=6)
    p6_2, f6_2, msv6_2, ms6_2 = psd.rescale(p, f, freq=f6)
    p12, f12, msv12, ms12 = psd.rescale(p6, f6, n_oct=12)
    i = int(1.0 / 2 ** (1 / 6))
    msv1 = np.sum(p[i:] * (f[1] - f[0]))
    assert abs(msv1 / msv3 - 1) < 0.12
    assert abs(msv1 / msv6 - 1) < 0.06
    assert abs(msv1 / msv12 - 1) < 0.03
    assert np.allclose(p6, p6_2)
    assert np.allclose(f6, f6_2)
    assert np.allclose(msv6, msv6_2)
    assert np.allclose(ms6, ms6_2)
    p_2, f_2, msv_2, ms_2 = psd.rescale(p, f, freq=f)
    assert np.allclose(p, p_2)
    assert np.allclose(f, f_2)
    msv1 = np.sum(p * (f[1] - f[0]))
    assert np.allclose(msv1, msv_2)

    P = np.vstack((p, p)).T
    P3, F3, MSV3, MS3 = psd.rescale(P, f)
    assert np.allclose(p3, P3[:, 0])
    assert np.allclose(f3, F3)
    assert np.allclose(msv3, MSV3[0])
    assert np.allclose(ms3, MS3[:, 0])
    assert np.allclose(p3, P3[:, 1])
    assert np.allclose(msv3, MSV3[1])
    assert np.allclose(ms3, MS3[:, 1])

    in_freq = np.arange(0, 10.1, 0.25)
    out_freq = np.arange(0, 10.1, 5)
    in_p = np.ones_like(in_freq)
    p, f, ms, mvs = psd.rescale(in_p, in_freq, freq=out_freq)
    assert np.allclose(p, [1.0, 1.0, 1.0])
    p, f, ms, mvs = psd.rescale(in_p, in_freq, freq=out_freq, extendends=False)
    assert np.allclose(p, [0.525, 1.0, 0.525])

    p, f, ms, mvs = psd.rescale(
        np.ones(1000), np.arange(1000), freq=np.arange(100, 800, 100), frange=(200, 500)
    )
    assert np.allclose(f, np.arange(200, 600, 100))


def test_spl():
    x = np.random.randn(100000)
    sr = 4000
    f, spl, oaspl = psd.spl(x, sr, sr, timeslice=len(x) / sr)
    # oaspl should be around 170.75 (since variance = 1):
    shouldbe = 10 * np.log10(1 / (2.9e-9) ** 2)
    abs(oaspl / shouldbe - 1) < 0.01
    oaspl1 = oaspl

    f, spl, oaspl = psd.spl(x, sr, sr, fs=0, timeslice=len(x) / sr)
    # oaspl should be around 170.75 (since variance = 1):
    abs(oaspl / shouldbe - 1) < 0.01

    f, spl, oaspl = psd.spl(x, sr)
    # oaspl should be greater than the one above:
    assert oaspl > oaspl1 * 1.01

    with pytest.raises(ValueError):
        psd.spl(x, sr, sr, timeslice=0.5)


def test_psd2time():
    spec = np.array([[20, 0.0768], [50, 0.48], [100, 0.48]])
    sig, sr = psd.psd2time(
        spec, ppc=10, fstart=35, fstop=70, df=0.01, winends=dict(portion=0.01)
    )

    # ensure that expand_method works:
    sig2, sr2 = psd.psd2time(
        spec,
        ppc=10,
        fstart=35,
        fstop=70,
        df=0.01,
        winends=dict(portion=0.01),
        expand_method="rescale",
    )

    f1, p1 = psd.psdmod(sig, sr, timeslice=f"{len(sig)}")
    f2, p2 = psd.psdmod(sig2, sr2, timeslice=f"{len(sig)}")

    assert np.trapezoid(p1, f1) < np.trapezoid(p2, f2)
    with pytest.raises(ValueError):
        psd.psd2time(
            spec,
            ppc=10,
            fstart=35,
            fstop=70,
            df=0.01,
            winends=dict(portion=0.01),
            expand_method="bad expand method",
        )

    multi_spec = np.array([[20, 0.0768, 0.01], [50, 0.48, 0.01], [100, 0.48, 0.01]])
    with pytest.raises(ValueError):
        psd.psd2time(multi_spec, ppc=10, fstart=35, fstop=70, df=0.01)

    assert np.allclose(700.0, sr)  # 70*10
    assert sig.size == 700 * 100
    f, p = signal.welch(sig, sr, nperseg=sr)
    pv = np.logical_and(f >= 37, f <= 68)
    fi = f[pv]
    psdi = p[pv]
    speci = psd.interp(spec, fi).flatten()
    assert abs(speci - psdi).max() < 0.05
    assert abs(np.trapezoid(psdi, fi) - np.trapezoid(speci, fi)) < 0.25

    spec = ([0.1, 5], [0.1, 0.1])
    sig, sr = psd.psd2time(
        spec, ppc=10, fstart=0.1, fstop=5, df=0.2, winends=dict(portion=0.01)
    )
    # df gets internally reset to fstart
    assert np.allclose(5 * 10.0, sr)
    assert sig.size == 50 * 10
    f, p = signal.welch(sig, sr, nperseg=sr)
    pv = np.logical_and(f >= 0.5, f <= 3.0)
    fi = f[pv]
    psdi = p[pv]
    speci = psd.interp(spec, fi).flatten()
    assert abs(speci - psdi).max() < 0.05
    assert abs(np.trapezoid(psdi, fi) - np.trapezoid(speci, fi)) < 0.065

    # ppc gets reset to 2 case:
    spec = np.array([[0.1, 2.0], [5, 2.0]])
    sig, sr = psd.psd2time(
        spec, ppc=1, fstart=0.1, fstop=5, df=0.2, winends=dict(portion=0.01)
    )
    assert (np.sum(np.mean(sig**2)) - 2.0 * 4.9) / (2.0 * 4.9) < 0.1

    # odd length FFT case:
    spec = np.array([[0.1, 2.0], [5, 2.0]])
    sig, sr, t = psd.psd2time(
        spec, ppc=3, fstart=0.2, fstop=5, df=0.2, winends=dict(portion=0.01), gettime=1
    )
    assert np.allclose(5 * 3, sr)
    assert sig.size == 15 * 5
    assert (np.sum(np.mean(sig**2)) - 2.0 * 4.8) / (2.0 * 4.8) < 0.1
    assert np.allclose(t, np.arange(15 * 5) / sr)


def test_psd2time2():
    spec = np.array([[20, 0.0768], [50, 0.48], [100, 0.48]])
    sig1, sr1 = psd.psd2time(
        spec,
        ppc=10,
        fstart=35,
        fstop=70,
        df=0.01,
        winends=dict(portion=0.01),
        rng=Generator(MT19937(1)),
    )

    sig2, sr2 = psd.psd2time(
        spec,
        ppc=10,
        fstart=35,
        fstop=70,
        df=0.01,
        winends=dict(portion=0.01),
        rng=Generator(MT19937(1)),
    )

    assert np.allclose(700.0, sr1)  # 70*10
    assert sig1.size == 700 * 100
    assert (sig1 == sig2).all()
    assert sr2 == sr1


def test_psdmod():
    TF = 30  # make a 30 second signal
    spec = [[20, 50], [1, 1]]
    sig, sr, t = psd.psd2time(
        spec,
        ppc=10,
        fstart=20,
        fstop=50,
        df=1 / TF,
        winends=dict(portion=10),
        gettime=True,
    )
    # sr = 500
    f, p = signal.welch(sig, sr, nperseg=sr)  # 1 second windows, df=1
    f2, p2 = psd.psdmod(sig, sr, nperseg=sr, timeslice=4, tsoverlap=0.5)
    f2b, p2b = psd.psdmod(sig, sr, nperseg=sr, timeslice="2000", tsoverlap=0.5)
    assert np.allclose(f2b, f2)
    assert np.allclose(p2b, p2)

    pv = np.logical_and(f2 > 24, f2 < 47)
    assert np.all(p2[pv] > p[pv])

    # mimic standard welch:
    f3, p3 = psd.psdmod(sig, sr, nperseg=sr, timeslice=30, tsoverlap=0.5)
    assert np.allclose(p3, p)

    # mimic maximax:
    f4, p4 = psd.psdmod(sig, sr, nperseg=sr)
    assert np.all(p4[pv] > p2[pv])

    # test the map output:
    f5, p5, pmap, t = psd.psdmod(sig, sr, getmap=1)
    assert np.allclose(p5, np.max(pmap, axis=1))
    tshouldbe = np.arange(0.5, 30.0 - 0.25, 0.5)
    assert np.allclose(t, tshouldbe)
