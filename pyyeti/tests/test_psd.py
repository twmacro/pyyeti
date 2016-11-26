import numpy as np
from pyyeti import psd
from nose.tools import *
import scipy.signal as signal


def test_interp():
    spec = np.array([[20, 0.5],
                     [50, 1.0]])
    freq = [15, 35, 60]
    psdlinear = psd.interp(spec, freq, linear=True).ravel()
    psdlog = psd.interp(spec, freq, linear=False).ravel()
    assert np.allclose(psdlinear, [0, 0.75, 0])
    assert np.allclose(psdlog, [0, 0.76352135927358911, 0])


def test_rescale():
    g = np.random.randn(10000)
    sr = 400
    f, p = signal.welch(g, sr, nperseg=sr)
    p3, f3, msv3, ms3 = psd.rescale(p, f)
    p6, f6, msv6, ms6 = psd.rescale(p, f, n_oct=6)
    p6_2, f6_2, msv6_2, ms6_2 = psd.rescale(p, f, freq=f6)
    p12, f12, msv12, ms12 = psd.rescale(p6, f6, n_oct=12)
    msv1 = np.sum(p*(f[1]-f[0]))
    assert abs(msv1/msv3 - 1) < .12
    assert abs(msv1/msv6 - 1) < .06
    assert abs(msv1/msv12 - 1) < .03
    assert np.allclose(p6, p6_2)
    assert np.allclose(f6, f6_2)
    assert np.allclose(msv6, msv6_2)
    assert np.allclose(ms6, ms6_2)
    p_2, f_2, msv_2, ms_2 = psd.rescale(p, f, freq=f)
    assert np.allclose(p, p_2)
    assert np.allclose(f, f_2)
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

    in_freq = np.arange(0, 10.1, .25)
    out_freq = np.arange(0, 10.1, 5)
    in_p = np.ones_like(in_freq)
    p, f, ms, mvs = psd.rescale(in_p, in_freq, freq=out_freq)
    assert np.allclose(p, [ 1.,  1.,  1.])
    p, f, ms, mvs = psd.rescale(in_p, in_freq, freq=out_freq,
                                extendends=False)
    assert np.allclose(p, [ 0.525,  1.   ,  0.525])


def test_spl():
    x = np.random.randn(100000)
    sr = 4000
    f, spl, oaspl = psd.spl(x, sr, sr, timeslice=len(x)/sr)
    # oaspl should be around 170.75 (since variance = 1):
    shouldbe = 10*np.log10(1/(2.9e-9)**2)
    abs(oaspl/shouldbe - 1) < .01
    oaspl1 = oaspl

    f, spl, oaspl = psd.spl(x, sr, sr, fs=0, timeslice=len(x)/sr)
    # oaspl should be around 170.75 (since variance = 1):
    abs(oaspl/shouldbe - 1) < .01

    f, spl, oaspl = psd.spl(x, sr)
    # oaspl should be greater than the one above:
    assert oaspl > oaspl1*1.01


def test_psd2time():
    spec = np.array([[20,  .0768],
                     [50,  .48],
                     [100, .48]])
    sig, sr = psd.psd2time(spec, ppc=10, fstart=35, fstop=70,
                              df=.01, winends=dict(portion=.01))
    assert np.allclose(700., sr)  # 70*10
    assert sig.size == 700*100
    f, p = signal.welch(sig, sr, nperseg=sr)
    pv = np.logical_and(f >= 37, f <= 68)
    fi = f[pv]
    psdi = p[pv]
    speci = psd.interp(spec, fi).flatten()
    assert abs(speci - psdi).max() < .05
    assert abs(np.trapz(psdi, fi) - np.trapz(speci, fi)) < .25

    spec = ([.1,  5],
            [.1,  .1])
    sig, sr = psd.psd2time(spec, ppc=10, fstart=.1, fstop=5,
                              df=.2, winends=dict(portion=.01))
    # df gets internally reset to fstart
    assert np.allclose(5*10., sr)
    assert sig.size == 50*10
    f, p = signal.welch(sig, sr, nperseg=sr)
    pv = np.logical_and(f >= .5, f <= 3.)
    fi = f[pv]
    psdi = p[pv]
    speci = psd.interp(spec, fi).flatten()
    assert abs(speci - psdi).max() < .05
    assert abs(np.trapz(psdi, fi) - np.trapz(speci, fi)) < .065

    # ppc gets reset to 2 case:
    spec = np.array([[.1,  2.],
                     [5,  2.]])
    sig, sr = psd.psd2time(spec, ppc=1, fstart=.1, fstop=5,
                              df=.2, winends=dict(portion=.01))
    assert (np.sum(np.mean(sig**2)) - 2.*4.9)/(2.*4.9) < .1

    # odd length FFT case:
    spec = np.array([[.1,  2.],
                     [5,  2.]])
    sig, sr, t = psd.psd2time(spec, ppc=3, fstart=.2, fstop=5,
                                 df=.2, winends=dict(portion=.01),
                                 gettime=1)
    assert np.allclose(5*3, sr)
    assert sig.size == 15*5
    assert (np.sum(np.mean(sig**2)) - 2.*4.8)/(2.*4.8) < .1
    assert np.allclose(t, np.arange(15*5)/sr)


def test_psdmod():
    TF = 30  # make a 30 second signal
    spec = [[20, 50], [1, 1]]
    sig, sr, t = psd.psd2time(spec, ppc=10, fstart=20, fstop=50,
                                 df=1/TF, winends=dict(portion=10),
                                 gettime=True)
    # sr = 500
    freq = np.arange(20., 50.1)
    f, p = signal.welch(sig, sr, nperseg=sr)  # 1 second windows, df=1
    f2, p2 = psd.psdmod(sig, sr, nperseg=sr, timeslice=4,
                        tsoverlap=0.5)
    pv = np.logical_and(f2 > 24, f2 < 47)
    assert np.all(p2[pv] > p[pv])

    # mimic standard welch:
    f3, p3 = psd.psdmod(sig, sr, nperseg=sr, timeslice=30,
                        tsoverlap=0.5)
    assert np.allclose(p3, p)

    # mimic maximax:
    f4, p4 = psd.psdmod(sig, sr, nperseg=sr)
    assert np.all(p4[pv] > p2[pv])

    # test the map output:
    f5, p5, pmap, t = psd.psdmod(sig, sr, getmap=1)
    assert np.allclose(p5, np.max(pmap, axis=1))
    tshouldbe = np.arange(.5, 30.-.25, .5)
    assert np.allclose(t, tshouldbe)
