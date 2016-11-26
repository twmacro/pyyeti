import numpy as np
from nose.tools import *
from pyyeti import psd
from pyyeti.fdepsd import fdepsd
import scipy.signal as signal


def compare(fde1, fde2):
    assert np.allclose(fde1.freq, fde2.freq)
    assert np.allclose(fde1.psd, fde2.psd)
    assert np.allclose(fde1.amp, fde2.amp)
    assert np.allclose(fde1.binamps, fde2.binamps)
    assert np.allclose(fde1.count, fde2.count)
    assert np.allclose(fde1.srs, fde2.srs)
    assert np.allclose(fde1.var, fde2.var)
    assert fde1.parallel in ('no', 'yes')
    assert fde2.parallel in ('no', 'yes')
    assert fde1.resp in ('absacce', 'pvelo')
    assert fde2.resp in ('absacce', 'pvelo')
    assert fde1.ncpu >= 1
    assert fde2.ncpu >= 1


def test_fdepsd_absacce():
    TF = 60  # make a 60 second signal
    sp = 1.
    spec = np.array([[20, sp], [50, sp]])
    sig, sr, t = psd.psd2time(spec, ppc=10, fstart=20, fstop=50,
                              df=1/TF, winends=dict(portion=10),
                              gettime=True)
    freq = np.arange(30., 50.1)
    q = 25
    fde_auto = fdepsd(sig, sr, freq, q)
    fde_no = fdepsd(sig, sr, freq, q, parallel='no')
    fde_yes = fdepsd(sig, sr, freq, q, parallel='yes')

    compare(fde_auto, fde_no)
    compare(fde_auto, fde_yes)
    pv = np.logical_and(freq > 32, freq < 45)
    assert abs(np.mean(fde_auto.psd[pv, :2], axis=0) - sp).max() < .22
    assert abs(np.mean(fde_auto.psd[pv, 2:], axis=0) - sp).max() < .12
    assert np.all(fde_auto.freq == freq)

    fde_none = fdepsd(sig, sr, freq, q, rolloff=None)
    fde_pre = fdepsd(sig, sr, freq, q, rolloff='prefilter')
    assert np.all(fde_none.psd <= fde_pre.psd)

    T0 = 120
    fde_T0 = fdepsd(sig, sr, freq, q, T0=T0)
    factor = (np.log(freq*60) / np.log(freq*T0))[:, None]
    assert np.allclose(fde_T0.psd[:, :2], fde_auto.psd[:, :2]*factor)
    assert np.all(fde_T0.psd[:, 2:] < fde_auto.psd[:, 2:])


def test_fdepsd_pvelo():
    TF = 60  # make a 60 second signal
    sp = 1.
    spec = np.array([[20, sp], [50, sp]])
    sig, sr, t = psd.psd2time(spec, ppc=10, fstart=20, fstop=50,
                              df=1/TF, winends=dict(portion=10),
                              gettime=True)
    freq = np.arange(30., 50.1)
    q = 25
    fde_auto = fdepsd(sig, sr, freq, q, resp='pvelo')
    fde_no = fdepsd(sig, sr, freq, q, resp='pvelo', parallel='no',
                    verbose=True)
    b, a = signal.butter(3, 5/(sr/2), 'high')
    sig5 = signal.lfilter(b, a, sig)
    fde_yes = fdepsd(sig5, sr, freq, q, resp='pvelo', parallel='yes',
                     hpfilter=None)

    compare(fde_auto, fde_no)
    compare(fde_auto, fde_yes)
    pv = np.logical_and(freq > 32, freq < 45)
    assert abs(np.mean(fde_auto.psd[pv, :2], axis=0) - sp).max() < .22
    assert abs(np.mean(fde_auto.psd[pv, 2:], axis=0) - sp).max() < .22
    assert np.all(fde_auto.freq == freq)

    fde_none = fdepsd(sig, sr, freq, q, resp='pvelo', rolloff=None)
    fde_pre = fdepsd(sig, sr, freq, q, resp='pvelo',
                     rolloff='prefilter')
    assert np.all(fde_none.psd <= fde_pre.psd)

    T0 = 120
    fde_T0 = fdepsd(sig, sr, freq, q, T0=T0, resp='pvelo')
    factor = (np.log(freq*60) / np.log(freq*T0))[:, None]
    assert np.allclose(fde_T0.psd[:, :2], fde_auto.psd[:, :2]*factor)
    assert np.all(fde_T0.psd[:, 2:] < fde_auto.psd[:, 2:])


def test_fdepsd_error():
    sig = [[1, 2, 3], [4, 5, 6]]
    sr = 100
    freq = [1, 2, 3]
    q = 20
    assert_raises(ValueError, fdepsd, sig, sr, freq, q)
    assert_raises(ValueError, fdepsd, freq, sr, freq, q, 'badresp')
