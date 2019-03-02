import numpy as np
from pyyeti import cyclecount
from pyyeti.rainflow.py_rain import rainflow
from nose.tools import *


def test_rainflow_1():
    a = np.arange(25000)
    a[::2] *= -1
    rf, os = cyclecount.rain.rainflow(a, getoffsets=True)
    rf2, os2 = rainflow(a, getoffsets=True)
    assert np.allclose(rf, rf2)
    assert np.all(os2 == os)

    # should all be half cycles, amp = .5, 1.5, ...,
    # means = [.5, -.5, .5, -.5 ...]
    amp = np.arange(len(a) - 1, dtype=float) + 0.5
    means = 0.5 * np.ones(len(a) - 1)
    means[1::2] *= -1.0
    assert np.allclose(amp, rf[:, 0])
    assert np.allclose(means, rf[:, 1])
    assert np.allclose(0.5, rf[:, 2])

    # offset starts = 0, 1, 2, ... len(a)-1
    # offset ends   = 1, 2, 3, ... len(a)
    assert np.all(np.arange(len(a) - 1) == os[:, 0])
    assert np.all(np.arange(1, len(a)) == os[:, 1])


def test_rainflow_2():
    a = np.arange(25000)
    a[::2] *= -1
    a = a[::-1]
    rf, os = cyclecount.rain.rainflow(a, getoffsets=True)
    rf2, os2 = rainflow(a, getoffsets=True)
    assert np.allclose(rf, rf2)
    assert np.all(os2 == os)

    # reverse of:
    #   should all be half cycles, amp = .5, 1.5, ...,
    #   means = [.5, -.5, .5, -.5 ...]
    amp = np.arange(len(a) - 1, dtype=float) + 0.5
    means = 0.5 * np.ones(len(a) - 1)
    means[1::2] *= -1.0
    assert np.allclose(amp[::-1], rf[:, 0])
    assert np.allclose(means[::-1], rf[:, 1])
    assert np.allclose(0.5, rf[:, 2])

    # offset starts = 0, 1, 2, ... len(a)-1
    # offset ends   = 1, 2, 3, ... len(a)
    assert np.all(np.arange(len(a) - 1) == os[:, 0])
    assert np.all(np.arange(1, len(a)) == os[:, 1])


def test_rainflow_3():
    a = np.ones(10)
    a[::2] *= -1
    rf, os = cyclecount.rain.rainflow(a, getoffsets=True)
    rf2, os2 = rainflow(a, getoffsets=True)
    assert np.allclose(rf, rf2)
    assert np.all(os2 == os)

    # should all be half cycles, amp = 1, means = 0
    assert np.allclose(1.0, rf[:, 0])
    assert np.allclose(0.0, rf[:, 1])
    assert np.allclose(0.5, rf[:, 2])

    # offset starts = 0, 1, 2, ... len(a)-1
    # offset ends   = 1, 2, 3, ... len(a)
    assert np.all(np.arange(len(a) - 1) == os[:, 0])
    assert np.all(np.arange(1, len(a)) == os[:, 1])


def test_rainflow_4():
    a = np.ones(10)
    a[::2] *= -1
    a[0] *= 2.0  # make rainflow not accept first point until the end
    rf, os = cyclecount.rain.rainflow(a, getoffsets=True)
    rf2, os2 = rainflow(a, getoffsets=True)
    assert np.allclose(rf, rf2)
    assert np.all(os2 == os)

    # should almost all full cycles, amp = 1, means = 0
    rf_shouldbe = [
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.5, -0.5, 0.5],
    ]  # range: -2 to +1 = 3
    os_shouldbe = [[1, 2], [3, 4], [5, 6], [7, 8], [0, 9]]

    assert np.allclose(rf_shouldbe, rf)
    assert np.allclose(os_shouldbe, os)


def test_rainflow_badinput():
    assert_raises(ValueError, cyclecount.rain.rainflow, 1.0)
    assert_raises(ValueError, rainflow, 1.0)
    assert_raises(ValueError, cyclecount.rain.rainflow, np.random.randn(2, 2))
    assert_raises(ValueError, rainflow, np.random.randn(2, 2))


def test_getbins_1():
    bb = cyclecount.getbins(4, 12, 4)
    assert np.allclose(bb, [3.992, 6.0, 8.0, 10.0, 12.0])
    bb = cyclecount.getbins(4, 4, 12)
    assert np.allclose(bb, [3.992, 6.0, 8.0, 10.0, 12.0])
    bb = cyclecount.getbins(4, 4, 12, right=False)
    assert np.allclose(bb, [4.0, 6.0, 8.0, 10.0, 12.008])
    assert_raises(ValueError, cyclecount.getbins, 4, 3.0, 3.0)


def test_getbins_2():
    mx = 100
    mn = 0
    bb = cyclecount.getbins(1, mx, mn)
    assert np.allclose(bb, [mn - 0.001 * (mx - mn), mx])
    bb = cyclecount.getbins(1, mx, mn, right=False)
    assert np.allclose(bb, [mn, mx + 0.001 * (mx - mn)])


def test_getbins_3():
    mx = 20.0
    mn = -5.0
    binpts = [-2.0, 4.0, 5.0, 8.0, 14.0]
    bb = cyclecount.getbins(binpts, mn, mx)
    assert np.allclose(bb, binpts)
    binpts = [0.5, 0.1, -10.0, 100]
    assert_raises(ValueError, cyclecount.getbins, binpts, mn, mx)


def test_sigcount_1():
    sig = np.arange(100)
    sig[::2] *= -1  # [0, 1, -2, 3, -4, ..., 99]
    # `sig` has 99 half-cycles; amplitude grows from 0.5 up to
    #  98.5; mean of each is either 0.5, -0.5
    table = cyclecount.sigcount(sig, 2, 2)
    #                  (0.402, 49.500]  (49.500, 98.500]
    # (-0.501, 0.000]             12.5              12.0
    # (0.000, 0.500]              12.5              12.5
    table2, ampb, aveb = cyclecount.sigcount(sig, 2, 2, retbins=True)
    assert np.all((table2 == table).values)
    assert np.all(table.index == ["(-0.501, 0.000]", "(0.000, 0.500]"])
    assert np.all(table.columns == ["(0.402, 49.500]", "(49.500, 98.500]"])
    assert np.allclose(table.values, [[12.5, 12.0], [12.5, 12.5]])
    assert np.allclose(ampb, [0.402, 49.500, 98.500])
    assert np.allclose(aveb, [-0.501, 0.000, 0.500])


def test_sigcount_2():
    sig = np.arange(100)
    sig[::2] *= -1  # [0, 1, -2, 3, -4, ..., 99]
    # `sig` has 99 half-cycles; amplitude grows from 0.5 up to
    #  98.5; mean of each is either 0.5, -0.5
    table = cyclecount.sigcount(sig, 2, 2, right=0)
    #                  [0.500, 49.500)  [49.500, 98.598)
    # [-0.500, 0.000)             12.0              12.5
    # [0.000, 0.501)              12.5              12.5
    table2, ampb, aveb = cyclecount.sigcount(sig, 2, 2, right=0, retbins=True)
    assert np.all((table2 == table).values)
    assert np.all(table.index == ["[-0.500, 0.000)", "[0.000, 0.501)"])
    assert np.all(table.columns == ["[0.500, 49.500)", "[49.500, 98.598)"])
    assert np.allclose(table.values, [[12.0, 12.5], [12.5, 12.5]])
    assert np.allclose(ampb, [0.500, 49.500, 98.598])
    assert np.allclose(aveb, [-0.500, 0.000, 0.501])
