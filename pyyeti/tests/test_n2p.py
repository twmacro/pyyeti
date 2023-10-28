import inspect
import numpy as np
import math
import scipy.linalg as la
from scipy.io import matlab
import io
import os
from pyyeti import nastran, cb
from pyyeti.nastran import n2p, op2, op4
import pytest


def conv_uset(uset):
    return n2p.make_uset(uset[:, :2].astype(int), uset[:, 2].astype(int), uset[:, 3:])


def test_rbgeom():
    x, y, z = 30.0, 10.0, 20.0
    grids = np.array([[0.0, 0.0, 0.0], [x, y, z]])
    rb = n2p.rbgeom(grids)
    rb_should_be = np.vstack((np.eye(6), np.eye(6)))
    rb_should_be[6, 4] = z
    rb_should_be[6, 5] = -y
    rb_should_be[7, 3] = -z
    rb_should_be[7, 5] = x
    rb_should_be[8, 3] = y
    rb_should_be[8, 4] = -x
    err = (np.abs(rb - rb_should_be)).max()
    assert err < 1.0e-14


def test_rbgeom_uset():
    #     m  -------------------------------------\
    #     s  ------------------------------\       > g --\
    #     o  -----------------------\       > n --/       \
    #     q  ----------------\       > f --/       \       \
    #     r  ---------\       > a --/       \       \       > p
    #     c  --\       > t --/       \       > fe    > ne  /
    #     b  ---> l --/               > d   /       /     /
    #     e  ------------------------/-----/-------/-----/
    uset = np.array(
        [
            [100, 1, 2097154, 5, 10, 15],  # b-set
            [100, 2, 2097154, 0, 1, 0],
            [100, 3, 2097154, 0, 0, 0],
            [100, 4, 2097154, 1, 0, 0],
            [100, 5, 2097154, 0, 1, 0],
            [100, 6, 2097154, 0, 0, 1],
            [200, 1, 4194304, 0, 0, 0],  # q-set
            [200, 2, 4194304, 0, 1, 0],
            [200, 3, 4194304, 0, 0, 0],
            [200, 4, 4194304, 1, 0, 0],
            [200, 5, 4194304, 0, 1, 0],
            [200, 6, 4194304, 0, 0, 1],
            [300, 1, 4, 10, 20, 30],  # o-set
            [300, 2, 4, 0, 1, 0],
            [300, 3, 4, 0, 0, 0],
            [300, 4, 4, 1, 0, 0],
            [300, 5, 4, 0, 1, 0],
            [300, 6, 4, 0, 0, 1],
            [400, 1, 1, 20, 30, 40],  # m-set
            [400, 2, 1, 0, 1, 0],
            [400, 3, 1, 0, 0, 0],
            [400, 4, 1, 1, 0, 0],
            [400, 5, 1, 0, 1, 0],
            [400, 6, 1, 0, 0, 1],
        ]
    )
    uset = conv_uset(uset)
    pv = n2p.mksetpv(uset, "a", "b")
    assert np.sum(pv) == 6
    pv = n2p.mksetpv(uset, "p", "b+q")
    assert np.sum(pv) == 12
    pv = n2p.mksetpv(uset, "p", "m+a")
    assert np.sum(pv) == 18
    pv = n2p.mksetpv(uset, "g", "q")

    assert np.all(
        uset.iloc[pv].index.get_level_values(0) == [200, 200, 200, 200, 200, 200]
    )
    rb = n2p.rbgeom_uset(uset, 300)
    rb_should_be = np.array(
        [
            [1, 0, 0, 0, -15, 10],
            [0, 1, 0, 15, 0, -5],
            [0, 0, 1, -10, 5, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 10, -10],
            [0, 1, 0, -10, 0, 10],
            [0, 0, 1, 10, -10, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )
    err = (np.abs(rb - rb_should_be)).max()
    assert err < 1.0e-14

    with io.StringIO() as f:
        tab = n2p.usetprt(f, uset, printsets="m, o, q, b, a, n, g")
        prt = f.getvalue()
    s = str(tab).split("\n")
    sbe = [
        "              m  o  q  b   a   n   g",
        "id  dof dof#                        ",
        "100 1   1     0  0  0  1   1   1   1",
        "    2   2     0  0  0  2   2   2   2",
        "    3   3     0  0  0  3   3   3   3",
        "    4   4     0  0  0  4   4   4   4",
        "    5   5     0  0  0  5   5   5   5",
        "    6   6     0  0  0  6   6   6   6",
        "200 1   7     0  0  1  0   7   7   7",
        "    2   8     0  0  2  0   8   8   8",
        "    3   9     0  0  3  0   9   9   9",
        "    4   10    0  0  4  0  10  10  10",
        "    5   11    0  0  5  0  11  11  11",
        "    6   12    0  0  6  0  12  12  12",
        "300 1   13    0  1  0  0   0  13  13",
        "    2   14    0  2  0  0   0  14  14",
        "    3   15    0  3  0  0   0  15  15",
        "    4   16    0  4  0  0   0  16  16",
        "    5   17    0  5  0  0   0  17  17",
        "    6   18    0  6  0  0   0  18  18",
        "400 1   19    1  0  0  0   0   0  19",
        "    2   20    2  0  0  0   0   0  20",
        "    3   21    3  0  0  0   0   0  21",
        "    4   22    4  0  0  0   0   0  22",
        "    5   23    5  0  0  0   0   0  23",
        "    6   24    6  0  0  0   0   0  24",
    ]
    assert s == sbe
    sbe = (
        "m-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      400-1      400-2      400-3      400-4      400-5      400-6\n"
        "\n"
        "o-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      300-1      300-2      300-3      300-4      300-5      300-6\n"
        "\n"
        "q-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      200-1      200-2      200-3      200-4      200-5      200-6\n"
        "\n"
        "b-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      100-1      100-2      100-3      100-4      100-5      100-6\n"
        "\n"
        "a-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      100-1      100-2      100-3      100-4      100-5      100-6      200-1      200-2      200-3      200-4 =    10\n"
        "    11=      200-5      200-6\n"
        "\n"
        "n-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      100-1      100-2      100-3      100-4      100-5      100-6      200-1      200-2      200-3      200-4 =    10\n"
        "    11=      200-5      200-6      300-1      300-2      300-3      300-4      300-5      300-6\n"
        "\n"
        "g-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      100-1      100-2      100-3      100-4      100-5      100-6      200-1      200-2      200-3      200-4 =    10\n"
        "    11=      200-5      200-6      300-1      300-2      300-3      300-4      300-5      300-6      400-1      400-2 =    20\n"
        "    21=      400-3      400-4      400-5      400-6\n"
        "\n"
    )
    assert prt == sbe

    with io.StringIO() as f:
        tab = n2p.usetprt(f, uset)
        prt = f.getvalue()
    sbe = (
        "m-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      400-1      400-2      400-3      400-4      400-5      400-6\n"
        "\n"
        "s-set, r-set, c-set, e-set\n"
        "      -None-\n"
        "\n"
        "o-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      300-1      300-2      300-3      300-4      300-5      300-6\n"
        "\n"
        "q-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      200-1      200-2      200-3      200-4      200-5      200-6\n"
        "\n"
        "b-set, l-set, t-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      100-1      100-2      100-3      100-4      100-5      100-6\n"
        "\n"
        "a-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      100-1      100-2      100-3      100-4      100-5      100-6      200-1      200-2      200-3      200-4 =    10\n"
        "    11=      200-5      200-6\n"
        "\n"
        "f-set, n-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      100-1      100-2      100-3      100-4      100-5      100-6      200-1      200-2      200-3      200-4 =    10\n"
        "    11=      200-5      200-6      300-1      300-2      300-3      300-4      300-5      300-6\n"
        "\n"
        "g-set\n"
        "             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n"
        "     1=      100-1      100-2      100-3      100-4      100-5      100-6      200-1      200-2      200-3      200-4 =    10\n"
        "    11=      200-5      200-6      300-1      300-2      300-3      300-4      300-5      300-6      400-1      400-2 =    20\n"
        "    21=      400-3      400-4      400-5      400-6\n"
        "\n"
    )

    assert prt == sbe
    assert n2p.usetprt(0, uset, printsets="r") is None


def test_rbgeom_uset_cylindrical():
    # some cylindrical coords:
    uset = np.array(
        [
            [100, 1, 2097154, 5, 10, 15],
            [100, 2, 2097154, 0, 1, 0],
            [100, 3, 2097154, 0, 0, 0],
            [100, 4, 2097154, 1, 0, 0],
            [100, 5, 2097154, 0, 1, 0],
            [100, 6, 2097154, 0, 0, 1],
            [200, 1, 4194304, 0, 0, 0],
            [200, 2, 4194304, 0, 1, 0],
            [200, 3, 4194304, 0, 0, 0],
            [200, 4, 4194304, 1, 0, 0],
            [200, 5, 4194304, 0, 1, 0],
            [200, 6, 4194304, 0, 0, 1],
            [300, 1, 4, 10, 20, 30],
            [300, 2, 4, 1, 2, 0],
            [300, 3, 4, 0, 0, 0],
            [300, 4, 4, 0, 0, 1],
            [300, 5, 4, 1, 0, 0],
            [300, 6, 4, 0, 1, 0],
            [400, 1, 1, 20, 30, 40],
            [400, 2, 1, 1, 2, 0],
            [400, 3, 1, 0, 0, 0],
            [400, 4, 1, 0, 0, 1],
            [400, 5, 1, 1, 0, 0],
            [400, 6, 1, 0, 1, 0],
        ]
    )
    uset = conv_uset(uset)
    rb = n2p.rbgeom_uset(uset)
    rb_should_be = np.array(
        [
            [1.0000, 0, 0, 0, 15.0000, -10.0000],
            [0, 1.0000, 0, -15.0000, 0, 5.0000],
            [0, 0, 1.0000, 10.0000, -5.0000, 0],
            [0, 0, 0, 1.0000, 0, 0],
            [0, 0, 0, 0, 1.0000, 0],
            [0, 0, 0, 0, 0, 1.0000],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0.5547, 0.8321, 0, -8.3205, 5.5470],
            [0, -0.8321, 0.5547, 36.0555, -5.5470, -8.3205],
            [1.0000, 0, 0, 0, 30.0000, -20.0000],
            [0, 0, 0, 0, 0.5547, 0.8321],
            [0, 0, 0, 0, -0.8321, 0.5547],
            [0, 0, 0, 1.0000, 0, 0],
            [0, 0.6000, 0.8000, -0.0000, -16.0000, 12.0000],
            [0, -0.8000, 0.6000, 50.0000, -12.0000, -16.0000],
            [1.0000, 0, 0, 0, 40.0000, -30.0000],
            [0, 0, 0, 0, 0.6000, 0.8000],
            [0, 0, 0, 0, -0.8000, 0.6000],
            [0, 0, 0, 1.0000, 0, 0],
        ]
    )

    err = (np.abs(rb - rb_should_be)).max()
    assert err < 1.0e-4

    rb = n2p.rbgeom_uset(uset, 300)
    rb_should_be = np.array(
        [
            [1.0000, 0, 0, 0, -15.0000, 10.0000],
            [0, 1.0000, 0, 15.0000, 0, -5.0000],
            [0, 0, 1.0000, -10.0000, 5.0000, 0],
            [0, 0, 0, 1.0000, 0, 0],
            [0, 0, 0, 0, 1.0000, 0],
            [0, 0, 0, 0, 0, 1.0000],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0.5547, 0.8321, 0, 0, 0],
            [0, -0.8321, 0.5547, 0, 0, 0],
            [1.0000, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.5547, 0.8321],
            [0, 0, 0, 0, -0.8321, 0.5547],
            [0, 0, 0, 1.0000, 0, 0],
            [0, 0.6000, 0.8000, 2.0000, -8.0000, 6.0000],
            [0, -0.8000, 0.6000, 14.0000, -6.0000, -8.0000],
            [1.0000, 0, 0, 0, 10.0000, -10.0000],
            [0, 0, 0, 0, 0.6000, 0.8000],
            [0, 0, 0, 0, -0.8000, 0.6000],
            [0, 0, 0, 1.0000, 0, 0],
        ]
    )

    err = (np.abs(rb - rb_should_be)).max()
    assert err < 1.0e-4


def test_rbgeom_uset_spherical():
    # add spherical:
    uset = np.array(
        [
            [100, 1, 2097154, 5.0000, 10.0000, 15.0000],
            [100, 2, 2097154, 1.0000, 2.0000, 0],
            [100, 3, 2097154, 0, 0, 0],
            [100, 4, 2097154, 0, 0, 1.0000],
            [100, 5, 2097154, 1.0000, 0, 0],
            [100, 6, 2097154, 0, 1.0000, 0],
            [200, 1, 4194304, 0, 0, 0],
            [200, 2, 4194304, 0, 1.0000, 0],
            [200, 3, 4194304, 0, 0, 0],
            [200, 4, 4194304, 1.0000, 0, 0],
            [200, 5, 4194304, 0, 1.0000, 0],
            [200, 6, 4194304, 0, 0, 1.0000],
            [300, 1, 4, 10.0000, 20.0000, 30.0000],
            [300, 2, 4, 2.0000, 3.0000, 0],
            [300, 3, 4, 2.0000, 1.9988, 0.0698],
            [300, 4, 4, 0.1005, 0.1394, 0.9851],
            [300, 5, 4, -0.7636, 0.6456, -0.0135],
            [300, 6, 4, -0.6379, -0.7509, 0.1714],
            [400, 1, 1, 20.0000, 30.0000, 40.0000],
            [400, 2, 1, 1.0000, 2.0000, 0],
            [400, 3, 1, 0, 0, 0],
            [400, 4, 1, 0, 0, 1.0000],
            [400, 5, 1, 1.0000, 0, 0],
            [400, 6, 1, 0, 1.0000, 0],
        ]
    )
    uset = conv_uset(uset)

    rb = n2p.rbgeom_uset(uset)
    rb_should_be = np.array(
        [
            [0, 0.5547, 0.8321, 0, -4.1603, 2.7735],
            [0, -0.8321, 0.5547, 18.0278, -2.7735, -4.1603],
            [1.0000, 0, 0, 0, 15.0000, -10.0000],
            [0, 0, 0, 0, 0.5547, 0.8321],
            [0, 0, 0, 0, -0.8321, 0.5547],
            [0, 0, 0, 1.0000, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0.2233, 0.5024, 0.8353, 1.6345, -1.6550, 0.5585],
            [-0.9692, 0.2060, 0.1351, -3.4774, -30.4266, 21.4436],
            [-0.1042, -0.8397, 0.5329, 35.8502, -8.4547, -6.3136],
            [0, 0, 0, 0.2233, 0.5024, 0.8353],
            [0, 0, 0, -0.9692, 0.2060, 0.1351],
            [0, 0, 0, -0.1042, -0.8397, 0.5329],
            [0, 0.6000, 0.8000, -0.0000, -16.0000, 12.0000],
            [0, -0.8000, 0.6000, 50.0000, -12.0000, -16.0000],
            [1.0000, 0, 0, 0, 40.0000, -30.0000],
            [0, 0, 0, 0, 0.6000, 0.8000],
            [0, 0, 0, 0, -0.8000, 0.6000],
            [0, 0, 0, 1.0000, 0, 0],
        ]
    )

    err = (np.abs(rb - rb_should_be)).max()
    assert err < 2.0e-3

    rb = n2p.rbgeom_uset(uset, 300)
    rb_should_be = np.array(
        [
            [0, 0.5547, 0.8321, 0, 4.1603, -2.7735],
            [0, -0.8321, 0.5547, -18.0278, 2.7735, 4.1603],
            [1.0000, 0, 0, 0, -15.0000, 10.0000],
            [0, 0, 0, 0, 0.5547, 0.8321],
            [0, 0, 0, 0, -0.8321, 0.5547],
            [0, 0, 0, 1.0000, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0.2233, 0.5024, 0.8353, 0, 0, 0],
            [-0.9692, 0.2060, 0.1351, 0, 0, 0],
            [-0.1042, -0.8397, 0.5329, 0, 0, 0],
            [0, 0, 0, 0.2233, 0.5024, 0.8353],
            [0, 0, 0, -0.9692, 0.2060, 0.1351],
            [0, 0, 0, -0.1042, -0.8397, 0.5329],
            [0, 0.6000, 0.8000, 2.0000, -8.0000, 6.0000],
            [0, -0.8000, 0.6000, 14.0000, -6.0000, -8.0000],
            [1.0000, 0, 0, 0, 10.0000, -10.0000],
            [0, 0, 0, 0, 0.6000, 0.8000],
            [0, 0, 0, 0, -0.8000, 0.6000],
            [0, 0, 0, 1.0000, 0, 0],
        ]
    )

    err = (np.abs(rb - rb_should_be)).max()
    assert err < 2.0e-3


def test_rbmove():
    grids = np.array([[0.0, 0.0, 0.0], [30.0, 10.0, 20.0]])
    rb0 = n2p.rbgeom(grids)
    rb1 = n2p.rbgeom(grids, [2.0, 4.0, -5.0])
    rb1_b = n2p.rbmove(rb0, [0.0, 0.0, 0.0], [2.0, 4.0, -5.0])
    assert np.all(rb1_b == rb1)


def test_replace_basic_cs():
    pth = os.path.dirname(inspect.getfile(cb))
    pth = os.path.join(pth, "tests", "nas2cam_csuper")

    # define new basic (so that old basic is at origin 100, 100, 100)
    """
           ^ Y_basic, X_new
           |
           |
    Y_new  |
    <------ -----> X_basic
          /
        /  Z_basic, Z_new


           ^ Z_new, X_old
           |
           |  / Y_new
           |/
           -----> X_new, Z_old
          /
        /  Y_old
    """

    new_cs_id = 50
    new_cs_in_basic = np.array([[100, 100, 100], [100, 100, 110], [100, 110, 100]])

    # Load the mass and stiffness from the .op4 file
    # This loads the data into a dict:
    mk = op4.load(os.path.join(pth, "inboard.op4"))
    # maa = mk["mxx"][0]
    kaa = mk["kxx"][0]

    # Get the USET table The USET table has the boundary DOF
    # information (id, location, coordinate system). This is needed
    # for superelements with an indeterminate interface. The nastran
    # module has the function bulk2uset which is handy for forming the
    # USET table from bulk data.

    uset, coords = nastran.bulk2uset(os.path.join(pth, "inboard.asm"))
    n = uset.shape[0]
    b = np.arange(n)
    kbb = kaa[np.ix_(b, b)]
    rb = n2p.rbgeom_uset(uset)

    frb = kbb @ rb
    frbres = rb.T @ frb

    assert abs(frb).max() < 0.01
    assert abs(frbres).max() < 0.1

    uset_new = n2p.replace_basic_cs(uset, new_cs_id, new_cs_in_basic)
    rb_new = n2p.rbgeom_uset(uset_new)
    frb_new = kbb @ rb_new
    frbres_new = rb_new.T @ frb_new

    assert abs(frb_new).max() < 0.01
    assert abs(frbres_new).max() < 0.1

    # check for ValueError on duplicate CS id:
    with pytest.raises(ValueError):
        n2p.replace_basic_cs(uset, 10, new_cs_in_basic)


def test_replace_basic_cs_2():
    cs10 = [[10, 1, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0]]

    #  first, make a uset table with node 1 at (0, 0, 0:
    uset0 = n2p.addgrid(None, 1, "b", 0, [0.0, 0.0, 0.0], 0)

    new_cs_id = 50
    new_cs_in_basic = np.array(
        [[10.0, 10.0, 10.0], [10.0, 10.0, 11.0], [11.0, 10.0, 10.0]]
    )
    uset1 = n2p.replace_basic_cs(uset0, new_cs_id, new_cs_in_basic)
    uset2 = n2p.replace_basic_cs(uset0, np.vstack(([new_cs_id, 1, 0], new_cs_in_basic)))

    assert np.all(uset1 == uset2)
    with pytest.raises(ValueError):
        n2p.replace_basic_cs(
            uset0,
            np.vstack(([new_cs_id, 0, 0], new_cs_in_basic)),
        )

    with pytest.raises(ValueError):
        n2p.replace_basic_cs(
            uset0,
            np.vstack(([new_cs_id, 1, 1], new_cs_in_basic)),
        )


def test_make_uset():
    # improper sized xyz:
    with pytest.raises(ValueError):
        n2p.make_uset([[1, 123456], [2, 0]], 1, [[1, 1, 1]])

    # wrong number of dof for grid 1:
    with pytest.raises(ValueError):
        n2p.make_uset([[1, 13456], [2, 0]], 1)

    # improper sized nasset:
    with pytest.raises(ValueError):
        n2p.make_uset(1, [1, 1])

    u = n2p.make_uset(
        dof=[[1, 123456], [2, 0]],
        nasset=[n2p.mkusetmask("b"), n2p.mkusetmask("q")],
        xyz=[[1, 2, 3], [0, 0, 0]],
    )
    mask = n2p.mkusetmask()
    b = mask["b"]
    q = mask["q"]
    s = mask["s"]

    sbe = np.array(
        [
            [1.0, 1.0, b, 1.0, 2.0, 3.0],
            [1.0, 2.0, b, 0.0, 1.0, 0.0],
            [1.0, 3.0, b, 0.0, 0.0, 0.0],
            [1.0, 4.0, b, 1.0, 0.0, 0.0],
            [1.0, 5.0, b, 0.0, 1.0, 0.0],
            [1.0, 6.0, b, 0.0, 0.0, 1.0],
            [2.0, 0.0, q, 0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(u.reset_index().values, sbe)

    u = n2p.make_uset(dof=sbe[:, :2], nasset=sbe[:, 2], xyz=sbe[:, 3:])
    assert np.allclose(u.reset_index().values, sbe)

    u = n2p.make_uset(
        dof=[[1, 123456], [2, 0]],
        nasset="b",
        xyz=[[1, 2, 3], [0, 0, 0]],
    )

    sbe = np.array(
        [
            [1.0, 1.0, b, 1.0, 2.0, 3.0],
            [1.0, 2.0, b, 0.0, 1.0, 0.0],
            [1.0, 3.0, b, 0.0, 0.0, 0.0],
            [1.0, 4.0, b, 1.0, 0.0, 0.0],
            [1.0, 5.0, b, 0.0, 1.0, 0.0],
            [1.0, 6.0, b, 0.0, 0.0, 1.0],
            [2.0, 0.0, b, 0.0, 0.0, 0.0],
        ]
    )
    assert np.allclose(u.reset_index().values, sbe)

    u = n2p.make_uset(dof=[[1, 123456], [2, 0]], nasset=n2p.mkusetmask("b"))

    sbe = np.array(
        [
            [1.0, 1.0, b, np.nan, np.nan, np.nan],
            [1.0, 2.0, b, np.nan, np.nan, np.nan],
            [1.0, 3.0, b, np.nan, np.nan, np.nan],
            [1.0, 4.0, b, np.nan, np.nan, np.nan],
            [1.0, 5.0, b, np.nan, np.nan, np.nan],
            [1.0, 6.0, b, np.nan, np.nan, np.nan],
            [2.0, 0.0, b, np.nan, np.nan, np.nan],
        ]
    )
    assert np.allclose(u.reset_index().values, sbe, equal_nan=True)

    dof = [
        [1, 123456],
        [2, 0],
        [3, 1],
        [3, 2],
        [3, 3],
        [3, 4],
        [3, 5],
        [3, 6],
        [4, 123456],
    ]
    xyz = [
        [1, 2, 3],
        [0, 0, 0],
        [4, 5, 6],
        [0, 1, 0],
        [2, 2, 2],
        [0, 1, 0],
        [0, 0, -1],
        [1, 0, 0],
        [10, 20, 30],
    ]

    u = n2p.make_uset(dof, n2p.mkusetmask("b"), xyz)
    sbe = np.array(
        [
            [1.0, 1.0, b, 1.0, 2.0, 3.0],
            [1.0, 2.0, b, 0.0, 1.0, 0.0],
            [1.0, 3.0, b, 0.0, 0.0, 0.0],
            [1.0, 4.0, b, 1.0, 0.0, 0.0],
            [1.0, 5.0, b, 0.0, 1.0, 0.0],
            [1.0, 6.0, b, 0.0, 0.0, 1.0],
            [2.0, 0.0, b, 0.0, 0.0, 0.0],
            [3.0, 1.0, b, 4.0, 5.0, 6.0],
            [3.0, 2.0, b, 0.0, 1.0, 0.0],
            [3.0, 3.0, b, 2.0, 2.0, 2.0],
            [3.0, 4.0, b, 0.0, 1.0, 0.0],
            [3.0, 5.0, b, 0.0, 0.0, -1.0],
            [3.0, 6.0, b, 1.0, 0.0, 0.0],
            [4.0, 1.0, b, 10.0, 20.0, 30.0],
            [4.0, 2.0, b, 0.0, 1.0, 0.0],
            [4.0, 3.0, b, 0.0, 0.0, 0.0],
            [4.0, 4.0, b, 1.0, 0.0, 0.0],
            [4.0, 5.0, b, 0.0, 1.0, 0.0],
            [4.0, 6.0, b, 0.0, 0.0, 1.0],
        ]
    )

    assert np.allclose(u.reset_index().values, sbe)

    dof = [
        [1, 123456],
        [2, 0],
        [3, 1],
        [3, 2],
        [3, 3],
        [3, 4],
        [3, 5],
        [3, 6],
        [4, 123456],
    ]
    nasset = [b, q, b, b, b, q, q, b, s]  # 1  # 2  # 3  # 4

    u = n2p.make_uset(dof, nasset, xyz)
    sbe = np.array(
        [
            [1.0, 1.0, b, 1.0, 2.0, 3.0],
            [1.0, 2.0, b, 0.0, 1.0, 0.0],
            [1.0, 3.0, b, 0.0, 0.0, 0.0],
            [1.0, 4.0, b, 1.0, 0.0, 0.0],
            [1.0, 5.0, b, 0.0, 1.0, 0.0],
            [1.0, 6.0, b, 0.0, 0.0, 1.0],
            [2.0, 0.0, q, 0.0, 0.0, 0.0],
            [3.0, 1.0, b, 4.0, 5.0, 6.0],
            [3.0, 2.0, b, 0.0, 1.0, 0.0],
            [3.0, 3.0, b, 2.0, 2.0, 2.0],
            [3.0, 4.0, q, 0.0, 1.0, 0.0],
            [3.0, 5.0, q, 0.0, 0.0, -1.0],
            [3.0, 6.0, b, 1.0, 0.0, 0.0],
            [4.0, 1.0, s, 10.0, 20.0, 30.0],
            [4.0, 2.0, s, 0.0, 1.0, 0.0],
            [4.0, 3.0, s, 0.0, 0.0, 0.0],
            [4.0, 4.0, s, 1.0, 0.0, 0.0],
            [4.0, 5.0, s, 0.0, 1.0, 0.0],
            [4.0, 6.0, s, 0.0, 0.0, 1.0],
        ]
    )

    u = n2p.make_uset(
        dof=[1, 2],
        nasset=[n2p.mkusetmask("b"), n2p.mkusetmask("q")],
        xyz=[[1, 2, 3], [0, 0, 0]],
    )

    sbe = np.array(
        [
            [1.0, 1.0, 2097154.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 2097154.0, 0.0, 1.0, 0.0],
            [1.0, 3.0, 2097154.0, 0.0, 0.0, 0.0],
            [1.0, 4.0, 2097154.0, 1.0, 0.0, 0.0],
            [1.0, 5.0, 2097154.0, 0.0, 1.0, 0.0],
            [1.0, 6.0, 2097154.0, 0.0, 0.0, 1.0],
            [2.0, 1.0, 4194304.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 4194304.0, 0.0, 1.0, 0.0],
            [2.0, 3.0, 4194304.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 4194304.0, 1.0, 0.0, 0.0],
            [2.0, 5.0, 4194304.0, 0.0, 1.0, 0.0],
            [2.0, 6.0, 4194304.0, 0.0, 0.0, 1.0],
        ]
    )

    assert np.allclose(u.reset_index().values, sbe)


def test_addgrid():
    # node 100 in basic is @ [5, 10, 15]
    # node 200 in cylindrical coordinate system is @
    # [r, th, z] = [32, 90, 10]
    cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    sphcoord = np.array([[2, 3, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])

    uset = None
    uset = n2p.addgrid(uset, 100, "b", 0, [5, 10, 15], 0)
    uset = n2p.addgrid(uset, 200, "b", cylcoord, [32, 90, 10], cylcoord)
    uset = n2p.addgrid(uset, 300, "b", sphcoord, [50, 90, 90], sphcoord)

    uset2 = n2p.addgrid(
        None,
        [100, 200, 300],
        "b",
        [0, cylcoord, sphcoord],
        [[5, 10, 15], [32, 90, 10], [50, 90, 90]],
        [0, cylcoord, sphcoord],
    )
    assert np.all((uset == uset2).values)

    # get coordinates of node 200 in basic:
    assert np.allclose(np.array([10.0, 0, 32.0]), n2p.getcoordinates(uset, 200, 0))
    # reverse:
    assert np.allclose([32, 90, 10], n2p.getcoordinates(uset, [[10.0, 0, 32.0]], 1))
    with pytest.raises(ValueError):
        n2p.addgrid(None, 555, "b", 555, [0, 0, 0], 555)
    with pytest.raises(ValueError):
        n2p.addgrid(uset, uset.index[0][0], "b", 0, [0, 0, 0], 0)
    uset = n2p.addgrid(None, 1, "brbccq", 0, [0, 0, 0], 0)
    b = n2p.mkusetmask("b")
    r = n2p.mkusetmask("r")
    c = n2p.mkusetmask("c")
    q = n2p.mkusetmask("q")
    sets = [b, r, b, c, c, q]
    assert np.all(uset["nasset"] == np.array(sets))


def test_getcoordinates():
    # node 100 in basic is @ [5, 10, 15]
    # node 200 in cylindrical coordinate system is @
    # [r, th, z] = [32, 90, 10]
    cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    sphcoord = np.array([[2, 3, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])
    uset = None
    uset = n2p.addgrid(uset, 100, "b", 0, [5, 10, 15], 0)
    uset = n2p.addgrid(uset, 200, "b", cylcoord, [32, 90, 10], cylcoord)
    uset = n2p.addgrid(uset, 300, "b", sphcoord, [50, 90, 45], sphcoord)
    np.set_printoptions(precision=2, suppress=True)

    # check coordinates of node 100:
    assert np.allclose(np.array([5.0, 10.0, 15.0]), n2p.getcoordinates(uset, 100, 0))
    rctcoord = np.array([[10, 1, 0], [-2, -8, 9], [-2, -8, 10], [0, -8, 9]])
    assert np.allclose(
        np.array([5.0 + 2.0, 10.0 + 8.0, 15.0 - 9.0]),
        n2p.getcoordinates(uset, 100, rctcoord),
    )
    r = np.hypot(10.0, 15.0)
    th = math.atan2(15.0, 10.0) * 180.0 / math.pi
    z = 5.0
    gc = n2p.getcoordinates(uset, 100, 1)
    assert np.allclose([r, th, z], gc)
    r = np.linalg.norm([5.0, 10.0, 15.0])
    th = math.atan2(np.hypot(15.0, 5.0), 10.0) * 180.0 / math.pi
    phi = math.atan2(5.0, 15.0) * 180.0 / math.pi
    assert np.allclose(np.array([r, th, phi]), n2p.getcoordinates(uset, 100, 2))

    # check coordinates of node 200:
    assert np.allclose(np.array([10.0, 0.0, 32.0]), n2p.getcoordinates(uset, 200, 0))
    assert np.allclose(np.array([32.0, 90.0, 10.0]), n2p.getcoordinates(uset, 200, 1))
    assert np.allclose(
        np.array([32.0, 90.0, 10.0]), n2p.getcoordinates(uset, 200, cylcoord)
    )
    r = np.hypot(10.0, 32.0)
    th = 90.0
    phi = math.atan2(10.0, 32.0) * 180 / math.pi
    assert np.allclose(np.array([r, th, phi]), n2p.getcoordinates(uset, 200, 2))
    assert np.allclose(np.array([r, th, phi]), n2p.getcoordinates(uset, 200, sphcoord))

    # check coordinates of node 300:
    xb = 50.0 / math.sqrt(2)
    yb = 0.0
    zb = xb
    assert np.allclose(np.array([xb, yb, zb]), n2p.getcoordinates(uset, 300, 0))
    assert np.allclose(np.array([zb, 90.0, xb]), n2p.getcoordinates(uset, 300, 1))
    assert np.allclose(np.array([50.0, 90.0, 45.0]), n2p.getcoordinates(uset, 300, 2))
    assert np.allclose(
        np.array([50.0, 90.0, 45.0]), n2p.getcoordinates(uset, 300, sphcoord)
    )

    # one more test to fill gap:
    sphcoord = np.array([[1, 3, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0]])
    uset = None
    uset = n2p.addgrid(uset, 100, "b", 0, [5, 10, 15], 0)
    uset = n2p.addgrid(uset, 200, "b", 0, [5, 10, 15], sphcoord)

    # get coordinates of node 100 in spherical (cid 1):
    R = np.linalg.norm([5, 10, 15])
    phi = math.atan2(10, 5) * 180 / math.pi
    xy_rad = np.linalg.norm([5, 10])
    th = 90.0 - math.acos(xy_rad / R) * 180 / math.pi
    assert np.allclose(n2p.getcoordinates(uset, 100, 1), [R, th, phi])

    with pytest.raises(ValueError):
        n2p.getcoordinates(uset, [[1, 2, 3, 4]], 0)

    with pytest.raises(ValueError):
        n2p.getcoordinates(uset, [[[1]]], 0)


def test_rbcoords():
    with pytest.raises(ValueError):
        n2p.rbcoords(np.random.randn(3, 4))
    with pytest.raises(ValueError):
        n2p.rbcoords(np.random.randn(13, 6))


def gettestuset():
    # z = x-basic; r = y-basic
    cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    sphcoord = np.array([[2, 3, 1], [2, 2, 2], [2, 7, 3], [-5, 19, 24]])
    coords = {
        100: [[5, 10, 15], "b", cylcoord],
        200: [[0, 0, 0], "q", 0],
        300: [[10, 20, 30], "o", sphcoord],
        400: [[20, 30, 40], "m", cylcoord],
    }
    uset = None
    coordref = {}
    for i, id in enumerate(sorted(coords)):
        loc = coords[id][0]
        dofset = coords[id][1]
        csys = coords[id][2]
        uset = n2p.addgrid(uset, id, dofset, csys, loc, csys, coordref)
    return uset


def test_mksetpv_mkdofpv():
    uset = gettestuset()
    with pytest.raises(ValueError):
        n2p.mksetpv(uset, "m", "b")
    pv, outdof = n2p.mkdofpv(uset, "f", [[100, 3], [200, 5], [300, 16], [300, 4]])
    assert np.all(pv == np.array([2, 10, 12, 17, 15]))
    assert np.all(
        outdof == np.array([[100, 3], [200, 5], [300, 1], [300, 6], [300, 4]])
    )
    bad_dof = [
        [100, 3],
        [200, 5],
        [300, 1],
        [300, 4],
        [400, 123],
    ]  # 400 123 is not in f-set
    with pytest.raises(ValueError):
        n2p.mkdofpv(uset, "f", bad_dof)

    # but it better work with strict off:
    pv, dof = n2p.mkdofpv(uset, "f", bad_dof, strict=0)
    uset_part = uset.loc[[(100, 3), (200, 5), (300, 1), (300, 4)]]
    assert np.all(uset_part == uset.iloc[pv])
    assert np.all(np.array([[100, 3], [200, 5], [300, 1], [300, 4]]) == dof)


def test_mkdovpv_2():
    u = n2p.make_uset(
        dof=[[1, 123456], [2, 0]],
        nasset=[n2p.mkusetmask("b"), n2p.mkusetmask("q")],
        xyz=[[1, 2, 3], [0, 0, 0]],
    )

    # assuming both are grids will fail on strict=True:
    with pytest.raises(ValueError):
        n2p.mkdofpv(u, "p", [1, 2], strict=True, grids_only=True)

    # but not if strict=False, and only the grid will be found:
    pv = n2p.mkdofpv(u, "p", [1, 2], strict=False, grids_only=True)[0]
    assert (pv == np.arange(6)).all()

    # this will successfully find everything:
    pv = n2p.mkdofpv(u, "p", [1, 2], strict=False, grids_only=False)[0]
    assert (pv == np.arange(7)).all()

    # exception if strict=True but we allow all type of DOF (cannot work)
    with pytest.raises(ValueError):
        n2p.mkdofpv(u, "p", [1, 2], strict=True, grids_only=False)


def test_mkcordcardinfo():
    uset = n2p.addgrid(None, 1, "b", 0, [0, 0, 0], 0)
    ci = n2p.mkcordcardinfo(uset)
    assert ci == {}
    with pytest.raises(ValueError):
        n2p.mkcordcardinfo(uset, 5)

    uset = gettestuset()
    with pytest.raises(ValueError):
        n2p.mkcordcardinfo(uset, 5)


# Testing formrbe3() is tough to do fully.  The docstring has a couple
# simple tests verified by results from an older code.  To be more
# complete, the following nonsense Nastran run was done to generate
# a bunch of GM matrices to compare against.  This also tests addgrid(),
# rbgeom_uset(), and nastran.bulk2uset().

# EIGR, 1, MGIV, 0., 1.
# $
#
# $1111111222222223333333344444444555555556666666677777777888888889999999900000000
# CORD2C  1       0       0.0     0.0     0.0     0.0     0.0     1.0
#         1.0     0.0     0.0
# $
# CORD2R  2       0       0.0     0.0     0.0     1.0     0.0     0.0
#         0.0     1.0     0.0
#
# CORD2C   8      7       9.97    12.0    0.456   -450.0  -13.0   21.5
#         -1.0    -81.0   4.0
#
# CORD2S   7      1       4.0     9.97    12.0    0.456   -450.0  -13.0
#         21.5    -1.0    -81.0
#
# GRID, 100, 1, 25.0, 0.0, 0.0, 1
# GRID, 200, 1, 25.0, 120.0, 0.0, 7
#
# GRID, 300, 1, 25.0, 240.0, 0.0
# GRID, 400, 2, 0.0, 0.0, 0.0, 2
#
# CORD2R   9      8       12.0    0.456   -450.0  -13.0   21.5    -1.0
#         -81.0   4.0     9.97
#
# CORD2R  10      9       12.0    0.456   -450.0  -13.0   21.5    -1.0
#         -81.0   4.0     9.97
#
# CORD2C  11      7       19.97   22.0    0.456   -450.0  -13.0   21.5
#         -1.0    -81.0   4.0
# CORD2S  12      7       4.0     99.7    12.0    0.888   -450.0  -13.0
#         21.5    -11.0   -811.0
#
# grid, 1, 7, 5., 5., 5., 8
# grid, 2, 8, 6., 6., 6., 7
# grid, 3, 9, 0., 0., 0., 9
#
# grid, 101, 10, 3., 4., 5., 10
# grid, 102, 11, 3., 4., 5., 10
# grid, 103, 12, 3., 4., 5., 10
#
# grid, 111, 10, 3., 54., 5., 11
# grid, 112, 11, 3., 54., 5., 11
# grid, 113, 12, 3., 54., 5., 11
#
# grid, 121, 10, 31., -4., 15., 12
# grid, 122, 11, 31., -4., 15., 12
# grid, 123, 12, 31., -4., 15., 12
# $grid, 124, 12, 31., 4., -165., 12
#
# CBAR, 1, 100, 100, 200, 1., 0., 0.
# CBAR, 2, 100, 200, 300, 1., 0., 0.
# CBAR, 3, 100, 300, 400, 1., 0., 0.
# CBAR, 4, 100, 400, 1, 1., 0., 0.
# CBAR, 5, 100, 1, 2, 1., 0., 0.
# CBAR, 6, 100, 2, 3, 1., 0., 0.
# CBAR, 7, 100, 3, 101, 1., 0., 0.
# CBAR, 8, 100, 101, 102, 1., 0., 0.
# CBAR, 9, 100, 102, 103, 1., 0., 0.
# CBAR, 10, 100, 103, 111, 1., 0., 0.
# CBAR, 11, 100, 111, 112, 1., 0., 0.
# CBAR, 12, 100, 112, 113, 1., 0., 0.
# CBAR, 13, 100, 113, 121, 1., 0., 0.
# CBAR, 14, 100, 121, 122, 1., 0., 0.
# CBAR, 15, 100, 122, 123, 1., 0., 0.
# $CBAR, 16, 100, 123, 124, 1., 0., 0.
#
# PBAR, 100, 300, 12.566, 12.566, 12.566, 25.133      $ TRUSS MEMBERS (4mm)
# PBAR, 101, 301, 201.06, 3217.0, 3217.0, 6434.0      $ ALPHA JOINT (16mm)
# $
# MAT1, 300, 6.894+7, 2.62+7, , 2.74-6  $ MATERIAL PROPERTIES (ALUMINUM)
# $                                          E    = 6.894e7
# $                                          G    = 2.62e7
# $                                          RHO  = 2.74e-6
# $                                          UNITS: mm, kg, s
# MAT1, 301, 6.894+7, 2.62+7, , 2.74-6
# $


def test_formrbe3_1():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     123456  2.3     123     100     2.5
    #         123     200     12.0    23      300     .5      34      400
    #         .4      456     1       2       3       5.5     136     101
    #         102     103     4.2     123456  111     112     113     .05
    #         25      121     122     123
    # load the data from above modes run:
    nasdata = matlab.loadmat("pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_1.mat")
    uset = conv_uset(nasdata["uset"][0][0][0])
    gm = nasdata["gm"][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata["drg"][0][0][0].T
    pyuset = nastran.bulk2uset(
        "pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_1.dat", follow_includes=False
    )[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(
        pyuset,
        124,
        123456,
        [
            [123, 2.3],
            100,
            [123, 2.5],
            200,
            [23, 12.0],
            300,
            [34, 0.5],
            400,
            [456, 0.4],
            [1, 2, 3],
            [136, 5.5],
            [101, 102, 103],
            [123456, 4.2],
            [111, 112, 113],
            [25, 0.05],
            [121, 122, 123],
        ],
    )
    pyuset["nasset"] = uset["nasset"]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_2():
    # grid, 124, 0, 31., 4., -165., 0
    # RBE3    1               124     123456  2.3     123     100     2.5
    #         123     200     12.0    23      300     .5      34      400
    #         .4      456     1       2       3       5.5     136     101
    #         102     103     4.2     123456  111     112     113     .05
    #         25      121     122     123
    # load the data from above modes run:
    nasdata = matlab.loadmat("pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_2.mat")
    uset = conv_uset(nasdata["uset"][0][0][0])
    gm = nasdata["gm"][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata["drg"][0][0][0].T
    pyuset = nastran.bulk2uset(
        "pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_2.dat", follow_includes=False
    )[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(
        pyuset,
        124,
        123456,
        [
            [123, 2.3],
            100,
            [123, 2.5],
            200,
            [23, 12.0],
            300,
            [34, 0.5],
            400,
            [456, 0.4],
            [1, 2, 3],
            [136, 5.5],
            [101, 102, 103],
            [123456, 4.2],
            [111, 112, 113],
            [25, 0.05],
            [121, 122, 123],
        ],
    )
    pyuset["nasset"] = uset["nasset"]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_3():
    # grid, 124, 7, 31., 4., -165., 7
    # RBE3    1               124     123456  2.3     123     100     2.5
    #         123     200     12.0    23      300     .5      34      400
    #         .4      456     1       2       3       5.5     136     101
    #         102     103     4.2     123456  111     112     113     .05
    #         25      121     122     123
    # load the data from above modes run:
    nasdata = matlab.loadmat("pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_3.mat")
    uset = conv_uset(nasdata["uset"][0][0][0])
    gm = nasdata["gm"][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata["drg"][0][0][0].T
    pyuset = nastran.bulk2uset(
        "pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_3.dat", follow_includes=False
    )[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(
        pyuset,
        124,
        123456,
        [
            [123, 2.3],
            100,
            [123, 2.5],
            200,
            [23, 12.0],
            300,
            [34, 0.5],
            400,
            [456, 0.4],
            [1, 2, 3],
            [136, 5.5],
            [101, 102, 103],
            [123456, 4.2],
            [111, 112, 113],
            [25, 0.05],
            [121, 122, 123],
        ],
    )
    pyuset["nasset"] = uset["nasset"]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_4():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     1346    2.6     123     100
    #         1.8     456     200
    # load the data from above modes run:
    nasdata = matlab.loadmat("pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_4.mat")
    uset = conv_uset(nasdata["uset"][0][0][0])
    gm = nasdata["gm"][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata["drg"][0][0][0].T
    pyuset = nastran.bulk2uset(
        "pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_4.dat", follow_includes=False
    )[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(pyuset, 124, 1346, [[123, 2.6], 100, [456, 1.8], 200])
    pyuset["nasset"] = uset["nasset"]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_UM_1():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     123456  2.3     123     100     2.5
    #         123     200     12.0    23      300     .5      34      400
    #         .4      456     1       2       3       5.5     136     101
    #         102     103     4.2     123456  111     112     113     .05
    #         25      121     122     123
    #         UM      100     1       200     2       300     3       111
    #         4       112     5       113     6

    # load the data from above modes run:
    nasdata = matlab.loadmat("pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_um_1.mat")
    uset = conv_uset(nasdata["uset"][0][0][0])
    gm = nasdata["gm"][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata["drg"][0][0][0].T
    pyuset = nastran.bulk2uset(
        "pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_um_1.dat", follow_includes=False
    )[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(
        pyuset,
        124,
        123456,
        [
            [123, 2.3],
            100,
            [123, 2.5],
            200,
            [23, 12.0],
            300,
            [34, 0.5],
            400,
            [456, 0.4],
            [1, 2, 3],
            [136, 5.5],
            [101, 102, 103],
            [123456, 4.2],
            [111, 112, 113],
            [25, 0.05],
            [121, 122, 123],
        ],
        [100, 2, 111, 2346, 122, 5],
    )
    pyuset["nasset"] = uset["nasset"]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_UM_2():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     123456  1.0     123456  100
    #         UM      124     123     200     456
    # load the data from above modes run:
    nasdata = matlab.loadmat("pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_um_2.mat")
    uset = conv_uset(nasdata["uset"][0][0][0])
    gm = nasdata["gm"][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata["drg"][0][0][0].T
    pyuset = nastran.bulk2uset(
        "pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_um_2.dat", follow_includes=False
    )[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(
        pyuset, 124, 123456, [[123, 2.6], 100, [456, 1.8], 200], [124, 123, 200, 456]
    )
    pyuset["nasset"] = uset["nasset"]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_UM_3():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     123456  2.6     123     100
    #         1.8     456     200
    #         UM      124     152346
    # load the data from above modes run:
    nasdata = matlab.loadmat("pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_um_3.mat")
    uset = conv_uset(nasdata["uset"][0][0][0])
    gm = nasdata["gm"][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata["drg"][0][0][0].T
    pyuset = nastran.bulk2uset(
        "pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_um_3.dat", follow_includes=False
    )[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(
        pyuset, 124, 123456, [[123, 2.6], 100, [456, 1.8], 200], [124, 152346]
    )
    pyuset["nasset"] = uset["nasset"]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_UM_4():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     1346    2.6     123     100
    #         1.8     456     200
    #         UM      124     6341
    # load the data from above modes run:
    # same as test 4:
    nasdata = matlab.loadmat("pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_4.mat")
    uset = conv_uset(nasdata["uset"][0][0][0])
    gm = nasdata["gm"][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata["drg"][0][0][0].T
    pyuset = nastran.bulk2uset(
        "pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_4.dat", follow_includes=False
    )[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(
        pyuset, 124, 1346, [[123, 2.6], 100, [456, 1.8], 200], [124, 6341]
    )
    pyuset["nasset"] = uset["nasset"]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_UM_5():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     1346    2.6     123     100
    #         1.8     456     200
    #         UM      100     12      200     56
    # load the data from above modes run:
    nasdata = matlab.loadmat("pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_um_5.mat")
    uset = conv_uset(nasdata["uset"][0][0][0])
    gm = nasdata["gm"][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata["drg"][0][0][0].T
    pyuset = nastran.bulk2uset(
        "pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_um_5.dat", follow_includes=False
    )[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(
        pyuset, 124, 1346, [[123, 2.6], 100, [456, 1.8], 200], [100, 12, 200, 56]
    )
    pyuset["nasset"] = uset["nasset"]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_UM_6():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     1346    2.6     123     100
    #         1.8     456     200
    #         UM      100     12      200     5       124     5
    # load the data from above modes run:
    nasdata = matlab.loadmat("pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_um_6.mat")
    uset = conv_uset(nasdata["uset"][0][0][0])
    gm = nasdata["gm"][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata["drg"][0][0][0].T
    pyuset = nastran.bulk2uset(
        "pyyeti/tests/nastran_gm_data/make_gm_nx9_rbe3_um_6.dat", follow_includes=False
    )[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(
        pyuset, 124, 1346, [[123, 2.6], 100, [456, 1.8], 200], [100, 12, 200, 5, 124, 6]
    )
    pyuset["nasset"] = uset["nasset"]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_upasetpv():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    pv = n2p.upasetpv(nas, 102)
    # csuper is:
    #   CSUPER, 102, 0, 103, 111, 19, 27, 9990001, 9990002
    # and grids 3, 11, 70 and spoints 1995001-1995022 are the other dof
    # in the model, so:
    # id    dof
    #  3    1-6
    #  11   7-12
    #  19   13-18
    #  27   19-24
    #  70   25-30
    #  103  31-36
    #  111  37-42
    #  1995001 43
    #   ...
    #  1995022 22+42 = 64
    #  9990001 65-70
    #  9990001 71-76
    # Therefore, pv better match this:
    shouldbe = (
        np.array(
            [
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
            ]
        )
        - 1
    )
    assert np.all(pv == shouldbe)


def test_upasetpv_2():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_extseout/nas2cam")
    pv = n2p.upasetpv(nas, 102)
    # se 102 seconct is:
    #    SECONCT      102       0              NO
    #                   3       3      11      11
    #                  19      19      27      27
    #             2995001 2995001 2995002 2995002
    #             2995003 2995003 2995004 2995004
    #             2995005 2995005 2995006 2995006
    #             2995007 2995007 2995008 2995008
    #
    # se 0 uset is:
    # id    dof
    # 70   1-6
    # 1995001-1995022   7-28
    #  3   29-34
    # 11   35-40
    # 19   41-46
    # 27   47-52
    # 2995001-2995008   53-60
    # Therefore, pv better match this:
    shouldbe = np.arange(29, 61) - 1
    assert np.all(pv == shouldbe)


def test_upqsetpv():
    nas_extseout = nastran.rdnas2cam("pyyeti/tests/nas2cam_extseout/nas2cam")
    pv_extseout = nastran.upqsetpv(nas_extseout, 0)

    ue = nas_extseout["uset"][0].index.get_level_values(0)
    assert np.all((ue > 200) == pv_extseout)

    nas_csuper = nastran.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    pv_csuper = nastran.upqsetpv(nas_csuper)

    uc = nas_csuper["uset"][0].index.get_level_values(0)
    # last four are dummy dof from grids:
    pv = uc > 200
    pv[-4:] = False
    assert np.all(pv == pv_csuper)

    with pytest.raises(ValueError):
        nastran.upqsetpv(nas_csuper, 1000)


def test_upa_upq_not_all_6():
    # The following model uses EXTSEOUT superelements and does not
    # include all 6 DOF for all boundary grids in the b-set. Test
    # that:

    # 'n1' is from msc nastran version 2017

    # - have EMAP but missing MAPS
    # need to update DMAP DBVIEW line from:
    # DBVIEW   MAPSX = MAPS       (WHERE SEID=CSEID AND PEID=0 AND
    #                              WILDCARD=TRUE) $
    # to:
    # DBVIEW   MAPSX = MAPS       (WHERE SEID=CSEID AND WILDCARD=TRUE) $

    n1 = op2.rdnas2cam("pyyeti/tests/nas2cam_extseout/nas2cam_notall6_msc2017")

    # 'n2' is from nx nastran version 2021
    # - missing EMAP but have MAPS
    # - nx won't write EMAP:
    #  *** USER INFORMATION MESSAGE 1207 (OUTPBN2)
    #      THE DATABLOCK EMAP    /EMAP     DEFINED AS NDDL TYPE UNST IS
    #         NOT SUPPORTED BY BYTE SWAPPING.
    #      THIS DATABLOCK WILL NOT BE BYTE SWAPPED

    n2 = op2.rdnas2cam("pyyeti/tests/nas2cam_extseout/nas2cam_notall6_nx2021")

    # So, patch them together to get one that works:
    n1["maps"][75] = n2["maps"][75]

    up_a = n2p.upasetpv(n1, 75)
    assert up_a.shape == (29,)
    assert (up_a == (n1["uset"][0]["nasset"] == 2).values.nonzero()[0]).all()

    u = n2p.formulvs(n1, 75)
    assert u.shape[0] == 29
    assert (u[:-10, 0] != 0.0).all()

    pvq2 = n2p.upqsetpv(n1)
    qset = np.array(
        [
            [7590001, 0],
            [7590002, 0],
            [7590003, 0],
            [7590004, 0],
            [7590005, 0],
            [7590006, 0],
            [7590007, 0],
            [7590008, 0],
            [7590009, 0],
            [7590010, 0],
        ]
    )
    assert (pvq2.nonzero()[0] == n2p.mkdofpv(n1["uset"][0], "p", qset)[0]).all()


def test_upa_upq_not_all_6_2():
    # The following model uses EXTSEOUT superelements and does not
    # include all 6 DOF for all boundary grids in the b-set. Test
    # that:

    n1 = op2.rdnas2cam("pyyeti/tests/nas2cam_extseout/nas2cam_notall6_msc2017_b")

    up_a = n2p.upasetpv(n1, 75)
    assert up_a.shape == (29,)
    assert (up_a == (n1["uset"][0]["nasset"] == 2).values.nonzero()[0]).all()

    u = n2p.formulvs(n1, 75)
    assert u.shape[0] == 29
    assert (u[:-10, 0] != 0.0).all()

    pvq2 = n2p.upqsetpv(n1)
    qset = np.array(
        [
            [7590001, 0],
            [7590002, 0],
            [7590003, 0],
            [7590004, 0],
            [7590005, 0],
            [7590006, 0],
            [7590007, 0],
            [7590008, 0],
            [7590009, 0],
            [7590010, 0],
        ]
    )
    assert (pvq2.nonzero()[0] == n2p.mkdofpv(n1["uset"][0], "p", qset)[0]).all()


def test_formtran1_seup():
    o4 = op4.OP4()
    tug1 = nastran.rddtipch("pyyeti/tests/nas2cam_extseout/outboard.pch")
    mug1 = o4.listload("pyyeti/tests/nas2cam_extseout/outboard.op4", "mug1")[1][0]
    grids = [[11, 123456], [45, 123456], [60, 123456], [1995002, 1]]
    pv, exp_dof = n2p.mkdofpv(tug1, "p", grids)
    MUG1 = mug1[pv, :]
    nas_csuper = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    exp_dof0 = exp_dof
    exp_dof0[-1, 1] = 0
    drm101, dof101 = n2p.formtran(nas_csuper, 101, exp_dof0)
    assert np.allclose(np.abs(drm101), np.abs(MUG1))


def test_formtran2_seup():
    o4 = op4.OP4()
    tug1 = nastran.rddtipch("pyyeti/tests/nas2cam_extseout/outboard.pch")
    mug1 = o4.listload("pyyeti/tests/nas2cam_extseout/outboard.op4", "mug1")[1][0]
    grids = [11, 45, 60]
    pv, exp_dof = n2p.mkdofpv(tug1, "p", grids)
    MUG1 = mug1[pv, :]
    nas_csuper = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    drm101, dof101 = n2p.formtran(nas_csuper, 101, exp_dof)
    assert np.allclose(np.abs(drm101), np.abs(MUG1))


def test_formtran3_seup():
    o4 = op4.OP4()
    tug1 = nastran.rddtipch("pyyeti/tests/nas2cam_extseout/outboard.pch")
    mug1 = o4.listload("pyyeti/tests/nas2cam_extseout/outboard.op4", "mug1")[1][0]
    grids = [[11, 45]]
    pv, exp_dof = n2p.mkdofpv(tug1, "p", grids)
    MUG1 = mug1[pv, :]
    nas_csuper = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    drm101, dof101 = n2p.formtran(nas_csuper, 101, exp_dof)
    assert np.allclose(np.abs(drm101), np.abs(MUG1))


def test_formtran4_seup():
    o4 = op4.OP4()
    tug1 = nastran.rddtipch("pyyeti/tests/nas2cam_extseout/outboard.pch")
    mug1 = o4.listload("pyyeti/tests/nas2cam_extseout/outboard.op4", "mug1")[1][0]
    grids = [[60, 4]]
    pv, exp_dof = n2p.mkdofpv(tug1, "p", grids)
    MUG1 = mug1[pv, :]
    nas_csuper = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    drm101, dof101 = n2p.formtran(nas_csuper, 101, exp_dof)
    assert np.allclose(np.abs(drm101), np.abs(MUG1))


def test_formtran5_seup():
    o4 = op4.OP4()
    tug1 = nastran.rddtipch("pyyeti/tests/nas2cam_extseout/outboard.pch")
    mug1 = o4.listload("pyyeti/tests/nas2cam_extseout/outboard.op4", "mug1")[1][0]
    # put recovery in non-ascending order:
    grids = [
        [11, 3],
        [60, 6],
        [60, 4],
        [1995002, 1],
        [11, 5],
        [11, 4],
        [60, 5],
        [45, 5],
        [45, 1],
        [11, 2],
        [60, 2],
        [45, 4],
        [60, 1],
        [45, 6],
        [45, 3],
        [11, 6],
        [11, 1],
        [60, 3],
        [45, 2],
    ]
    pv, exp_dof = n2p.mkdofpv(tug1, "p", grids)
    MUG1 = mug1[pv, :]
    exp_dof0 = exp_dof.copy()
    pv = exp_dof0[:, 0] == 1995002
    exp_dof0[pv, 1] = 0
    nas_csuper = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    drm101, dof101 = n2p.formtran(nas_csuper, 101, exp_dof0)
    assert np.allclose(np.abs(drm101), np.abs(MUG1))
    assert np.all(exp_dof == np.array(grids))


def test_formtran1_se0():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_extseout/nas2cam")
    grids = [[3, 123456], [27, 123456], [70, 123456], [2995004, 0], [2995005, 0]]
    drm_a, dof_a = n2p.formtran(nas, 0, grids)  # use phg
    nas.pop("phg", None)
    drm_b, dof_b = n2p.formtran(nas, 0, grids)  # use pha
    assert np.allclose(drm_a, drm_b)
    dof = np.array(
        [
            [3, 1],
            [3, 2],
            [3, 3],
            [3, 4],
            [3, 5],
            [3, 6],
            [27, 1],
            [27, 2],
            [27, 3],
            [27, 4],
            [27, 5],
            [27, 6],
            [70, 1],
            [70, 2],
            [70, 3],
            [70, 4],
            [70, 5],
            [70, 6],
            [2995004, 0],
            [2995005, 0],
        ]
    )
    assert np.all(dof_b == dof)


def test_formtran2_se0():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_extseout/nas2cam")
    grids = [
        [2995004, 0],
        [70, 3],
        [3, 3],
        [70, 5],
        [27, 4],
        [70, 6],
        [27, 6],
        [70, 4],
        [27, 2],
        [70, 2],
        [3, 2],
        [70, 1],
    ]
    drm_a, dof_a = n2p.formtran(nas, 0, grids)  # use phg
    nas.pop("phg", None)
    drm_b, dof_b = n2p.formtran(nas, 0, grids)  # use pha
    assert np.allclose(drm_a, drm_b)
    assert np.all(dof_b == np.array(grids))


def test_formtran3_se0():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_extseout/nas2cam")
    grids = [
        [2995004, 0],
        [70, 3],
        [3, 3],
        [70, 5],
        [27, 4],
        [70, 6],
        [27, 6],
        [70, 4],
        [27, 2],
        [70, 2],
        [3, 2],
        [70, 1],
    ]
    drm_a, dof_a = n2p.formtran(nas, 0, grids)
    drm_b_gset, dof_b = n2p.formtran(nas, 0, grids, gset=True)
    drm_b = np.dot(drm_b_gset, nas["phg"][0])
    assert np.allclose(drm_a, drm_b)
    assert np.all(dof_b == np.array(grids))


def test_formulvs_1():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_extseout/nas2cam")
    ulvs = n2p.formulvs(nas, 101, 101)
    assert ulvs == 1.0
    ulvs = n2p.formulvs(nas, 101)
    assert np.allclose(ulvs, nas["ulvs"][101])
    ulvs = n2p.formulvs(nas, 102)
    assert np.allclose(ulvs, nas["ulvs"][102])
    ulvs = n2p.formulvs(nas, 101, shortcut=0, keepcset=0)
    assert np.allclose(ulvs, nas["ulvs"][101])
    ulvs = n2p.formulvs(nas, 102, shortcut=0, keepcset=0)
    assert np.allclose(ulvs, nas["ulvs"][102])
    assert 1.0 == n2p.formulvs(nas, seup=101, sedn=101)
    old = nas["ulvs"]
    del nas["ulvs"]
    n2p.addulvs(nas, 101, 102)
    for se in [101, 102]:
        assert np.allclose(old[se], nas["ulvs"][se])

    old = {i: v for i, v in nas["ulvs"].items()}
    n2p.addulvs(nas, 101, 102)
    for se in [101, 102]:
        assert id(old[se]) == id(nas["ulvs"][se])

    n2p.addulvs(nas, 101, 102, shortcut=False)
    for se in [101, 102]:
        assert id(old[se]) != id(nas["ulvs"][se])
        assert np.allclose(old[se], nas["ulvs"][se])


def test_formulvs_2():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_extseout/nas2cam")
    ulvsgset = n2p.formulvs(nas, 101, gset=True)
    ulvs = np.dot(ulvsgset, nas["phg"][0])
    assert np.allclose(ulvs, nas["ulvs"][101])

    ulvsgset = n2p.formulvs(nas, 102, gset=True)
    ulvs = np.dot(ulvsgset, nas["phg"][0])
    assert np.allclose(ulvs, nas["ulvs"][102])


def test_rdnas2cam_no_se():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam/no_se_nas2cam")
    assert np.all(nas["selist"] == [0, 0])


def test_formulvs_multilevel():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam/with_se_nas2cam")
    old = nas["ulvs"]
    del nas["ulvs"]
    ses = [100, 200, 300, 400]
    n2p.addulvs(nas, *ses)
    for se in ses:
        assert np.allclose(old[se], nas["ulvs"][se])
    u300_100 = n2p.formulvs(nas, seup=300, sedn=100, keepcset=True)
    u300 = u300_100.dot(nas["ulvs"][100])
    assert np.allclose(u300, nas["ulvs"][300])
    u300_100 = n2p.formulvs(nas, seup=300, sedn=100, keepcset=False)
    assert u300_100.shape[1] < nas["ulvs"][100].shape[0]


def test_new_dnids_entry():
    nas_old = op2.rdnas2cam("pyyeti/tests/nas2cam/with_se_nas2cam")
    nas_new = op2.rdnas2cam("pyyeti/tests/nas2cam/with_se_convemap_nas2cam")
    assert len(nas_new["dnids"]) == len(nas_old["dnids"])
    for k, v in nas_old["dnids"].items():
        assert np.allclose(nas_new["dnids"][k], v)


def test_formdrm_1():
    grids = [[11, 123456], [45, 123456], [60, 123456], [1995002, 0]]
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    drm101, dof101 = n2p.formtran(nas, 101, grids)
    ulvs = n2p.formulvs(nas, 101)
    DRM_A = np.dot(drm101, ulvs)

    DRM_B, dof101_b = n2p.formdrm(nas, 101, grids)
    assert np.allclose(DRM_A, DRM_B)
    assert np.all(dof101 == dof101_b)

    del nas["phg"]
    with pytest.raises(RuntimeError, match=r"neither nas\['phg'\]\[0\]"):
        ulvs = n2p.formulvs(nas, 101)
    with pytest.raises(RuntimeError):
        n2p.formulvs(nas, 101)


def test_formdrm_2():
    grids = [[11, 123456], [45, 123456], [60, 123456], [1995002, 0]]
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    drm101, dof101 = n2p.formtran(nas, 101, grids, gset=True)
    ulvs = n2p.formulvs(nas, 101, gset=True)
    DRM_A = np.dot(drm101, ulvs)

    DRM_B, dof101_b = n2p.formdrm(nas, 101, grids, gset=True)
    assert np.allclose(DRM_A, DRM_B)
    assert np.all(dof101 == dof101_b)


def test_formdrm_oset_sset():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam/with_se_nas2cam")
    drm, dof = n2p.formdrm(nas, 100, 11)  # just o-set
    pv = n2p.mksetpv(nas["uset"][0], "g", "a")
    pha = nas["phg"][0][pv]
    del nas["phg"]
    nas["pha"] = {0: pha}
    drm, dof = n2p.formdrm(nas, 0, 306)  # just s-set


def test_formdrm_noqset():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam/with_se_nas2cam")
    # modify 100 so it is a static reduction only:
    del nas["lambda"][100]
    del nas["goq"]
    # move 'q' set into 's' set:
    s = n2p.mkusetmask("s")
    q = n2p.mksetpv(nas["uset"][100], "g", "q")
    nas["uset"][100].iloc[q, 0] = s
    drm, dof = n2p.formdrm(nas, seup=100, sedn=100, dof=11)


def test_fromdrm_null_cset():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    ulvs = n2p.formulvs(nas, 102, keepcset=1, shortcut=0)
    nas["ulvs"] = {102: ulvs}
    drm1, dof = n2p.formdrm(nas, seup=102, sedn=0, dof=3)

    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    ulvs = n2p.formulvs(nas, 102, keepcset=0, shortcut=0)
    nas["ulvs"] = {102: ulvs}
    drm2, dof = n2p.formdrm(nas, seup=102, sedn=0, dof=3)
    assert np.allclose(drm1, drm2)


def test_build_coords():
    #      [cid, ctype, refcid, a1, a2, a3, b1, b2, b3, c1, c2, c3]
    cords = [
        [10, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [20, 2, 10, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    ]
    dct1 = n2p.build_coords(cords)

    cords = [
        [10, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [20, 2, 10, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [20, 2, 10, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    ]
    dct2 = n2p.build_coords(cords)
    assert dir(dct1) == dir(dct2)
    for k in dct1:
        assert np.all(dct1[k] == dct2[k])

    cords = [
        [10, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [20, 2, 10, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [20, 2, 10, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    ]

    with pytest.raises(RuntimeError):
        n2p.build_coords(cords)

    cords = [
        [10, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [20, 2, 5, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    ]

    with pytest.raises(RuntimeError):
        n2p.build_coords(cords)


def test_rbmodes_allq():
    # node 100 in basic is @ [5, 10, 15]
    # node 200 in cylindrical coordinate system is @
    # [r, th, z] = [32, 90, 10]
    cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    sphcoord = np.array([[2, 3, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])
    uset = None
    uset = n2p.addgrid(uset, 100, "q", 0, [5, 10, 15], 0)
    uset = n2p.addgrid(uset, 200, "q", cylcoord, [32, 90, 10], cylcoord)
    uset = n2p.addgrid(uset, 300, "q", sphcoord, [50, 90, 90], sphcoord)
    rb = n2p.rbgeom_uset(uset)
    assert np.all(rb == 0)

    uset2 = None
    uset2 = n2p.addgrid(uset2, 100, "q", 0, [5, 10, 15], 0)
    uset2 = n2p.addgrid(uset2, 300, "q", sphcoord, [50, 90, 90], sphcoord)
    uset2 = n2p.addgrid(uset2, 200, "q", cylcoord, [32, 90, 10], cylcoord)
    assert np.allclose(uset, uset2.sort_index())


def test_formdrm_go_warnings():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam/with_se_nas2cam")
    # make sure that got and goq is not present for se 300
    try:
        del nas["got"][300]
    except KeyError:
        pass
    try:
        del nas["goq"][300]
    except KeyError:
        pass
    q = sum(n2p.mksetpv(nas["uset"][300], "g", "q"))
    o = sum(n2p.mksetpv(nas["uset"][300], "g", "o"))
    t = sum(n2p.mksetpv(nas["uset"][300], "g", "t"))
    goq = np.zeros((o, q))
    got = np.zeros((o, t))

    nas["got"][300] = got
    with pytest.warns(RuntimeWarning, match=r"nas\['goq'\]\[300\] not found"):
        drm, dof = n2p.formdrm(nas, 300, [38, 39])
    del nas["got"][300]

    nas["goq"][300] = goq
    with pytest.warns(RuntimeWarning, match=r"nas\['got'\]\[300\] not found"):
        drm, dof = n2p.formdrm(nas, 300, [38, 39])
    del nas["goq"][300]


def test_badrbe3_error():
    # put some grids on the x-axis and build a bad rbe3 to test for
    x = np.arange(0, 5, 1.0)
    n = x.shape[0]
    y = np.zeros(n)
    # the 'poorly conditioned' message:
    uset = n2p.addgrid(None, np.arange(1, n + 1), "b", 0, np.column_stack((x, y, y)), 0)
    uset = n2p.addgrid(uset, 100, "b", 0, [5, 0, 0], 0)

    with pytest.warns(RuntimeWarning, match="matrix is poorly conditioned"):
        with pytest.raises(la.LinAlgError):
            n2p.formrbe3(uset, 100, 123456, [123, [1, 2, 3, 4, 5]])


def test_badrbe3_warn():
    # put some grids on the x-axis and build a bad rbe3 to test for
    # the 'poorly conditioned' message:
    x = np.arange(0, 5, 1.0)
    n = x.shape[0]
    y = np.zeros(n)
    z = np.zeros(n)
    for i in range(len(x)):
        j = i * 6
        if i == 4:
            _z = 0.00000000000001
        else:
            _z = 0.0
        z[i] = _z
    uset = n2p.addgrid(None, np.arange(1, n + 1), "b", 0, np.column_stack((x, y, z)), 0)
    uset = n2p.addgrid(uset, 100, "b", 0, [5, 5, 5], 0)
    with pytest.warns(RuntimeWarning, match="matrix is poorly conditioned"):
        rbe3 = n2p.formrbe3(uset, 100, 123456, [123, [1, 2, 3, 4, 5]])


def test_rbe3_badum():
    x = np.arange(0, 2, 1.0)
    n = x.shape[0]
    y = np.zeros(n)
    z = np.zeros(n)
    uset = n2p.addgrid(None, np.arange(1, n + 1), "b", 0, np.column_stack((x, y, z)), 0)
    uset = n2p.addgrid(uset, 100, "b", 0, [5, 5, 5], 0)
    with pytest.raises(ValueError, match="incorrect size of m-set"):
        rbe3 = n2p.formrbe3(uset, 100, 123456, [123456, [1, 2]], [1, 234])


def test_bad_se():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam/with_se_nas2cam")
    with pytest.raises(ValueError):
        n2p._findse(nas, 345)


def test_sph_zero_theta():
    sphcoord = np.array([[2, 3, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0]])
    uset = n2p.addgrid(None, 1, "b", 0, [0, 0, 0], sphcoord)
    rb = n2p.rbgeom_uset(uset)
    rb2 = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert np.allclose(rb, rb2)


def test_find_xyz_triples1():
    rb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 15.0, -10.0],
            [0.0, 1.0, 0.0, -15.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 10.0, -5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 15.0, -10.0],
            [0.0, 1.0, 0.0, -15.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 10.0, -5.0, 0.0],
        ]
    )
    c = 1 / np.sqrt(2)
    T = np.array([[1.0, 0.0, 0.0], [0.0, c, c], [0.0, -c, c]])
    rb[-3:] = 10 * T @ rb[-3:]

    rb_2 = rb.copy()

    sbe = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 15.0, -10.0],
            [0.0, 1.0, 0.0, -15.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 10.0, -5.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [10.0, 0.0, 0.0, 0.0, 150.0, -100.0],
            [0.0, 7.07, 7.07, -35.36, -35.36, 35.36],
            [0.0, -7.07, 7.07, 176.78, -35.36, -35.36],
        ]
    )
    assert abs(rb - sbe).max() < 0.01

    mats = {"rb": rb}
    trips = n2p.find_xyz_triples(rb, get_trans=True, mats=mats)
    assert trips.outmats is not mats

    assert np.all(
        trips.pv == np.array([True, True, True, False, False, False, True, True, True])
    )

    sbe = np.array(
        [
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
        ]
    )
    assert np.allclose(trips.coords, sbe, equal_nan=True)

    sbe = np.array([1.0, 1.0, 1.0, np.nan, np.nan, np.nan, 10.0, 10.0, 10.0])
    assert np.allclose(trips.scales, sbe, equal_nan=True)
    assert len(trips.Ts) == 2

    assert np.allclose(
        trips.Ts[0], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert (
        abs(
            trips.Ts[1]
            - np.array([[0.1, 0.0, 0.0], [0.0, 0.0707, -0.0707], [0.0, 0.0707, 0.0707]])
        ).max()
        < 0.0001
    )

    assert np.allclose(
        trips.outmats["rb"],
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 15.0, -10.0],
                [0.0, 1.0, 0.0, -15.0, 0.0, 5.0],
                [0.0, 0.0, 1.0, 10.0, -5.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 15.0, -10.0],
                [0.0, 1.0, 0.0, -15.0, 0.0, 5.0],
                [0.0, 0.0, 1.0, 10.0, -5.0, 0.0],
            ]
        ),
    )

    pv = trips.pv.nonzero()[0]
    for j, Tcurr in enumerate(trips.Ts):
        pvcurr = pv[j * 3 : j * 3 + 3]
        rb[pvcurr] = Tcurr @ rb[pvcurr]

    assert np.allclose(
        rb,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 15.0, -10.0],
                [0.0, 1.0, 0.0, -15.0, 0.0, 5.0],
                [0.0, 0.0, 1.0, 10.0, -5.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 15.0, -10.0],
                [0.0, 1.0, 0.0, -15.0, 0.0, 5.0],
                [0.0, 0.0, 1.0, 10.0, -5.0, 0.0],
            ]
        ),
    )

    # test in "inplace" option:
    mats = {"rb": rb_2}
    trips2 = n2p.find_xyz_triples(rb_2, get_trans=True, mats=mats, inplace=True)
    assert trips2.outmats is mats
    assert rb_2 is mats["rb"]
    assert np.allclose(rb, rb_2)
    assert len(trips2.Ts) == 2
    for t1, t2 in zip(trips.Ts, trips2.Ts):
        assert np.allclose(t1, t2)
    for name in ("pv", "coords", "scales"):
        assert np.allclose(getattr(trips, name), getattr(trips2, name), equal_nan=True)


def test_find_xyz_triples2():
    rb = np.array(
        [
            [0.99, 0.0, 0.0, 0.0, 15.0, -10.0],
            [0.0, 1.0, 0.0, -15.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 10.0, -5.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 35.0, -10.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 15.0, -10.0],
            [0.0, 1.0, 0.0, -15.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 10.0, -5.0, 0.0],
        ]
    )

    trips = n2p.find_xyz_triples(rb, tol=0.01)

    loc = np.array(
        [
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
        ]
    )

    assert np.allclose(trips.coords, loc, atol=0.1, equal_nan=True)

    trips = n2p.find_xyz_triples(rb, tol=0.001)

    loc = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
        ]
    )

    assert np.allclose(trips.coords, loc, equal_nan=True)


def test_find_xyz_triples3():
    rb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 15.0, -10.0],
            [0.0, 1.0, 0.0, -15.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 10.0, -5.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 35.0, -10.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 1e-6],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 15.0, -10.0],
            [0.0, 1.0, 0.0, -15.0, 0.0, 5.0],
            [0.0, 0.0, 1.0, 10.0, -5.0, 0.0],
        ]
    )

    trips = n2p.find_xyz_triples(rb, tol=0.01)
    coords = np.array(
        [
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
            [np.nan, np.nan, np.nan],
            [0.0, -5.0e-07, 0.0],
            [0.0, -5.0e-07, 0.0],
            [0.0, -5.0e-07, 0.0],
            [np.nan, np.nan, np.nan],
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
            [5.0, 10.0, 15.0],
        ]
    )

    assert np.allclose(trips.coords, coords, equal_nan=True)
    assert np.allclose(trips.model_scale, 15.0)
