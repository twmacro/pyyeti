import numpy as np
import math
import scipy.linalg as la
from scipy.io import matlab
import io
import os
from pyyeti import nastran
from pyyeti.nastran import n2p, op2, op4
from nose.tools import *


def test_rbgeom():
    x, y, z = 30., 10., 20.
    grids = np.array([[0., 0., 0.], [x, y, z]])
    rb = n2p.rbgeom(grids)
    rb_should_be = np.vstack((np.eye(6), np.eye(6)))
    rb_should_be[6, 4] = z
    rb_should_be[6, 5] = -y
    rb_should_be[7, 3] = -z
    rb_should_be[7, 5] = x
    rb_should_be[8, 3] = y
    rb_should_be[8, 4] = -x
    err = (np.abs(rb - rb_should_be)).max()
    assert err < 1.e-14


def test_rbgeom_uset():
    #     M  -------------------------------------\
    #     S  ------------------------------\       > G --\
    #     O  -----------------------\       > N --/       \
    #     Q  ----------------\       > F --/       \       \
    #     R  ---------\       > A --/       \       \       > P
    #     C  --\       > T --/       \       > FE    > NE  /
    #     B  ---> L --/               > D   /       /     /
    #     E  ------------------------/-----/-------/-----/
    uset = np.array([
        [100,        1,  2097154,        5,       10,       15],  # b-set
        [100,        2,  2097154,        0,        1,        0],
        [100,        3,  2097154,        0,        0,        0],
        [100,        4,  2097154,        1,        0,        0],
        [100,        5,  2097154,        0,        1,        0],
        [100,        6,  2097154,        0,        0,        1],
        [200,        1,  4194304,        0,        0,        0],  # q-set
        [200,        2,  4194304,        0,        1,        0],
        [200,        3,  4194304,        0,        0,        0],
        [200,        4,  4194304,        1,        0,        0],
        [200,        5,  4194304,        0,        1,        0],
        [200,        6,  4194304,        0,        0,        1],
        [300,        1,        4,       10,       20,       30],  # o-set
        [300,        2,        4,        0,        1,        0],
        [300,        3,        4,        0,        0,        0],
        [300,        4,        4,        1,        0,        0],
        [300,        5,        4,        0,        1,        0],
        [300,        6,        4,        0,        0,        1],
        [400,        1,        1,       20,       30,       40],  # m-set
        [400,        2,        1,        0,        1,        0],
        [400,        3,        1,        0,        0,        0],
        [400,        4,        1,        1,        0,        0],
        [400,        5,        1,        0,        1,        0],
        [400,        6,        1,        0,        0,        1]])
    pv = n2p.mksetpv(uset, 'a', 'b')
    assert np.sum(pv) == 6
    pv = n2p.mksetpv(uset, 'p', 'b+q')
    assert np.sum(pv) == 12
    pv = n2p.mksetpv(uset, 'p', 'm+a')
    assert np.sum(pv) == 18
    pv = n2p.mksetpv(uset, 'g', 'q')
    assert np.all(uset[pv, 0] == [200, 200, 200, 200, 200, 200])
    rb = n2p.rbgeom_uset(uset, 300)
    rb_should_be = np.array([
        [1,    0,    0,    0,  -15,   10],
        [0,    1,    0,   15,    0,   -5],
        [0,    0,    1,  -10,    5,    0],
        [0,    0,    0,    1,    0,    0],
        [0,    0,    0,    0,    1,    0],
        [0,    0,    0,    0,    0,    1],
        [0,    0,    0,    0,    0,    0],
        [0,    0,    0,    0,    0,    0],
        [0,    0,    0,    0,    0,    0],
        [0,    0,    0,    0,    0,    0],
        [0,    0,    0,    0,    0,    0],
        [0,    0,    0,    0,    0,    0],
        [1,    0,    0,    0,    0,    0],
        [0,    1,    0,    0,    0,    0],
        [0,    0,    1,    0,    0,    0],
        [0,    0,    0,    1,    0,    0],
        [0,    0,    0,    0,    1,    0],
        [0,    0,    0,    0,    0,    1],
        [1,    0,    0,    0,   10,  -10],
        [0,    1,    0,  -10,    0,   10],
        [0,    0,    1,   10,  -10,    0],
        [0,    0,    0,    1,    0,    0],
        [0,    0,    0,    0,    1,    0],
        [0,    0,    0,    0,    0,    1]])
    err = (np.abs(rb - rb_should_be)).max()
    assert err < 1.e-14

    fname = 'delete.this'
    n2p.usetprt(fname, uset, form=2, perpage=8,
                printsets='m, o, q, b, a, n, g')
    sbe = (' DOF #     GRID    DOF     M   O   Q   B   A   N   G\n'
           '-------  --------  ---     --  --  --  --  --  --  --\n'
           '      1       100   1                   1   1   1   1\n'
           '      2       100   2                   2   2   2   2\n'
           '      3       100   3                   3   3   3   3\n'
           '      4       100   4                   4   4   4   4\n'
           '      5       100   5                   5   5   5   5\n'
           '      6       100   6                   6   6   6   6\n'
           '{} DOF #     GRID    DOF     M   O   Q   B   A   N   G\n'
           '-------  --------  ---     --  --  --  --  --  --  --\n'
           '      7       200   1               1       7   7   7\n'
           '      8       200   2               2       8   8   8\n'
           '      9       200   3               3       9   9   9\n'
           '     10       200   4               4      10  10  10\n'
           '     11       200   5               5      11  11  11\n'
           '     12       200   6               6      12  12  12\n'
           '{} DOF #     GRID    DOF     M   O   Q   B   A   N   G\n'
           '-------  --------  ---     --  --  --  --  --  --  --\n'
           '     13       300   1           1              13  13\n'
           '     14       300   2           2              14  14\n'
           '     15       300   3           3              15  15\n'
           '     16       300   4           4              16  16\n'
           '     17       300   5           5              17  17\n'
           '     18       300   6           6              18  18\n'
           '{} DOF #     GRID    DOF     M   O   Q   B   A   N   G\n'
           '-------  --------  ---     --  --  --  --  --  --  --\n'
           '     19       400   1       1                      19\n'
           '     20       400   2       2                      20\n'
           '     21       400   3       3                      21\n'
           '     22       400   4       4                      22\n'
           '     23       400   5       5                      23\n'
           '     24       400   6       6                      24\n'.
           format(chr(12), chr(12), chr(12)))

    with open(fname) as f:
        tbl = ''.join(f.readlines())
    assert tbl == sbe

    n2p.usetprt(fname, uset, form=0, perpage=8)
    sbe = ('M-set\n'
           '             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n'
           '     1=      400-1      400-2      400-3      400-4      400-5      400-6\n'
           '\n'
           'S-set, R-set, C-set, E-set\n'
           '      -None-\n'
           '\n'
           '{}O-set\n'
           '             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n'
           '     1=      300-1      300-2      300-3      300-4      300-5      300-6\n'
           '\n'
           'Q-set\n'
           '             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n'
           '     1=      200-1      200-2      200-3      200-4      200-5      200-6\n'
           '\n'
           '{}B-set, L-set, T-set\n'
           '             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n'
           '     1=      100-1      100-2      100-3      100-4      100-5      100-6\n'
           '\n'
           'A-set\n'
           '             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n'
           '     1=      100-1      100-2      100-3      100-4      100-5      100-6      200-1      200-2      200-3      200-4 =    10\n'
           '    11=      200-5      200-6\n'
           '\n'
           '{}F-set, N-set\n'
           '             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n'
           '     1=      100-1      100-2      100-3      100-4      100-5      100-6      200-1      200-2      200-3      200-4 =    10\n'
           '    11=      200-5      200-6      300-1      300-2      300-3      300-4      300-5      300-6\n'
           '\n'
           'G-set\n'
           '             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n'
           '     1=      100-1      100-2      100-3      100-4      100-5      100-6      200-1      200-2      200-3      200-4 =    10\n'
           '{}G-set (continued)\n'
           '             -1-        -2-        -3-        -4-        -5-        -6-        -7-        -8-        -9-       -10-\n'
           '    11=      200-5      200-6      300-1      300-2      300-3      300-4      300-5      300-6      400-1      400-2 =    20\n'
           '    21=      400-3      400-4      400-5      400-6\n'
           '\n'.format(chr(12), chr(12), chr(12), chr(12)))

    with open(fname) as f:
        tbl = ''.join(f.readlines())
    os.remove(fname)
    assert tbl == sbe

    with io.StringIO() as f:
        n2p.usetprt(f, uset, form=4, perpage=8,
                    printsets='m, b')
        tbl = f.getvalue()
    sbe = (' DOF #     GRID    DOF     M     B\n'
           '-------  --------  ---     ----  ----\n'
           '      1       100   1               1\n'
           '      2       100   2               2\n'
           '      3       100   3               3\n'
           '      4       100   4               4\n'
           '      5       100   5               5\n'
           '      6       100   6               6\n'
           '{} DOF #     GRID    DOF     M     B\n'
           '-------  --------  ---     ----  ----\n'
           '     19       400   1         1\n'
           '     20       400   2         2\n'
           '     21       400   3         3\n'
           '     22       400   4         4\n'
           '     23       400   5         5\n'
           '     24       400   6         6\n'.format(chr(12)))
    assert tbl == sbe
    assert n2p.usetprt(0, uset, printsets='r') is None
    with io.StringIO() as f:
        n2p.usetprt(f, uset, printsets='b,fe', form=1)
        tbl = f.getvalue()
    sbe = (' DOF #     GRID    DOF     B   FE\n'
           '-------  --------  ---     --  --\n'
           '      1       100   1       1   1\n'
           '      2       100   2       2   2\n'
           '      3       100   3       3   3\n'
           '      4       100   4       4   4\n'
           '      5       100   5       5   5\n'
           '      6       100   6       6   6\n'
           '      7       200   1           7\n'
           '      8       200   2           8\n'
           '      9       200   3           9\n'
           '     10       200   4          10\n'
           '     11       200   5          11\n'
           '     12       200   6          12\n'
           '     13       300   1          13\n'
           '     14       300   2          14\n'
           '     15       300   3          15\n'
           '     16       300   4          16\n'
           '     17       300   5          17\n'
           '     18       300   6          18\n')
    assert sbe == tbl


def test_rbgeom_uset_cylindrical():
    # some cylindrical coords:
    uset = np.array([
        [100,        1,  2097154,        5,       10,       15],
        [100,        2,  2097154,        0,        1,        0],
        [100,        3,  2097154,        0,        0,        0],
        [100,        4,  2097154,        1,        0,        0],
        [100,        5,  2097154,        0,        1,        0],
        [100,        6,  2097154,        0,        0,        1],
        [200,        1,  4194304,        0,        0,        0],
        [200,        2,  4194304,        0,        1,        0],
        [200,        3,  4194304,        0,        0,        0],
        [200,        4,  4194304,        1,        0,        0],
        [200,        5,  4194304,        0,        1,        0],
        [200,        6,  4194304,        0,        0,        1],
        [300,        1,        4,       10,       20,       30],
        [300,        2,        4,        1,        2,        0],
        [300,        3,        4,        0,        0,        0],
        [300,        4,        4,        0,        0,        1],
        [300,        5,        4,        1,        0,        0],
        [300,        6,        4,        0,        1,        0],
        [400,        1,        1,       20,       30,       40],
        [400,        2,        1,        1,        2,        0],
        [400,        3,        1,        0,        0,        0],
        [400,        4,        1,        0,        0,        1],
        [400,        5,        1,        1,        0,        0],
        [400,        6,        1,        0,        1,        0]])

    rb = n2p.rbgeom_uset(uset)
    rb_should_be = np.array([
        [1.0000,         0,         0,         0,   15.0000,  -10.0000],
        [0,    1.0000,         0,  -15.0000,         0,    5.0000],
        [0,         0,    1.0000,   10.0000,   -5.0000,         0],
        [0,         0,         0,    1.0000,         0,         0],
        [0,         0,         0,         0,    1.0000,         0],
        [0,         0,         0,         0,         0,    1.0000],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,    0.5547,    0.8321,         0,   -8.3205,    5.5470],
        [0,   -0.8321,    0.5547,   36.0555,   -5.5470,   -8.3205],
        [1.0000,         0,         0,         0,   30.0000,  -20.0000],
        [0,         0,         0,         0,    0.5547,    0.8321],
        [0,         0,         0,         0,   -0.8321,    0.5547],
        [0,         0,         0,    1.0000,         0,         0],
        [0,    0.6000,    0.8000,   -0.0000,  -16.0000,   12.0000],
        [0,   -0.8000,    0.6000,   50.0000,  -12.0000,  -16.0000],
        [1.0000,         0,         0,         0,   40.0000,  -30.0000],
        [0,         0,         0,         0,    0.6000,    0.8000],
        [0,         0,         0,         0,   -0.8000,    0.6000],
        [0,         0,         0,    1.0000,         0,         0]])

    err = (np.abs(rb - rb_should_be)).max()
    assert err < 1.e-4

    rb = n2p.rbgeom_uset(uset, 300)
    rb_should_be = np.array([
        [1.0000,         0,         0,         0,  -15.0000,   10.0000],
        [0,    1.0000,         0,   15.0000,         0,   -5.0000],
        [0,         0,    1.0000,  -10.0000,    5.0000,         0],
        [0,         0,         0,    1.0000,         0,         0],
        [0,         0,         0,         0,    1.0000,         0],
        [0,         0,         0,         0,         0,    1.0000],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,    0.5547,    0.8321,         0,         0,         0],
        [0,   -0.8321,    0.5547,         0,         0,         0],
        [1.0000,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,    0.5547,    0.8321],
        [0,         0,         0,         0,   -0.8321,    0.5547],
        [0,         0,         0,    1.0000,         0,         0],
        [0,    0.6000,    0.8000,    2.0000,   -8.0000,    6.0000],
        [0,   -0.8000,    0.6000,   14.0000,   -6.0000,   -8.0000],
        [1.0000,         0,         0,         0,   10.0000,  -10.0000],
        [0,         0,         0,         0,    0.6000,    0.8000],
        [0,         0,         0,         0,   -0.8000,    0.6000],
        [0,         0,         0,    1.0000,         0,         0]])

    err = (np.abs(rb - rb_should_be)).max()
    assert err < 1.e-4


def test_rbgeom_uset_spherical():
    # add spherical:
    uset = np.array([
        [100,     1,   2097154,         5.0000,        10.0000,        15.0000],
        [100,     2,   2097154,         1.0000,         2.0000,              0],
        [100,     3,   2097154,              0,              0,              0],
        [100,     4,   2097154,              0,              0,         1.0000],
        [100,     5,   2097154,         1.0000,              0,              0],
        [100,     6,   2097154,              0,         1.0000,              0],
        [200,     1,   4194304,              0,              0,              0],
        [200,     2,   4194304,              0,         1.0000,              0],
        [200,     3,   4194304,              0,              0,              0],
        [200,     4,   4194304,         1.0000,              0,              0],
        [200,     5,   4194304,              0,         1.0000,              0],
        [200,     6,   4194304,              0,              0,         1.0000],
        [300,     1,         4,        10.0000,        20.0000,        30.0000],
        [300,     2,         4,         2.0000,         3.0000,              0],
        [300,     3,         4,         2.0000,         1.9988,         0.0698],
        [300,     4,         4,         0.1005,         0.1394,         0.9851],
        [300,     5,         4,        -0.7636,         0.6456,        -0.0135],
        [300,     6,         4,        -0.6379,        -0.7509,         0.1714],
        [400,     1,         1,        20.0000,        30.0000,        40.0000],
        [400,     2,         1,         1.0000,         2.0000,              0],
        [400,     3,         1,              0,              0,              0],
        [400,     4,         1,              0,              0,         1.0000],
        [400,     5,         1,         1.0000,              0,              0],
        [400,     6,         1,              0,         1.0000,              0]])

    rb = n2p.rbgeom_uset(uset)
    rb_should_be = np.array([
        [0,    0.5547,    0.8321,         0,   -4.1603,    2.7735],
        [0,   -0.8321,    0.5547,   18.0278,   -2.7735,   -4.1603],
        [1.0000,         0,         0,         0,   15.0000,  -10.0000],
        [0,         0,         0,         0,    0.5547,    0.8321],
        [0,         0,         0,         0,   -0.8321,    0.5547],
        [0,         0,         0,    1.0000,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0.2233,    0.5024,    0.8353,    1.6345,   -1.6550,    0.5585],
        [-0.9692,    0.2060,    0.1351,   -3.4774,  -30.4266,   21.4436],
        [-0.1042,   -0.8397,    0.5329,   35.8502,   -8.4547,   -6.3136],
        [0,         0,         0,    0.2233,    0.5024,    0.8353],
        [0,         0,         0,   -0.9692,    0.2060,    0.1351],
        [0,         0,         0,   -0.1042,   -0.8397,    0.5329],
        [0,    0.6000,    0.8000,   -0.0000,  -16.0000,   12.0000],
        [0,   -0.8000,    0.6000,   50.0000,  -12.0000,  -16.0000],
        [1.0000,         0,         0,         0,   40.0000,  -30.0000],
        [0,         0,         0,         0,    0.6000,    0.8000],
        [0,         0,         0,         0,   -0.8000,    0.6000],
        [0,         0,         0,    1.0000,         0,         0]])

    err = (np.abs(rb - rb_should_be)).max()
    assert err < 2.e-3

    rb = n2p.rbgeom_uset(uset, 300)
    rb_should_be = np.array([
        [0,    0.5547,    0.8321,         0,    4.1603,   -2.7735],
        [0,   -0.8321,    0.5547,  -18.0278,    2.7735,    4.1603],
        [1.0000,         0,         0,         0,  -15.0000,   10.0000],
        [0,         0,         0,         0,    0.5547,    0.8321],
        [0,         0,         0,         0,   -0.8321,    0.5547],
        [0,         0,         0,    1.0000,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0,         0,         0,         0,         0,         0],
        [0.2233,    0.5024,    0.8353,         0,         0,         0],
        [-0.9692,    0.2060,    0.1351,         0,         0,         0],
        [-0.1042,   -0.8397,    0.5329,         0,         0,         0],
        [0,         0,         0,    0.2233,    0.5024,    0.8353],
        [0,         0,         0,   -0.9692,    0.2060,    0.1351],
        [0,         0,         0,   -0.1042,   -0.8397,    0.5329],
        [0,    0.6000,    0.8000,    2.0000,   -8.0000,    6.0000],
        [0,   -0.8000,    0.6000,   14.0000,   -6.0000,   -8.0000],
        [1.0000,         0,         0,         0,   10.0000,  -10.0000],
        [0,         0,         0,         0,    0.6000,    0.8000],
        [0,         0,         0,         0,   -0.8000,    0.6000],
        [0,         0,         0,    1.0000,         0,         0]])

    err = (np.abs(rb - rb_should_be)).max()
    assert err < 2.e-3


def test_rbmove():
    grids = np.array([[0., 0., 0.], [30., 10., 20.]])
    rb0 = n2p.rbgeom(grids)
    rb1 = n2p.rbgeom(grids, [2., 4., -5.])
    rb1_b = n2p.rbmove(rb0, [0., 0., 0.], [2., 4., -5.])
    assert np.all(rb1_b == rb1)


def test_addgrid():
    # node 100 in basic is @ [5, 10, 15]
    # node 200 in cylindrical coordinate system is @
    # [r, th, z] = [32, 90, 10]
    cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    sphcoord = np.array([[2, 3, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])
    uset = None
    uset = n2p.addgrid(uset, 100, 'b', 0, [5, 10, 15], 0)
    uset = n2p.addgrid(uset, 200, 'b', cylcoord, [32, 90, 10], cylcoord)
    uset = n2p.addgrid(uset, 300, 'b', sphcoord, [50, 90, 90], sphcoord)
    # get coordinates of node 200 in basic:
    assert np.allclose(np.array([10., 0, 32.]),
                       n2p.getcoords(uset, 200, 0))
    # reverse:
    assert np.allclose([32, 90, 10],
                       n2p.getcoords(uset, [10., 0, 32.], 1))
    assert_raises(ValueError, n2p.addgrid, None, 555, 'b',
                  555, [0, 0, 0], 555)
    assert_raises(ValueError, n2p.addgrid, uset, uset[0, 0], 'b',
                  0, [0, 0, 0], 0)
    uset = n2p.addgrid(None, 1, 'brbccq', 0, [0, 0, 0], 0)
    b = n2p.mkusetmask('b')
    r = n2p.mkusetmask('r')
    c = n2p.mkusetmask('c')
    q = n2p.mkusetmask('q')
    sets = [b, r, b, c, c, q]
    print(uset[:, 2])
    print(np.array(sets))
    assert np.all(uset[:, 2] == np.array(sets))


def test_getcoords():
    # node 100 in basic is @ [5, 10, 15]
    # node 200 in cylindrical coordinate system is @
    # [r, th, z] = [32, 90, 10]
    cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    sphcoord = np.array([[2, 3, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])
    uset = None
    uset = n2p.addgrid(uset, 100, 'b', 0, [5, 10, 15], 0)
    uset = n2p.addgrid(uset, 200, 'b', cylcoord, [32, 90, 10], cylcoord)
    uset = n2p.addgrid(uset, 300, 'b', sphcoord, [50, 90, 45], sphcoord)
    np.set_printoptions(precision=2, suppress=True)

    # check coordinates of node 100:
    assert np.allclose(np.array([5., 10., 15.]), n2p.getcoords(uset, 100, 0))
    rctcoord = np.array([[10, 1, 0], [-2, -8, 9], [-2, -8, 10], [0, -8, 9]])
    assert np.allclose(np.array([5. + 2., 10. + 8., 15. - 9.]),
                       n2p.getcoords(uset, 100, rctcoord))
    r = np.hypot(10., 15.)
    th = math.atan2(15., 10.) * 180. / math.pi
    z = 5.
    gc = n2p.getcoords(uset, 100, 1)
    assert np.allclose([r, th, z], gc)
    r = np.linalg.norm([5., 10., 15.])
    th = math.atan2(np.hypot(15., 5.), 10.) * 180. / math.pi
    phi = math.atan2(5., 15.) * 180. / math.pi
    assert np.allclose(np.array([r, th, phi]), n2p.getcoords(uset, 100, 2))

    # check coordinates of node 200:
    assert np.allclose(np.array([10., 0., 32.]), n2p.getcoords(uset, 200, 0))
    assert np.allclose(np.array([32., 90., 10.]), n2p.getcoords(uset, 200, 1))
    assert np.allclose(np.array([32., 90., 10.]),
                       n2p.getcoords(uset, 200, cylcoord))
    r = np.hypot(10., 32.)
    th = 90.
    phi = math.atan2(10., 32.) * 180 / math.pi
    assert np.allclose(np.array([r, th, phi]), n2p.getcoords(uset, 200, 2))
    assert np.allclose(np.array([r, th, phi]),
                       n2p.getcoords(uset, 200, sphcoord))

    # check coordinates of node 300:
    xb = 50. / math.sqrt(2)
    yb = 0.
    zb = xb
    assert np.allclose(np.array([xb, yb, zb]), n2p.getcoords(uset, 300, 0))
    assert np.allclose(np.array([zb, 90., xb]), n2p.getcoords(uset, 300, 1))
    assert np.allclose(np.array([50., 90., 45.]), n2p.getcoords(uset, 300, 2))
    assert np.allclose(np.array([50., 90., 45.]),
                       n2p.getcoords(uset, 300, sphcoord))

    # one more test to fill gap:
    sphcoord = np.array([[1, 3, 0], [0, 0, 0], [0, 0, 1],
                         [1, 0, 0]])
    uset = None
    uset = n2p.addgrid(uset, 100, 'b', 0, [5, 10, 15], 0)
    uset = n2p.addgrid(uset, 200, 'b', 0, [5, 10, 15], sphcoord)

    # get coordinates of node 100 in spherical (cid 1):
    R = np.linalg.norm([5, 10, 15])
    phi = math.atan2(10, 5) * 180 / math.pi
    xy_rad = np.linalg.norm([5, 10])
    th = 90. - math.acos(xy_rad / R) * 180 / math.pi
    assert np.allclose(n2p.getcoords(uset, 100, 1), [R, th, phi])


def test_rbcoords():
    assert_raises(ValueError, n2p.rbcoords, np.random.randn(3, 4))
    assert_raises(ValueError, n2p.rbcoords, np.random.randn(13, 6))


def gettestuset():
    # z = x-basic; r = y-basic
    cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    sphcoord = np.array([[2, 3, 1], [2, 2, 2], [2, 7, 3], [-5, 19, 24]])
    coords = {100: [[5, 10, 15], 'b', cylcoord],
              200: [[0, 0, 0],   'q', 0],
              300: [[10, 20, 30], 'o', sphcoord],
              400: [[20, 30, 40], 'm', cylcoord]}
    n = len(coords)
    uset = np.zeros((n * 6, 6))
    coordref = {}
    for i, id in enumerate(sorted(coords)):
        j = i * 6
        loc = coords[id][0]
        dofset = coords[id][1]
        csys = coords[id][2]
        uset[j:j + 6, :] = n2p.addgrid(None, id, dofset, csys, loc,
                                       csys, coordref)
    return uset


def test_mksetpv_mkdofpv():
    uset = gettestuset()
    assert_raises(ValueError, n2p.mksetpv, uset, 'm', 'b')
    pv, outdof = n2p.mkdofpv(uset, 'f', [[100, 3], [200, 5],
                                         [300, 16], [300, 4]])
    assert np.all(pv == np.array([2, 10, 12, 17, 15]))
    assert np.all(outdof == np.array([[100,   3],
                                      [200,   5],
                                      [300,   1],
                                      [300,   6],
                                      [300,   4]]))
    assert_raises(ValueError, n2p.mkdofpv, uset, 'f',
                  [[100, 3], [200, 5], [300, 1], [300, 4], [400, 123]])


def test_coordcardinfo():
    uset = np.zeros((2, 2))
    ci = n2p.coordcardinfo(uset)
    assert ci == {}
    uset2 = n2p.addgrid(None, 1, 'b', 0, [0, 0, 0], 0)
    ci = n2p.coordcardinfo(uset2)
    assert ci == {}
    assert_raises(ValueError, n2p.coordcardinfo, uset, 5)

    uset = gettestuset()
    assert_raises(ValueError, n2p.coordcardinfo, uset, 5)

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
    nasdata = matlab.loadmat('pyyeti/tests/nastran_gm_data/'
                             'make_gm_nx9_rbe3_1.mat')
    uset = nasdata['uset'][0][0][0]
    gm = nasdata['gm'][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata['drg'][0][0][0].T
    pyuset = nastran.bulk2uset('pyyeti/tests/nastran_gm_data/'
                               'make_gm_nx9_rbe3_1.dat')[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(pyuset, 124, 123456,
                        [[123, 2.3], 100,
                         [123, 2.5], 200,
                         [23, 12.],  300,
                         [34, .5],   400,
                         [456, .4], [1, 2, 3],
                         [136, 5.5], [101, 102, 103],
                         [123456, 4.2], [111, 112, 113],
                         [25, .05], [121, 122, 123]])
    pyuset[:, 2] = uset[:, 2]  # so set-membership gets ignored
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
    nasdata = matlab.loadmat('pyyeti/tests/nastran_gm_data/'
                             'make_gm_nx9_rbe3_2.mat')
    uset = nasdata['uset'][0][0][0]
    gm = nasdata['gm'][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata['drg'][0][0][0].T
    pyuset = nastran.bulk2uset('pyyeti/tests/nastran_gm_data/'
                               'make_gm_nx9_rbe3_2.dat')[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(pyuset, 124, 123456,
                        [[123, 2.3], 100,
                         [123, 2.5], 200,
                         [23, 12.],  300,
                         [34, .5],   400,
                         [456, .4], [1, 2, 3],
                         [136, 5.5], [101, 102, 103],
                         [123456, 4.2], [111, 112, 113],
                         [25, .05], [121, 122, 123]])
    pyuset[:, 2] = uset[:, 2]  # so set-membership gets ignored
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
    nasdata = matlab.loadmat('pyyeti/tests/nastran_gm_data/'
                             'make_gm_nx9_rbe3_3.mat')
    uset = nasdata['uset'][0][0][0]
    gm = nasdata['gm'][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata['drg'][0][0][0].T
    pyuset = nastran.bulk2uset('pyyeti/tests/nastran_gm_data/'
                               'make_gm_nx9_rbe3_3.dat')[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(pyuset, 124, 123456,
                        [[123, 2.3], 100,
                         [123, 2.5], 200,
                         [23, 12.],  300,
                         [34, .5],   400,
                         [456, .4], [1, 2, 3],
                         [136, 5.5], [101, 102, 103],
                         [123456, 4.2], [111, 112, 113],
                         [25, .05], [121, 122, 123]])
    pyuset[:, 2] = uset[:, 2]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_4():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     1346    2.6     123     100
    #         1.8     456     200
    # load the data from above modes run:
    nasdata = matlab.loadmat('pyyeti/tests/nastran_gm_data/'
                             'make_gm_nx9_rbe3_4.mat')
    uset = nasdata['uset'][0][0][0]
    gm = nasdata['gm'][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata['drg'][0][0][0].T
    pyuset = nastran.bulk2uset('pyyeti/tests/nastran_gm_data/'
                               'make_gm_nx9_rbe3_4.dat')[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(pyuset, 124, 1346,
                        [[123, 2.6], 100,
                         [456, 1.8], 200])
    pyuset[:, 2] = uset[:, 2]  # so set-membership gets ignored
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
    nasdata = matlab.loadmat('pyyeti/tests/nastran_gm_data/'
                             'make_gm_nx9_rbe3_um_1.mat')
    uset = nasdata['uset'][0][0][0]
    gm = nasdata['gm'][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata['drg'][0][0][0].T
    pyuset = nastran.bulk2uset('pyyeti/tests/nastran_gm_data/'
                               'make_gm_nx9_rbe3_um_1.dat')[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(pyuset, 124, 123456,
                        [[123, 2.3], 100,
                         [123, 2.5], 200,
                         [23, 12.],  300,
                         [34, .5],   400,
                         [456, .4], [1, 2, 3],
                         [136, 5.5], [101, 102, 103],
                         [123456, 4.2], [111, 112, 113],
                         [25, .05], [121, 122, 123]],
                        [100, 2, 111, 2346, 122, 5])
    pyuset[:, 2] = uset[:, 2]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_UM_2():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     123456  1.0     123456  100
    #         UM      124     123     200     456
    # load the data from above modes run:
    nasdata = matlab.loadmat('pyyeti/tests/nastran_gm_data/'
                             'make_gm_nx9_rbe3_um_2.mat')
    uset = nasdata['uset'][0][0][0]
    gm = nasdata['gm'][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata['drg'][0][0][0].T
    pyuset = nastran.bulk2uset('pyyeti/tests/nastran_gm_data/'
                               'make_gm_nx9_rbe3_um_2.dat')[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(pyuset, 124, 123456,
                        [[123, 2.6], 100,
                         [456, 1.8], 200],
                        [124, 123, 200, 456])
    pyuset[:, 2] = uset[:, 2]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_UM_3():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     123456  2.6     123     100
    #         1.8     456     200
    #         UM      124     152346
    # load the data from above modes run:
    nasdata = matlab.loadmat('pyyeti/tests/nastran_gm_data/'
                             'make_gm_nx9_rbe3_um_3.mat')
    uset = nasdata['uset'][0][0][0]
    gm = nasdata['gm'][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata['drg'][0][0][0].T
    pyuset = nastran.bulk2uset('pyyeti/tests/nastran_gm_data/'
                               'make_gm_nx9_rbe3_um_3.dat')[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(pyuset, 124, 123456,
                        [[123, 2.6], 100,
                         [456, 1.8], 200],
                        [124, 152346])
    pyuset[:, 2] = uset[:, 2]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_UM_4():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     1346    2.6     123     100
    #         1.8     456     200
    #         UM      124     6341
    # load the data from above modes run:
    # same as test 4:
    nasdata = matlab.loadmat('pyyeti/tests/nastran_gm_data/'
                             'make_gm_nx9_rbe3_4.mat')
    uset = nasdata['uset'][0][0][0]
    gm = nasdata['gm'][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata['drg'][0][0][0].T
    pyuset = nastran.bulk2uset('pyyeti/tests/nastran_gm_data/'
                               'make_gm_nx9_rbe3_4.dat')[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(pyuset, 124, 1346,
                        [[123, 2.6], 100,
                         [456, 1.8], 200],
                        [124, 6341])
    pyuset[:, 2] = uset[:, 2]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_UM_5():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     1346    2.6     123     100
    #         1.8     456     200
    #         UM      100     12      200     56
    # load the data from above modes run:
    nasdata = matlab.loadmat('pyyeti/tests/nastran_gm_data/'
                             'make_gm_nx9_rbe3_um_5.mat')
    uset = nasdata['uset'][0][0][0]
    gm = nasdata['gm'][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata['drg'][0][0][0].T
    pyuset = nastran.bulk2uset('pyyeti/tests/nastran_gm_data/'
                               'make_gm_nx9_rbe3_um_5.dat')[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(pyuset, 124, 1346,
                        [[123, 2.6], 100,
                         [456, 1.8], 200],
                        [100, 12, 200, 56])
    pyuset[:, 2] = uset[:, 2]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_formrbe3_UM_6():
    # grid, 124, 12, 31., 4., -165., 12
    # RBE3    1               124     1346    2.6     123     100
    #         1.8     456     200
    #         UM      100     12      200     5       124     5
    # load the data from above modes run:
    nasdata = matlab.loadmat('pyyeti/tests/nastran_gm_data/'
                             'make_gm_nx9_rbe3_um_6.mat')
    uset = nasdata['uset'][0][0][0]
    gm = nasdata['gm'][0][0][0]
    pv = np.any(gm, axis=0)
    gmmod = gm[:, pv]
    drg = nasdata['drg'][0][0][0].T
    pyuset = nastran.bulk2uset('pyyeti/tests/nastran_gm_data/'
                               'make_gm_nx9_rbe3_um_6.dat')[0]
    pydrg = n2p.rbgeom_uset(pyuset, 124)
    assert np.allclose(drg, pydrg)
    pygm = n2p.formrbe3(pyuset, 124, 1346,
                        [[123, 2.6], 100,
                         [456, 1.8], 200],
                        [100, 12, 200, 5, 124, 6])
    pyuset[:, 2] = uset[:, 2]  # so set-membership gets ignored
    assert np.allclose(uset, pyuset)
    assert np.allclose(gmmod, pygm)


def test_upasetpv():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
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
    shouldbe = np.array([31, 32, 33, 34, 35, 36,
                         37, 38, 39, 40, 41, 42,
                         13, 14, 15, 16, 17, 18,
                         19, 20, 21, 22, 23, 24,
                         65, 66, 67, 68, 69, 70,
                         71, 72, 73, 74, 75, 76]) - 1
    assert np.all(pv == shouldbe)


def test_upqsetpv():
    nas_extseout = nastran.rdnas2cam(
        'pyyeti/tests/nas2cam_extseout/nas2cam')
    pv_extseout = nastran.upqsetpv(nas_extseout, 0)

    ue = nas_extseout['uset'][0]
    assert np.all((ue[:, 0] > 200) == pv_extseout)

    nas_csuper = nastran.rdnas2cam(
        'pyyeti/tests/nas2cam_csuper/nas2cam')
    pv_csuper = nastran.upqsetpv(nas_csuper)

    uc = nas_csuper['uset'][0]
    # last four are dummy dof from grids:
    pv = (uc[:, 0] > 200)
    pv[-4:] = False
    assert np.all(pv == pv_csuper)

    assert_raises(ValueError, nastran.upqsetpv, nas_csuper, 1000)


def test_formtran1_seup():
    o4 = op4.OP4()
    tug1 = nastran.rddtipch('pyyeti/tests/nas2cam_extseout/outboard.pch')
    mug1 = o4.listload(
        'pyyeti/tests/nas2cam_extseout/outboard.op4', 'mug1')[1][0]
    grids = [[11, 123456],
             [45, 123456],
             [60, 123456],
             [1995002, 1]]
    pv, exp_dof = n2p.mkdofpv(tug1, 'p', grids)
    MUG1 = mug1[pv, :]
    nas_csuper = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    exp_dof0 = exp_dof
    exp_dof0[-1, 1] = 0
    drm101, dof101 = n2p.formtran(nas_csuper, 101, exp_dof0)
    assert np.allclose(np.abs(drm101), np.abs(MUG1))


def test_formtran2_seup():
    o4 = op4.OP4()
    tug1 = nastran.rddtipch('pyyeti/tests/nas2cam_extseout/outboard.pch')
    mug1 = o4.listload(
        'pyyeti/tests/nas2cam_extseout/outboard.op4', 'mug1')[1][0]
    grids = [11, 45, 60]
    pv, exp_dof = n2p.mkdofpv(tug1, 'p', grids)
    MUG1 = mug1[pv, :]
    nas_csuper = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    drm101, dof101 = n2p.formtran(nas_csuper, 101, exp_dof)
    assert np.allclose(np.abs(drm101), np.abs(MUG1))


def test_formtran3_seup():
    o4 = op4.OP4()
    tug1 = nastran.rddtipch('pyyeti/tests/nas2cam_extseout/outboard.pch')
    mug1 = o4.listload(
        'pyyeti/tests/nas2cam_extseout/outboard.op4', 'mug1')[1][0]
    grids = [[11, 45]]
    pv, exp_dof = n2p.mkdofpv(tug1, 'p', grids)
    MUG1 = mug1[pv, :]
    nas_csuper = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    drm101, dof101 = n2p.formtran(nas_csuper, 101, exp_dof)
    assert np.allclose(np.abs(drm101), np.abs(MUG1))


def test_formtran4_seup():
    o4 = op4.OP4()
    tug1 = nastran.rddtipch('pyyeti/tests/nas2cam_extseout/outboard.pch')
    mug1 = o4.listload(
        'pyyeti/tests/nas2cam_extseout/outboard.op4', 'mug1')[1][0]
    grids = [[60, 4]]
    pv, exp_dof = n2p.mkdofpv(tug1, 'p', grids)
    MUG1 = mug1[pv, :]
    nas_csuper = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    drm101, dof101 = n2p.formtran(nas_csuper, 101, exp_dof)
    assert np.allclose(np.abs(drm101), np.abs(MUG1))


def test_formtran5_seup():
    o4 = op4.OP4()
    tug1 = nastran.rddtipch('pyyeti/tests/nas2cam_extseout/outboard.pch')
    mug1 = o4.listload(
        'pyyeti/tests/nas2cam_extseout/outboard.op4', 'mug1')[1][0]
    # put recovery in non-ascending order:
    grids = [[11,       3],
             [60,       6],
             [60,       4],
             [1995002,       1],
             [11,       5],
             [11,       4],
             [60,       5],
             [45,       5],
             [45,       1],
             [11,       2],
             [60,       2],
             [45,       4],
             [60,       1],
             [45,       6],
             [45,       3],
             [11,       6],
             [11,       1],
             [60,       3],
             [45,       2]]
    pv, exp_dof = n2p.mkdofpv(tug1, 'p', grids)
    MUG1 = mug1[pv, :]
    exp_dof0 = exp_dof.copy()
    pv = exp_dof0[:, 0] == 1995002
    exp_dof0[pv, 1] = 0
    nas_csuper = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    drm101, dof101 = n2p.formtran(nas_csuper, 101, exp_dof0)
    assert np.allclose(np.abs(drm101), np.abs(MUG1))
    assert np.all(exp_dof == np.array(grids))


def test_formtran1_se0():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_extseout/nas2cam')
    grids = [[3, 123456],
             [27, 123456],
             [70, 123456],
             [2995004, 0],
             [2995005, 0]]
    drm_a, dof_a = n2p.formtran(nas, 0, grids)  # use phg
    nas.pop('phg', None)
    drm_b, dof_b = n2p.formtran(nas, 0, grids)  # use pha
    assert np.allclose(drm_a, drm_b)
    dof = np.array([[3,       1],
                    [3,       2],
                    [3,       3],
                    [3,       4],
                    [3,       5],
                    [3,       6],
                    [27,       1],
                    [27,       2],
                    [27,       3],
                    [27,       4],
                    [27,       5],
                    [27,       6],
                    [70,       1],
                    [70,       2],
                    [70,       3],
                    [70,       4],
                    [70,       5],
                    [70,       6],
                    [2995004,       0],
                    [2995005,       0]])
    assert np.all(dof_b == dof)


def test_formtran2_se0():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_extseout/nas2cam')
    grids = [[2995004,       0],
             [70,       3],
             [3,       3],
             [70,       5],
             [27,       4],
             [70,       6],
             [27,       6],
             [70,       4],
             [27,       2],
             [70,       2],
             [3,       2],
             [70,       1]]
    drm_a, dof_a = n2p.formtran(nas, 0, grids)  # use phg
    nas.pop('phg', None)
    drm_b, dof_b = n2p.formtran(nas, 0, grids)  # use pha
    assert np.allclose(drm_a, drm_b)
    assert np.all(dof_b == np.array(grids))


def test_formtran3_se0():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_extseout/nas2cam')
    grids = [[2995004,       0],
             [70,       3],
             [3,       3],
             [70,       5],
             [27,       4],
             [70,       6],
             [27,       6],
             [70,       4],
             [27,       2],
             [70,       2],
             [3,       2],
             [70,       1]]
    drm_a, dof_a = n2p.formtran(nas, 0, grids)
    drm_b_gset, dof_b = n2p.formtran(nas, 0, grids, gset=True)
    drm_b = np.dot(drm_b_gset, nas['phg'][0])
    assert np.allclose(drm_a, drm_b)
    assert np.all(dof_b == np.array(grids))


def test_formulvs_1():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_extseout/nas2cam')
    ulvs = n2p.formulvs(nas, 101, 101)
    assert ulvs == 1.
    ulvs = n2p.formulvs(nas, 101)
    assert np.allclose(ulvs, nas['ulvs'][101])
    ulvs = n2p.formulvs(nas, 102)
    assert np.allclose(ulvs, nas['ulvs'][102])
    ulvs = n2p.formulvs(nas, 101, shortcut=0, keepcset=0)
    assert np.allclose(ulvs, nas['ulvs'][101])
    ulvs = n2p.formulvs(nas, 102, shortcut=0, keepcset=0)
    assert np.allclose(ulvs, nas['ulvs'][102])
    assert 1. == n2p.formulvs(nas, seup=101, sedn=101)
    old = nas['ulvs']
    del nas['ulvs']
    n2p.addulvs(nas, 101, 102)
    for se in [101, 102]:
        assert np.allclose(old[se], nas['ulvs'][se])

    old = {i: v for i, v in nas['ulvs'].items()}
    n2p.addulvs(nas, 101, 102)
    for se in [101, 102]:
        assert id(old[se]) == id(nas['ulvs'][se])

    n2p.addulvs(nas, 101, 102, shortcut=False)
    for se in [101, 102]:
        assert id(old[se]) != id(nas['ulvs'][se])
        assert np.allclose(old[se], nas['ulvs'][se])


def test_formulvs_2():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_extseout/nas2cam')
    ulvsgset = n2p.formulvs(nas, 101, gset=True)
    ulvs = np.dot(ulvsgset, nas['phg'][0])
    assert np.allclose(ulvs, nas['ulvs'][101])

    ulvsgset = n2p.formulvs(nas, 102, gset=True)
    ulvs = np.dot(ulvsgset, nas['phg'][0])
    assert np.allclose(ulvs, nas['ulvs'][102])


def test_rdnas2cam_no_se():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam/no_se_nas2cam')
    assert np.all(nas['selist'] == [0, 0])


def test_formulvs_multilevel():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam/with_se_nas2cam')
    old = nas['ulvs']
    del nas['ulvs']
    ses = [100, 200, 300, 400]
    n2p.addulvs(nas, *ses)
    for se in ses:
        assert np.allclose(old[se], nas['ulvs'][se])
    u300_100 = n2p.formulvs(nas, seup=300, sedn=100, keepcset=True)
    u300 = u300_100.dot(nas['ulvs'][100])
    assert np.allclose(u300, nas['ulvs'][300])
    u300_100 = n2p.formulvs(nas, seup=300, sedn=100, keepcset=False)
    assert u300_100.shape[1] < nas['ulvs'][100].shape[0]


def test_formdrm_1():
    grids = [[11, 123456],
             [45, 123456],
             [60, 123456],
             [1995002, 0]]
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    drm101, dof101 = n2p.formtran(nas, 101, grids)
    ulvs = n2p.formulvs(nas, 101)
    DRM_A = np.dot(drm101, ulvs)

    DRM_B, dof101_b = n2p.formdrm(nas, 101, grids)
    assert np.allclose(DRM_A, DRM_B)
    assert np.all(dof101 == dof101_b)

    del nas['phg']
    with assert_raises(RuntimeError) as cm:
        ulvs = n2p.formulvs(nas, 101)
    the_msg = str(cm.exception)
    assert 0 == the_msg.find("neither nas['phg'][0]")
    assert_raises(RuntimeError, n2p.formulvs, nas, 101)


def test_formdrm_2():
    grids = [[11, 123456],
             [45, 123456],
             [60, 123456],
             [1995002, 0]]
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    drm101, dof101 = n2p.formtran(nas, 101, grids, gset=True)
    ulvs = n2p.formulvs(nas, 101, gset=True)
    DRM_A = np.dot(drm101, ulvs)

    DRM_B, dof101_b = n2p.formdrm(nas, 101, grids, gset=True)
    assert np.allclose(DRM_A, DRM_B)
    assert np.all(dof101 == dof101_b)


def test_formdrm_oset_sset():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam/with_se_nas2cam')
    drm, dof = n2p.formdrm(nas, 100, 11)  # just o-set
    pv = n2p.mksetpv(nas['uset'][0], 'g', 'a')
    pha = nas['phg'][0][pv]
    del nas['phg']
    nas['pha'] = {0: pha}
    drm, dof = n2p.formdrm(nas, 0, 306)   # just s-set


def test_formdrm_noqset():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam/with_se_nas2cam')
    # modify 100 so it is a static reduction only:
    del nas['lambda'][100]
    del nas['goq']
    # move 'q' set into 's' set:
    s = n2p.mkusetmask('s')
    q = n2p.mksetpv(nas['uset'][100], 'g', 'q')
    nas['uset'][100][q, 2] = s
    drm, dof = n2p.formdrm(nas, seup=100, sedn=100, dof=11)


def test_fromdrm_null_cset():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    ulvs = n2p.formulvs(nas, 102, keepcset=1, shortcut=0)
    nas['ulvs'] = {102: ulvs}
    drm1, dof = n2p.formdrm(nas, seup=102, sedn=0, dof=3)

    nas = op2.rdnas2cam('pyyeti/tests/nas2cam_csuper/nas2cam')
    ulvs = n2p.formulvs(nas, 102, keepcset=0, shortcut=0)
    nas['ulvs'] = {102: ulvs}
    drm2, dof = n2p.formdrm(nas, seup=102, sedn=0, dof=3)
    assert np.allclose(drm1, drm2)


def test_build_coords():
    #      [cid, ctype, refcid, a1, a2, a3, b1, b2, b3, c1, c2, c3]
    cords = [[10, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
             [20, 2, 10, 0, 0, 0, 0, 0, 1, 1, 0, 0]]
    dct1 = n2p.build_coords(cords)

    cords = [[10, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
             [20, 2, 10, 0, 0, 0, 0, 0, 1, 1, 0, 0],
             [20, 2, 10, 0, 0, 0, 0, 0, 1, 1, 0, 0]]
    dct2 = n2p.build_coords(cords)
    assert dir(dct1) == dir(dct2)
    for k in dct1:
        assert np.all(dct1[k] == dct2[k])

    cords = [[10, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
             [20, 2, 10, 0, 0, 0, 0, 0, 1, 1, 0, 0],
             [20, 2, 10, 0, 0, 0, 0, 0, 1, 1, 1, 0]]

    assert_raises(RuntimeError, n2p.build_coords, cords)

    cords = [[10, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
             [20, 2, 5, 0, 0, 0, 0, 0, 1, 1, 0, 0]]

    assert_raises(RuntimeError, n2p.build_coords, cords)


def test_rbmodes_allq():
    # node 100 in basic is @ [5, 10, 15]
    # node 200 in cylindrical coordinate system is @
    # [r, th, z] = [32, 90, 10]
    cylcoord = np.array([[1, 2, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    sphcoord = np.array([[2, 3, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])
    uset = None
    uset = n2p.addgrid(uset, 100, 'q', 0, [5, 10, 15], 0)
    uset = n2p.addgrid(uset, 200, 'q', cylcoord, [32, 90, 10], cylcoord)
    uset = n2p.addgrid(uset, 300, 'q', sphcoord, [50, 90, 90], sphcoord)
    rb = n2p.rbgeom_uset(uset)
    assert np.all(rb == 0)

    uset2 = None
    uset2 = n2p.addgrid(uset2, 100, 'q', 0, [5, 10, 15], 0)
    uset2 = n2p.addgrid(uset2, 300, 'q', sphcoord, [50, 90, 90], sphcoord)
    uset2 = n2p.addgrid(uset2, 200, 'q', cylcoord, [32, 90, 10], cylcoord)
    assert np.allclose(uset, uset2)


def test_formdrm_go_warnings():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam/with_se_nas2cam')
    # make sure that got and goq is not present for se 300
    try:
        del nas['got'][300]
    except KeyError:
        pass
    try:
        del nas['goq'][300]
    except KeyError:
        pass
    q = sum(n2p.mksetpv(nas['uset'][300], "g", "q"))
    o = sum(n2p.mksetpv(nas['uset'][300], "g", "o"))
    t = sum(n2p.mksetpv(nas['uset'][300], "g", "t"))
    goq = np.zeros((o, q))
    got = np.zeros((o, t))

    # without the following, when pandas is imported, we get:
    # ERROR: test_n2p_nose.test_badrbe3_error
    # ----------------------------------------------------------------------
    # Traceback (most recent call last):
    #   File "/home/macro/anaconda3/lib/python3.5/site-packages/nose/case.py", line 198, in runTest
    #     self.test(*self.arg)
    #   File "/home/macro/code/pyyeti/pyyeti/tests/test_n2p_nose.py", line 1380, in test_badrbe3_error
    #     with assert_warns(RuntimeWarning) as cm:
    #   File "/home/macro/anaconda3/lib/python3.5/unittest/case.py", line 225, in __enter__
    #     for v in sys.modules.values():
    # RuntimeError: dictionary changed size during iteration
    #
    # Reported here: https://github.com/pytest-dev/pytest/issues/1288
    import sys
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}

    nas['got'][300] = got
    with assert_warns(RuntimeWarning) as cm:
        drm, dof = n2p.formdrm(nas, 300, [38, 39])
    the_warning = str(cm.warning)
    assert 0 == the_warning.find("nas['goq'][300] not found")
    del nas['got'][300]

    nas['goq'][300] = goq
    with assert_warns(RuntimeWarning) as cm:
        drm, dof = n2p.formdrm(nas, 300, [38, 39])
    the_warning = str(cm.warning)
    assert 0 == the_warning.find("nas['got'][300] not found")
    del nas['goq'][300]


def test_badrbe3_error():
    # put some grids on the x-axis and build a bad rbe3 to test for
    # the 'poorly conditioned' message:
    x = np.arange(0, 5, 1.)
    uset = np.zeros((6 * len(x), 6), float)
    for i in range(len(x)):
        j = i * 6
        uset[j:j + 6, :] = n2p.addgrid(None, i + 1, 'b',
                                       0, [x[i], 0, 0], 0)
    uset = n2p.addgrid(uset, 100, 'b', 0, [5, 5, 5], 0)
    with assert_warns(RuntimeWarning) as cm:
        assert_raises(la.LinAlgError, n2p.formrbe3, uset, 100, 123456,
                      [123, [1, 2, 3, 4, 5]])
    the_warning = str(cm.warning)
    assert 0 == the_warning.find("matrix is poorly conditioned")


def test_badrbe3_warn():
    # put some grids on the x-axis and build a bad rbe3 to test for
    # the 'poorly conditioned' message:
    x = np.arange(0, 5, 1.)
    uset = np.zeros((6 * len(x), 6), float)
    for i in range(len(x)):
        j = i * 6
        if i == 4:
            z = 0.00000000000001
        else:
            z = 0.
        uset[j:j + 6, :] = n2p.addgrid(None, i + 1, 'b',
                                       0, [x[i], 0, z], 0)
    uset = n2p.addgrid(uset, 100, 'b', 0, [5, 5, 5], 0)
    with assert_warns(RuntimeWarning) as cm:
        rbe3 = n2p.formrbe3(uset, 100, 123456,
                            [123, [1, 2, 3, 4, 5]])
    the_warning = str(cm.warning)
    assert 0 == the_warning.find("matrix is poorly conditioned")


def test_rbe3_badum():
    x = np.arange(0, 2, 1.)
    uset = np.zeros((6 * len(x), 6), float)
    for i in range(len(x)):
        j = i * 6
        uset[j:j + 6, :] = n2p.addgrid(None, i + 1, 'b',
                                       0, [x[i], 0, 0], 0)
    uset = n2p.addgrid(uset, 100, 'b', 0, [5, 5, 5], 0)
    with assert_raises(ValueError) as cm:
        rbe3 = n2p.formrbe3(uset, 100, 123456,
                            [123456, [1, 2]], [1, 234])
    the_msg = str(cm.exception)
    assert 0 == the_msg.find("incorrect size of M-set")


def test_bad_se():
    nas = op2.rdnas2cam('pyyeti/tests/nas2cam/with_se_nas2cam')
    assert_raises(ValueError, n2p._findse, nas, 345)


def test_sph_zero_theta():
    sphcoord = np.array([[2, 3, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0]])
    uset = n2p.addgrid(None, 1, 'b', 0, [0, 0, 0], sphcoord)
    rb = n2p.rbgeom_uset(uset)
    rb2 = np.array([[0., 0., 1., 0., 0., 0.],
                    [1., 0., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 1.],
                    [0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 1., 0.]])
    assert np.allclose(rb, rb2)
