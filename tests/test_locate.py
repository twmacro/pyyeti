import numpy as np
from pyyeti import locate
from nose.tools import *


def test_find_rows():
    a = [[1, 2, 3], [4, 5, 6]]
    b = [1, 2, 3, 4]
    assert locate.find_rows(a, b).size == 0


def test_mat_intersect():
    a, b = locate.mat_intersect([1, 2, 3, 4, 5], [4, 2, 8])
    assert np.all(a == np.array([3, 1]))
    assert np.all(b == np.array([0, 1]))

    a, b = locate.mat_intersect([1, 2, 3, 4, 5], [4, 2, 8], keep=1)
    assert np.all(a == np.array([1, 3]))
    assert np.all(b == np.array([1, 0]))

    a, b = locate.mat_intersect([1, 2, 3, 4, 5], [4, 2, 8], keep=2)
    assert np.all(a == np.array([3, 1]))
    assert np.all(b == np.array([0, 1]))

    a, b = locate.mat_intersect([1, 2, 3], [1, 2, 3, 1], keep=2)
    assert np.all(a == [0, 1, 2, 0])
    assert np.all(b == [0, 1, 2, 3])

    a, b = locate.mat_intersect([1, 2, 3], [1, 2, 3, 1], keep=1)
    assert np.all(a == [0, 1, 2])
    assert np.all(b == [0, 1, 2])

    a, b = locate.mat_intersect([1, 2, 3], [1, 2, 3, 1])
    assert np.all(a == [0, 1, 2])
    assert np.all(b == [0, 1, 2])

    a, b = locate.mat_intersect([1, 2, 3], [100, 200, 300, 100])
    assert len(a) == len(b) == 0

    # now with floating point:
    a, b = locate.mat_intersect([1.0, 2, 3, 4, 5], [4, 2, 8.0])
    assert np.all(a == np.array([3, 1]))
    assert np.all(b == np.array([0, 1]))

    a, b = locate.mat_intersect([1, 2, 3, 4, 5], [4, 2, 8.0], keep=1)
    assert np.all(a == np.array([1, 3]))
    assert np.all(b == np.array([1, 0]))

    a, b = locate.mat_intersect([1, 2, 3, 4, 5], [4, 2.0, 8], keep=2)
    assert np.all(a == np.array([3, 1]))
    assert np.all(b == np.array([0, 1]))

    a, b = locate.mat_intersect([1, 2, 3], [1, 2, 3, 1.0], keep=2)
    assert np.all(a == [0, 1, 2, 0])
    assert np.all(b == [0, 1, 2, 3])

    a, b = locate.mat_intersect([1, 2, 3], [1, 2, 3, 1.0], keep=1)
    assert np.all(a == [0, 1, 2])
    assert np.all(b == [0, 1, 2])

    a, b = locate.mat_intersect([1, 2, 3], [1, 2, 3, 1.0])
    assert np.all(a == [0, 1, 2])
    assert np.all(b == [0, 1, 2])

    a, b = locate.mat_intersect([1, 2, 3], [100, 200, 300, 100.0])
    assert len(a) == len(b) == 0


def test_locate_misc():
    mat1 = np.array([[7, 3], [6, 8], [4, 0], [9, 2], [1, 5]])
    mat2 = np.array([[9, 2], [1, 5], [7, 3]])
    pv1, pv2 = locate.mat_intersect(mat1, mat2)
    assert np.all(np.array([3, 4, 0]) == pv1)
    assert np.all(np.array([0, 1, 2]) == pv2)
    pv1, pv2 = locate.mat_intersect(mat1, mat2, 1)
    assert np.all(np.array([0, 3, 4]) == pv1)
    assert np.all(np.array([2, 0, 1]) == pv2)
    pv1, pv2 = locate.mat_intersect(mat1, mat2, 2)
    assert np.all(np.array([3, 4, 0]) == pv1)
    assert np.all(np.array([0, 1, 2]) == pv2)
    pv = np.array([0, 3, 5])
    tf = locate.index2bool(pv, 8)
    assert np.all(np.array([True, False, False, True, False, True, False, False]) == tf)
