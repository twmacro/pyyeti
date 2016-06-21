import numpy as np
import h5py
import tempfile
import os
from pyyeti.pp import PP
from nose.tools import *


def test_h5py():
    # write and read a file:
    f = tempfile.NamedTemporaryFile(delete=False)
    name = f.name
    f.close()

    a = np.arange(18).reshape(2, 3, 3)
    b = np.arange(3)
    with h5py.File(name, 'w') as F:
        F.create_dataset('varA', data=a)
        F.create_dataset('varB', data=b)

    with h5py.File(name, 'r') as F:
        o = PP(F)
    os.remove(name)

    sbe = ["<class 'h5py._hl.files.File'>[n=2]",
           "    'varA': H5 int64 ndarray 18 elems: (2, 3, 3)",
           "    'varB': H5 int64 ndarray 3 elems: (3,) [0 1 2]",
           ]

    assert o.output == '\n'.join(sbe)+'\n'


def test_long_list():
    o = PP(['val {}'.format(i) for i in range(1000)])
    assert o.output == ("[n=1000]: ['val 0', 'val 1', 'val 2',"
                        " 'val 3', 'val 4', 'val 5', 'val 6', "
                        "'v ...\n")


def test_long_string():
    o = PP('a'*100)
    assert o.output == ("'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ...\n")


def test_array():
    np.set_printoptions(edgeitems=3,infstr='inf',
                        linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)
    o = PP(np.arange(9).reshape(3, -1)+5.005)
    assert o.output == ('float64 ndarray 9 elems: (3, 3) [[  5.005 '
                        '  6.005   7. <...> 5] [ 11.005  12.005  '
                        '13.005]]\n')
