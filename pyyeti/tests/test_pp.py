import numpy as np
import h5py
import tempfile
import os
from collections import OrderedDict
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

    assert o.output.replace('int32', 'int64') == '\n'.join(sbe)+'\n'


def test_long_list():
    o = PP(['val {}'.format(i) for i in range(1000)])
    assert o.output == ("[n=1000]: ['val 0', 'val 1', 'val 2', "
                        "... , 'val 24', 'val 25', 'val 26',  ...]\n")

def test_long_string():
    o = PP('a'*100)
    assert o.output == ("'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ... "
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\n")


def test_array():
    np.set_printoptions(edgeitems=3,infstr='inf',
                        linewidth=75, nanstr='nan', precision=8,
                        suppress=False, threshold=1000, formatter=None)
    o = PP(np.arange(9).reshape(3, -1)+5.005)
    assert o.output == ('float64 ndarray 9 elems: (3, 3) [[  5.005 '
                        '  6.005   7. <...> 5] [ 11.005  12.005  '
                        '13.005]]\n')


def test_ordered_dict():
    d = OrderedDict()
    d['z'] = 'should be first'
    d['a'] = 2
    d['b'] = ('third item', 1)
    d[90] = 'key 90 is 4th'
    d['Z'] = 'Z is 5'
    d['A'] = 'A is after Z'
    d[0] = 'last item'
    o = PP(d)
    sbe = ["<class 'collections.OrderedDict'>[n=7]",
           "    'z': 'should be first'",
           "    'a': 2",
           "    'b': [n=2]: ('third item', 1)",
           "    90 : 'key 90 is 4th'",
           "    'Z': 'Z is 5'",
           "    'A': 'A is after Z'",
           "    0  : 'last item'",
           '']
    assert o.output.split('\n') == sbe
