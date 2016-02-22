import numpy as np
from pyyeti import cb
import os
import sys
from io import StringIO
from pyyeti import writer
from nose.tools import *


def test_vecwrite():
    s = 'test string'
    i = 5
    v = ['short string', 'a bit longer string']
    r = np.array([[1.1, 1.2, 1.3], [10.1, 10.2, 10.3]])
    frm = '{:2}, {:=^25} : ' + '  {:6.2f}'*3 + chr(10)
    r2 = np.vstack((r, [20.1, 20.2, 20.3]))
    assert_raises(ValueError, writer.vecwrite, sys.stdout,
                  frm, i, v, r2)


def test_vecwrite_2file():
    from pyyeti import writer
    import sys
    import numpy as np
    r = np.array([1.2, 45.8])
    s = 'test string'
    i = [5]
    v = ['short string', 'a bit longer string']
    frm = '{:3}, {:5.1f}, {:<25}, {}' + chr(10)
    writer.vecwrite('temp.writer', frm, i, r, v, s)
    try:
        with open('temp.writer') as f:
            txt = f.read()
    finally:
        os.remove('temp.writer')
    sbe = ('  5,   1.2, short string             , test string\n'
           '  5,  45.8, a bit longer string      , test string\n')
    assert txt==sbe
    
    
def test_formheader():
    descs = ['Item 1', 'A different item']
    mx = np.array([[1.2, 2.3], [3.4, 4.5]]) * 1000
    time = np.array([[1.234], [2.345]])
    formats = ['{:<25s}', '{:10.2f}', '{:8.3f}']
    widths  = [25, 10, 8]
    assert_raises(ValueError, writer.formheader, 44, widths, formats,
                  sep=[4, 5, 2], just=0)
    headers = [['The']*3, ['Descriptions', 'Maximum', 'Time', 'BAD']]
    assert_raises(ValueError, writer.formheader, headers, widths,
                  formats, sep=[4, 5, 2], just=0)
    headers = [['The']*3, ['Descriptions', 'Maximum', 'Time']]
    assert_raises(ValueError, writer.formheader, headers, [25, 10],
                  formats, sep=[4, 5, 2], just=0)
    h, u, f = writer.formheader(headers, widths, formats,
                                sep=[4, 5, 2], just=0)
    fout = StringIO()
    fout.write(h[0])
    fout.write(h[1])
    fout.write(u)
    writer.vecwrite(fout, f, descs, mx, time)
    s = fout.getvalue()
    sbe = ('               The                   The        The\n'
           '           Descriptions            Maximum      Time\n'
           '    -------------------------     ----------  --------\n'
           '    Item 1                           1200.00  2300.000\n'
           '    A different item                 3400.00  4500.000\n')
    assert sbe==s

    headers = ['Descriptions', 'Maximum', 'Time']
    h, u, f = writer.formheader(headers, widths, formats,
                                sep=[4, 5, 2], just=0)
    fout = StringIO()
    fout.write(h)
    fout.write(u)
    writer.vecwrite(fout, f, descs, mx, time)
    s = fout.getvalue()
    sbe = (
           '           Descriptions            Maximum      Time\n'
           '    -------------------------     ----------  --------\n'
           '    Item 1                           1200.00  2300.000\n'
           '    A different item                 3400.00  4500.000\n')
    assert sbe==s
    
    headers = ['Descriptions', 'Maximum', 'Time']
    h, u, f = writer.formheader(headers, widths, formats,
                                sep=2, just=('l', 'c', 'r'))
    fout = StringIO()
    fout.write(h)
    fout.write(u)
    writer.vecwrite(fout, f, descs, mx, time)
    s = fout.getvalue()
    sbe = (
           '  Descriptions                Maximum        Time\n'
           '  -------------------------  ----------  --------\n'
           '  Item 1                        1200.00  2300.000\n'
           '  A different item              3400.00  4500.000\n')
    assert sbe==s
