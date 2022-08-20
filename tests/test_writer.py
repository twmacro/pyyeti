import numpy as np
from pyyeti import cb
import os
import sys
from io import StringIO
from pyyeti import writer
import pytest


def test_vecwrite():
    s = "test string"
    i = 5
    v = ["short string", "a bit longer string"]
    r = np.array([[1.1, 1.2, 1.3], [10.1, 10.2, 10.3]])
    frm = "{:2}, {:=^25} : " + "  {:6.2f}" * 3 + chr(10)
    r2 = np.vstack((r, [20.1, 20.2, 20.3]))
    with pytest.raises(ValueError):
        writer.vecwrite(sys.stdout, frm, i, v, r2)


def test_vecwrite_2file():
    r = np.array([1.2, 45.8])
    s = "test string"
    i = [5]
    v = ["short string", "a bit longer string"]
    frm = "{:3}, {:5.1f}, {:<25}, {}" + chr(10)
    writer.vecwrite("temp.writer", frm, i, r, v, s)
    try:
        with open("temp.writer") as f:
            txt = f.read()
    finally:
        os.remove("temp.writer")
    sbe = (
        "  5,   1.2, short string             , test string\n"
        "  5,  45.8, a bit longer string      , test string\n"
    )
    assert txt == sbe


def test_vecwrite_slice():
    r = np.array([1.2, 45.8, 2.4, 12.3, 6.5])
    m = np.arange(10).reshape(5, -1)  # [[0, 1], [2, 3] ...
    s = "test"
    i = [5]
    v = ["short string", "a bit longer string", "3rd", "fourth", "number 5"]
    frm = "{:3}, {:5.1f}, {:<25}, {:1d}-{:1d}, {}" + chr(10)
    with StringIO() as f:
        writer.vecwrite(f, frm, i, r, v, m, s)
        txt = f.getvalue()
    sbe = (
        "  5,   1.2, short string             , 0-1, test\n"
        "  5,  45.8, a bit longer string      , 2-3, test\n"
        "  5,   2.4, 3rd                      , 4-5, test\n"
        "  5,  12.3, fourth                   , 6-7, test\n"
        "  5,   6.5, number 5                 , 8-9, test\n"
    )
    assert txt == sbe

    with StringIO() as f:
        writer.vecwrite(f, frm, i, r, v, m, s, so=slice(1, 5, 2))
        txt = f.getvalue()
    sbe = (  #'  5,   1.2, short string             , 0-1, test\n'
        "  5,  45.8, a bit longer string      , 2-3, test\n"
        #'  5,   2.4, 3rd                      , 4-5, test\n'
        "  5,  12.3, fourth                   , 6-7, test\n"
        #'  5,   6.5, number 5                 , 8-9, test\n'
    )
    assert txt == sbe

    with StringIO() as f:
        writer.vecwrite(f, frm, i, r, v, m, s, so=slice(1, None, 2))
        txt = f.getvalue()
    sbe = (  #'  5,   1.2, short string             , 0-1, test\n'
        "  5,  45.8, a bit longer string      , 2-3, test\n"
        #'  5,   2.4, 3rd                      , 4-5, test\n'
        "  5,  12.3, fourth                   , 6-7, test\n"
        #'  5,   6.5, number 5                 , 8-9, test\n'
    )
    assert txt == sbe

    with StringIO() as f:
        writer.vecwrite(f, frm, i, r, v, m, s, so=slice(3))
        txt = f.getvalue()
    sbe = (
        "  5,   1.2, short string             , 0-1, test\n"
        "  5,  45.8, a bit longer string      , 2-3, test\n"
        "  5,   2.4, 3rd                      , 4-5, test\n"
        #'  5,  12.3, fourth                   , 6-7, test\n'
        #'  5,   6.5, number 5                 , 8-9, test\n'
    )
    assert txt == sbe

    with StringIO() as f:
        writer.vecwrite(f, frm, i, r, v, m, s, so=slice(3, None))
        txt = f.getvalue()
    sbe = (  #'  5,   1.2, short string             , 0-1, test\n'
        #'  5,  45.8, a bit longer string      , 2-3, test\n'
        #'  5,   2.4, 3rd                      , 4-5, test\n'
        "  5,  12.3, fourth                   , 6-7, test\n"
        "  5,   6.5, number 5                 , 8-9, test\n"
    )
    assert txt == sbe

    v = list("too long to be compatible")
    with StringIO() as f:
        with pytest.raises(ValueError):
            writer.vecwrite(f, frm, i, r, v, m, s, so=slice(1, None, 2))

    with StringIO() as f:
        writer.vecwrite(f, frm, i, r, v, m, s, so=slice(5))
        txt = f.getvalue()
    sbe = (
        "  5,   1.2, t                        , 0-1, test\n"
        "  5,  45.8, o                        , 2-3, test\n"
        "  5,   2.4, o                        , 4-5, test\n"
        "  5,  12.3,                          , 6-7, test\n"
        "  5,   6.5, l                        , 8-9, test\n"
    )
    assert txt == sbe


def test_formheader():
    descs = ["Item 1", "A different item"]
    mx = np.array([[1.2, 2.3], [3.4, 4.5]]) * 1000
    time = np.array([[1.234], [2.345]])
    formats = ["{:<25s}", "{:10.2f}", "{:8.3f}"]
    widths = [25, 10, 8]
    with pytest.raises(ValueError):
        writer.formheader(44, widths, formats, sep=[4, 5, 2], just=0)
    headers = [["The"] * 3, ["Descriptions", "Maximum", "Time", "BAD"]]
    with pytest.raises(ValueError):
        writer.formheader(headers, widths, formats, sep=[4, 5, 2], just=0)
    headers = [["The"] * 3, ["Descriptions", "Maximum", "Time"]]
    with pytest.raises(ValueError):
        writer.formheader(headers, [25, 10], formats, sep=[4, 5, 2], just=0)
    hu, f = writer.formheader(headers, widths, formats, sep=[4, 5, 2], just=0)
    with StringIO() as fout:
        fout.write(hu)
        writer.vecwrite(fout, f, descs, mx, time)
        s = fout.getvalue()
    sbe = (
        "               The                   The        The\n"
        "           Descriptions            Maximum      Time\n"
        "    -------------------------     ----------  --------\n"
        "    Item 1                           1200.00  2300.000\n"
        "    A different item                 3400.00  4500.000\n"
    )
    assert sbe == s

    headers = ["Descriptions", "Maximum", "Time"]
    hu, f = writer.formheader(headers, widths, formats, sep=[4, 5, 2], just=0)
    with StringIO() as fout:
        fout.write(hu)
        writer.vecwrite(fout, f, descs, mx, time)
        s = fout.getvalue()
    sbe = (
        "           Descriptions            Maximum      Time\n"
        "    -------------------------     ----------  --------\n"
        "    Item 1                           1200.00  2300.000\n"
        "    A different item                 3400.00  4500.000\n"
    )
    assert sbe == s

    headers = ["Descriptions", "Maximum", "Time"]
    hu, f = writer.formheader(headers, widths, formats, sep=2, just=("l", "c", "r"))
    with StringIO() as fout:
        fout.write(hu)
        writer.vecwrite(fout, f, descs, mx, time)
        s = fout.getvalue()
    sbe = (
        "  Descriptions                Maximum        Time\n"
        "  -------------------------  ----------  --------\n"
        "  Item 1                        1200.00  2300.000\n"
        "  A different item              3400.00  4500.000\n"
    )
    assert sbe == s
