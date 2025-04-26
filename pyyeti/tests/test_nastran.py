import math
import numpy as np
import os
import tempfile
import warnings
from io import StringIO
from pathlib import Path
import matplotlib.pyplot as plt
from pyyeti import nastran
from pyyeti.nastran import op2, n2p, op4
import pytest


def test_rdcards():
    a = nastran.rdcards(
        "pyyeti/tests/nas2cam_extseout/assemble.out", "CCC", no_data_return="no CCC"
    )
    assert a == "no CCC"

    with pytest.raises(ValueError):
        nastran.rdcards(
            "pyyeti/tests/nas2cam_extseout/assemble.out",
            "grid",
            return_var="bad option",
        )


def test_rdcards2():
    fs = StringIO(
        """
$
PARAM,POST,-1
EIGR           1    AHOU                          100000

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
TABLED1        1
            0.01     1.0   150.0    1.0     ENDT
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

$$$INCLUDE 'outboard.blk'
$ GRID           1       0      0.      0.    300.       0
GRID*                  1               0      0.00000000      0.00000000
*           300.00000000               0
$GRID           2       0    300.      0.    300.       0
grid, 2, 0,  300.,  0., 300.,  0
$$$
$111111122222222333333334444444455555555666666667777777788888888
RBE2    1001    330     123456  33
$ last line
"""
    )

    lst = nastran.rdcards(
        fs,
        r"[a-z]+[*]*",
        return_var="list",
        regex=True,
        keep_name=True,
        keep_comments=True,
    )
    sbe = [
        "$\n",
        ["PARAM", "POST", -1],
        ["EIGR", 1, "AHOU", "", "", "", 100000],
        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n",
        ["TABLED1", 1, "", "", "", "", "", "", "", 0.01, 1.0, 150.0, 1.0, "ENDT"],
        "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n",
        "$$$INCLUDE 'outboard.blk'\n",
        "$ GRID           1       0      0.      0.    300.       0\n",
        ["GRID*", 1, 0, 0.0, 0.0, 300.0, 0],
        "$GRID           2       0    300.      0.    300.       0\n",
        ["grid", 2, 0, 300.0, 0.0, 300.0, 0],
        "$$$\n",
        "$111111122222222333333334444444455555555666666667777777788888888\n",
        ["RBE2", 1001, 330, 123456, 33],
        "$ last line\n",
    ]
    assert lst == sbe

    fs = StringIO(
        """
$ starting comment
DTI     SELOAD         1       2
dti     seload         3       4
$ a comment for testing
dti,seload,5,6
DTI, SELOAD, , 8.0, 'a'
DTI,SETREE,100,0
$ ending comment
    """
    )

    lst = nastran.rdcards(
        fs,
        r"DTI(,\s*|\s+)SELOAD",
        regex=True,
        return_var="list",
        keep_name=True,
        keep_comments=True,
    )
    sbe = [
        "$ starting comment\n",
        ["DTI", "SELOAD", 1, 2],
        ["dti", "seload", 3, 4],
        "$ a comment for testing\n",
        ["dti", "seload", 5, 6],
        ["DTI", "SELOAD", "", 8.0, "'a'"],
        "$ ending comment\n",
    ]
    assert sbe == lst


def _wtfile(path, contents):
    with open(path, "wt") as fobj:
        fobj.write(contents)


def test_rdcards_with_includes_errors():
    file1 = "GRID,1,0,10.0,0.0,0.0\nINCLUDE 'file2.bdf'\nGRID,3,0,30.0,0.0,0.0\n"
    with tempfile.TemporaryDirectory() as tempdir_path:
        file1_path = os.path.join(tempdir_path, "file1.bdf")

        # INCLUDE has no quotes
        _wtfile(file1_path, file1.replace("'file2.bdf'", "file2.bdf"))
        with pytest.raises(ValueError, match=r"Invalid INCLU.*no quote.*file2"):
            nastran.rdcards(file1_path, "grid", follow_includes=True)

        # INCLUDE has opening but no closing quote
        _wtfile(file1_path, file1.replace("'file2.bdf'", "'file2.bdf"))
        with pytest.raises(ValueError, match=r"Invalid INCLU.*no ending quote.*file2"):
            nastran.rdcards(file1_path, "grid", follow_includes=True)

        # Symbol used in INCLUDE, but is not defined
        _wtfile(file1_path, file1.replace("file2.bdf", "MODEL_DIR:file2.bdf"))
        regex = r"Symbol.*in INCLUDE.*not defined.*model_dir"
        with pytest.raises(ValueError, match=regex):
            nastran.rdcards(file1_path, "grid", follow_includes=True)

        # Symbol doesn't have a length >1 so it cannot be distinguished from a Windows
        # drive letter
        symbols = {"C": r"C:\dirC"}
        with pytest.raises(ValueError, match=r"Symbols.*length >1.*got c"):
            nastran.rdcards(file1_path, "grid", include_symbols=symbols)

        # Path on INCLUDE statement does not exist
        _wtfile(file1_path, file1.replace("file2.bdf", "zzzz.bdf"))
        with pytest.raises(FileNotFoundError, match=r"zzzz.bdf"):
            nastran.rdcards(file1_path, "grid")


def test_rdcards_with_includes():
    file1 = (
        "GRID,1,0,10.0,0.0,0.0\n"
        "GRID         101\n"  # not a proper grid, just testing a multi-line card
        "+       ABC\n"
        "$ This is a comment\n"
        "\n"
        "INCLUDE 'file2.\n"  # include statement spans multiple lines
        "bdf'\n"
        "GRID,3,0,30.0,0.0,0.0\n"
    )
    file2 = "GRID,2,0,20.0,0.0,0.0\nINCLUDE 'file3.bdf'"
    file3 = "GRID,201,0,2001.0,0.0,0.0\n$ Another comment"

    def check_results(cards):
        assert len(cards) == 5
        for card in cards:
            assert card[0] == "GRID"
        assert cards[0][1] == 1
        assert cards[1][1] == 101
        assert cards[1][9] == "ABC"
        assert cards[2][1] == 2
        assert cards[3][1] == 201
        assert math.isclose(cards[3][3], 2001.0)
        assert cards[4][1] == 3
        assert math.isclose(cards[4][3], 30.0)

    # all files in same directory
    with tempfile.TemporaryDirectory() as tempdir_path:
        file1_path = os.path.join(tempdir_path, "file1.bdf")
        _wtfile(file1_path, file1)
        _wtfile(os.path.join(tempdir_path, "file2.bdf"), file2)
        _wtfile(os.path.join(tempdir_path, "file3.bdf"), file3)
        # follow_includes is True
        cards = nastran.rdcards(
            file1_path,
            "grid",
            blank=None,
            return_var="list",
            keep_name=True,
            follow_includes=True,
        )
        check_results(cards)

        # follow_includes is False
        cards = nastran.rdcards(
            file1_path,
            "grid",
            blank=None,
            return_var="list",
            keep_name=True,
            follow_includes=False,
        )
        assert len(cards) == 3
        assert [card[1] for card in cards] == [1, 101, 3]

        # keep_comments is True
        cards = nastran.rdcards(
            file1_path,
            "grid",
            blank=None,
            return_var="list",
            keep_name=True,
            keep_comments=True,
            follow_includes=True,
        )
        assert cards[2].startswith("$ This is a comm")
        assert cards[5].startswith("$ Another comm")
        check_results([cards[i] for i in [0, 1, 3, 4, 6]])

    # file1 in subdirectory, file2 and file3 in parent dir
    with tempfile.TemporaryDirectory() as tempdir_path:
        subdir_path = os.path.join(tempdir_path, "subdir")
        os.mkdir(subdir_path)
        file1_path = os.path.join(subdir_path, "file1.bdf")
        # update to use relative path
        _wtfile(file1_path, file1.replace("file2", "../file2"))
        # update to use relative path
        # note that path is relative to file1, even though the statement is in
        # file2, this is consistent with Nastran when the INCLUDE has directories
        # in addition to a filename
        _wtfile(
            os.path.join(tempdir_path, "file2.bdf"), file2.replace("file3", "../file3")
        )
        _wtfile(os.path.join(tempdir_path, "file3.bdf"), file3)
        # follow_includes is True
        cards = nastran.rdcards(
            file1_path,
            "grid",
            blank=None,
            return_var="list",
            keep_name=True,
            follow_includes=True,
        )
        check_results(cards)

    # file1 in subdirectory, file2 and file3 in parent dir
    with tempfile.TemporaryDirectory() as tempdir_path:
        subdir_path = os.path.join(tempdir_path, "subdir")
        os.mkdir(subdir_path)
        file1_path = os.path.join(subdir_path, "file1.bdf")
        # update to use relative path
        _wtfile(file1_path, file1.replace("file2", "../file2"))
        # In file2, the "INCLUDE 'file3.bdf'" statement has no directories, so file3 is
        # treated as if it's in the same directory as file2.
        _wtfile(os.path.join(tempdir_path, "file2.bdf"), file2)
        _wtfile(os.path.join(tempdir_path, "file3.bdf"), file3)
        # follow_includes is True
        cards = nastran.rdcards(
            file1_path,
            "grid",
            blank=None,
            return_var="list",
            keep_name=True,
            follow_includes=True,
        )
        check_results(cards)

    # file1 and file3 in parent_dir, file2 in subdirectory
    with tempfile.TemporaryDirectory() as tempdir_path:
        file1_path = os.path.join(tempdir_path, "file1.bdf")
        file3_path = os.path.join(tempdir_path, "file3.bdf")
        subdir_path = os.path.join(tempdir_path, "subdir")
        os.mkdir(subdir_path)
        file2_path = os.path.join(subdir_path, "file2.bdf")
        # file2 and file3 are both included from file1
        file1_mod = file1.replace(
            "INCLUDE 'file2.\nbdf'", "INCLUDE 'subdir/file2.bdf'\nINCLUDE 'file3.bdf'"
        )
        _wtfile(file1_path, file1_mod)
        _wtfile(file2_path, file2.replace("INCLUDE", "$ INCLUDE"))
        _wtfile(file3_path, file3)
        cards = nastran.rdcards(
            file1_path,
            "grid",
            blank=None,
            return_var="list",
            keep_name=True,
            follow_includes=True,
        )
        check_results(cards)

    # file1 in parent_dir, file2 and file3 in subdirectory, includes use symbols
    with tempfile.TemporaryDirectory() as tempdir_path:
        file1_path = os.path.join(tempdir_path, "file1.bdf")
        # update to use symbol
        _wtfile(file1_path, file1.replace("file2", "SUBDIR:file2"))
        subdir_path = os.path.join(tempdir_path, "subdir")
        os.mkdir(subdir_path)
        _wtfile(
            os.path.join(subdir_path, "file2.bdf"),
            file2.replace("file3", "SUBDIR:../file3"),
        )
        _wtfile(os.path.join(tempdir_path, "file3.bdf"), file3)
        symbols = {"SUBDIR": subdir_path}
        cards = nastran.rdcards(
            file1_path,
            "grid",
            blank=None,
            return_var="list",
            keep_name=True,
            follow_includes=True,
            include_symbols=symbols,
        )
        check_results(cards)

    # file2 is empty
    with tempfile.TemporaryDirectory() as tempdir_path:
        file1_path = os.path.join(tempdir_path, "file1.bdf")
        _wtfile(file1_path, file1)
        _wtfile(os.path.join(tempdir_path, "file2.bdf"), "")
        cards = nastran.rdcards(
            file1_path,
            "grid",
            blank=None,
            return_var="list",
            keep_name=True,
            follow_includes=True,
        )
        assert len(cards) == 3
        assert cards[0][1] == 1
        assert cards[1][1] == 101
        assert cards[2][1] == 3


def _wtfile_x93_x94(path, contents):
    with open(path, "wt") as fobj:
        fobj.write(contents)
    with open(path, "ab") as fobj:
        fobj.write(b"$ \x93\x94\n")


def test_encoding_with_include():
    # "$ \x93\x94\n" characters will be appended to each file
    # - These characters cannot be read in using utf8 encoding, but
    #   latin1 works

    file1 = "GRID,1,0,10.0,0.0,0.0\nINCLUDE 'file2.bdf'\nGRID,4,0,40.0,0.0,0.0\n"
    file2 = "GRID,2,0,20.0,0.0,0.0\nINCLUDE 'file3.bdf'\n"
    file3 = "GRID,3,0,30.0,0.0,0.0\n"

    # all files in same directory
    with tempfile.TemporaryDirectory() as tempdir_path:
        file1_path = os.path.join(tempdir_path, "file1.bdf")
        _wtfile_x93_x94(file1_path, file1)
        _wtfile_x93_x94(os.path.join(tempdir_path, "file2.bdf"), file2)
        _wtfile_x93_x94(os.path.join(tempdir_path, "file3.bdf"), file3)
        grids = nastran.rdgrids(file1_path, encoding="latin1")
        with pytest.raises(
            UnicodeDecodeError, match=r"'utf-8' codec can't decode byte 0x93"
        ):
            grids = nastran.rdgrids(file1_path)

    tf = grids == [
        [1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    assert tf.all()


@pytest.mark.parametrize(
    ("string", "match"),
    [
        ("SET 1111 = 1,2,3,", r"Invalid SET.*EOF before.*1111"),
        ("SET 1112 = 1,2,3,\n4,5,6,", r"Invalid SET.*EOF before.*1112"),
        ("SET 1113 = 1,2,3,\n4,5,6,\n\n", r"Invalid SET.*EOF before.*1113"),
        ("SET 1114 = 1,2,3,\nGRID\n4,5,6\n", r"Invalid SET.*cannot.*int.*1114"),
        ("SET 1115 = 1,2,QQQ", r"Invalid SET.*cannot.*int.*1115"),
        ("SET 1116 = 1,2,QQQ THRU 5", r"Invalid SET.*cannot.*int.*1116"),
        ("SET 1117 = 1,2,5 THRU QQQ", r"Invalid SET.*cannot.*int.*1117"),
    ],
)
def test_rdsets_errors(string, match):
    with pytest.raises(ValueError, match=match):
        nastran.rdsets(StringIO(string))


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        ("SET 101 = 1", {101: [1]}),  # one item, no newline
        ("SET 101 = 1\n", {101: [1]}),  # one item, with newline
        ("SET 101=1", {101: [1]}),  # no spaces around equals sign, no newline
        ("SET 101=1\n", {101: [1]}),  # no spaces around equals sign, with newline
        ("SET101=1\n", {101: [1]}),  # no spaces anywhere
        ("  SET  101  =  1", {101: [1]}),  # multiple embedded spaces
        ("set 101 = 1", {101: [1]}),  # lower case set
        ("SET 101 = 1,2", {101: [1, 2]}),  # two items, no newline
        ("SET 101=1,2", {101: [1, 2]}),  # two items, no spaces, no newline
        ("SET 101=1,2\n", {101: [1, 2]}),  # two items, no spaces, with newline
        ("SET 101 = 1 thru 3", {101: [1, 2, 3]}),  # thru, no newline
        ("SET 101=1 thru 3", {101: [1, 2, 3]}),  # thru, no spaces, no newline
        ("SET 101=1 thru 3\n", {101: [1, 2, 3]}),  # thru, no spaces, with newline
        (
            # set in bulk data, should be ignored
            "SET 101 = 1, 3, 6 THRU 8\nBEGIN BULK\nSET 102 = 99",
            {101: [1, 3, 6, 7, 8]},
        ),
        (
            # set in bulk data, should be ignored, 'begin bulk' has leading space
            "SET 101 = 1, 3, 6 THRU 8\n begin bulk\nSET 102 = 99",
            {101: [1, 3, 6, 7, 8]},
        ),
        (
            # 'begin bulk' is commented out
            "SET 101 = 1, 3, 6 THRU 8\n$BEGIN BULK\nSET 102 = 99",
            {101: [1, 3, 6, 7, 8], 102: [99]},
        ),
        (
            # two sets
            (
                "SET 111 = 1,3 , 6 THRU 8, 10,\n"
                "  11,13 thru   15\n"
                "   set 112=  10101,10102\n"  # leading spaces and lower case
            ),
            {111: [1, 3, 6, 7, 8, 10, 11, 13, 14, 15], 112: [10101, 10102]},
        ),
        (
            # embedded blank line in set
            (
                " SET 111 = 1,3 , 10,  \n"  # leading and trailing spaces
                "  11,13 thru 15,\n"
                "\n"  # embedded blank line
                " 18, 19 \n"  # trailing space
            ),
            {111: [1, 3, 10, 11, 13, 14, 15, 18, 19]},
        ),
    ],
)
def test_rdsets(string, expected):
    sets = nastran.rdsets(StringIO(string))
    assert sets == expected


def test_rdsets_with_includes_errors():
    file1 = "SET 111 = 1,3,5 THRU 8\nINCLUDE 'file2.bdf'\n"
    with tempfile.TemporaryDirectory() as tempdir_path:
        file1_path = os.path.join(tempdir_path, "file1.bdf")

        # Path on INCLUDE statement does not exist
        _wtfile(file1_path, file1.replace("file2.bdf", "zzzz.bdf"))
        with pytest.raises(FileNotFoundError, match=r"zzzz.bdf"):
            nastran.rdsets(file1_path)


def test_rdsets_with_includes():
    file1 = (
        "SET 111 = 1,3,5 THRU 8\n"
        "DISPLACEMENT(PLOT) = 111\n"
        "INCLUDE 'file2.bdf'\n"
        "SET 333=1001"
    )
    file2 = "SET 222 = 101, 201 THRU 203\n"

    with tempfile.TemporaryDirectory() as tempdir_path:
        file1_path = os.path.join(tempdir_path, "file1.bdf")
        _wtfile(file1_path, file1)
        _wtfile(os.path.join(tempdir_path, "file2.bdf"), file2)

        # follow_includes is True
        sets = nastran.rdsets(file1_path)
        assert sets == {111: [1, 3, 5, 6, 7, 8], 222: [101, 201, 202, 203], 333: [1001]}

        # follow_includes is False
        sets = nastran.rdsets(file1_path, follow_includes=False)
        assert sets == {111: [1, 3, 5, 6, 7, 8], 333: [1001]}

    # INCLUDE statement spans two lines
    with tempfile.TemporaryDirectory() as tempdir_path:
        file1_path = os.path.join(tempdir_path, "file1.bdf")
        _wtfile(file1_path, file1.replace("'file1.bdf'", "'file1.\nbdf'"))
        _wtfile(os.path.join(tempdir_path, "file2.bdf"), file2)
        sets = nastran.rdsets(file1_path)
        assert sets == {111: [1, 3, 5, 6, 7, 8], 222: [101, 201, 202, 203], 333: [1001]}

    # file1 in subdirectory, file2 in parent dir
    with tempfile.TemporaryDirectory() as tempdir_path:
        subdir_path = os.path.join(tempdir_path, "subdir")
        os.mkdir(subdir_path)
        file1_path = os.path.join(subdir_path, "file1.bdf")
        # update to use relative path
        _wtfile(file1_path, file1.replace("file2", "../file2"))
        _wtfile(os.path.join(tempdir_path, "file2.bdf"), file2)
        sets = nastran.rdsets(file1_path)
        assert sets == {111: [1, 3, 5, 6, 7, 8], 222: [101, 201, 202, 203], 333: [1001]}

    # file1 in subdirectory, file2 in parent dir, includes use symbols
    with tempfile.TemporaryDirectory() as tempdir_path:
        subdir_path = os.path.join(tempdir_path, "subdir")
        os.mkdir(subdir_path)
        file1_path = os.path.join(subdir_path, "file1.bdf")
        # update to use symbol
        _wtfile(file1_path, file1.replace("file2", "SUBDIR:file2"))
        _wtfile(os.path.join(tempdir_path, "file2.bdf"), file2)
        symbols = {"SUBDIR": os.path.abspath(tempdir_path)}
        sets = nastran.rdsets(file1_path, include_symbols=symbols)
        assert sets == {111: [1, 3, 5, 6, 7, 8], 222: [101, 201, 202, 203], 333: [1001]}


def test_rdsymbols():
    # Posix paths
    f = StringIO(
        "BUFFSIZE=65535\n"
        "SYMBOL=FEMDIR='/home/loads/FEM/V1'\n"
        "SYMBOL=DMAPDIR='/home/cla/cla/nastran_files/alter'"
    )
    symbols = nastran.rdsymbols(f)
    assert len(symbols) == 2
    assert symbols["femdir"] == "/home/loads/FEM/V1"
    assert symbols["dmapdir"] == "/home/cla/cla/nastran_files/alter"

    # Windows paths
    f = StringIO(
        "BUFFSIZE=65535\n"
        "SYMBOL=FEMDIR='c:\\home\\loads\\FEM Folder\\V1'\n"  # path includes a space
        "SYMBOL=DMAPDIR='d:\\home\\cla\\cla\\nastran_files\\alter'"
    )
    symbols = nastran.rdsymbols(f)
    assert len(symbols) == 2
    assert symbols["femdir"] == "c:\\home\\loads\\FEM Folder\\V1"
    assert symbols["dmapdir"] == "d:\\home\\cla\\cla\\nastran_files\\alter"

    # Empty file
    f = StringIO("")
    symbols = nastran.rdsymbols(f)
    assert symbols == {}


def test_rdgrids():
    file1 = "GRID,1,0,10.0,0.0,0.0\nINCLUDE 'file2.bdf'\nGRID,3,0,30.0,0.0,0.0\n"
    file2 = "GRID,2,0,20.0,0.0,0.0"
    with tempfile.TemporaryDirectory() as tempdir_path:
        file1_path = os.path.join(tempdir_path, "file1.bdf")
        _wtfile(file1_path, file1)
        _wtfile(os.path.join(tempdir_path, "file2.bdf"), file2)
        # follow_includes=False
        grids = nastran.rdgrids(file1_path, follow_includes=False)
        np.testing.assert_array_equal(grids[:, 0], [1, 3])  # grid ID
        np.testing.assert_allclose(grids[:, 2], [10.0, 30.0])  # x coord
        # follow_includes=True
        grids = nastran.rdgrids(file1_path)
        np.testing.assert_array_equal(grids[:, 0], [1, 2, 3])  # grid ID
        np.testing.assert_allclose(grids[:, 2], [10.0, 20, 30.0])  # x coord


def test_wtgrids():
    xyz = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])
    with StringIO() as f:
        nastran.wtgrids(f, [100, 200], xyz=xyz, cd=10, form="{:8.2f}", ps=123, seid=100)
        s = f.getvalue()
    assert s == (
        "GRID         100       0    0.10    0.20"
        "    0.30      10     123     100\n"
        "GRID         200       0    1.10    1.20"
        "    1.30      10     123     100\n"
    )

    with StringIO() as f:
        nastran.wtgrids(
            f, [100, 200], xyz=xyz, cd=10, form="{:16.2f}", ps=123, seid=100
        )
        s = f.getvalue()
    assert s == (
        "GRID*                100               0"
        "            0.10            0.20\n"
        "*                   0.30              10"
        "             123             100\n"
        "GRID*                200               0"
        "            1.10            1.20\n"
        "*                   1.30              10"
        "             123             100\n"
    )
    with pytest.raises(ValueError):
        nastran.wtgrids(1, 100, form="{:9f}")


def test_wttabled1():
    t = np.arange(0, 1, 0.05)
    d = np.sin(2 * np.pi * 3 * t)
    with StringIO() as f:
        nastran.wttabled1(f, 4000, t, d, form="{:16.2f}{:16.5f}")
        s = f.getvalue()
    sbe = (
        "TABLED1*            4000\n"
        "*\n"
        "*                   0.00         0.00000            0.05         0.80902\n"
        "*                   0.10         0.95106            0.15         0.30902\n"
        "*                   0.20        -0.58779            0.25        -1.00000\n"
        "*                   0.30        -0.58779            0.35         0.30902\n"
        "*                   0.40         0.95106            0.45         0.80902\n"
        "*                   0.50         0.00000            0.55        -0.80902\n"
        "*                   0.60        -0.95106            0.65        -0.30902\n"
        "*                   0.70         0.58779            0.75         1.00000\n"
        "*                   0.80         0.58779            0.85        -0.30902\n"
        "*                   0.90        -0.95106            0.95        -0.80902\n"
        "*       ENDT\n"
    )
    assert s == sbe
    with pytest.raises(ValueError):
        nastran.wttabled1(1, 10, [1, 2], 1)
    with pytest.raises(ValueError):
        nastran.wttabled1(1, 10, [1, 2], [1, 2], form="{:9f}{:9f}")


def test_rdtabled1():
    tab = """TABLED1,    1
  , 0.39700, 0.00066, 0.39708, 0.00064, 0.39717, 0.00062, 0.39725, 0.00059,
  , 0.39733, 0.00057, 0.39742, 0.00054, 0.39750, 0.00051, 0.39758, 0.00048,
  , 0.39767, 0.00046, 0.39775, 0.00043, 0.39783, 0.00040, 0.39792, 0.00037,
  , 0.39800, 0.00035, 0.39808, 0.00032, 0.39817, 0.00030, 0.39825, 0.00027,
  , 0.39833, 0.00025, 0.39842, 0.00022, 0.39850, 0.00020, 0.39858, 0.00018,
  , 0.39867, 0.00016, 0.39875, 0.00014, 0.39883, 0.00012, 0.39892, 0.00010,
  , 0.39900, 0.00009, 0.39908, 0.00007, 0.39917, 0.00006, 0.39925, 0.00005,
  , 0.39933, 0.00004, 0.39942, 0.00003, 0.39950, 0.00002, 0.39958, 0.00001,
  , 0.39967, 0.00001, 0.39975, 0.00000, 0.39983, 0.00000,15.00000, 0.00000,
  ,    ENDT
"""
    with StringIO(tab) as f:
        dct = nastran.rdtabled1(f)
    with StringIO(tab) as f:
        lines = f.readlines()
    mat = np.array(
        [[float(num) for num in line[3:-2].split(",")] for line in lines[1:-1]]
    ).ravel()
    t = mat[::2]
    d = mat[1::2]
    assert np.allclose(dct[1][:, 0], t)
    assert np.allclose(dct[1][:, 1], d)


def test_rdtabled1_2():
    tab = """newname,    1
  , 0.39700, 0.00066, 0.39708, 0.00064, 0.39717, 0.00062, 0.39725, 0.00059,
  , 0.39733, 0.00057, 0.39742, 0.00054, 0.39750, 0.00051, 0.39758, 0.00048,
  , 0.39767, 0.00046, 0.39775, 0.00043, 0.39783, 0.00040, 0.39792, 0.00037,
  , 0.39800, 0.00035, 0.39808, 0.00032, 0.39817, 0.00030, 0.39825, 0.00027,
  , 0.39833, 0.00025, 0.39842, 0.00022, 0.39850, 0.00020, 0.39858, 0.00018,
  , 0.39867, 0.00016, 0.39875, 0.00014, 0.39883, 0.00012, 0.39892, 0.00010,
  , 0.39900, 0.00009, 0.39908, 0.00007, 0.39917, 0.00006, 0.39925, 0.00005,
  , 0.39933, 0.00004, 0.39942, 0.00003, 0.39950, 0.00002, 0.39958, 0.00001,
  , 0.39967, 0.00001, 0.39975, 0.00000, 0.39983, 0.00000,15.00000, 0.00000,
  ,    ENDT
"""
    with StringIO(tab) as f:
        dct = nastran.rdtabled1(f, "newname")
    with StringIO(tab) as f:
        lines = f.readlines()
    mat = np.array(
        [[float(num) for num in line[3:-2].split(",")] for line in lines[1:-1]]
    ).ravel()
    t = mat[::2]
    d = mat[1::2]
    assert np.allclose(dct[1][:, 0], t)
    assert np.allclose(dct[1][:, 1], d)


def test_rdwtbulk():
    with StringIO() as f:
        nastran.rdwtbulk("pyyeti/tests/nas2cam_csuper/inboard.out", f)
        s = f.getvalue()
    with open("pyyeti/tests/nas2cam_csuper/yeti_outputs/inboard_yeti.bulk") as f:
        sy = f.read()
    assert s == sy

    with StringIO() as f:
        nastran.rdwtbulk("pyyeti/tests/nas2cam_csuper/fake_bulk.out", f)
        s = f.getvalue()
    with open("pyyeti/tests/nas2cam_csuper/yeti_outputs/fake_bulk.blk") as f:
        sy = f.read()
    assert s == sy


def test_bulk2uset():
    xyz = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]])
    with StringIO() as f:
        nastran.wtgrids(f, [100, 200], xyz=xyz)
        u, c = nastran.bulk2uset(f)

    uset = n2p.addgrid(None, 100, "b", 0, xyz[0], 0)
    uset = n2p.addgrid(uset, 200, "b", 0, xyz[1], 0)
    assert np.allclose(uset, u)
    coord = {0: np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])}
    assert coord.keys() == c.keys()
    assert np.allclose(coord[0], c[0])

    blk = """
CORD2R  10      0       0.0     0.0     0.0     1.0     0.0     0.0
        0.0     1.0     0.0
"""

    with StringIO(blk) as f:
        uset, cord = nastran.bulk2uset(f)
    assert uset.size == 0
    assert len(cord) == 2


def test_uset2bulk():
    xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    with StringIO() as f:
        nastran.wtgrids(f, [100, 200, 300], xyz=xyz)
        uset, cords = nastran.bulk2uset(f)

    new_cs_in_basic = np.array(
        [[0.0, 100.0, 0.0], [0.0, 100.0, 1.0], [0.0, 110.0, 0.0]]
    )
    uset2 = n2p.replace_basic_cs(uset, 10, new_cs_in_basic)

    with StringIO() as f:
        nastran.uset2bulk(f, uset2)
        # s = f.getvalue()
        uset3, cords = nastran.bulk2uset(f)

    assert np.allclose(uset2, uset3)

    coordinates = [[0.0, 100.0, 0.0], [0.0, 110.0, 0.0], [0.0, 120.0, 0.0]]

    assert np.allclose(uset3.loc[(slice(None), 1), "x":], coordinates)

    """
               ^ X_10
               |
               |
        Y_10   |
        <------                    ---
              /                     |
            /  Z_10                 |
                                    |
                                   100
               ^ Y_basic            |
               |                    |
               |                    |
               |                    |
                -----> X_basic     ---
              /
            /  Z_basic
    """

    # transform from 10 to basic:
    T = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(uset3.iloc[3:6, 1:], T)

    # Could get T from n2p.build_coords:
    # T = n2p.build_coords([10, 1, 0, *new_cs_in_basic.ravel()])[10][2:]


def test_asm2uset():
    asm1 = """
$ SE101 ASSEMBLY FILE FOR RESIDUAL RUN...INCLUDE IN BULK DATA
$
SEBULK       101  EXTOP4          MANUAL                     101
SECONCT      101       0              NO
               3       3      11      11      19      19      27      27
$
$ COORDINATE SYSTEM DATA
$
$ Coordinate 10:
CORD2R*               10               0  0.00000000e+00  0.00000000e+00*
*         0.00000000e+00  1.00000000e+00  0.00000000e+00  0.00000000e+00*
*         0.00000000e+00  1.00000000e+00  0.00000000e+00
$
$ BOUNDARY GRID DATA
$
GRID*                  3               0    600.00000000      0.00000000
*           300.00000000               0
GRID*                 11               0    600.00000000    300.00000000
*           300.00000000              10
GRID*                 19               0    600.00000000    300.00000000
*             0.00000000               0
GRID*                 27               0    600.00000000      0.00000000
*             0.00000000               0
$
SECONCT      101       0              NO
         9900101    THRU 9900122 9900101    THRU 9900122
$
SPOINT   9900101    THRU 9900122
"""
    with StringIO(asm1) as f:
        uset1, cord1, bset1 = nastran.asm2uset(f)
        cords1 = nastran.rdcord2cards(f)

    # make the uset manually for testing:
    rng = range(9900101, 9900123)
    dof = [[3, 123456], [11, 123456], [19, 123456], [27, 123456]] + [
        [i, 0] for i in rng
    ]
    nasset = np.zeros(4 + 22, np.int64)
    nasset[:4] = n2p.mkusetmask("b")
    nasset[4:] = n2p.mkusetmask("q")
    xyz = np.array(
        [
            [600.0, 0.0, 300.0],
            [600.0, 300.0, 300.0],
            [600.0, 300.0, 0.0],
            [600.0, 0.0, 0.0],
        ]
        + [[0.0, 0.0, 0.0] for i in rng]
    )

    uset1_man = n2p.make_uset(dof=dof, nasset=nasset, xyz=xyz)

    # fix up grid 11 coords:
    uset1_man.loc[(11, 2), "x"] = 10
    uset1_man.loc[(11, 4):(11, 6), "x":"z"] = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    # assert uset1.equals(uset1_man)
    assert np.allclose(uset1.reset_index(), uset1_man.reset_index())
    assert (bset1 == n2p.mksetpv(uset1, "a", "b")).all()

    assert len(cords1) == len(cord1)
    for k, v in cords1.items():
        assert np.allclose(cord1[k], v)

    asm2 = """
$ SE101 ASSEMBLY FILE FOR RESIDUAL RUN...INCLUDE IN BULK DATA
$
$1111111222222223333333344444444555555556666666677777777888888889999999900000000
SEBULK       101  EXTOP4          MANUAL                     101
SECONCT      101       0              NO
               3       3     110     110      19      19      27      27
$
$ COORDINATE SYSTEM DATA
$
$ Coordinate 10:
CORD2R*               10               0  0.00000000e+00  0.00000000e+00*
*         0.00000000e+00  1.00000000e+00  0.00000000e+00  0.00000000e+00*
*         0.00000000e+00  1.00000000e+00  0.00000000e+00
$
$ BOUNDARY GRID DATA
$
GRID*                  3               0    600.00000000      0.00000000
*           300.00000000               0
GRID*                 19               0    600.00000000    300.00000000
*             0.00000000               0
GRID*                 27               0    600.00000000      0.00000000
*             0.00000000               0
$
SPOINT   110
"""

    with StringIO(asm2) as f:
        uset2, cord2, bset2 = nastran.asm2uset(f)
        cords2 = nastran.rdcord2cards(f)

    # make the uset manually for testing:
    dof = [[3, 123456], [110, 0], [19, 123456], [27, 123456]]
    nasset = np.zeros(4, np.int64)
    nasset[:] = n2p.mkusetmask("b")
    nasset[1] = n2p.mkusetmask("q")
    xyz = np.array(
        [[600.0, 0.0, 300.0], [0.0, 0.0, 0.0], [600.0, 300.0, 0.0], [600.0, 0.0, 0.0]]
    )

    uset2_man = n2p.make_uset(dof=dof, nasset=nasset, xyz=xyz)
    # assert uset2.equals(uset2_man)
    assert np.allclose(uset2.reset_index(), uset2_man.reset_index())
    assert (bset2 == n2p.mksetpv(uset2, "a", "b")).all()

    assert len(cords2) == len(cord2)
    for k, v in cords2.items():
        assert np.allclose(cord2[k], v)


def test_asm2uset_2():
    u, c, b = nastran.asm2uset("pyyeti/tests/nas2cam_extseout/reduced_bset_notall6.asm")
    m = op4.read("pyyeti/tests/nas2cam_extseout/reduced_bset_notall6.op4")

    assert u.shape[0] == 29

    q = ~b
    assert ((np.diag(m["maa"]) == 1.0) == q).all()

    up, cp, bp, pv = nastran.asm2uset(
        "pyyeti/tests/nas2cam_extseout/reduced_bset_notall6.asm", try_rdextrn="pv_only"
    )

    uf, cf, bf = nastran.asm2uset(
        "pyyeti/tests/nas2cam_extseout/reduced_bset_notall6.asm", try_rdextrn=False
    )

    assert up.equals(uf)
    assert up.loc[pv].equals(u)


def test_rdcord2cards():
    cylcoord = np.array([[50, 2, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    sphcoord = np.array([[51, 3, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]])

    uset = n2p.addgrid(
        None,
        [100, 200, 300],
        "b",
        [0, cylcoord, sphcoord],
        [[5, 10, 15], [32, 90, 10], [50, 90, 90]],
        [0, cylcoord, sphcoord],
    )

    with StringIO() as f:
        nastran.uset2bulk(f, uset)
        cords = nastran.rdcord2cards(f)
        u, c = nastran.bulk2uset(f)

    assert len(cords) == len(c)
    for k, v in cords.items():
        assert np.allclose(c[k], v)


def test_rdcord2cards2():
    s1 = """
$1111111222222223333333344444444555555556666666677777777888888889999999900000000
CORD2R       501       0   300.0    4.0 -10.0000  .56000 .200000   -10.7+
+       310.0000     4.0   -11.7
"""

    s2 = """
$1111111222222223333333344444444555555556666666677777777888888889999999900000000
CORD2R       501           300.0    4.0 -10.0000  .56000 .200000   -10.7+
+       310.0000     4.0   -11.7
"""

    with StringIO(s1) as f:
        cords1 = nastran.rdcord2cards(f)

    with StringIO(s2) as f:
        cords2 = nastran.rdcord2cards(f)

    assert np.all(cords1[501] == cords2[501])


def test_rdcord2cards_errors():
    strs = [
        """
    $1111111222222223333333344444444555555556666666677777777888888889999999900000000
    CORD2R       501           300.0    4.0 -10.0000  .56000 .200000   -10.7+
    +       310.0000     4.0   -11.7    1.0
    """,
        """
    $1111111222222223333333344444444555555556666666677777777888888889999999900000000
    CORD2R       501           300.0    4.0 -10.0000  .56000 .200000   -10.7+
    +       310.0a00     4.0   -11.7
    """,
    ]

    for i, s in enumerate(strs):
        with StringIO(s) as f:
            try:
                nastran.rdcord2cards(f)
            except ValueError as e:
                if i == 0:
                    assert e.args[0].startswith("expected 12")
                else:
                    assert e.args[0].startswith("could not convert")


def test_rdcord2cards_13fields():
    # nastran apparently is okay with an empty 13th field:
    s = """
$11111112222222233333333444444445555555566666666777777778888888899999999
CORD2R*         10020001          4000012404.74921408901-9.0898126100692
*       33.94550978374462404.74921408901613.371535432961-2289.1118668606*
*       -.25675367686563-9.089812610069233.9455097837446
*
$11111112222222233333333444444445555555566666666777777778888888899999999
CORD2C*           400001                0.000000000000000.00000000000000
*       0.000000000000000.000000000000000.000000000000001.00000000000000
*       1.000000000000000.000000000000000.00000000000000
*
"""

    with StringIO(s) as f:
        c = nastran.rdcord2cards(f)

    assert 10020001 in c

    # adding a 0.0 should cause trouble:
    s = """
$11111112222222233333333444444445555555566666666777777778888888899999999
CORD2R*         10020001          4000012404.74921408901-9.0898126100692
*       33.94550978374462404.74921408901613.371535432961-2289.1118668606*
*       -.25675367686563-9.089812610069233.94550978374460.0
*
$11111112222222233333333444444445555555566666666777777778888888899999999
CORD2C*           400001                0.000000000000000.00000000000000
*       0.000000000000000.000000000000000.000000000000001.00000000000000
*       1.000000000000000.000000000000000.00000000000000
*
"""

    with StringIO(s) as f:
        try:
            c = nastran.rdcord2cards(f)
        except ValueError as e:
            assert e.args[0].startswith("expected 12")


def test_wtextseout():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    se = 101
    maa = nas["maa"][se]
    kaa = nas["kaa"][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas["uset"][se]
    bset = n2p.mksetpv(uset, "p", "b")
    usetb = nas["uset"][se].iloc[bset]
    b = n2p.mksetpv(uset, "a", "b")
    q = ~b
    b = np.nonzero(b)[0]
    baa = np.zeros_like(maa)
    baa[q, q] = 2 * 0.05 * np.sqrt(kaa[q, q])
    filename = "_wtextseout_test_"

    # test the additional writing of matrices:
    mug1 = np.arange(12).reshape(3, 4)
    mef1 = 10 * mug1
    try:
        nastran.wtextseout(
            filename,
            se=101,
            maa=maa,
            kaa=kaa,
            baa=baa,
            bset=b,
            uset=usetb,
            spoint1=9900101,
            mug1=mug1,
            mef1=mef1,
        )
        names, mats, f, t = op4.load(filename + ".op4", into="list")
        all_names = [
            "kaa",
            "maa",
            "baa",
            "k4xx",
            "pa",
            "gpxx",
            "gdxx",
            "rvax",
            "va",
            "mug1",
            "mug1o",
            "mes1",
            "mes1o",
            "mee1",
            "mee1o",
            "mgpf",
            "mgpfo",
            "mef1",
            "mef1o",
            "mqg1",
            "mqg1o",
            "mqmg1",
            "mqmg1o",
        ]
        assert names == all_names

        for name, mat in zip(names, mats):
            if name in ["maa", "kaa", "baa", "mug1", "mef1"]:
                assert np.allclose(mat, eval(name))
            elif name == "pa":
                assert np.allclose(mat, np.zeros((maa.shape[0], 1)))
            elif name == "va":
                assert np.allclose(mat, np.ones((maa.shape[0], 1)))
            else:
                assert mat.shape == (1, 1)
                assert mat[0, 0] == 0.0
    finally:
        for ext in (".asm", ".pch", ".op4"):
            if os.path.exists(filename + ext):
                os.remove(filename + ext)

    try:
        namelist = ["maa", "kxx"]
        nastran.wtextseout(
            filename,
            se=101,
            kxx=kaa,
            maa=maa,
            bset=b,
            uset=usetb,
            spoint1=9900101,
            namelist=namelist,
            mug1=mug1,
        )
        names, mats, f, t = op4.load(filename + ".op4", into="list")
        assert names == namelist
    finally:
        for ext in (".asm", ".pch", ".op4"):
            if os.path.exists(filename + ext):
                os.remove(filename + ext)


def test_wtextseout2():
    name = Path("pyyeti/tests/nas2cam_extseout")
    mats = op4.read(name / "outboard.op4")
    uset, coords, bset = nastran.asm2uset(name / "outboard.asm")

    pchold = name / "outboard.pch"
    tug1 = nastran.rddtipch(pchold)

    se = nastran.rdcards(
        pchold,
        "begin +super",
        regex=True,
        keep_name=True,
        return_var="list",
    )[0][-1]

    qset = ~bset
    spoint1 = uset.iloc[qset].index[0][0]

    filename = "_wtextseout_test_"

    namelist = list(mats)

    # test the DTI write:
    try:
        nastran.wtextseout(
            filename,
            se=se,
            bset=bset.nonzero()[0],
            uset=uset,
            spoint1=spoint1,
            namelist=namelist,
            tug1=tug1,
            **mats,
        )

        names, mats2, f, t = op4.load(filename + ".op4", into="list")
        assert names == namelist

        pchnew = filename + ".pch"
        for i in range(3):
            old = nastran.rdcards(
                pchold,
                f"DTI +TUG1 +{i}",
                regex=True,
                keep_name=True,
                return_var="list",
            )
            new = nastran.rdcards(
                pchnew,
                f"DTI +TUG1 +{i}",
                regex=True,
                keep_name=True,
                return_var="list",
            )
            assert new == old

        tug1_new = nastran.rddtipch(filename + ".pch")
        assert (tug1_new == tug1).all()

    finally:
        for ext in (".asm", ".pch", ".op4"):
            if os.path.exists(filename + ext):
                os.remove(filename + ext)

    with pytest.raises(ValueError):
        mug1 = mats["mug1"]
        mats["mug1"] = [[33, 12.0]]
        try:
            nastran.wtextseout(
                filename,
                se=se,
                bset=bset.nonzero()[0],
                uset=uset,
                spoint1=spoint1,
                namelist=namelist,
                tug1=tug1,
                **mats,
            )

        finally:
            for ext in (".asm", ".pch", ".op4"):
                if os.path.exists(filename + ext):
                    os.remove(filename + ext)

    with pytest.raises(RuntimeError):
        mats["mug1"] = mug1
        del mats[namelist[0]]
        nastran.wtextseout(
            filename,
            se=se,
            bset=bset.nonzero()[0],
            uset=uset,
            spoint1=spoint1,
            namelist=namelist,
            tug1=tug1,
            **mats,
        )


def test_wtextseout3():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_csuper/nas2cam")
    se = 101
    maa = nas["maa"][se]
    kaa = nas["kaa"][se]
    pv = np.any(maa, axis=0)
    pv = np.ix_(pv, pv)
    maa = maa[pv]
    kaa = kaa[pv]

    uset = nas["uset"][se]
    bset = n2p.mksetpv(uset, "p", "b")
    usetb = nas["uset"][se].iloc[bset]
    b = n2p.mksetpv(uset, "a", "b")

    q = ~b
    b = np.nonzero(b)[0]
    q = np.nonzero(q)[0]

    center = np.mean(usetb.iloc[::6, 1:], axis=0)
    rb = n2p.rbgeom_uset(usetb, center.values)

    # transform to single pt on centerline:
    # [b, q]_old = T*[b, q]_new
    #            = [[rb, 0], [0, I]] * [b, q]_new
    T = np.zeros((len(b) + len(q), 6 + len(q)))
    T[: len(b), :6] = rb
    T[len(b) :, 6:] = np.eye(len(q))

    kaa = T.T @ kaa @ T
    maa = T.T @ maa @ T
    b = np.arange(6)

    filename = "_wtextseout_test_"
    usetnew = n2p.make_uset(999, nasset="b", xyz=center.values)
    try:
        nastran.wtextseout(
            filename,
            se=101,
            maa=maa,
            kaa=kaa,
            baa=0.01 * kaa,
            bset=b,
            uset=usetnew,
            spoint1=9900101,
            forms=None,
        )

        names, mats, fu, t = op4.load(filename + ".op4", into="list")
    finally:
        for ext in (".asm", ".pch", ".op4"):
            if os.path.exists(filename + ext):
                os.remove(filename + ext)

    try:
        nastran.wtextseout(
            filename,
            se=101,
            maa=maa,
            kaa=kaa,
            baa=0.01 * kaa,
            bset=b,
            uset=usetnew,
            spoint1=9900101,
            forms="symmetric",
        )

        names, mats, fs, t = op4.load(filename + ".op4", into="list")
    finally:
        for ext in (".asm", ".pch", ".op4"):
            if os.path.exists(filename + ext):
                os.remove(filename + ext)

    assert fu[0] == 1
    assert fs[0] == 6


def test_rdeigen():
    e1 = nastran.rdeigen("pyyeti/tests/nas2cam_csuper/assemble.out")
    e2 = nastran.rdeigen("pyyeti/tests/nas2cam_csuper/assemble.out", use_pandas=False)

    sbe = np.array(
        [
            2.776567e-05,
            1.754059e-05,
            1.183176e-05,
            1.708013e-05,
            2.299500e-05,
            4.592735e-05,
            1.699652e00,
            1.768612e00,
            1.857731e00,
            3.439703e00,
            7.024192e00,
            7.025385e00,
            1.072738e01,
            1.098313e01,
            1.389833e01,
            1.448323e01,
            1.466003e01,
            1.526510e01,
            2.519912e01,
            2.530912e01,
            2.925036e01,
            4.243738e01,
            4.311826e01,
            4.689425e01,
            4.780881e01,
            6.915960e01,
            8.182875e01,
            9.652563e01,
            9.655103e01,
            9.999950e01,
            1.746837e02,
            1.889342e02,
            1.996603e02,
            2.436533e02,
            2.839537e02,
            3.144806e02,
            4.254619e02,
            4.504501e02,
            5.460081e02,
            6.784015e02,
            7.837016e02,
            8.376910e02,
            8.747553e02,
            8.875936e02,
            9.451668e02,
            9.907786e02,
            1.020666e03,
            1.065056e03,
            1.360919e03,
            1.407037e03,
            1.675989e03,
            1.837844e03,
            1.970020e03,
            5.281664e03,
        ]
    )
    assert np.allclose(sbe, e1[0].cycles)
    assert np.allclose(e2[0], e1[0].values)


def test_rdeigen2():
    # hand-crafted example to trip an old error:
    data = """
1    SYSTEM MODES                                                              JUNE  10, 2015  NX NASTRAN  5/ 1/14   PAGE    72
                                                                                                        SUPERELEMENT 20
0

                                              R E A L   E I G E N V A L U E S
                                         (BEFORE AUGMENTATION OF RESIDUAL VECTORS)
   MODE    EXTRACTION      EIGENVALUE            RADIANS             CYCLES            GENERALIZED         GENERALIZED
    NO.       ORDER                                                                       MASS              STIFFNESS
        1         1       -3.043520E-08        1.744569E-04        2.776567E-05        1.000000E+00       -3.043520E-08
        2         2       -1.214641E-08        1.102107E-04        1.754059E-05        1.000000E+00       -1.214641E-08
        3         3        5.526609E-09        7.434117E-05        1.183176E-05        1.000000E+00        5.526609E-09
        4         4        1.151707E-08        1.073176E-04        1.708013E-05        1.000000E+00        1.151707E-08
        5         5        2.087500E-08        1.444818E-04        2.299500E-05        1.000000E+00        2.087500E-08
        6         6        8.327270E-08        2.885701E-04        4.592735E-05        1.000000E+00        8.327270E-08
1    SYSTEM MODES                                                              JUNE  10, 2015  NX NASTRAN  5/ 1/14   PAGE    73
                                                                                                        SUPERELEMENT 20
0
1    SYSTEM MODES                                                              JUNE  10, 2015  NX NASTRAN  5/ 1/14   PAGE    74
                                                                                                        SUPERELEMENT 0
0

                                              R E A L   E I G E N V A L U E S
                                         (AFTER AUGMENTATION OF RESIDUAL VECTORS)
   MODE    EXTRACTION      EIGENVALUE            RADIANS             CYCLES            GENERALIZED         GENERALIZED
    NO.       ORDER                                                                       MASS              STIFFNESS
        1         1        5.440398E+04        2.332466E+02        3.712235E+01        1.000000E+00        5.440398E+04
        2         2        5.579406E+04        2.362077E+02        3.759362E+01        1.000000E+00        5.579406E+04
        3         3        4.037157E+05        6.353862E+02        1.011249E+02        1.000000E+00        4.037157E+05
        4         4        3.110918E+06        1.763780E+03        2.807142E+02        1.000000E+00        3.110918E+06
        5         5        4.394972E+06        2.096419E+03        3.336554E+02        1.000000E+00        4.394972E+06
        6         6        5.312899E+06        2.304973E+03        3.668478E+02        1.000000E+00        5.312899E+06
        7         7        5.829790E+06        2.414496E+03        3.842789E+02        1.000000E+00        5.829790E+06
        8         8        6.409046E+06        2.531609E+03        4.029181E+02        1.000000E+00        6.409046E+06
"""
    with StringIO(data) as f:
        e = nastran.rdeigen(f)

    cyc20 = [
        2.776567e-05,
        1.754059e-05,
        1.183176e-05,
        1.708013e-05,
        2.299500e-05,
        4.592735e-05,
    ]
    cyc0 = [
        3.712235e01,
        3.759362e01,
        1.011249e02,
        2.807142e02,
        3.336554e02,
        3.668478e02,
        3.842789e02,
        4.029181e02,
    ]
    assert np.allclose(e[20]["cycles"].values, cyc20)
    assert np.allclose(e[0]["cycles"].values, cyc0)


def test_rdeigen3():
    fname = "pyyeti/tests/nas2cam/with_se.out"
    dct = nastran.rdeigen(fname)
    eig300_none = nastran.rdeigen(
        fname, search_strings=("processing of superelement +300", "after augmentation")
    )
    eig300 = nastran.rdeigen(
        fname,
        search_strings=("processing of superelement +300", "after augmentation"),
        regex=True,
    )

    assert eig300_none is None
    assert eig300.equals(dct[300])

    eig300a = nastran.rdeigen(
        fname,
        search_strings=("processing of superelement +300", "after augmentation"),
        regex=True,
        use_pandas=False,
    )

    assert np.allclose(eig300, eig300a)
    assert isinstance(eig300a, np.ndarray)

    eig200 = nastran.rdeigen(
        fname,
        search_strings="AFTER AUGMENTATION",
    )
    assert eig200.equals(dct[200])


def test_wtqcset():
    with StringIO() as f:
        nastran.wtqcset(f, 990001, 5)
        assert f.getvalue() == ("QSET1      12345  990001\nCSET1          6  990001\n")

    with StringIO() as f:
        nastran.wtqcset(f, 990001, 6)
        assert f.getvalue() == ("QSET1     123456  990001\n")

    with StringIO() as f:
        nastran.wtqcset(f, 990001, 7)
        assert f.getvalue() == (
            "QSET1     123456  990001\n"
            "QSET1          1  990002\n"
            "CSET1      23456  990002\n"
        )

    with StringIO() as f:
        nastran.wtqcset(f, 990001, 12)
        assert f.getvalue() == ("QSET1     123456  990001 THRU     990002\n")


def test_wtrbe3_errors():
    # Ind_List has an odd length
    with pytest.raises(ValueError, match=r"Ind_List.*even length"):
        nastran.wtrbe3(1, 100, 9900, 123456, [1, 2, 3])
    # UM_List has an odd length
    with pytest.raises(ValueError, match=r"UM_List.*even length"):
        nastran.wtrbe3(1, 100, 9900, 123456, [1, 2], [101, 1, 102])


def test_wtrbe3():
    eid = 100
    GRID_dep, DOF_dep = 9900, 123456
    Ind_List = [123, [9901, 9902, 9903, 9904]]
    with StringIO() as f:
        nastran.wtrbe3(f, eid, GRID_dep, DOF_dep, Ind_List)
        expected = (
            "RBE3         100            9900  123456      1.     123    9901    9902\n"
            "+           9903    9904\n"
        )
        assert f.getvalue() == expected

    eid = 100
    GRID_dep, DOF_dep = 9900, 123456
    Ind_List = [[123, 0.8], [9901, 9902, 9903, 9904], 23456, 9905]
    alpha = "1.0E-8"
    with StringIO() as f:
        nastran.wtrbe3(f, eid, GRID_dep, DOF_dep, Ind_List, alpha=alpha)
        expected = (
            "RBE3         100            9900  123456      .8     123    9901    9902\n"
            "+           9903    9904      1.   23456    9905\n"
            "+       ALPHA       1.-8\n"
        )
        assert f.getvalue() == expected

    eid = 100
    GRID_dep, DOF_dep = 9900, 123456
    Ind_List = [
        123,  # DOF
        [9901, 9902, 9903, 9904],  # grids
    ]
    UM_List = [9901, 12, 9902, 3, 9903, 12]
    alpha = 2.0e-8
    with StringIO() as f:
        nastran.wtrbe3(f, eid, GRID_dep, DOF_dep, Ind_List, UM_List, alpha)
        expected = (
            "RBE3         100            9900  123456      1.     123    9901    9902\n"
            "+           9903    9904\n"
            "+       UM          9901      12    9902       3    9903      12\n"
            "+       ALPHA       2.-8\n"
        )
        assert f.getvalue() == expected

    eid = 100
    GRID_dep, DOF_dep = 9900, 123456
    Ind_List = [
        123,  # DOF
        [9901, 9902, 9903, 9904],  # grids
        [123456, 1.2],  # DOF and weight
        [450001, 200],  # grids
        345,  # DOF
        [9905],  # grids
    ]
    UM_List = [9901, 12, 9902, 3, 9903, 12, 904, 3]
    alpha = "6.5e-8"
    with StringIO() as f:
        nastran.wtrbe3(f, eid, GRID_dep, DOF_dep, Ind_List, UM_List, alpha)
        expected = (
            "RBE3         100            9900  123456      1.     123    9901    9902\n"
            "+           9903    9904     1.2  123456  450001     200      1.     345\n"
            "+           9905\n"
            "+       UM          9901      12    9902       3    9903      12        \n"
            "+                    904       3\n"
            "+       ALPHA      6.5-8\n"
        )
        assert f.getvalue() == expected


def test_wtseset_bad_inputs():
    f = StringIO()
    superid = 101
    grids = []
    with pytest.raises(ValueError, match=r"grids.*length.*>0"):
        nastran.wtxset1(f, superid, grids)


def test_wtset_bad_inputs():
    f = StringIO()
    setid = 101
    ids = []
    with pytest.raises(ValueError, match=r"ids.*length.*>0"):
        nastran.wtxset1(f, setid, ids)


def test_wtset():
    with StringIO() as f:
        nastran.wtset(f, 101, [1, 2, 3, 4])
        expected = "SET 101 = 1 THRU 4"
        assert f.getvalue() == expected

    with StringIO() as f:
        nastran.wtset(f, 101, [9, 2, 4, 6, 8, 10, 12, 14, 15, 16], max_length=35)
        expected = "SET 101 = 9, 2, 4, 6, 8, 10, 12, \n" "14 THRU 16"
        assert f.getvalue() == expected

    with StringIO() as f:
        nastran.wtset(f, 101, [9, 2, 3, 4, 6, 8, 10, 12, 14, 15, 16], max_length=35)
        expected = "SET 101 = 9, 2 THRU 4, 6, 8, 10, \n" "12, 14 THRU 16"
        assert f.getvalue() == expected


def test_rdgpwg():
    # get third table:
    s1 = "W E I G H T"
    mass, cg, ref, Is = nastran.rdgpwg(
        "pyyeti/tests/nas2cam_extseout/assemble.out", [s1, s1]
    )
    r = 0
    m = np.array(
        [
            [
                3.345436e00,
                1.598721e-13,
                -1.132427e-12,
                -1.873559e-10,
                5.018153e02,
                -5.018153e02,
            ],
            [
                1.622036e-13,
                3.345436e00,
                -1.922240e-12,
                -5.018153e02,
                2.731554e-09,
                2.118899e03,
            ],
            [
                -1.133316e-12,
                -1.928013e-12,
                3.345436e00,
                5.018153e02,
                -2.118899e03,
                -1.996398e-09,
            ],
            [
                -1.874909e-10,
                -5.018153e02,
                5.018153e02,
                5.433826e05,
                -3.178349e05,
                -3.178349e05,
            ],
            [
                5.018153e02,
                2.734168e-09,
                -2.118899e03,
                -3.178349e05,
                2.441110e06,
                -7.527230e04,
            ],
            [
                -5.018153e02,
                2.118899e03,
                -1.992703e-09,
                -3.178349e05,
                -7.527230e04,
                2.772279e06,
            ],
        ]
    )
    c = np.array(
        [
            [3.345436e00, -5.600344e-11, 1.500000e02, 1.500000e02],
            [3.345436e00, 6.333702e02, 8.165016e-10, 1.500000e02],
            [3.345436e00, 6.333702e02, 1.500000e02, -5.967527e-10],
        ]
    )
    i = np.array(
        [
            [3.928379e05, 5.339971e-07, 6.432529e-07],
            [5.339971e-07, 1.023790e06, -2.849381e-06],
            [6.432529e-07, -2.849381e-06, 1.354959e06],
        ]
    )
    assert np.allclose(m, mass)
    assert np.allclose(c, cg)
    assert r == ref
    assert np.allclose(i, Is)

    a = nastran.rdgpwg("pyyeti/tests/nas2cam_extseout/assemble.out", "asdfsadfasdf")
    for i in a:
        assert i is None

    a = nastran.rdgpwg(
        "pyyeti/tests/nas2cam_extseout/assemble.out", (s1, s1, "END OF JOB")
    )
    for i in a:
        assert i is None


def test_fsearch():
    with open("pyyeti/tests/nas2cam_extseout/assemble.out") as f:
        a, p = nastran.fsearch(f, "asdfadfadfadsfasf")
    assert a is None
    assert p is None


def test_wtmpc_bad_inputs():
    f = StringIO()
    setid = 101
    gid_dof_d = np.array([21, 1], "i8")
    coeff_d = -1.0
    gid_dof_i = np.array([[31, 1], [31, 2], [31, 3]], "i8")
    coeffs_i = np.array([0.25, 0.60, 0.15], "f8")

    setid = 0
    with pytest.raises(ValueError, match=r"setid.*>0"):
        nastran.wtmpc(f, setid, gid_dof_d, coeff_d, gid_dof_i, coeffs_i)
    setid = 101

    gid_dof_d = np.array([0, 0, 0], "i8")
    with pytest.raises(ValueError, match=r"gid_dof.*length 2"):
        nastran.wtmpc(f, setid, gid_dof_d, coeff_d, gid_dof_i, coeffs_i)
    gid_dof_d = np.array([21, 1], "i8")

    coeffs_i = np.array([], "f8")
    with pytest.raises(ValueError, match=r"must have.*one.*indep.*DOF"):
        nastran.wtmpc(f, setid, gid_dof_d, coeff_d, gid_dof_i, coeffs_i)
    coeffs_i = np.array([0.25, 0.60, 0.15], "f8")

    gid_dof_i = np.array([[31, 1], [31, 2]], "i8")
    with pytest.raises(ValueError, match=r"rows in gid_dof_i.*coeffs_i"):
        nastran.wtmpc(f, setid, gid_dof_d, coeff_d, gid_dof_i, coeffs_i)

    gid_dof_i = np.array([[31, 1, 0], [31, 2, 0], [31, 3, 0]], "i8")
    with pytest.raises(ValueError, match=r"gid_dof_i.*two columns"):
        nastran.wtmpc(f, setid, gid_dof_d, coeff_d, gid_dof_i, coeffs_i)


def test_wtmpc_1coeff():
    with StringIO() as f:
        setid = 101
        id_dof_d = np.array([21, 1])
        coeff_d = -1.0
        id_dof_i = np.array([[31, 1]])
        coeffs_i = np.array([0.75])
        nastran.wtmpc(f, setid, id_dof_d, coeff_d, id_dof_i, coeffs_i)
        s = f.getvalue()
    sbe = (
        "MPC*                 101              21               1             -1.\n"
        "*                     31               1             .75\n"
    )
    assert s == sbe


def test_wtmpc_2coeff():
    with StringIO() as f:
        setid = 101
        id_dof_d = np.array([21, 1])
        coeff_d = -1.0
        id_dof_i = np.array([[31, 1], [31, 2]])
        coeffs_i = np.array([0.75, 0.25])
        nastran.wtmpc(f, setid, id_dof_d, coeff_d, id_dof_i, coeffs_i)
        s = f.getvalue()
    sbe = (
        "MPC*                 101              21               1             -1.\n"
        "*                     31               1             .75                *\n"
        "*                                     31               2             .25\n"
        "*\n"
    )
    assert s == sbe


def test_wtmpc_3coeff():
    with StringIO() as f:
        setid = 101
        id_dof_d = np.array([21, 1])
        coeff_d = -1.0
        id_dof_i = np.array([[31, 1], [31, 2], [32, 5]])
        coeffs_i = np.array([0.75, 0.25, 1.65])
        nastran.wtmpc(f, setid, id_dof_d, coeff_d, id_dof_i, coeffs_i)
        s = f.getvalue()
    sbe = (
        "MPC*                 101              21               1             -1.\n"
        "*                     31               1             .75                *\n"
        "*                                     31               2             .25\n"
        "*                     32               5            1.65\n"
    )
    assert s == sbe


def test_find_sequence_bad_input():
    from pyyeti.nastran.bulk import _find_sequence

    with pytest.raises(ValueError, match=r"start out of bounds.*4,.*5"):
        _find_sequence([1, 2, 3, 4], 5)
    with pytest.raises(ValueError, match=r"start out of bounds.*4,.*-1"):
        _find_sequence([1, 2, 3, 4], -1)


@pytest.mark.parametrize(
    ("seq", "start", "expected"),
    [
        ([1, 2, 3, 5, 7], 0, 2),
        ([1, 2, 3, 5, 7], 1, 2),
        ([1, 2, 3, 5, 7], 2, 2),
        ([1, 2, 3, 5, 7], 3, 3),
        ([1, 2, 3, 5, 7, 8], 4, 5),
    ],
)
def test_find_sequence(seq, start, expected):
    from pyyeti.nastran.bulk import _find_sequence

    output = _find_sequence(seq, start)
    assert output == expected


def test_wtspoints_bad_inputs():
    f = StringIO()
    spoints = []
    with pytest.raises(ValueError, match=r"spoints.*length.*>0"):
        nastran.wtspoints(f, spoints)


def test_wtspoints():
    with StringIO() as f:
        spoints = list(range(1001, 1016))
        nastran.wtspoints(f, spoints)
        s = f.getvalue()
    sbe = "SPOINT      1001THRU        1015\n"
    assert s == sbe

    with StringIO() as f:
        spoints = [1001, 1003, 1004, 1005, 1007, 1009]
        nastran.wtspoints(f, spoints)
        s = f.getvalue()
    sbe = (
        "SPOINT      1001\n"
        "SPOINT      1003THRU        1005\n"
        "SPOINT      1007    1009\n"
    )
    assert s == sbe

    with StringIO() as f:
        spoints = [1001, 1002, 1004, 1006, 1007, 1009]
        nastran.wtspoints(f, spoints)
        s = f.getvalue()
    sbe = (
        "SPOINT      1001THRU        1002\n"
        "SPOINT      1004\n"
        "SPOINT      1006THRU        1007\n"
        "SPOINT      1009\n"
    )
    assert s == sbe

    with StringIO() as f:
        spoints = list(reversed(range(1001, 1018)))
        nastran.wtspoints(f, spoints)
        s = f.getvalue()
    sbe = (
        "SPOINT      1017    1016    1015    1014    1013    1012    1011    1010\n"
        "SPOINT      1009    1008    1007    1006    1005    1004    1003    1002\n"
        "SPOINT      1001\n"
    )
    assert s == sbe


def test_wtxset1_bad_inputs():
    f = StringIO()
    dof = 123456
    grids = []
    with pytest.raises(ValueError, match=r"grids.*length.*>0"):
        nastran.wtxset1(f, dof, grids)


def test_wtxset1():
    with StringIO() as f:
        dof = 123456
        grids = [1001, 1003, 1005]
        nastran.wtxset1(f, dof, grids)
        s = f.getvalue()
    sbe = "BSET1     123456    1001    1003    1005\n"
    assert s == sbe

    with StringIO() as f:
        dof = 123456
        grids = [1001, 1002, 1003]
        nastran.wtxset1(f, dof, grids)
        s = f.getvalue()
    sbe = "BSET1     123456    1001THRU        1003\n"
    assert s == sbe

    with StringIO() as f:
        dof = 123456
        grids = [1001, 1003, 1005, 1006, 1007, 1009]
        nastran.wtxset1(f, dof, grids, name="QSET1")
        s = f.getvalue()
    sbe = (
        "QSET1     123456    1001    1003\n"
        "QSET1     123456    1005THRU        1007\n"
        "QSET1     123456    1009\n"
    )
    assert s == sbe

    with StringIO() as f:
        dof = 123
        grids = list(reversed(range(1001, 1010)))
        nastran.wtxset1(f, dof, grids)
        s = f.getvalue()
    sbe = (
        "BSET1        123    1009    1008    1007    1006    1005    1004    1003\n"
        "BSET1        123    1002    1001\n"
    )
    assert s == sbe


def test_wtrspline():
    with pytest.raises(ValueError):
        nastran.wtrspline(1, 1, 1)
    ids = np.array(
        [
            [100, 1],
            [101, 0],
            [102, 0],
            [103, 1],
            [104, 0],
            [105, 1],
            [106, 0],
            [107, 0],
            [108, 0],
            [109, 1],
            [110, 0],
            [111, 0],
            [112, 1],
        ]
    )
    with StringIO() as f:
        nastran.wtrspline(f, 10, ids)
        s = f.getvalue()

    sbe = (
        "RSPLINE       10     0.1     100     101  123456     102  123456     103\n"
        "RSPLINE       11     0.1     103     104  123456     105\n"
        "RSPLINE       12     0.1     105     106  123456     107  123456     108\n"
        "          123456     109\n"
        "RSPLINE       13     0.1     109     110  123456     111  123456     112\n"
    )
    assert sbe == s

    # test for first and last must be independent error:
    ids[-1, 1] = 0
    with pytest.raises(ValueError):
        nastran.wtrspline(1, 10, ids)

    # test for "no independents" error:
    ids[:, 1] = 1
    with pytest.raises(ValueError):
        nastran.wtrspline(1, 10, ids)


def test_wtrspline_rings():
    theta1 = np.arange(0, 359, 360 / 5) * np.pi / 180
    rad1 = 50.0
    sta1 = 0.0
    n1 = len(theta1)
    ring1 = np.vstack(
        (
            np.arange(1, n1 + 1),  # ID
            sta1 * np.ones(n1),  # x
            rad1 * np.cos(theta1),  # y
            rad1 * np.sin(theta1),
        )
    ).T  # z
    theta2 = np.arange(10, 359, 360 / 7) * np.pi / 180
    rad2 = 45.0
    sta2 = 1.0
    n2 = len(theta2)
    ring2 = np.vstack(
        (
            np.arange(1, n2 + 1) + 100,  # ID
            sta2 * np.ones(n2),  # x
            rad2 * np.cos(theta2),  # y
            rad2 * np.sin(theta2),
        )
    ).T  # z

    uset1 = (
        "Name 1",
        n2p.addgrid(None, ring1[:, 0].astype(int), "b", 0, ring1[:, 1:], 0),
    )
    uset2 = (
        "Name 2",
        n2p.addgrid(None, ring2[:, 0].astype(int), "b", 0, ring2[:, 1:], 0),
    )

    fig = plt.figure("rspline demo", figsize=(8, 6))
    fig.clf()
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    with StringIO() as f:
        nastran.wtrspline_rings(
            f, ring1, ring2, 1001, 2001, makeplot=ax, independent="ring1"
        )
        u, c = nastran.bulk2uset(f)
        rsplines1 = nastran.rdcards(f, "RSPLINE")

    with StringIO() as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            nastran.wtrspline_rings(
                f, uset1, uset2, 1001, 2001, makeplot=ax, independent="n amE 2"
            )
        u2, c2 = nastran.bulk2uset(f)
        rsplines2 = nastran.rdcards(f, "RSPLINE")

    assert np.allclose(u, u2)
    for k, v in c2.items():
        assert np.allclose(c[k], v)

    # x coord of new grids should be same as ring 2 ... 1.0:
    assert np.allclose(1, u.loc[(slice(None), 1), "x"])

    # y, z coords should be like ring1, but with reduced radius to
    # match ring 2:
    assert np.allclose(ring1[:, 2:] * rad2 / rad1, u.loc[(slice(None), 1), "y":"z"])

    # the local coord:
    #  z_local is x
    #  x_local points through node 1 ... which is on y
    to_local = [[0, 1.0, 0], [0, 0, 1], [1.0, 0, 0]]
    assert np.allclose(c[10010][-3:].T, to_local)

    # rsplines1 should have the 1001 series of numbers being
    # independent:
    assert (rsplines1[:, 2] > 1000).all()

    # rsplines2 should have the 101 series of numbers being
    # independent:
    assert (rsplines2[:, 2] < 1000).all()

    with pytest.raises(ValueError):
        nastran.wtrspline_rings(
            1,
            uset1,
            uset2,
            1001,
            2001,
            makeplot="no",
            independent="badoption",
        )

    uset1 = uset1[1]
    with pytest.raises(ValueError):
        nastran.wtrspline_rings(
            1,
            uset1[:-1],
            uset2,
            1001,
            2001,
            makeplot="no",
        )

    uset3 = None
    for row in ring2:
        uset3 = n2p.addgrid(uset3, int(row[0]), "b", 0, [row[3], row[1], row[2]], 0)
    with pytest.raises(ValueError):
        nastran.wtrspline_rings(1, uset1, uset3, 1001, 2001, makeplot="no")


def test_wtcoordcards():
    with StringIO() as f:
        nastran.wtcoordcards(f, None)
        assert f.getvalue() == ""
        nastran.wtcoordcards(f, {})
        assert f.getvalue() == ""


def test_mknast():
    name = "_test_mknast_.sh"
    with tempfile.TemporaryDirectory() as tempdir_path:
        path = os.path.join(tempdir_path, name)
        with pytest.warns(RuntimeWarning, match=r"file.*not found"):
            nastran.mknast(
                path,
                stoponfatal="yes",
                files=["tt.py", "tt", "doruns.sh", "subd/t.t"],
                before="# BEFORE",
                after="# AFTER",
                top="# TOP",
                bottom="# BOTTOM",
            )
        with open(path) as f:
            s = f.read().splitlines()

    sbe = [
        "#!/bin/sh",
        "cd --",
        "",
        "# TOP",
        "",
        "# ******** File tt.py ********",
        "# BEFORE",
        "  nast9p1 'batch=no' 'tt.py'",
        "  if [ X != X`grep -l '[*^][*^][*^].*FATAL' tt.out` ] ; then",
        "    exit",
        "  fi",
        "# AFTER",
        "",
        "# ******** File tt ********",
        "# BEFORE",
        "  nast9p1 'batch=no' 'tt'",
        "  if [ X != X`grep -l '[*^][*^][*^].*FATAL' tt.out` ] ; then",
        "    exit",
        "  fi",
        "# AFTER",
        "",
        "# ******** File doruns.sh ********",
        "# BEFORE",
        "  nast9p1 'batch=no' 'doruns.sh'",
        "  if [ X != X`grep -l '[*^][*^][*^].*FATAL' doruns.out` ] ; then",
        "    exit",
        "  fi",
        "# AFTER",
        "",
        "# ******** File subd/t.t ********",
        "  cd subd",
        "# BEFORE",
        "  nast9p1 'batch=no' 't.t'",
        "  if [ X != X`grep -l '[*^][*^][*^].*FATAL' t.out` ] ; then",
        "    exit",
        "  fi",
        "# AFTER",
        "  cd --",
        "# BOTTOM",
    ]

    for i, st in enumerate(sbe):
        if st == "  cd --":
            assert s[i][:4] == "  cd"
        elif st == "cd --":
            assert s[i][:2] == "cd"
        else:
            print(s[i])
            print(st)
            assert s[i] == st


def test_rddtipch():
    d = nastran.rddtipch("pyyeti/tests/nas2cam_csuper/fake_dtipch.pch", "TEF1")
    dof = [
        (10, 8),
        (97, 8),
        (3140051, 8),
        (3000108, 77),
        (3000113, 77),
        (3000299, 77),
        (3000310, 77),
        (3000330, 77),
    ]
    n = 0
    for i in dof:
        n += i[1]
    sbe = np.empty((n, 2), dtype=np.int64)
    n = 0
    for i in dof:
        sbe[n : n + i[1], 0] = i[0]
        sbe[n : n + i[1], 1] = np.arange(1, i[1] + 1)
        n += i[1]
    assert np.all(d == sbe)


def test_rddmig():
    dct = nastran.rddmig("pyyeti/tests/nastran_dmig_data/matrix_factory.pch")
    dct2 = nastran.rddmig("pyyeti/tests/nastran_dmig_data/matrix.op2")

    for key, val in dct.items():
        val2 = dct2["m" + key]
        assert np.allclose(val, val2)

    assert np.all((dct["ident"] == np.eye(dct["ident"].shape[0])).values)
    pattern_mat = np.empty(dct["patrn"].shape)
    for i in range(dct["patrn"].shape[1]):
        pattern_mat[:, i] = i + 1
    assert np.all((pattern_mat == dct["patrn"]).values)

    assert sorted(dct.keys()) == ["cmplx", "ident", "patrn", "randm"]

    dct = nastran.rddmig(
        "pyyeti/tests/nastran_dmig_data/matrix_factory.pch", ("patrn", "randm")
    )
    dct2 = nastran.rddmig(
        "pyyeti/tests/nastran_dmig_data/matrix.op2", ("mpatrn", "mrandm")
    )

    for key, val in dct.items():
        val2 = dct2["m" + key]
        assert np.allclose(val, val2)

    assert sorted(dct.keys()) == ["patrn", "randm"]


def test_rddmig2():
    dct = nastran.rddmig("pyyeti/tests/nastran_dmig_data/matrix.op2", "mrandm")

    # chop out some DOF so we can test 'expanded':
    slc = (slice(None), slice(3))
    mrandm = dct["mrandm"].loc[slc, slc]

    with StringIO() as f:
        nastran.wtdmig(f, dict(mrandm=mrandm))
        default = nastran.rddmig(f)
        expanded = nastran.rddmig(f, expanded=True)

    assert np.allclose(expanded["mrandm"].sum().sum(), default["mrandm"].sum().sum())
    assert default["mrandm"].shape == (12, 12)
    assert expanded["mrandm"].shape == (21, 21)
    assert np.allclose(mrandm, default["mrandm"])
    assert np.allclose(mrandm, expanded["mrandm"].loc[slc, slc])
    slc2 = (slice(None), slice(4, None))
    assert np.all(expanded["mrandm"].loc[slc2, slc2] == 0.0)

    # test symmetric writing/reading:
    symm = np.arange(12)[:, None] * np.arange(12)
    symm = symm.T @ symm
    default["mrandm"].iloc[:, :] = symm.astype(float)

    with StringIO() as f:
        nastran.wtdmig(f, default)
        dct = nastran.rddmig(f)
        s = f.getvalue()

    assert np.allclose(dct["mrandm"], default["mrandm"].iloc[1:, 1:])
    assert s.startswith(
        "DMIG    MRANDM         0       6       2       0       0"
        "              12\n"
        "DMIG*   MRANDM                         1               2\n"
        "*                      1               2 5.060000000D+02\n"
        "*                      1               3 1.012000000D+03\n"
        "*                      2               1 1.518000000D+03\n"
        "*                      2               2 2.024000000D+03\n"
        "*                      2               3 2.530000000D+03\n"
        "*                      3               1 3.036000000D+03\n"
        "*                      3               2 3.542000000D+03\n"
        "*                      3               3 4.048000000D+03\n"
        "*                     10               0 4.554000000D+03\n"
        "*                     11               0 5.060000000D+03\n"
        "*                     12               0 5.566000000D+03\n"
        "DMIG*   MRANDM                         1               3\n"
    )

    assert s.endswith(
        "DMIG*   MRANDM                        10               0\n"
        "*                     10               0 4.098600000D+04\n"
        "*                     11               0 4.554000000D+04\n"
        "*                     12               0 5.009400000D+04\n"
        "DMIG*   MRANDM                        11               0\n"
        "*                     11               0 5.060000000D+04\n"
        "*                     12               0 5.566000000D+04\n"
        "DMIG*   MRANDM                        12               0\n"
        "*                     12               0 6.122600000D+04\n"
    )


@pytest.mark.parametrize(
    ("c", "match"),
    [
        (1234567, r"invalid input"),
        (1023456, r"invalid input"),
    ],
)
def test_integer_to_dofs_errors(c: int, match: str):
    from pyyeti.nastran import bulk

    with pytest.raises(ValueError, match=match):
        bulk._integer_to_dofs(c)


@pytest.mark.parametrize(
    ("c", "expected"),
    [
        (123456, [1, 2, 3, 4, 5, 6]),
        (325, [2, 3, 5]),
        (65421, [1, 2, 4, 5, 6]),
        (0, [0]),
    ],
)
def test_integer_to_dofs(c: int, expected: list):
    from pyyeti.nastran import bulk

    output = bulk._integer_to_dofs(c)
    assert output == expected


@pytest.mark.parametrize(
    ("string", "expected"),
    [
        # no errors with empty file
        ("", []),
        # no errors with invalid card type
        ("RVDOF2,0,1003,1004", []),
        # RVDOF input, two lines
        (
            "RVDOF,1002,456,1001,123\nRVDOF,1003,0",
            [
                (1001, 1),
                (1001, 2),
                (1001, 3),
                (1002, 4),
                (1002, 5),
                (1002, 6),
                (1003, 0),
            ],
        ),
        # verify no error when card contains empty fields
        (
            "RVDOF,1002,456,1001,123,",
            [(1001, 1), (1001, 2), (1001, 3), (1002, 4), (1002, 5), (1002, 6)],
        ),
        # RVDOF1 without a THRU
        ("RVDOF1,13,1001,1002", [(1001, 1), (1001, 3), (1002, 1), (1002, 3)]),
        # RVDOF1 with a THRU
        ("RVDOF1,0,1001,THRU,1003", [(1001, 0), (1002, 0), (1003, 0)]),
        # mixed RVDOF and RVDOF1 inputs
        (
            "RVDOF,1002,456,1001,123\nRVDOF1,0,1003,THRU,1005",
            [
                (1001, 1),
                (1001, 2),
                (1001, 3),
                (1002, 4),
                (1002, 5),
                (1002, 6),
                (1003, 0),
                (1004, 0),
                (1005, 0),
            ],
        ),
        (
            # verify no error when card contains empty fields
            "RVDOF1,0,1003,1006,1005,,",
            [(1003, 0), (1005, 0), (1006, 0)],
        ),
        (
            # RVDOF1 card with only one grid
            "RVDOF1,0,1003",
            [(1003, 0)],
        ),
    ],
)
def test_rdrvdof(string: str, expected: list):
    output = nastran.rdrvdof(StringIO(string))
    assert output == expected


def test_rdseconct():
    s = """
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
$ ASSEMBLY PUNCH (.ASM) FILE FOR EXTERNAL SUPERELEMENT      200
$ -------------------------------------------------------------
$
$ THIS FILE CONTAINING BULK DATA ENTRIES PERTAINING TO
$ EXTERNAL SUPERELEMENT      200 IS MEANT FOR INCLUSION
$ ANYWHERE IN THE MAIN BULK DATA PORTION OF THE ASSEMBLY RUN
$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$--------------------------- COLUMN NUMBERS ----------------------------
$00000000111111111122222222223333333333444444444455555555556666666666777
$23456789012345678901234567890123456789012345678901234567890123456789012
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$
SEBULK       200  EXTOP4          MANUAL                      32
$
SECONCT      200       0
               5       5       7       7      16      16      18      18
              27      27      29      29
         9920001    THRU 9920020 9920001    THRU 9920020
SPOINT   9920001    THRU 9920020
$
$ BOUNDARY GRID DATA
$
GRID    5               4.      2.      0.
GRID    7               6.      2.      0.
GRID    16              4.      1.      0.
GRID    18              6.      1.      0.
GRID    27              4.      0.      0.
GRID    29              6.      0.      0.
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
"""
    a, b = nastran.rdseconct(StringIO(s))
    assert (a == b).all()
    sbe = np.r_[5, 7, 16, 18, 27, 29, 9920001:9920021]
    assert (a == sbe).all()


def test_wrap_text_lines():
    from pyyeti.nastran.bulk import _wrap_text_lines

    # bad input
    with pytest.raises(ValueError, match=r"length.*sep.*max_length"):
        _wrap_text_lines([], 5, "123456")

    # arbitrary strings
    values = (
        (([], 12, "/"), []),
        ((["abcdefghijkl"], 12, "/"), ["abcdefghijkl"]),
        ((["abcdefghijkl"], 10, "/"), ["abcdefghi", "jkl"]),
        (
            (["a" * 45, "b" * 8, "c" * 6], 24, ","),
            ["a" * 23, "a" * 22 + ",", "b" * 8 + "," + "c" * 6],
        ),
        (
            (["a" * 12, "b" * 11], 24, ","),
            ["a" * 12 + "," + "b" * 11],
        ),
        (
            (["a" * 12, "b" * 11, "c"], 24, ","),
            ["a" * 12 + ",", "b" * 11 + ",c"],
        ),
        (
            (["a" * 10, "b" * 10, "c"], 24, "++"),
            ["a" * 10 + "++" + "b" * 10 + "++", "c"],
        ),
        (
            (["a" * 11, "b" * 10, "c"], 24, "++"),
            ["a" * 11 + "++", "b" * 10 + "++c"],
        ),
    )
    for (lines, max_length, sep), expected in values:
        output = _wrap_text_lines(lines, max_length, sep)
        assert output == expected

    # paths
    values = (
        (
            ("/home/abcdefg/this/is/a/long/path/modes.op4", 14),
            ["/home/abcdefg/", "this/is/a/", "long/path/", "modes.op4"],
        ),
        (
            ("/home/abcd/this/is/a/long/path/modes.op4", 15),
            ["/home/abcd/", "this/is/a/long/", "path/modes.op4"],
        ),
        (
            ("/home/abcd/this/is/aa/long/path/modes.op4", 15),
            ["/home/abcd/", "this/is/aa/", "long/path/", "modes.op4"],
        ),
        (
            ("/home/abcd/this/is/a/long/path/modes.op4", 20),
            ["/home/abcd/this/is/", "a/long/path/", "modes.op4"],
        ),
    )
    for (path, max_length), expected in values:
        lines = path.split("/")
        output = _wrap_text_lines(lines, max_length, "/")
        assert output == expected


def test_relative_paths():
    from pyyeti.nastran.bulk import _relative_path

    values = (
        # Windows, no current_path
        ((r"C:\direc1\bulk.bdf", None), "c:/direc1/bulk.bdf"),
        # Windows, current_path
        ((r"C:\direc1\bulk.bdf", r"c:\direc1"), "bulk.bdf"),
        # Windows, different drive letters
        ((r"C:\direc1\bulk.bdf", r"D:\direc1"), "c:/direc1/bulk.bdf"),
        # Posix, no current_path
        (("/direc1/bulk.bdf", None), "/direc1/bulk.bdf"),
        # Posix, current_path
        (("/direc1/Bulk.bdf", "/direc1/direc2"), "../Bulk.bdf"),
    )
    for (path, current_path), expected in values:
        output = _relative_path(path, current_path)
        assert output == expected


def test_wtinclude_bad_input():
    f = StringIO()
    with pytest.raises(ValueError, match=r"max_length.*>=24"):
        nastran.wtinclude(f, "bulk.bdf", max_length=20)


def test_wtinclude():
    values = (
        ((r"c:\direc\bulk.bdf", r"c:\direc"), "INCLUDE 'bulk.bdf'\n"),
        # different capitalization
        ((Path(r"d:\direc\bulk.bdf"), Path(r"D:\direc")), "INCLUDE 'bulk.bdf'\n"),
        ((r"c:\direc\bulk.bdf", r"c:\\"), "INCLUDE 'direc/bulk.bdf'\n"),
        ((r"c:\dir1\dir2\bulk.bdf", r"c:\dir1\dir3"), "INCLUDE '../dir2/bulk.bdf'\n"),
        (("/dir1/dir2/bulk3.bdf", "/dir1"), "INCLUDE 'dir2/bulk3.bdf'\n"),
        # no current path, windows
        ((r"c:\direc1\direc2\bulk.bdf", None), "INCLUDE 'c:/direc1/direc2/bulk.bdf'\n"),
        # no current path, posix
        (("/direc1/direc2/bulk.bdf", None), "INCLUDE '/direc1/direc2/bulk.bdf'\n"),
        # different drive letters, always use absolute path
        ((r"c:\direc1\bulk.bdf", r"d:\direc1"), "INCLUDE 'c:/direc1/bulk.bdf'\n"),
        # long path, wrap at separators, default max_length
        (
            (
                (
                    r"c:\direc1\direc2\direc3\direc4\direc5\direc6\direc7\direc8\direc9"
                    r"\direc10\direc11\bulk.bdf"
                ),
                r"c:\\",
            ),
            (
                "INCLUDE 'direc1/direc2/direc3/direc4/direc5/direc6/direc7/direc8/direc9/\n"
                "direc10/direc11/bulk.bdf'\n"
            ),
        ),
        # long path, wrap in middle of directory name
        (
            ("/direc1/thisisareallylongdirectoryname/bulk3.bdf", None, 25),
            "INCLUDE '/direc1/\nthisisareallylongdirecto\nryname/bulk3.bdf'\n",
        ),
    )
    for args, expected in values:
        f = StringIO()
        nastran.wtinclude(f, *args)
        assert f.getvalue() == expected


def test_wtassign_bad_input():
    f = StringIO()
    with pytest.raises(ValueError, match=r"invalid assign_type.*'zzz'"):
        nastran.wtassign(f, "zzz", "output.op4")
    with pytest.raises(ValueError, match=r"max_length.*\>=24"):
        nastran.wtassign(f, "input4", "output.op4", max_length=20)


def test_wtassign():
    values = (
        # no params, no current_path
        (
            ("input2", "/direc1/direc2/input.op2"),
            "ASSIGN INPUTT2 = '/direc1/direc2/input.op2'\n",
        ),
        # no params, no current_path, Windows path
        (
            ("input2", r"c:\direc1\direc2\input.op2"),
            "ASSIGN INPUTT2 = 'c:/direc1/direc2/input.op2'\n",
        ),
        # params, current_path
        (
            (
                "input2",
                Path("/direc1/direc2/input.op2"),
                {"unit": 101},
                Path("/direc1/direc3"),
            ),
            "ASSIGN INPUTT2 = '../direc2/input.op2',UNIT=101\n",
        ),
        # params, no curren_path, wrap inside path
        (
            (
                "output4",
                "/direc1/direc2/output4.op4",
                {"unit": 101, "delete": None},
                None,
                24,
            ),
            ("ASSIGN OUTPUT4 = '/dire,\nc1/direc2/output4.op4',\nUNIT=101,DELETE\n"),
        ),
        # params, no current_path, wrap outside path
        (
            (
                "output4",
                "/direc1/direc2/output4.op4",
                {"unit": 101, "delete": None},
                None,
                50,
            ),
            ("ASSIGN OUTPUT4 = '/direc1/direc2/output4.op4',\nUNIT=101,DELETE\n"),
        ),
        # params, current_path, wrap outside path
        (
            (
                "output4",
                "/direc1/dir2/out4.op4",
                {"unit": 102, "delete": None},
                "/direc1/direc3/direc4",
                50,
            ),
            ("ASSIGN OUTPUT4 = '../../dir2/out4.op4',UNIT=102,\nDELETE\n"),
        ),
        # UNIT=11 doesn't fit on first line
        (
            (
                "output4",
                "../../../../../../../tmp/tmp8rjamzez/kdjj.op4",
                {"unit": 11, "formatted": None},
                None,
                71,
            ),
            (
                "ASSIGN OUTPUT4 = '../../../../../../../tmp/tmp8rjamzez/kdjj.op4',"
                "\nUNIT=11,FORMATTED\n"
            ),
        ),
        # UNIT=11 would fit on first line, but comma does not
        (
            (
                "output4",
                "../../../../../../../tmp/tmp8rjamzez/kdjj.op4",
                {"unit": 11, "formatted": None},
                None,
                72,
            ),
            (
                "ASSIGN OUTPUT4 = '../../../../../../../tmp/tmp8rjamzez/kdjj.op4',"
                "\nUNIT=11,FORMATTED\n"
            ),
        ),
        # with max_length=73, the UNIT=11 and comma now fit on first line
        (
            (
                "output4",
                "../../../../../../../tmp/tmp8rjamzez/kdjj.op4",
                {"unit": 11, "formatted": None},
                None,
                73,
            ),
            (
                "ASSIGN OUTPUT4 = '../../../../../../../tmp/tmp8rjamzez/kdjj.op4',"
                "UNIT=11,\nFORMATTED\n"
            ),
        ),
    )
    for args, expected in values:
        f = StringIO()
        nastran.wtassign(f, *args)
        assert f.getvalue() == expected


def test_wttabdmp1_bad_inputs():
    f = StringIO()
    setid = 101
    # not enough terms in freq/damp
    freq, damp = [1.0], [0.01]
    with pytest.raises(ValueError, match=r"freq.*length.*\>=2"):
        nastran.wttabdmp1(f, setid, freq, damp)
    # length mismatch
    freq, damp = [1.0, 10.0], [0.01, 0.01, 0.01]
    with pytest.raises(ValueError, match=r"freq.*damp.*same length"):
        nastran.wttabdmp1(f, setid, freq, damp)
    # invalid damping type
    freq, damp = [1.0, 10.0], [0.01, 0.01]
    damping_type = "zzz"
    with pytest.raises(ValueError, match=r"damping_type.*got 'ZZZ'"):
        nastran.wttabdmp1(f, setid, freq, damp, damping_type)


def test_wttabdmp1_two_terms():
    f = StringIO()
    setid = 101
    freq = [0.1, 2.5]
    damp = [0.01, 0.015]
    nastran.wttabdmp1(f, setid, freq, damp)
    expected = (
        "TABDMP1*             101CRIT                                            \n"
        "*                                                                       *\n"
        "*                     .1             .01             2.5            .015\n"
        "*       ENDT            \n"
    )
    assert f.getvalue() == expected


def test_wttabdmp1_three_terms():
    f = StringIO()
    setid = 101.0  # setid is a float, verify it's written as an integer
    freq = [0.1, 2.5, 3]  # freq[2] is an integer, verify it's written as a real
    damp = [0.01, 0.015, 0.015]
    damping_type = "g"
    nastran.wttabdmp1(f, setid, freq, damp, damping_type)
    expected = (
        "TABDMP1*             101G                                               \n"
        "*                                                                       *\n"
        "*                     .1             .01             2.5            .015\n"
        "*                     3.            .015ENDT            \n"
    )
    assert f.getvalue() == expected


def test_wttabdmp1_four_terms():
    f = StringIO()
    setid = 101
    freq = [0.1, 2.5, 3.2, 3.4]
    damp = [0.01, 0.015, 0.015, 0.016]
    damping_type = "q"
    nastran.wttabdmp1(f, setid, freq, damp, damping_type)
    expected = (
        "TABDMP1*             101Q                                               \n"
        "*                                                                       *\n"
        "*                     .1             .01             2.5            .015\n"
        "*                    3.2            .015             3.4            .016*\n"
        "*       ENDT            \n"
        "*\n"  # trailing newline so card had even number of lines
    )
    assert f.getvalue() == expected


def test_wttload1_bad_input():
    f = StringIO()
    setid, excite_id = 201, 202
    delay = 0.05
    excite_type = "zzz"
    tabledi_id = 1501
    with pytest.raises(ValueError, match=r"excite_type.*got 'ZZZ'"):
        nastran.wttload1(f, setid, excite_id, delay, excite_type, tabledi_id)


def test_wttload1_case1():
    f = StringIO()
    setid, excite_id = 201, 202
    delay = 0.05  # float, represents a time value
    excite_type = "load"
    tabledi_id = 1501.0  # float, verify it's written as an integer
    nastran.wttload1(f, setid, excite_id, delay, excite_type, tabledi_id)
    expected = "TLOAD1       201     202     .05LOAD        1501\n"
    assert f.getvalue() == expected


def test_wttload1_case2():
    f = StringIO()
    setid, excite_id = 201.0, 203
    delay = 1504  # integer, represents a DELAY card
    excite_type = "load"
    tabledi_id = 1502
    nastran.wttload1(f, setid, excite_id, delay, excite_type, tabledi_id)
    expected = "TLOAD1       201     203    1504LOAD        1502\n"
    assert f.getvalue() == expected


def test_wttload2_bad_input():
    f = StringIO()
    setid, excite_id = 201, 202
    delay = 0.05
    excite_type = "zzz"
    t1, t2 = 0.0, 1.0
    with pytest.raises(ValueError, match=r"excite_type.*got 'ZZZ'"):
        nastran.wttload2(f, setid, excite_id, delay, excite_type, t1, t2)
    excite_type = "load"
    t1, t2 = 1.0, 1.0
    with pytest.raises(ValueError, match=r"t2 must.*greater.*t1"):
        nastran.wttload2(f, setid, excite_id, delay, excite_type, t1, t2)


def test_wttload2_case1():
    f = StringIO()
    setid, excite_id = 201, 202
    delay = 0.05  # float, represents a time value
    excite_type = "load"
    t1, t2 = 0.1, 0.2
    nastran.wttload2(f, setid, excite_id, delay, excite_type, t1, t2)
    expected = (
        "TLOAD2       201     202     .05LOAD          .1      .2      0.      0.\n"
        "+             0.      0.\n"
    )
    assert f.getvalue() == expected


def test_wttload2_case2():
    fobj = StringIO()
    setid, excite_id = 201, 202
    delay = 2  # integer, represents a DELAY card
    excite_type = "load"
    t1, t2 = 0.1, 2  # integer, verify it's converted to real
    f, p, c, b = 11, 12, 13, 14  # integers, verify all are converted to real
    nastran.wttload2(fobj, setid, excite_id, delay, excite_type, t1, t2, f, p, c, b)
    expected = (
        "TLOAD2       201     202       2LOAD          .1      2.     11.     12.\n"
        "+            13.     14.\n"
    )
    assert fobj.getvalue() == expected


def test_wtconm2_case1():
    f = StringIO()
    eid, gid, cid = 1, 2, 3
    mass = 5000
    I_diag = [6000, 7000, 8000]  # verify these are written as real
    I_offdiag = [6500, 8500, 7500]
    offset = [3, 2, 1]
    nastran.wtconm2(f, eid, gid, cid, mass, I_diag, I_offdiag, offset)
    expected = (
        "CONM2*                 1               2               3           5000.\n"
        "*                     3.              2.              1.                *\n"
        "*                  6000.           6500.           7000.           8500.\n"
        "*                  7500.           8000.\n"
    )
    assert f.getvalue() == expected


def test_wtconm2_case2():
    # defaults for I_offdiag and offset
    f = StringIO()
    eid, gid, cid = 1, 2, 3.0  # verify this is written as an integer
    mass = 5000.0
    I_diag = [6000, 7000, 8000]
    nastran.wtconm2(f, eid, gid, cid, mass, I_diag)
    expected = (
        "CONM2*                 1               2               3           5000.\n"
        "*                     0.              0.              0.                *\n"
        "*                  6000.              0.           7000.              0.\n"
        "*                     0.           8000.\n"
    )
    assert f.getvalue() == expected


def test_wtcard8_bad_inputs():
    f = StringIO()
    # card name too long
    fields = ("ABCDEFGHI", 0, 0)
    with pytest.raises(ValueError, match=r"card name.*\<8.*ABCDEFGHI"):
        nastran.wtcard8(f, fields)

    # unsupported field type
    fields = ("ABC", 0, [1, 2, 3])
    with pytest.raises(TypeError, match=r"unsupported field type.*list"):
        nastran.wtcard8(f, fields)


def test_wtcard8():
    values = (
        # 3 fields
        (("ABC", 0, "", 1.5), "ABC            0             1.5\n"),
        # 7 fields, NumPy types
        (
            (
                np.str_("ABC"),
                np.uint32(10),
                np.uint64(11),
                np.int32(12),
                np.int64(13),
                np.float32(1.0),
                np.float64(2.0),
            ),
            "ABC           10      11      12      13      1.      2.\n",
        ),
        # 4 fields
        (("ABC", 0, "", 1.5, -2.0), "ABC            0             1.5     -2.\n"),
        # 4 fields
        (
            ("ABC", 0, "", 1.5, -2.0, "", "DEF", 1.2e7, 3.4e-5),
            "ABC            0             1.5     -2.        DEF        1.2+7 .000034\n",
        ),
        # 9 fields
        (
            ("ABC", 0, "", 1.5, -2.0, "", "DEF", 1.2e7, 3.4e-5, -3.4e-12),
            "ABC            0             1.5     -2.        DEF        1.2+7 .000034\n"
            "+        -3.4-12\n",
        ),
        # 17 fields
        (
            (
                "ABC",
                0,
                "",
                1.5,
                -2.0,
                "",
                "DEF",
                1.2e7,
                3.4e-5,
                -3.4e-12,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
            "ABC            0             1.5     -2.        DEF        1.2+7 .000034\n"
            "+        -3.4-12       0       0       0       0       0       0       0\n"
            "+              0\n",
        ),
    )
    for fields, expected in values:
        f = StringIO()
        nastran.wtcard8(f, fields)
        assert f.getvalue() == expected


def test_wtcard16_bad_inputs():
    f = StringIO()
    # card name doesn't end in asterisk
    fields = ("GRID", 0, 0)
    with pytest.raises(ValueError, match=r"card name.*asterisk"):
        nastran.wtcard16(f, fields)

    # card name too long
    fields = ("ABCDEFGHI*", 0, 0)
    with pytest.raises(ValueError, match=r"card name.*\<8.*ABCDEFGHI"):
        nastran.wtcard16(f, fields)

    # unsupported field type
    fields = ("ABC*", 0, [1, 2, 3])
    with pytest.raises(TypeError, match=r"unsupported field type.*list"):
        nastran.wtcard16(f, fields)


def test_wtcard16():
    values = (
        # three fields
        (
            ("ABC*", 0, "", 1.5),
            ("ABC*                   0                             1.5\n" "*\n"),
        ),
        # 7 fields, NumPy types
        (
            (
                np.str_("ABC*"),
                np.uint32(10),
                np.uint64(11),
                np.int32(12),
                np.int64(13),
                np.float32(1.0),
                np.float64(2.0),
            ),
            "ABC*                  10              11              12              13\n"
            "*                     1.              2.\n",
        ),
        # four fields
        (
            ("ABC*", 0, "", 1.0, -2.0),
            (
                "ABC*                   0                              1.             -2.\n"
                "*\n"
            ),
        ),
        # six fields
        (
            ("GRID*", 7100285, 7100003, 1.24499e3, 5.4e1, -3.94e3, 7100004),
            (
                "GRID*            7100285         7100003         1244.99             54.\n"
                "*                 -3940.         7100004\n"
            ),
        ),
        # eight fields
        (
            ("ABC*", 0, "", 1.0, 2.0, "", 3.0, 4.0e-2, 550),
            (
                "ABC*                   0                              1.              2.\n"
                "*                                     3.             .04             550\n"
            ),
        ),
        # nine fields
        (
            ("ABC*", 0, "", 1.0, 2.0, "", 3.0, 4.0, 550, "ABC"),
            (
                "ABC*                   0                              1.              2.\n"
                "*                                     3.              4.             550*\n"
                "*       ABC             \n"
                "*\n"
            ),
        ),
    )
    for fields, expected in values:
        f = StringIO()
        nastran.wtcard16(f, fields)
        assert f.getvalue() == expected


def test_wtcard16d_bad_inputs():
    f = StringIO()
    # card name doesn't end in asterisk
    fields = ("GRID", 0, 0)
    with pytest.raises(ValueError, match=r"card name.*asterisk"):
        nastran.wtcard16d(f, fields)

    # card name too long
    fields = ("ABCDEFGHI*", 0, 0)
    with pytest.raises(ValueError, match=r"card name.*\<8.*ABCDEFGHI"):
        nastran.wtcard16d(f, fields)

    # unsupported field type
    fields = ("ABC*", 0, [1, 2, 3])
    with pytest.raises(TypeError, match=r"unsupported field type.*list"):
        nastran.wtcard16d(f, fields)


def test_wtcard16d():
    values = (
        # three fields
        (
            ("ABC*", 0, "", 1.5),
            ("ABC*                   0                          1.5D+0\n" "*\n"),
        ),
        # 7 fields, NumPy types
        (
            (
                np.str_("ABC*"),
                np.uint32(10),
                np.uint64(11),
                np.int32(12),
                np.int64(13),
                np.float32(1.0),
                np.float64(2.0),
            ),
            "ABC*                  10              11              12              13\n"
            "*                  1.D+0           2.D+0\n",
        ),
        # four fields
        (
            ("ABC*", 0, "", 1.0, -2.0),
            (
                "ABC*                   0                           1.D+0          -2.D+0\n"
                "*\n"
            ),
        ),
        # six fields
        (
            ("GRID*", 7100285, 7100003, 1.24499e3, 5.4e1, -3.94e3, 7100004),
            (
                "GRID*            7100285         7100003      1.24499D+3          5.4D+1\n"
                "*               -3.94D+3         7100004\n"
            ),
        ),
        # eight fields
        (
            ("ABC*", 0, "", 1.0, 2.0, "", 3.0, 4.0e-2, 550),
            (
                "ABC*                   0                           1.D+0           2.D+0\n"
                "*                                  3.D+0           4.D-2             550\n"
            ),
        ),
        # nine fields
        (
            ("ABC*", 0, "", 1.0, 2.0, "", 3.0, 4.0, 550, "ABC"),
            (
                "ABC*                   0                           1.D+0           2.D+0\n"
                "*                                  3.D+0           4.D+0             550*\n"
                "*       ABC             \n"
                "*\n"
            ),
        ),
    )
    for fields, expected in values:
        f = StringIO()
        nastran.wtcard16d(f, fields)
        assert f.getvalue() == expected


def test_format_scientific8():
    from pyyeti.nastran.bulk import _format_scientific8

    small_exponent = -17
    large_exponent = 17
    nums = (
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            0.11,
            0.123451,
            0.123459,
            -0.123431,
            -0.123459,
            0.000034,
            -0.000034,
            -0.009,
            0.000000000000000000000001,
            -0.000000000000000000000001,
        ]
        + [9.0 / 11.0 * 10**x for x in range(small_exponent, large_exponent + 1)]
        + [-9.0 / 11.0 * 10**x for x in range(small_exponent, large_exponent + 1)]
    )
    expecteds = (
        [
            "      0.",
            "      0.",
            "    1.+0",
            "   -1.+0",
            "    1.-1",
            "   1.1-1",
            "1.2345-1",
            "1.2346-1",
            "-1.234-1",
            "-1.235-1",
            "   3.4-5",
            "  -3.4-5",
            "   -9.-3",
            "   1.-24",
            "  -1.-24",
        ]
        + [f"8.182{i:+d}" for i in range(-18, -9)]
        + [f"8.1818{i:+d}" for i in range(-9, 10)]
        + [f"8.182{i:+d}" for i in range(10, 17)]
        + [f"-8.18{i:+d}" for i in range(-18, -9)]
        + [f"-8.182{i:+d}" for i in range(-9, 10)]
        + [f"-8.18{i:+d}" for i in range(10, 17)]
    )
    assert len(nums) == len(expecteds)
    for num, expected in zip(nums, expecteds):
        output = _format_scientific8(num)
        assert len(output) == 8
        assert output == expected


def test_format_float8():
    small_exponent = -17
    large_exponent = 17
    nums = (
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            0.11,
            0.123451,
            0.123459,
            -0.123431,
            -0.123459,
            0.000034,
            -0.000034,
            -0.009,
            0.000000000000000000000001,
            -0.000000000000000000000001,
        ]
        + [9.0 / 11.0 * 10**x for x in range(small_exponent, large_exponent + 1)]
        + [-9.0 / 11.0 * 10**x for x in range(small_exponent, large_exponent + 1)]
    )
    expecteds = (
        [
            "      0.",
            "      0.",
            "      1.",
            "     -1.",
            "      .1",
            "     .11",
            " .123451",
            " .123459",
            "-.123431",
            "-.123459",
            " .000034",
            "  -3.4-5",
            "   -.009",
            "   1.-24",
            "  -1.-24",
        ]
        + [f"8.182{i:+d}" for i in range(-18, -9)]
        + [f"8.1818{i:+d}" for i in range(-9, -3)]
        + [
            ".0081818",
            ".0818182",
            ".8181818",
            "8.181818",
            "81.81818",
            "818.1818",
            "8181.818",
            "81818.18",
            "818181.8",
            "8181818.",
        ]
        + [f"8.1818{i:+d}" for i in range(7, 10)]
        + [f"8.182{i:+d}" for i in range(10, 17)]
        + [f"-8.18{i:+d}" for i in range(-18, -9)]
        + [f"-8.182{i:+d}" for i in range(-9, -2)]
        + [
            "-.081818",
            "-.818182",
            "-8.18182",
            "-81.8182",
            "-818.182",
            "-8181.82",
            "-81818.2",
            "-818182.",
        ]
        + [f"-8.182{i:+d}" for i in range(6, 10)]
        + [f"-8.18{i:+d}" for i in range(10, 17)]
    )
    assert len(nums) == len(expecteds)
    for num, expected in zip(nums, expecteds):
        output = nastran.format_float8(num)
        assert len(output) == 8
        assert output == expected


def test_format_float8_many():
    for start in range(-13, 13):
        nums = np.logspace(
            start, start + 1, num=1000, endpoint=True, base=10.0, dtype="f8"
        )
        for num in nums:
            output = nastran.format_float8(num)
            assert len(output) == 8
            assert "." in output


def test_format_scientific16():
    from pyyeti.nastran.bulk import _format_scientific16

    small_exponent = -17
    large_exponent = 17
    nums = (
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            0.11,
            0.123451,
            0.123459,
            -0.123431,
            -0.123459,
            0.000034,
            -0.000034,
            -0.009,
            0.000000000000000000000001,
            -0.000000000000000000000001,
        ]
        + [9.0 / 11.0 * 10**x for x in range(small_exponent, large_exponent + 1)]
        + [-9.0 / 11.0 * 10**x for x in range(small_exponent, large_exponent + 1)]
    )
    expecteds = (
        [
            "              0.",
            "              0.",
            "            1.+0",
            "           -1.+0",
            "            1.-1",
            "           1.1-1",
            "       1.23451-1",
            "       1.23459-1",
            "      -1.23431-1",
            "      -1.23459-1",
            "           3.4-5",
            "          -3.4-5",
            "           -9.-3",
            "           1.-24",
            "          -1.-24",
        ]
        + [f"8.18181818182{i:+d}" for i in range(-18, -9)]
        + [f"8.181818181818{i:+d}" for i in range(-9, 10)]
        + [f"8.18181818182{i:+d}" for i in range(10, 17)]
        + [f"-8.1818181818{i:+d}" for i in range(-18, -9)]
        + [f"-8.18181818182{i:+d}" for i in range(-9, 10)]
        + [f"-8.1818181818{i:+d}" for i in range(10, 17)]
    )
    assert len(nums) == len(expecteds)
    for num, expected in zip(nums, expecteds):
        output = _format_scientific16(num)
        assert len(output) == 16
        assert output == expected


def test_format_float16():
    small_exponent = -17
    large_exponent = 17
    nums = (
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            0.11,
            0.123451,
            0.123459,
            -0.123431,
            -0.123459,
            0.000034,
            -0.000034,
            -0.009,
            0.000000000000000000000001,
            -0.000000000000000000000001,
        ]
        + [9.0 / 11.0 * 10**x for x in range(small_exponent, large_exponent + 1)]
        + [-9.0 / 11.0 * 10**x for x in range(small_exponent, large_exponent + 1)]
    )
    expecteds = (
        [
            "              0.",
            "              0.",
            "              1.",
            "             -1.",
            "              .1",
            "             .11",
            "         .123451",
            "         .123459",
            "        -.123431",
            "        -.123459",
            "         .000034",
            "        -.000034",
            "           -.009",
            "           1.-24",
            "          -1.-24",
        ]
        + [f"8.18181818182{i:+d}" for i in range(-18, -9)]
        + [f"8.181818181818{i:+d}" for i in range(-9, -3)]
        + [
            ".008181818181818",
            ".081818181818182",
            ".818181818181818",
            "8.18181818181818",
            "81.8181818181818",
            "818.181818181818",
            "8181.81818181818",
            "81818.1818181818",
            "818181.818181818",
            "8181818.18181818",
            "81818181.8181818",
            "818181818.181818",
            "8181818181.81818",
            "81818181818.1818",
            "818181818181.818",
            "8181818181818.18",
            "81818181818181.8",
            "818181818181818.",
            "8.18181818182+15",
            "8.18181818182+16",
        ]
        + [f"-8.1818181818{i:+d}" for i in range(-18, -9)]
        + [f"-8.18181818182{i:+d}" for i in range(-9, -2)]
        + [
            "-.08181818181818",
            "-.81818181818182",
            "-8.1818181818182",
            "-81.818181818182",
            "-818.18181818182",
            "-8181.8181818182",
            "-81818.181818182",
            "-818181.81818182",
            "-8181818.1818182",
            "-81818181.818182",
            "-818181818.18182",
            "-8181818181.8182",
            "-81818181818.182",
            "-818181818181.82",
            "-8181818181818.2",
            "-81818181818182.",
            "-8.1818181818+14",
            "-8.1818181818+15",
            "-8.1818181818+16",
        ]
    )
    assert len(nums) == len(expecteds)
    for num, expected in zip(nums, expecteds):
        output = nastran.format_float16(num)
        assert len(output) == 16
        assert output == expected


def test_format_float16_many():
    for start in range(-13, 13):
        nums = np.logspace(
            start, start + 1, num=1000, endpoint=True, base=10.0, dtype="f8"
        )
        for num in nums:
            output = nastran.format_float16(num)
            assert len(output) == 16
            assert "." in output


def test_format_double16():
    small_exponent = -17
    large_exponent = 17
    nums = (
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.1,
            0.11,
            0.123451,
            0.123459,
            -0.123431,
            -0.123459,
            0.000034,
            -0.000034,
            -0.009,
            0.000000000000000000000001,
            -0.000000000000000000000001,
        ]
        + [9.0 / 11.0 * 10**x for x in range(small_exponent, large_exponent + 1)]
        + [-9.0 / 11.0 * 10**x for x in range(small_exponent, large_exponent + 1)]
    )
    expecteds = (
        [
            "           0.D+0",
            "           0.D+0",
            "           1.D+0",
            "          -1.D+0",
            "           1.D-1",
            "          1.1D-1",
            "      1.23451D-1",
            "      1.23459D-1",
            "     -1.23431D-1",
            "     -1.23459D-1",
            "          3.4D-5",
            "         -3.4D-5",
            "          -9.D-3",
            "          1.D-24",
            "         -1.D-24",
        ]
        + [f"8.1818181818D{i:+d}" for i in range(-18, -9)]
        + [f"8.18181818182D{i:+d}" for i in range(-9, 10)]
        + [f"8.1818181818D{i:+d}" for i in range(10, 17)]
        + [f"-8.181818182D{i:+d}" for i in range(-18, -9)]
        + [f"-8.1818181818D{i:+d}" for i in range(-9, 10)]
        + [f"-8.181818182D{i:+d}" for i in range(10, 17)]
    )
    assert len(nums) == len(expecteds)
    for num, expected in zip(nums, expecteds):
        output = nastran.format_double16(num)
        assert len(output) == 16
        assert output == expected


def test_format_double16_many():
    for start in range(-13, 13):
        nums = np.logspace(
            start, start + 1, num=1000, endpoint=True, base=10.0, dtype="f8"
        )
        for num in nums:
            output = nastran.format_double16(num)
            assert len(output) == 16
            assert "." in output
            assert "D" in output
