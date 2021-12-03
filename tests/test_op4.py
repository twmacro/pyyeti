import numpy as np
from pyyeti.nastran import op4
from scipy.io import matlab
import os
from glob import glob
import tempfile
import scipy.sparse as sp
from nose.tools import *


def _check_badname_cm(cm):
    assert (
        cm.warnings[0].message.args[0]
        == "Output4 file has matrix name: '1mat'. Changing to 'm0'."
    )

    assert (
        cm.warnings[1].message.args[0]
        == "Output4 file has matrix name: ''. Changing to 'm1'."
    )

    assert (
        cm.warnings[2].message.args[0]
        == "Output4 file has matrix name: '09'. Changing to 'm2'."
    )


def _rdop4_tst(o4):
    matfile = "tests/nastran_op4_data/r_c_rc.mat"
    filenames = glob("tests/nastran_op4_data/*.op4")
    nocomp = [
        "cdbin",
        "rdbin",
        "csbin",
        "rsbin",
        "cd",
        "rd",
        "cs",
        "rs",
        "x100000",
        "cdbin_ascii_sparse_bigmat",
        "cdbin_ascii_sparse_nonbigmat",
    ]
    nocomp = [s + ".op4" for s in nocomp]
    m = matlab.loadmat(matfile)

    badname_translate = {"m0": "rmat", "m1": "cmat", "m2": "rcmat"}

    for filename in filenames:
        # output of dir not checked, but it should work on all these files:
        op4.dir(filename, verbose=False)

        basename = os.path.basename(filename)
        if basename in nocomp:
            continue
        if basename.startswith("big"):
            continue

        if filename.find("badname") > -1:
            with assert_warns(RuntimeWarning) as cm:
                dct = o4.dctload(filename)
            _check_badname_cm(cm)

            with assert_warns(RuntimeWarning) as cm:
                names, mats, forms, mtypes = o4.listload(filename)
            _check_badname_cm(cm)

            with assert_warns(RuntimeWarning) as cm:
                names2, sizes, forms2, mtypes2 = o4.dir(filename, verbose=False)
            _check_badname_cm(cm)
        else:
            dct = o4.dctload(filename)
            names, mats, forms, mtypes = o4.listload(filename)
            names2, sizes, forms2, mtypes2 = o4.dir(filename, verbose=False)
        assert sorted(dct.keys()) == sorted(names)
        assert names == names2
        assert forms == forms2
        assert mtypes == mtypes2
        for mat, sz in zip(mats, sizes):
            assert mat.shape == sz
        for nm in dct:
            matnm = badname_translate.get(nm, nm)
            if nm[-1] == "s":
                matnm = nm[:-1]
            assert np.allclose(m[matnm], dct[nm][0])
            pos = names.index(nm)
            assert np.allclose(m[matnm], mats[pos])
            assert dct[nm][1] == forms[pos]
            assert dct[nm][2] == mtypes[pos]

        nm2 = nm = "rcmat"
        if filename.find("single") > -1:
            nm2 = "rcmats"
        if filename.find("badname") > -1:
            nm2 = "m2"
            with assert_warns(RuntimeWarning) as cm:
                dct = o4.dctload(filename, nm2)
                name, mat, *_ = o4.listload(filename, [nm2])
        else:
            dct = o4.dctload(filename, [nm2])
            name, mat, *_ = o4.listload(filename, nm2)
        assert np.allclose(m[nm], dct[nm2][0])
        assert np.allclose(m[nm], mat[0])


def test_rdop4():
    # unittest bug avoidance:
    import sys

    for v in list(sys.modules.values()):
        if getattr(v, "__warningregistry__", None):
            v.__warningregistry__ = {}

    o4 = op4.OP4()
    _rdop4_tst(o4)


def test_rdop4_zero_rowscutoff():
    o4 = op4.OP4()
    o4._rowsCutoff = 0
    _rdop4_tst(o4)


def test_rdop4_partb():
    filenames = glob("tests/nastran_op4_data/*other")
    file1 = filenames[0]
    filenames = filenames[1:]
    o4 = op4.OP4()
    dct = o4.dctload(file1)
    for filename in filenames:
        dct2 = o4.dctload(filename)
        assert set(dct2.keys()) == set(dct.keys())
        for nm in dct:
            for j in range(3):
                assert np.allclose(dct2[nm][j], dct[nm][j])


def test_wtop4():
    matfile = "tests/nastran_op4_data/r_c_rc.mat"
    o4 = op4.OP4()
    m = matlab.loadmat(matfile)
    names = ["rmat", "cmat", "rcmat"]
    mats = []
    wtdct = {}
    for nm in names:
        mats.append(m[nm])
        wtdct[nm] = m[nm]
    # write(filename, names, matrices=None,
    #       binary=True, digits=16, endian='')
    filenames = [
        ["tests/nastran_op4_data/temp_ascii.op4", False, ""],
        ["tests/nastran_op4_data/temp_le.op4", True, "<"],
        ["tests/nastran_op4_data/temp_be.op4", True, ">"],
    ]
    for item in filenames:
        filename = item[0]
        binary = item[1]
        endian = item[2]
        o4.write(filename, names, mats, binary=binary, endian=endian)
        names2, sizes, forms, mtypes = o4.dir(filename, verbose=False)
        assert names2 == names
        dct = o4.dctload(filename)
        for nm in dct:
            assert np.allclose(m[nm], dct[nm][0])
        o4.write(filename, wtdct, binary=binary, endian=endian)
        dct = o4.dctload(filename)
        for nm in dct:
            assert np.allclose(m[nm], dct[nm][0])
    # clean up:
    for item in filenames:
        os.remove(item[0])


def test_wtop4_2():
    matfile = "tests/nastran_op4_data/r_c_rc.mat"
    m = matlab.loadmat(matfile)
    names = ["rmat", "cmat", "rcmat"]
    mats = []
    wtdct = {}
    for nm in names:
        mats.append(m[nm])
        wtdct[nm] = m[nm]
    # write(filename, names, matrices=None,
    #       binary=True, digits=16, endian='')
    filenames = [
        ["tests/nastran_op4_data/temp_ascii.op4", False, ""],
        ["tests/nastran_op4_data/temp_le.op4", True, "<"],
        ["tests/nastran_op4_data/temp_be.op4", True, ">"],
    ]
    for item in filenames:
        filename = item[0]
        binary = item[1]
        endian = item[2]
        op4.write(filename, names, mats, binary=binary, endian=endian)
        names2, sizes, forms, mtypes = op4.dir(filename, verbose=False)
        assert names2 == names
        dct = op4.load(filename)
        for nm in dct:
            assert np.allclose(m[nm], dct[nm][0])
        op4.save(filename, wtdct, binary=binary, endian=endian)
        dct = op4.load(filename, into="dct")
        for nm in dct:
            assert np.allclose(m[nm], dct[nm][0])

        dct = op4.read(filename)
        for nm in dct:
            assert np.allclose(m[nm], dct[nm])
        op4.save(filename, wtdct, binary=binary, endian=endian)
        dct = op4.read(filename, into="dct")
        for nm in dct:
            assert np.allclose(m[nm], dct[nm])

    # clean up:
    for item in filenames:
        os.remove(item[0])


def test_wtop4_single():
    matfile = "tests/nastran_op4_data/r_c_rc.mat"
    o4 = op4.OP4()
    m = matlab.loadmat(matfile)
    name = "rmat"
    mat = m[name]
    # write(filename, names, matrices=None,
    #       binary=True, digits=16, endian='')
    filenames = [["tests/nastran_op4_data/temp_ascii.op4", False, ""]]
    for item in filenames:
        filename = item[0]
        binary = item[1]
        endian = item[2]
        o4.write(
            filename, name, mat, binary=binary, endian=endian, forms=45
        )  # 45 is not actually a valid setting
        dct = o4.dctload(filename)
        for nm in dct:
            assert nm == name
            assert np.allclose(m[nm], dct[nm][0])
            assert dct[nm][1] == 45

    # clean up:
    for item in filenames:
        os.remove(item[0])


def test_wtop4_nonbigmat_binary():
    filenames = glob("tests/nastran_op4_data/*.op4") + glob(
        "tests/nastran_op4_data/*.op4.other"
    )
    o4 = op4.OP4()
    for name in filenames:
        if name.find("badname") != -1:
            continue
        data = o4.listload(name)
        o4.write("temp.op4", data[0], data[1], sparse="nonbigmat")
        data2 = o4.listload("temp.op4")
        assert data[0] == data2[0]
        for d1, d2 in zip(data[1], data2[1]):
            assert np.all(d1 == d2)
        os.remove("temp.op4")


def test_wtop4_bigmat_binary():
    filenames = glob("tests/nastran_op4_data/*.op4") + glob(
        "tests/nastran_op4_data/*.op4.other"
    )
    o4 = op4.OP4()
    for name in filenames:
        if name.find("badname") != -1:
            continue
        data = o4.listload(name)
        o4.write("temp.op4", data[0], data[1], sparse="bigmat")
        data2 = o4.listload("temp.op4")
        assert data[0] == data2[0]
        for d1, d2 in zip(data[1], data2[1]):
            assert np.all(d1 == d2)
        os.remove("temp.op4")


def test_wtop4_nonbigmat_ascii():
    filenames = glob("tests/nastran_op4_data/*.op4") + glob(
        "tests/nastran_op4_data/*.op4.other"
    )
    o4 = op4.OP4()
    for name in filenames:
        if name.find("badname") != -1:
            continue
        data = o4.listload(name)
        o4.write("temp.op4", data[0], data[1], sparse="nonbigmat", binary=False)
        data2 = o4.listload("temp.op4")
        assert data[0] == data2[0]
        for d1, d2 in zip(data[1], data2[1]):
            assert np.all(d1 == d2)
        os.remove("temp.op4")


def test_wtop4_bigmat_ascii():
    filenames = glob("tests/nastran_op4_data/*.op4") + glob(
        "tests/nastran_op4_data/*.op4.other"
    )
    o4 = op4.OP4()
    for name in filenames:
        if name.find("badname") != -1:
            continue
        data = o4.listload(name)
        o4.write("temp.op4", data[0], data[1], sparse="bigmat", binary=False)
        data2 = o4.listload("temp.op4")
        assert data[0] == data2[0]
        for d1, d2 in zip(data[1], data2[1]):
            assert np.all(d1 == d2)
        os.remove("temp.op4")


def test_wtop4_bigmat_ascii_1():
    filenames = glob("tests/nastran_op4_data/*.op4") + glob(
        "tests/nastran_op4_data/*.op4.other"
    )
    o4 = op4.OP4()
    for name in filenames[:1]:
        if name.find("badname") != -1:
            continue
        data = o4.load(name, into="list")
        o4.write("temp.op4", data[0], data[1], sparse="bigmat", binary=False)
        data2 = o4.load("temp.op4", into="list")
        assert data[0] == data2[0]
        for d1, d2 in zip(data[1], data2[1]):
            assert np.all(d1 == d2)
        os.remove("temp.op4")


def test_wtop4_bigmat_ascii_2():
    filenames = glob("tests/nastran_op4_data/*.op4") + glob(
        "tests/nastran_op4_data/*.op4.other"
    )
    for name in filenames[:1]:
        if name.find("badname") != -1:
            continue
        data = op4.load(name, into="list")
        op4.write("temp.op4", data[0], data[1], sparse="bigmat", binary=False)
        data2 = op4.load("temp.op4", into="list")
        assert data[0] == data2[0]
        for d1, d2 in zip(data[1], data2[1]):
            assert np.all(d1 == d2)
        os.remove("temp.op4")


def test_non_float64():
    i8 = np.array([1, 2, 0, 4], np.int8)
    i16 = i8.astype(np.int16)
    i32 = i8.astype(np.int32)
    i64 = i8.astype(np.int64)
    f32 = i8.astype(np.float32)
    c32 = (f32 + 1j * f32).astype(np.complex64)
    o4 = op4.OP4()
    for mat in [i8, i16, i32, i64, f32, c32]:
        o4.write("temp.op4", dict(mat=mat))
        mat2 = o4.dctload("temp.op4", "mat")["mat"][0]
        assert np.all(mat2 == mat)
        os.remove("temp.op4")


def test_wtop4_single_save():
    matfile = "tests/nastran_op4_data/r_c_rc.mat"
    o4 = op4.OP4()
    m = matlab.loadmat(matfile)
    name = "rmat"
    mat = m[name]
    # write(filename, names, matrices=None,
    #       binary=True, digits=16, endian='')
    filenames = [["tests/nastran_op4_data/temp_ascii.op4", False, ""]]
    for item in filenames:
        filename = item[0]
        binary = item[1]
        endian = item[2]
        o4.save(filename, name, mat, binary=binary, endian=endian)
        dct = o4.dctload(filename)
        for nm in dct:
            assert np.allclose(m[nm], dct[nm][0])
    # clean up:
    for item in filenames:
        os.remove(item[0])


def test_wtop4_single_save_1():
    matfile = "tests/nastran_op4_data/r_c_rc.mat"
    o4 = op4.OP4()
    m = matlab.loadmat(matfile)
    name = "rmat"
    mat = m[name]
    # write(filename, names, matrices=None,
    #       binary=True, digits=16, endian='')
    filenames = [["tests/nastran_op4_data/temp_ascii.op4", False, ""]]
    for item in filenames:
        filename = item[0]
        binary = item[1]
        endian = item[2]
        o4.save(filename, name, mat, binary=binary, endian=endian)
        dct = o4.load(filename, into="dct")
        for nm in dct:
            assert np.allclose(m[nm], dct[nm][0])
    # clean up:
    for item in filenames:
        os.remove(item[0])


def test_wtop4_single_2():
    matfile = "tests/nastran_op4_data/r_c_rc.mat"
    m = matlab.loadmat(matfile)
    name = "rmat"
    mat = m[name]
    # write(filename, names, matrices=None,
    #       binary=True, digits=16, endian='')
    filenames = [["tests/nastran_op4_data/temp_ascii.op4", False, ""]]
    for item in filenames:
        filename = item[0]
        binary = item[1]
        endian = item[2]
        op4.write(filename, name, mat, binary=binary, endian=endian)
        dct = op4.read(filename)
        for nm in dct:
            assert np.allclose(m[nm], dct[nm])
    # clean up:
    for item in filenames:
        os.remove(item[0])


def test_wtop4_single_3():
    matfile = "tests/nastran_op4_data/r_c_rc.mat"
    assert_raises(ValueError, op4.load, matfile, into="badstring")


def test_wtop4_single_4():
    matfile = "tests/nastran_op4_data/r_c_rc.mat"
    o4 = op4.OP4()
    assert_raises(ValueError, o4.load, matfile, into="badstring")


def test_forced_bigmat():
    mat = np.zeros((1000000, 1))
    o4 = op4.OP4()
    o4.save("temp.op4", dict(mat=mat), sparse="nonbigmat")
    m = o4.load("temp.op4", into="dct")
    assert np.all(mat == m["mat"][0])

    o4.save("temp.op4", dict(mat=mat), sparse="nonbigmat", binary=False)
    m = o4.load("temp.op4", into="dct")
    assert np.all(mat == m["mat"][0])
    os.remove("temp.op4")


def test_i64():
    filenames1 = ["cdbin", "rdbin", "csbin", "rsbin"]
    filenames2 = ["cd", "rd", "cs", "rs"]
    for f1, f2 in zip(filenames1, filenames2):
        dct1 = op4.load("tests/nastran_op4_data/" + f1 + ".op4")
        dct2 = op4.load("tests/nastran_op4_data/" + f2 + ".op4")
        assert set(dct1.keys()) == set(dct2.keys())
        for nm in dct1:
            for j in range(2):
                assert np.allclose(dct1[nm][j], dct2[nm][j])


def test_bad_sparse():
    matfile = "temp1.op4"
    r = 1.2
    assert_raises(ValueError, op4.save, matfile, dict(r=r), sparse="badsparsestring")
    assert_raises(
        ValueError, op4.save, matfile, dict(r=r), sparse="badsparsestring", binary=False
    )


def test_bad_dimensions():
    matfile = "temp2.op4"
    r = np.ones((2, 2, 2))
    assert_raises(ValueError, op4.save, matfile, dict(r=r))


def test_sparse_read():
    direc = "tests/nastran_op4_data/"
    fnames = glob(direc + "*.op4") + glob(direc + "*.other")

    for fname in fnames:
        m = op4.read(fname)
        if fname.endswith("cdbin_ascii_sparse_nonbigmat.op4"):
            # this has an all-zeros matrix that is written the same as
            # the dense format (so the sparse form is not used by
            # default)
            del m["c2"]
        m2 = op4.read(fname, sparse=None)
        m3 = op4.read(fname, sparse=True)
        m4 = op4.read(fname, sparse=False)
        m5 = op4.read(fname, sparse=(None, sp.coo_matrix.tocsr))
        if "bigmat" in fname:
            for k, v in m.items():
                assert sp.issparse(m2[k])
                assert sp.isspmatrix_csr(m5[k])
                assert np.all(m2[k].toarray() == v)
                assert np.all(m5[k].toarray() == v)
        else:
            for k, v in m.items():
                assert np.all(m2[k] == v)
                assert np.all(m5[k] == v)

        for k, v in m.items():
            assert sp.issparse(m3[k])
            assert np.all(m3[k].toarray() == v)
            assert np.all(m4[k] == v)


def write_read(m, binary, sparse):
    f = tempfile.NamedTemporaryFile(delete=False)
    name = f.name
    f.close()
    try:
        op4.write(name, m, binary=binary, sparse=sparse)
        m2 = op4.read(name, sparse=None)
    finally:
        os.remove(name)
    return m2


def test_sparse_write():
    fnames = [
        "tests/nastran_op4_data/cdbin.op4",
        "tests/nastran_op4_data/rs.op4",
        "tests/nastran_op4_data/r_c_rc.op4",
        "tests/nastran_op4_data/double_bigmat_le.op4",
        "tests/nastran_op4_data/double_nonbigmat_be.op4",
        "tests/nastran_op4_data/single_dense_be.op4",
        # the nx nastran v9 version of the following files does not
        # make NR negative ...  not really a big deal, but the tests
        # below would need to be changed. So, just using the pyYeti
        # versions directly (nastran read them fine)
        "tests/nastran_op4_data/big_bigmat_ascii.op4",
        "tests/nastran_op4_data/big_bigmat_binary.op4",
        "tests/nastran_op4_data/big_dense_ascii.op4",
        "tests/nastran_op4_data/big_dense_binary.op4",
    ]

    for fname in fnames:
        for rd_sparse in (True, False, None):
            m = op4.read(fname, sparse=rd_sparse)
            m2 = write_read(m, binary=True, sparse="dense")
            m3 = write_read(m, binary=False, sparse="dense")
            m4 = write_read(m, binary=False, sparse="nonbigmat")
            m5 = write_read(m, binary=True, sparse="nonbigmat")
            m6 = write_read(m, binary=False, sparse="bigmat")
            m7 = write_read(m, binary=True, sparse="bigmat")
            m8 = write_read(m, binary=False, sparse="auto")
            m9 = write_read(m, binary=True, sparse="auto")

            for k, v in m.items():
                if rd_sparse:
                    assert sp.issparse(v)
                elif rd_sparse is None and "bigmat" in fname:
                    if "nonbigmat" in fname and not sp.issparse(v):
                        # v could be dense if all zeros
                        assert np.all(v == 0.0)
                    else:
                        assert sp.issparse(v)
                va = v.toarray() if sp.issparse(v) else v

                assert np.all(va == m2[k])
                assert np.all(va == m3[k])

                if sp.issparse(m4[k]):
                    assert np.all(va == m4[k].A)
                else:
                    assert np.allclose(va, m4[k])
                    assert np.all(m4[k] == 0.0)

                if sp.issparse(m5[k]):
                    assert np.all(va == m5[k].A)
                else:
                    assert np.allclose(va, m5[k])
                    assert np.all(m5[k] == 0.0)

                assert sp.issparse(m6[k])
                assert np.all(va == m6[k].A)

                assert sp.issparse(m7[k])
                assert np.all(va == m7[k].A)

                if rd_sparse or (rd_sparse is None and sp.issparse(v)):
                    assert sp.issparse(m8[k])
                    assert np.all(va == m8[k].A)
                    assert sp.issparse(m9[k])
                    assert np.all(va == m9[k].A)
                else:
                    assert np.all(va == m8[k])
                    assert np.all(va == m9[k])


def test_large_sparse():
    data = [2.3, 5, -100.4]
    rows = [2, 500000, 3500000]
    cols = [3750000, 500000, 4999999]
    a = sp.csr_matrix((data, (rows, cols)), shape=(5000000, 5000000))

    f = tempfile.NamedTemporaryFile(delete=False)
    name = f.name
    f.close()
    try:
        op4.write(name, dict(a=a))
        a2 = op4.read(name, sparse=None)
    finally:
        os.remove(name)

    assert sp.issparse(a2["a"])
    a2 = a2["a"].tocsr()
    for i, j, v in zip(rows, cols, data):
        assert np.allclose(a2[i, j], v)


def test_large_rows_dense():
    fname = "tests/nastran_op4_data/x100000.op4"
    m = op4.read(fname)
    x = np.zeros((100000, 1))
    x[45678] = 1.0
    assert np.allclose(m["x"], x)


def test_premature_eof_warning():
    a = np.random.randn(10, 10)
    f = tempfile.NamedTemporaryFile(delete=False)
    fname = f.name
    f.close()

    try:
        op4.write(fname, {"a": a})
        bufr = open(fname, "rb").read()
        open(fname, "wb").write(bufr[:-6])

        with assert_warns(RuntimeWarning) as cm:
            a2 = op4.read(fname)
            assert cm.warnings[0].message.args[0].startswith("Premature end-of-file")

    finally:
        os.remove(fname)

    assert (a2["a"] == a).all()


def test_invalid_write_name():
    a = 1.0

    f = tempfile.NamedTemporaryFile(delete=False)
    name = f.name
    f.close()
    with assert_warns(RuntimeWarning) as cm:
        try:
            op4.write(name, {"4badname": a})
        finally:
            os.remove(name)
    assert (
        cm.warnings[0].message.args[0]
        == "Matrix for output4 write has name: '4badname'. Changing to 'm0'."
    )


def test_too_long_write_name():
    a = 1.0

    f = tempfile.NamedTemporaryFile(delete=False)
    name = f.name
    f.close()
    with assert_warns(RuntimeWarning) as cm:
        try:
            op4.write(name, {"name_is_too_long": a})
        finally:
            os.remove(name)
    assert cm.warnings[0].message.args[0] == (
        "Matrix for output4 write has name: 'name_is_too_long'. "
        "Truncating to 'name_is_'."
    )
