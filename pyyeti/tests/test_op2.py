import math
import os
import re
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from pyyeti import ytools, nastran, locate
from pyyeti.nastran import op4, op2, bulk
from scipy.io import matlab
import pytest


def runcomp(nas, m):
    matnas = m["nas"]
    for name in matnas.dtype.names:
        if isinstance(nas[name], dict):
            for k in nas[name]:
                matnas_key = "k{}".format(k)
                m1 = matnas[name][0][0][matnas_key][0][0]
                m2 = nas[name][k]
                if name == "rfmodes" and len(m2) > 0:
                    m1 -= 1.0
                if isinstance(m2, np.ndarray) and m2.ndim == 1:
                    m1 = m1.flatten()
                if name == "maps" and len(m2) > 0:
                    m1[:, 0] -= 1
                if name == "uset":
                    m2 = m2.reset_index().values
                assert np.allclose(m1, m2)
        else:
            m1 = matnas[name][0][0]
            m2 = nas[name]
            assert np.allclose(m1, m2)

    # check cstm2:
    for k in nas["cstm"]:
        prem1 = nas["cstm2"][k]
        prem2 = nas["cstm"][k]
        for i, j in enumerate(prem2[:, 0]):
            m1 = prem1[int(j)]  # 5x3
            m2 = np.zeros((5, 3))
            m2[0, :2] = prem2[i, :2]
            m2[1:, :] = prem2[i, 2:].reshape((4, 3))
            assert np.allclose(m1, m2)


def test_n2c_csuper():
    nas = op2.rdnas2cam(Path("pyyeti/tests/nas2cam_csuper/nas2cam"))
    # nas.keys()
    # dict_keys(['rfmodes', 'fgravh', 'lambda', 'phg', 'dnids', 'nrb',
    # 'maps', 'kaa', 'cstm', 'maa', 'fgravg', 'uset', 'selist', 'upids',
    # 'cstm2'])
    m = matlab.loadmat("pyyeti/tests/nas2cam_csuper/nas2cam.mat")
    # In [41]: m['nas'].dtype.names
    # Out[41]:
    # ('cstm',
    #  'dnids',
    #  'fgravg',
    #  'fgravh',
    #  'kaa',
    #  'lambda',
    #  'maa',
    #  'maps',
    #  'nrb',
    #  'phg',
    #  'rfmodes',
    #  'selist',
    #  'upids',
    #  'uset')
    runcomp(nas, m)


def test_n2c_extseout():
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam_extseout/nas2cam")
    m = matlab.loadmat("pyyeti/tests/nas2cam_extseout/nas2cam.mat")
    runcomp(nas, m)


def test_n2c_error():
    with pytest.raises(ValueError):
        op2.rdnas2cam(
            "pyyeti/tests/nas2cam_extseout/assemble.op2",
            "pyyeti/tests/nas2cam_extseout/nas2cam.op2",
        )


def test_drm12_reader():
    import numpy as np

    o4 = op4.OP4()

    drm12 = "pyyeti/tests/nastran_drm12/drm12"
    mats = o4.dctload(drm12 + ".op4")
    dsorted = op2.procdrm12(drm12, dosort=True)
    # just to exercise more code:
    with op2.OP2("pyyeti/tests/nastran_drm12/drm12.op2") as o2:
        dkeys = o2.rddrm2op2(1)

    # check desc:
    assert np.all(["T1", "T2", "T3"] * 3 == dsorted["DTM_desc"])
    assert np.all(["T1", "T2", "T3"] * 3 == dsorted["ATM_desc"])
    spcf_desc = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    assert np.all(spcf_desc * 4 == dsorted["SPCF_desc"])
    stress = [
        "CBAR Bending Stress 1 - End A",  # 2
        "CBAR Bending Stress 2 - End A",  # 3
        "CBAR Bending Stress 3 - End A",  # 4
        "CBAR Bending Stress 4 - End A",  # 5
        "CBAR Axial Stress",  # 6
        "CBAR Max. Bend. Stress -End A",  # 7
        "CBAR Min. Bend. Stress -End A",  # 8
        "CBAR M.S. Tension",  # 9
        "CBAR Bending Stress 1 - End B",  # 10
        "CBAR Bending Stress 2 - End B",  # 11
        "CBAR Bending Stress 3 - End B",  # 12
        "CBAR Bending Stress 4 - End B",  # 13
        "CBAR Max. Bend. Stress -End B",  # 14
        "CBAR Min. Bend. Stress -End B",  # 15
        "CBAR M.S. Compression",
    ]  # 16
    assert np.all(stress * 2 == dsorted["STM_desc"])

    force = [
        "CBAR Bending Moment 1 - End A",  # 2
        "CBAR Bending Moment 2 - End A",  # 3
        "CBAR Bending Moment 1 - End B",  # 4
        "CBAR Bending Moment 2 - End B",  # 5
        "CBAR Shear 1",  # 6
        "CBAR Shear 2",  # 7
        "CBAR Axial Force",  # 8
        "CBAR Torque",
    ]  # 9
    assert np.all(force * 2 + force[-2:] == dsorted["LTM_desc"])

    # check id_dof:
    ids = np.array([[12] * 3, [14] * 3, [32] * 3]).reshape((1, -1)).T
    dof = np.array([[1, 2, 3] * 3]).T
    iddof = np.hstack((ids, dof))
    assert np.all(iddof == dsorted["DTM_id_dof"])
    assert np.all(iddof == dsorted["ATM_id_dof"])

    ids = np.array([[3] * 6, [11] * 6, [19] * 6, [27] * 6]).reshape((1, -1)).T
    dof = np.array([[1, 2, 3, 4, 5, 6] * 4]).T
    iddof = np.hstack((ids, dof))
    assert np.all(iddof == dsorted["SPCF_id_dof"])

    ids = np.array([[11] * 15, [89] * 15]).reshape((1, -1)).T
    dof = np.array([[i for i in range(2, 17)] * 2]).T
    iddof = np.hstack((ids, dof))
    assert np.all(iddof == dsorted["STM_id_dof"])

    ids = np.array([[11] * 8 + [23] * 8 + [28] * 2]).reshape((1, -1)).T
    dof = np.array([[i for i in range(2, 10)] * 2 + [8, 9]]).T
    iddof = np.hstack((ids, dof))
    assert np.all(iddof == dsorted["LTM_id_dof"])

    # check drms:
    # manually getting rows from TOUGV1, etc in .out file:
    rows = np.array([13, 14, 15, 19, 20, 21, 49, 50, 51]) - 1
    assert np.all(mats["mougs1"][0][rows] == dsorted["DTMD"])
    assert np.all(mats["mougd1"][0][rows] == dsorted["DTMA"])
    assert np.all(mats["mougv1"][0][rows] == dsorted["ATM"])
    assert np.all(mats["moqgs1"][0] == dsorted["SPCFD"])
    assert np.all(mats["moqgd1"][0] == dsorted["SPCFA"])

    rows = np.array([i for i in range(1, 17)] + [23, 24]) - 1
    assert np.all(mats["moefs1"][0][rows] == dsorted["LTMD"])
    assert np.all(mats["moefd1"][0][rows] == dsorted["LTMA"])

    assert np.all(mats["moess1"][0] == dsorted["STMD"])
    assert np.all(mats["moesd1"][0] == dsorted["STMA"])

    draw = op2.procdrm12(drm12, dosort=False)

    # check desc:
    assert np.all(["T1", "T2", "T3"] * 3 == draw["DTM_desc"])
    assert np.all(["T3", "T1", "T2"] + ["T1", "T2", "T3"] * 2 == draw["ATM_desc"])
    spcf_desc = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    assert np.all(spcf_desc * 4 == draw["SPCF_desc"])
    stress = [
        "CBAR Bending Stress 1 - End A",  # 2
        "CBAR Bending Stress 2 - End A",  # 3
        "CBAR Bending Stress 3 - End A",  # 4
        "CBAR Bending Stress 4 - End A",  # 5
        "CBAR Axial Stress",  # 6
        "CBAR Max. Bend. Stress -End A",  # 7
        "CBAR Min. Bend. Stress -End A",  # 8
        "CBAR M.S. Tension",  # 9
        "CBAR Bending Stress 1 - End B",  # 10
        "CBAR Bending Stress 2 - End B",  # 11
        "CBAR Bending Stress 3 - End B",  # 12
        "CBAR Bending Stress 4 - End B",  # 13
        "CBAR Max. Bend. Stress -End B",  # 14
        "CBAR Min. Bend. Stress -End B",  # 15
        "CBAR M.S. Compression",
    ]  # 16
    assert np.all(stress * 2 == draw["STM_desc"])

    force = [
        "CBAR Bending Moment 1 - End A",  # 2
        "CBAR Bending Moment 2 - End A",  # 3
        "CBAR Bending Moment 1 - End B",  # 4
        "CBAR Bending Moment 2 - End B",  # 5
        "CBAR Shear 1",  # 6
        "CBAR Shear 2",  # 7
        "CBAR Axial Force",  # 8
        "CBAR Torque",
    ]  # 9
    assert np.all(force + force[-1:] + force[-2:-1] + force == draw["LTM_desc"])

    # check id_dof:
    ids = np.array([[14] * 3, [12] * 3, [32] * 3]).reshape((1, -1)).T
    dof = np.array([[1, 2, 3] * 3]).T
    iddof = np.hstack((ids, dof))
    assert np.all(iddof == draw["DTM_id_dof"])

    dof = np.array([[3, 1, 2] + [1, 2, 3] * 2]).T
    iddof = np.hstack((ids, dof))
    assert np.all(iddof == draw["ATM_id_dof"])

    ids = np.array([[3] * 6, [11] * 6, [19] * 6, [27] * 6]).reshape((1, -1)).T
    dof = np.array([[1, 2, 3, 4, 5, 6] * 4]).T
    iddof = np.hstack((ids, dof))
    assert np.all(iddof == draw["SPCF_id_dof"])

    ids = np.array([[89] * 15, [11] * 15]).reshape((1, -1)).T
    dof = np.array([[i for i in range(2, 17)] * 2]).T
    iddof = np.hstack((ids, dof))
    assert np.all(iddof == draw["STM_id_dof"])

    ids = np.array([[23] * 8 + [28] * 2 + [11] * 8]).reshape((1, -1)).T
    dof = np.array([[i for i in range(2, 10)] + [9, 8] + [i for i in range(2, 10)]]).T
    iddof = np.hstack((ids, dof))
    assert np.all(iddof == draw["LTM_id_dof"])

    # check drms:
    # manually getting rows from TOUGV1, etc in .out file:
    rows = np.array([19, 20, 21, 13, 14, 15, 49, 50, 51]) - 1
    assert np.all(mats["mougs1"][0][rows] == draw["DTMD"])
    assert np.all(mats["mougd1"][0][rows] == draw["DTMA"])
    rows = np.array([21, 19, 20, 13, 14, 15, 49, 50, 51]) - 1
    assert np.all(mats["mougv1"][0][rows] == draw["ATM"])
    assert np.all(mats["moqgs1"][0] == draw["SPCFD"])
    assert np.all(mats["moqgd1"][0] == draw["SPCFA"])

    rows = np.array([9, 10, 11, 12, 13, 14, 15, 16, 24, 23, 1, 2, 3, 4, 5, 6, 7, 8]) - 1
    assert np.all(mats["moefs1"][0][rows] == draw["LTMD"])
    assert np.all(mats["moefd1"][0][rows] == draw["LTMA"])

    rows = np.array([i for i in range(16, 31)] + [i for i in range(1, 16)]) - 1
    assert np.all(mats["moess1"][0][rows] == draw["STMD"])
    assert np.all(mats["moesd1"][0][rows] == draw["STMA"])


def test_codefuncs():
    with op2.OP2("pyyeti/tests/nastran_drm12/drm12.op2") as o:
        assert o.CodeFuncs[1](7) == 1
        assert o.CodeFuncs[1](3002) == 2
        assert o.CodeFuncs[2](123) == 23
        assert o.CodeFuncs[3](123) == 123
        assert o.CodeFuncs[4](22) == 2
        assert o.CodeFuncs[5](15) == 5
        assert o.CodeFuncs[6](8) == 0
        assert o.CodeFuncs[6](9) == 0
        assert o.CodeFuncs[6](7) == 1
        assert o.CodeFuncs[7](222) == 0
        assert o.CodeFuncs[7](2222) == 0
        assert o.CodeFuncs[7](1222) == 1
        assert o.CodeFuncs[7](3222) == 1
        assert o.CodeFuncs[7](5222) == 2
        funccode = 123456
        val = 18 & (funccode & 65535)
        assert o.CodeFuncs["big"](funccode, 18) == val

        assert o._check_code(22, [4], [[2]], "test")
        with pytest.warns(RuntimeWarning):
            o._check_code(22, [4], [[42]], "test")
        assert o._check_code(18, [funccode], [[val]], "test")

        with pytest.warns(RuntimeWarning):
            o._check_code(18, [funccode], [[1]], "test")

        with pytest.raises(ValueError):
            o._check_code(22, [4, 4], [[2]], "test")
        with pytest.raises(ValueError):
            o._check_code(18, [77], [[val]], "test")


def test_bigend():
    # most files are little-endian ... test a big-endian file:
    nas = op2.rdnas2cam("pyyeti/tests/nas2cam/bigend_nas2cam")
    # 1st 3 rows of the GM matrix should have 1/3 in x, y, z
    # positions: [[ 1/3,   0,   0, 0, 0, 0, 1/3,   0,   0, ...],
    #             [   0, 1/3,   0, 0, 0, 0,   0, 1/3,   0, ...],
    #             [   0,   0, 1/3, 0, 0, 0,   0,   0, 1/3, ...]]
    # - there are 18 columns (3 grids being averaged)
    # use broadcasting to get the 1/3 values out:
    rows = [[0], [1], [2]]
    cols = ytools.mkpattvec([0, 6, 12], 3, 1)
    gmvals = nas["gm"][0][rows, cols]
    assert np.allclose(gmvals, 1 / 3)


def test_notop2():
    with pytest.raises(ValueError):
        op2.OP2("pyyeti/tests/nas2cam/no_se.dat")


def test_rdop2mats():
    dct2 = op4.load("pyyeti/tests/nastran_op4_data/r_c_rc.op4")
    dr = "pyyeti/tests/nastran_op2_data/"
    for cut in (0, 30000):
        for name in ("double_le.op2", "double_be.op2"):
            with op2.OP2(dr + name) as o2:
                o2._rowsCutoff = cut
                dct = o2.rdop2mats()
            assert np.allclose(dct2["rmat"][0], dct["ZUZR01"])
            assert np.allclose(dct2["cmat"][0], dct["ZUZR02"])
            assert np.allclose(dct2["rcmat"][0], dct["ZUZR03"])

        for name in ("single_le.op2", "single_be.op2"):
            with op2.OP2(dr + name) as o2:
                o2._rowsCutoff = cut
                dct = o2.rdop2mats()
            assert np.allclose(dct2["rmat"][0], dct["ZUZR04"])
            assert np.allclose(dct2["cmat"][0], dct["ZUZR05"])
            assert np.allclose(dct2["rcmat"][0], dct["ZUZR06"])

    with op2.OP2(dr + "double_le.op2") as o2:
        dct = o2.rdop2mats(["zuzr01", "zuzr03"])
        assert np.allclose(dct2["rmat"][0], dct["ZUZR01"])
        assert np.allclose(dct2["rcmat"][0], dct["ZUZR03"])
        assert len(dct) == 2

    dct = op2.rdmats(dr + "double_le.op2", ["zuzr01", "zuzr03"])
    assert np.allclose(dct2["rmat"][0], dct["ZUZR01"])
    assert np.allclose(dct2["rcmat"][0], dct["ZUZR03"])
    assert len(dct) == 2

    with op2.OP2(dr + "double_le.op2") as o2:
        d = o2.directory()
    assert sorted(d.keys()) == ["CASECC", "ZUZR01", "ZUZR02", "ZUZR03"]


def test_rdop2tload():
    sbe = np.zeros((5, 30), dtype=np.int64)
    sbe[0] = range(1001, 1031)
    sbe[1] = range(1101, 1131)
    sbe[4] = 1000
    tload = op2.OP2("pyyeti/tests/nas2cam_extseout/inboard.op2").rdop2tload()
    assert np.all(tload[:5] == sbe)

    tload = op2.OP2("pyyeti/tests/nas2cam_extseout/inboard_v2007.op2").rdop2tload()
    assert np.all(tload[:5] == sbe)

    with op2.OP2("pyyeti/tests/nas2cam_extseout/inboard.op2") as o2:
        o2._rowsCutoff = 0
        tload = o2.rdop2tload()
    assert np.all(tload[:5] == sbe)


def test_rdop2gpwg_not_present():
    gpwg = op2.OP2("pyyeti/tests/nastran_op2_data/rdop2gpwg_2.op2").rdop2gpwg()
    assert gpwg is None


def test_rdop2gpwg():
    gpwg = op2.OP2("pyyeti/tests/nastran_op2_data/rdop2gpwg_1.op2").rdop2gpwg()
    assert isinstance(gpwg, dict)
    keys = ["m66", "s", "mass3", "mass", "cg33", "cg", "Is", "Iq", "q"]
    m66, s, mass3, mass, cg33, cg, Is, Iq, q = [gpwg[key] for key in keys]
    assert m66.dtype == np.dtype("f4")
    np.testing.assert_allclose(
        m66[0, :], [5.005019e1, 0.0, 0.0, 0.0, 1.000585e1, -5.004597], atol=1e-5
    )
    np.testing.assert_allclose(
        m66[2, :], [0.0, 0.0, 5.005019e1, 5.004597, -1.711721e2, 0.0], atol=1e-5
    )
    np.testing.assert_allclose(
        m66[5, 3:6], [-1.951159e1, -2.75115e1, 1.122445e3], atol=1e-3
    )
    np.testing.assert_allclose(s, np.eye(3))
    np.testing.assert_allclose(mass3, [5.005019e1] * 3, atol=1e-5)
    assert math.isclose(mass, 5.005019e1, abs_tol=1e-5)
    np.testing.assert_allclose(cg33[0, :], [0.0, 9.999156e-2, 1.999163e-1], atol=1e-5)
    np.testing.assert_allclose(cg33[1, :], [3.42, 0.0, 1.999163e-1], atol=1e-5)
    np.testing.assert_allclose(cg33[2, :], [3.42, 9.999156e-2, 0.0], atol=1e-5)
    np.testing.assert_allclose(cg, [3.42, 9.999156e-2, 1.999163e-1], atol=1e-5)
    np.testing.assert_allclose(Is[0, :], [1.550258e2, -8.106795, -1.47085e1], atol=1e-4)
    np.testing.assert_allclose(Is[2, 1:3], [2.6511e1, 5.365345e2], atol=1e-5)
    np.testing.assert_allclose(Iq, [5.02794e2, 1.542326e2, 5.570665e2], atol=1e-4)
    np.testing.assert_allclose(q[0, 0:2], [-4.41407e-2, 9.98883e-1], atol=1e-5)
    np.testing.assert_allclose(q[2, 1:3], [-4.016433e-2, 7.950108e-1], atol=1e-5)


def test_rdop2opg_not_present():
    opg = op2.OP2("pyyeti/tests/nastran_op2_data/rdop2gpwg_2.op2").rdop2opg()
    assert opg is None


def test_rdop2opg():
    opg = op2.OP2("pyyeti/tests/nastran_op2_data/rdop2opg_1.op2").rdop2opg()
    grid_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 21, 22])
    assert isinstance(opg, dict)
    assert len(opg) == 2
    # subcase 1001
    ls1001 = opg[1001]
    assert isinstance(ls1001, pd.DataFrame)
    assert ls1001.shape == (grid_ids.shape[0], 6)
    np.testing.assert_array_equal(ls1001.index, grid_ids)
    np.testing.assert_array_equal(ls1001.columns, [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(ls1001.loc[[1, 2, 3, 4, 6, 7, 8, 22]], 0.0)
    # forces applied in csys 0, but reported in csys 11 (grid 5 output csys)
    # these values were calculated with Femap
    np.testing.assert_allclose(
        ls1001.loc[5], [-9.1490, 6.0960, -0.3665, -9.9808, 6.6502, -0.3998], atol=1e-3
    )
    np.testing.assert_allclose(ls1001.loc[21], [5.0, 6.0, 0.0, 0.0, 7.0, 8.0])
    # subcase 1002
    ls1002 = opg[1002]
    assert isinstance(ls1002, pd.DataFrame)
    assert ls1002.shape == (grid_ids.shape[0], 6)
    np.testing.assert_array_equal(ls1002.index, grid_ids)
    np.testing.assert_array_equal(ls1002.columns, [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(ls1002.loc[[2, 3, 4, 5, 6, 7, 21, 22]], 0.0)
    np.testing.assert_allclose(ls1002.loc[1], [0.0, 0.2, 0.0, 100.0, 200.0, 300.0])
    np.testing.assert_allclose(ls1002.loc[8], [1.1, 1.2, 1.3, 0.0, 0.0, 0.0])


def test_rdpostop2():
    post = op2.rdpostop2("pyyeti/tests/nas2cam_extseout/inboard.op2", 1, 1, 1, 1)

    dof = post["mats"]["ougv1"][0]["dof"]
    lam = post["mats"]["ougv1"][0]["lambda"]
    ougv1 = post["mats"]["ougv1"][0]["ougv1"]

    o4 = op4.load("pyyeti/tests/nas2cam_extseout/inboard.op4")
    mug1 = o4["mug1"][0]
    tug1 = nastran.rddtipch("pyyeti/tests/nas2cam_extseout/inboard.pch")
    tef1 = nastran.rddtipch("pyyeti/tests/nas2cam_extseout/inboard.pch", "TEF1")
    tes1 = nastran.rddtipch("pyyeti/tests/nas2cam_extseout/inboard.pch", "TES1")

    # ougv1, oef1, oes1 ... they don't have the constraint modes
    # or the resflex modes! How can they be useful? Anyway, this
    # checks the values present:

    # mug1 has 24 b-set ... get first 3 modes (rest are resflex):
    modes = mug1[:, 24:27]

    pv = locate.mat_intersect(tug1, dof)[0]
    assert np.allclose(modes[pv], ougv1)

    assert np.allclose(o4["mef1"][0][:, 24:27], post["mats"]["oef1"][0][0])
    assert np.all(post["mats"]["oef1"][0][1][:, 0] == tef1[:, 0])
    assert np.all(post["mats"]["oef1"][0][1][:, 1] == 34)

    pv = np.ones(15, bool)
    pv[5:7] = False
    pv[11:15] = False
    pv = np.hstack((pv, pv, pv))

    assert np.allclose(o4["mes1"][0][:, 24:27], post["mats"]["oes1"][0][0][pv])
    assert np.all(post["mats"]["oes1"][0][1][pv, 0] == tes1[:, 0])
    assert np.all(post["mats"]["oes1"][0][1][:, 1] == 34)

    with op2.OP2("pyyeti/tests/nas2cam_extseout/inboard.op2") as o2:
        o2._rowsCutoff = 0
        fpos = o2.dbdct["OUGV1"][0].start
        assert fpos == o2.dbnames["OUGV1"][0][0][0]
        o2._fileh.seek(fpos)
        name, trailer, dbtype = o2.rdop2nt()
        oug = o2._rdop2ougv1("OUGV1")

    assert np.all(oug["ougv1"] == ougv1)
    assert np.all(oug["dof"] == dof)
    assert np.all(oug["lambda"] == lam)


def verify_one_match(records, regexes):
    """Verify that at least one warning matches each regex."""
    for regex in regexes:
        match_found = False
        for record in records:
            match_found = (
                re.search(regex, record.message.args[0]) is not None or match_found
            )
        assert match_found, f"No matching warning found: {regex}"


def test_rdpostop2_assemble():
    with pytest.warns(RuntimeWarning) as records:
        post = op2.rdpostop2("pyyeti/tests/nas2cam_extseout/assemble.op2", 1, 1)
        verify_one_match(
            records, [r"ACODE.*52.*not accept", r"TCODE.*1001.*not accept"]
        )
    assert np.all(post["selist"] == [[101, 0], [102, 0], [0, 0]])
    sebulk = np.array(
        [[101, 5, -1, 2, 0.0, 1, 3, 101], [102, 5, -1, 2, 0.0, 1, 3, 102]]
    )
    assert np.all(post["sebulk"] == sebulk)

    with op2.OP2("pyyeti/tests/nas2cam_extseout/assemble.op2") as o2:
        o2._rowsCutoff = 0
        o2.set_position("GEOM1S", 2)
        name, trailer, dbtype = o2.rdop2nt()
        # (cords, sebulk, selist, seload, seconct) = o2._rdop2geom1cord2()
        dct = o2._rdop2geom1cord2()
        assert np.all(post["selist"] == dct["selist"])
        assert np.all(post["sebulk"] == dct["sebulk"])


def test_rdpostop2_jsc_cla_model():
    post = op2.rdpostop2("pyyeti/tests/nas2cam_extseout/cla_test_model.op2")

    assert len(post["geom1"]) == len(post["geom1_list"]) == 2

    grid = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0],
            [3.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            [4.0, 0.0, 3.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            [5.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert (post["geom1"][0]["grid"] == grid).all()

    extrn = np.array(
        [
            [5, 123456],
            [90560001, 1],
            [90560002, 1],
            [90560003, 1],
            [90560004, 1],
            [90560005, 1],
            [90560006, 1],
            [-1, -1],
        ]
    )
    assert (post["geom1"][56]["extrn"] == extrn).all()


def test_rdpostop2_many_nodes():
    post = op2.rdpostop2("pyyeti/tests/nas2cam_extseout/cant_beam.op2")
    assert (post["geom1"][0]["grid"][:, 0].astype(int) == np.arange(8222) + 1).all()


def test_rdop2record():
    with op2.OP2("pyyeti/tests/nas2cam_extseout/inboard.op2") as o2:
        fpos = o2.dbdct["DYNAMICS"][0].start

        fh = o2.file_handle()
        fh.seek(fpos)
        name, trailer, dbtype = o2.rdop2nt()
        o2._rowsCutoff = 40000
        r1 = o2.rdop2record()

        fh.seek(fpos)
        name, trailer, dbtype = o2.rdop2nt()
        o2._rowsCutoff = 0
        r2 = o2.rdop2record()
        assert np.all(r1 == r2)

        fh.seek(fpos)
        name, trailer, dbtype = o2.rdop2nt()
        with pytest.raises(ValueError):
            o2.rdop2record("badform")

        fh.seek(fpos)
        name, trailer, dbtype = o2.rdop2nt()
        assert len(o2.rdop2tabheaders()) == 3


def test_prtdir():
    with op2.OP2("pyyeti/tests/nas2cam_extseout/inboard.op2") as o2:
        o2.dblist = []
        o2.prtdir()
        o2.directory(verbose=1, redo=1)


def compdict(d1, d2, is_dataframe=False):
    assert sorted(d1.keys()) == sorted(d2.keys())
    for k in d1:
        if isinstance(d1[k], dict):
            compdict(d1[k], d2[k], k == "uset")
        elif is_dataframe:
            assert np.all((d1[k] == d2[k]).values)
        else:
            assert np.all(d1[k] == d2[k])


def test_rdop2emap():
    with op2.OP2("pyyeti/tests/nas2cam_csuper/nas2cam.op2") as o2:
        nas1 = o2.rdn2cop2()
        o2._rowsCutoff = 0
        nas2 = o2.rdn2cop2()
        compdict(nas1, nas2)


def test_empty_file_error():
    f = tempfile.NamedTemporaryFile(delete=False)
    fname = f.name
    f.close()

    try:
        with pytest.raises(RuntimeError):
            op2.OP2(fname)
    finally:
        os.remove(fname)


def test_rdparampost_old():
    post = op2.rdparampost_old(
        "pyyeti/tests/nas2cam_extseout/assemble.op2", get_all=True
    )
    assert np.all(post["selist"] == [[101, 0], [102, 0], [0, 0]])
    sebulk = np.array(
        [[101, 5, -1, 2, 0.0, 1, 3, 101], [102, 5, -1, 2, 0.0, 1, 3, 102]]
    )
    assert np.all(post["sebulk"] == sebulk)

    selist = np.array([[101, 0], [102, 0], [0, 0]])
    assert np.all(post["geom1"][0]["selist"] == selist)
    assert np.all(post["geom1"][101]["grid"][:, 0] == [3, 11, 19, 27])

    assert post["khh"].shape == (54, 54)
    assert post["lama"].shape == (54, 7)

    pa = op2.rdparampost_old(
        "pyyeti/tests/nas2cam_extseout/assemble.op2", get_all=True, which="all"
    )

    assert np.all(pa["selist"] == selist)
    assert np.all(pa["geom1"][0]["selist"] == selist)
    assert np.all(pa["geom1"][101]["grid"][:, 0] == [3, 11, 19, 27])

    assert len(pa["khh"]) == 1
    assert pa["khh"][0].shape == (54, 54)
    assert pa["lama"][0].shape == (54, 7)
    assert np.allclose(pa["khh"][0], post["khh"])
    assert len(pa["ougv1"][0]) == 3

    # another test file:
    with pytest.warns(RuntimeWarning) as records:
        post = op2.rdparampost_old(
            "pyyeti/tests/nas2cam_extseout/inboard_v2007.op2", get_all=True
        )
        verify_one_match(records, [r"have 1 'GEOM1' data blocks and 0 'BGPDT'"])

    assert np.allclose(post["ougv1"]["lambda"], post["lama"][:, 2])
    assert np.allclose(
        post["geom1"][0]["grid"],
        bulk.rdgrids("pyyeti/tests/nas2cam_extseout/inboard_v2007.dat"),
    )

    assert post["oes1x1"][0].shape > (0, 0)
    assert post["oef1x"][0].shape > (0, 0)

    with pytest.warns(RuntimeWarning) as records:
        pa = op2.rdparampost_old(
            "pyyeti/tests/nas2cam_extseout/inboard_v2007.op2", get_all=True, which="all"
        )
        verify_one_match(records, [r"have 1 'GEOM1' data blocks and 0 'BGPDT'"])

    assert pa["tload"].shape == (8, 30)
    assert np.all(pa["tload"] == post["tload"])
    assert np.all(pa["oef1x"][0][0] == post["oef1x"][0])
    assert np.all(pa["ougv1"][0]["lambda"] == post["ougv1"]["lambda"])

    # a third test file:

    p = op2.rdparampost_old(
        "pyyeti/tests/nastran_op2_data/example_with_tugd.op2",
        get_all=True,
    )

    assert np.allclose(p["ogpwg"]["mass"], 625798.31)
    assert p["tload"].shape == (6, 47)
    assert np.all(p["tugd"][-1] == [1718, 6, 1])
    assert np.all(p["tefd"][-1] == [1104, 77, 2])

    pa = op2.rdparampost_old(
        "pyyeti/tests/nastran_op2_data/example_with_tugd.op2",
        get_all=True,
        verbose=True,
        which="all",
    )

    assert np.allclose(pa["ogpwg"][0]["mass"], 625798.31)
    assert pa["tload"].shape == (6, 47)
    assert np.all(pa["tugd"][0][-1] == [1718, 6, 1])
    assert np.all(pa["tefd"][0][-1] == [1104, 77, 2])

    # test rdparampost_old on a file w/o GEOM1 (not really a param,post,-1
    # file, but still want it to do what it can):
    pa = op2.rdparampost_old(
        "pyyeti/tests/nastran_op2_data/rdop2gpwg_1.op2",
        get_all=True,
    )
    gpwg = op2.OP2("pyyeti/tests/nastran_op2_data/rdop2gpwg_1.op2").rdop2gpwg()
    for key, val in gpwg.items():
        assert np.allclose(pa["ogpwg"][key], val)
    assert len(pa["geom1"]) == len(pa["bgpdt"]) == 0
    assert "lama" in pa
    assert "ougv1" in pa


def test_rdparampost_old_uset():
    p = op2.rdparampost_old("pyyeti/tests/nas2cam_extseout/inboard.op2")
    u, c, b = bulk.asm2uset("pyyeti/tests/nas2cam_extseout/inboard.asm")
    assert np.all(p["uset"].loc[u.loc[b].index] == u.loc[b])


def test_rdparampost():
    post = op2.rdparampost("pyyeti/tests/nas2cam_extseout/assemble.op2", get_all=True)
    assert np.all(post["selist"] == [[101, 0], [102, 0], [0, 0]])
    sebulk = np.array(
        [[101, 5, -1, 2, 0.0, 1, 3, 101], [102, 5, -1, 2, 0.0, 1, 3, 102]]
    )
    assert np.all(post["sebulk"] == sebulk)

    selist = np.array([[101, 0], [102, 0], [0, 0]])
    assert np.all(post["geom1"][0]["selist"] == selist)
    assert np.all(post["geom1"][101]["grid"][:, 0] == [3, 11, 19, 27])

    assert post["khh"].shape == (54, 54)
    assert post["lama"].shape == (54, 7)

    pa = op2.rdparampost(
        "pyyeti/tests/nas2cam_extseout/assemble.op2", get_all=True, which="all"
    )

    assert np.all(pa["selist"] == selist)
    assert np.all(pa["geom1"][0]["selist"] == selist)
    assert np.all(pa["geom1"][101]["grid"][:, 0] == [3, 11, 19, 27])

    assert len(pa["khh"]) == 1
    assert pa["khh"][0].shape == (54, 54)
    assert pa["lama"][0].shape == (54, 7)
    assert np.allclose(pa["khh"][0], post["khh"])
    assert isinstance(pa["ougv1"][0], pd.DataFrame)

    # another test file:
    with pytest.warns(RuntimeWarning) as records:
        post = op2.rdparampost(
            "pyyeti/tests/nas2cam_extseout/inboard_v2007.op2", get_all=True
        )
        verify_one_match(records, [r"have 1 'GEOM1' data blocks and 0 'BGPDT'"])

    assert np.allclose(post["ougv1"].columns.get_level_values(1), post["lama"][:, 4])
    assert np.allclose(
        post["geom1"][0]["grid"],
        bulk.rdgrids("pyyeti/tests/nas2cam_extseout/inboard_v2007.dat"),
    )

    assert post["oes1x1"].shape > (0, 0)
    assert post["oef1x"].shape > (0, 0)

    with pytest.warns(RuntimeWarning) as records:
        pa = op2.rdparampost(
            "pyyeti/tests/nas2cam_extseout/inboard_v2007.op2", get_all=True, which="all"
        )
        verify_one_match(records, [r"have 1 'GEOM1' data blocks and 0 'BGPDT'"])

    assert pa["tload"].shape == (8, 30)
    assert np.all(pa["tload"] == post["tload"])
    assert pa["oef1x"][0].equals(post["oef1x"])

    # a third test file:
    p = op2.rdparampost(
        "pyyeti/tests/nastran_op2_data/example_with_tugd.op2",
        get_all=True,
    )

    assert np.allclose(p["ogpwg"]["mass"], 625798.31)
    assert p["tload"].shape == (6, 47)
    assert np.all(p["tugd"][-1] == [1718, 6, 1])
    assert np.all(p["tefd"][-1] == [1104, 77, 2])

    pa = op2.rdparampost(
        "pyyeti/tests/nastran_op2_data/example_with_tugd.op2",
        get_all=True,
        verbose=True,
        which="all",
    )

    assert np.allclose(pa["ogpwg"][0]["mass"], 625798.31)
    assert pa["tload"].shape == (6, 47)
    assert np.all(pa["tugd"][0][-1] == [1718, 6, 1])
    assert np.all(pa["tefd"][0][-1] == [1104, 77, 2])

    # test rdparampost on a file w/o GEOM1 (not really a param,post,-1
    # file, but still want it to do what it can):
    pa = op2.rdparampost(
        "pyyeti/tests/nastran_op2_data/rdop2gpwg_1.op2",
        get_all=True,
    )
    gpwg = op2.OP2("pyyeti/tests/nastran_op2_data/rdop2gpwg_1.op2").rdop2gpwg()
    for key, val in gpwg.items():
        assert np.allclose(pa["ogpwg"][key], val)
    assert len(pa["geom1"]) == len(pa["bgpdt"]) == 0
    assert "lama" in pa
    assert "ougv1" in pa


def test_rdparampost_uset():
    p = op2.rdparampost("pyyeti/tests/nas2cam_extseout/inboard.op2")
    u, c, b = bulk.asm2uset("pyyeti/tests/nas2cam_extseout/inboard.asm")
    assert np.all(p["uset"].loc[u.loc[b].index] == u.loc[b])


def test_rdparampost_64bit():
    p32 = op2.rdparampost(
        "pyyeti/tests/nastran_op2_data/tugd_2020_32bit.op2",
        get_all=True,
    )
    p64 = op2.rdparampost(
        "pyyeti/tests/nastran_op2_data/tugd_2020_64bit.op2",
        get_all=True,
    )

    for name in (
        "dynamics",
        "tugd",
        "tefd",
        "tload",
    ):
        assert (p32[name] == p64[name]).all()
