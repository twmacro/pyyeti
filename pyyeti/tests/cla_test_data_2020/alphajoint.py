import os
from pyyeti import cla
from pyyeti.nastran import n2p


def alphajoint(sol, nas, Vars, se):
    return Vars[se]["alphadrm"] @ sol.a


def get_drdefs(nas, sc):
    drdefs = cla.DR_Def(sc["drdefs"].defaults)

    @cla.DR_Def.addcat
    def _():
        se = 0
        name = "alphajoint"
        desc = "Alpha-Joint Acceleration"
        units = "mm/sec^2, rad/sec^2"
        labels = ["Alpha-Joint {:2s}".format(i) for i in "X,Y,Z,RX,RY,RZ".split(",")]
        drms = {"alphadrm": n2p.formdrm(nas, 0, 33)[0]}
        srsopts = dict(eqsine=1, ic="steady")
        histpv = 1  # second row
        srspv = [1]
        drfile = os.path.abspath(__file__)
        drdefs.add(**locals())

    return drdefs
