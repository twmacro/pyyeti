import sys

sys.path.insert(0, "/home/loads/twidrick/code/pyyeti")

import numpy as np
from pyyeti.nastran import op4


allz = np.zeros((66_001, 4))
sparse = allz + 0.0
sparse[4500, 0] = 9.8
sparse[4505, 0] = -9.8
sparse[12, 1] = 1.2
sparse[16, 1] = -1.2
sparse[55_000, 2] = 5.5
sparse[55_002, 2] = -5.5
allzc = allz - 1j * allz
sparsec = sparse - 1j * sparse

mats = dict(allz=allz, allzc=allzc, sparse=sparse, sparsec=sparsec)

for binary in ("ascii", "binary"):
    for sparse in ("dense", "bigmat"):
        op4.write(
            f"big_{sparse}_{binary}.op4",
            mats,
            binary=binary == "binary",
            sparse=sparse,
        )
