import numpy as np
from setuptools import Extension, setup

# This extension module can not be defined in pyproject.toml because of the required call
# to np.get_include().
setup(
    ext_modules=[
        Extension(
            name="pyyeti.rainflow.c_rain",
            sources=["pyyeti/rainflow/c_rain.c"],
            include_dirs=[np.get_include()],
            optional=True,
        ),
    ],
)
