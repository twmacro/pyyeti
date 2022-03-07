from setuptools import setup, find_packages, Extension
from distutils.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
import numpy
import pyyeti
import os


# the following is here so matplotlib will not open figures during
# "python setup.py nosetests" ... but don't make this a hard
# requirement
try:
    import matplotlib as mpl
except ImportError:
    pass
else:
    mpl.interactive(False)
    mpl.use("Agg")

ext_errors = (
    CCompilerError,
    DistutilsExecError,
    DistutilsPlatformError,
    IOError,
    ValueError,
)


def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


long_description = read("README.md")
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Programming Language :: C",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: Implementation :: CPython",
    "Natural Language :: English",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering",
]


def check_dependencies():
    install_requires = []
    packages = ["numpy", "scipy", "matplotlib", "pandas", "xlsxwriter", "h5py"]
    for package in packages:
        try:
            exec(f"import {package}")
        except ImportError:
            install_requires.append(package)
    return install_requires


class BuildFailed(Exception):
    pass


class ve_build_ext(build_ext):
    # This class allows C extension building to fail.
    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            raise BuildFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except ext_errors:
            raise BuildFailed()


def run_setup(with_binary):
    if with_binary:
        kw = dict(
            ext_modules=[
                Extension("pyyeti.rainflow.c_rain", ["pyyeti/rainflow/c_rain.c"])
            ],
            cmdclass=dict(build_ext=ve_build_ext),
            include_dirs=[numpy.get_include()],
        )
    else:
        kw = {}

    install_requires = check_dependencies()
    setup(
        name="pyyeti",
        version=pyyeti.__version__,
        url="http://github.com/twmacro/pyyeti/",
        license="BSD",
        author="Tim Widrick",
        install_requires=install_requires,
        author_email="twmacro@gmail.com",
        description=("Tools mostly related to structural dynamics"),
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages(),
        scripts=["scripts/lsop2", "scripts/lsop4"],
        include_package_data=True,
        data_files=[
            (
                "tests/nas2cam_csuper",
                [
                    "tests/nas2cam_csuper/nas2cam.op2",
                    "tests/nas2cam_csuper/nas2cam.op4",
                    "tests/nas2cam_csuper/inboard.op4",
                    "tests/nas2cam_csuper/inboard.asm",
                ],
            ),
            (
                "tests/nastran_drm12",
                [
                    "tests/nastran_drm12/inboard_nas2cam.op2",
                    "tests/nastran_drm12/inboard_nas2cam.op4",
                    "tests/nastran_drm12/drm12.op2",
                    "tests/nastran_drm12/drm12.op4",
                ],
            ),
        ],
        platforms="any",
        setup_requires=["nose>=1.0"],
        test_suite="nose.collector",
        tests_require=["nose"],
        classifiers=CLASSIFIERS,
        **kw,
    )


if __name__ == "__main__":
    try:
        run_setup(True)
    except BuildFailed:
        BUILD_EXT_WARNING = """
Warning:

   The C rainflow extension could not be compiled; only plain Python
   rainflow counter will be available. Note: the Python version will
   be sped up with `numba.jit(nopython=True)` if possible; tests show
   speeds on par with compiled C version.

"""
        print("*" * 86)
        print(BUILD_EXT_WARNING)
        print("Failure information, if any, is above.")
        print("I'm retrying the build without the C extension now.")
        print("*" * 86)

        run_setup(False)

        print("*" * 86)
        print(BUILD_EXT_WARNING)
        print("Plain-Python installation succeeded.")
        print("*" * 86)
