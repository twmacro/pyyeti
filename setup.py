from setuptools import setup, find_packages, Extension
from distutils.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, \
    DistutilsPlatformError
import numpy
import pyyeti
import os

# the following would be good, but is soooooo slow!
# import matplotlib as mpl
# mpl.use('Agg')

ext_errors = (CCompilerError, DistutilsExecError,
              DistutilsPlatformError, IOError, ValueError)

def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

long_description = read('README.rst')
CLASSIFIERS=[
    'Development Status :: 4 - Beta',
    'Programming Language :: C',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: Implementation :: CPython',
    'Natural Language :: English',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Topic :: Scientific/Engineering',
]

def check_dependencies():
    install_requires = []
    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')
    try:
        import matplotlib
    except ImportError:
        install_requires.append('matplotlib')
    try:
        import pandas
    except ImportError:
        install_requires.append('pandas')

    if os.getenv('READTHEDOCS'):
        install_requires.append('sphinx>=1.3.1')
        install_requires.append('jupyter-core>=4')
        install_requires.append('notebook>=4')

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
            ext_modules = [
                Extension("pyyeti.rainflow.c_rain",
                          ["pyyeti/rainflow/c_rain.c"])
            ],
            cmdclass=dict(build_ext=ve_build_ext),
            include_dirs=[numpy.get_include()],
        )
    else:
        kw = {}

    install_requires = check_dependencies()
    setup(
        name='pyyeti',
        version=pyyeti.__version__,
        url='http://github.com/twmacro/pyyeti/',
        license='BSD',
        author='Tim Widrick',
        install_requires=install_requires,
        author_email='twmacro@gmail.com',
        description=('Tools mostly related '
                     'to structural dynamics'),
        long_description=long_description,
        packages=find_packages(),
        #    packages=['pyyeti', 'pyyeti/rainflow'],
        include_package_data=True,
        platforms='any',
        #    test_suite='nose.collector',
        tests_require=['nose'],
        classifiers=CLASSIFIERS,
        **kw
    )

if __name__ == "__main__":
    try:
        run_setup(True)
    except BuildFailed:
        BUILD_EXT_WARNING = ("WARNING: The C rainflow extension could not "
                             "be compiled; fast rainflow is not enabled.")
        print('*' * 86)
        print(BUILD_EXT_WARNING)
        print("Failure information, if any, is above.")
        print("I'm retrying the build without the C extension now.")
        print('*' * 86)

        run_setup(False)

        print('*' * 86)
        print(BUILD_EXT_WARNING)
        print("Plain-Python installation succeeded.")
        print('*' * 86)
