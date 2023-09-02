#!/bin/sh

# rm -fr .coverag*
coverage erase
python -m pytest --cov --doctest-modules pyyeti --ignore=pyyeti/tests
python -m pytest --cov --cov-append pyyeti/tests
coverage report
