#!/bin/sh

# rm -fr .coverag*
coverage erase
python -m pytest --cov --doctest-modules pyyeti
python -m pytest --cov --cov-append tests
coverage report 
