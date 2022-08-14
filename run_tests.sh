#!/bin/sh

# rm -fr .coverag*
coverage erase
pytest --cov --doctest-modules pyyeti
pytest --cov --cov-append tests
coverage report 
