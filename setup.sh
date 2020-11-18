#! /bin/bash

# This should be executed once, inside the Singularity container

cd pixel_nms && python3 setup.py build_ext --inplace
