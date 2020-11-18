# SORTS and SORTS+RReID

This repository contains the source code for the paper *Real-Time and Online Segmentation Multi-Target Tracking with Track Revival Re-Identification* which will be published at VISAPP 2021 (link to paper will appear here after publication, unless forgotten).

SORTS and SORTS+RReID are based on [SORT](https://github.com/abewley/sort), copied at 2020-05-27.

## Requirements
- A modern Linux PC with an NVidia GPU with appropriate drivers
- [Singularity](https://sylabs.io/docs/)

## Instructions
- `git clone https://github.com/ahrnbom/sorts`
- `cd sorts`
- `./run_singularity.sh build`
- `./run_singularity.sh bash`

Once you are inside a `bash` shell in the Singularity container, you can run e.g. 
- `python demo.py PATH/TO/VIDEO/FILE.mp4`

