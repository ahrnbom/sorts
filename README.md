# SORTS and SORTS+RReID

This repository contains the source code for the paper *Real-Time and Online Segmentation Multi-Target Tracking with Track Revival Re-Identification* which was published at VISAPP 2021. The paper is available [here](https://www.scitepress.org/Link.aspx?doi=10.5220/0010190907770784).

SORTS and SORTS+RReID are based on [SORT](https://github.com/abewley/sort), copied at 2020-05-27.

## Requirements
- A modern Linux PC with an NVidia GPU with appropriate drivers. Tested with CUDA 11.4. 
- [Singularity](https://sylabs.io/docs/)

## Instructions
- `git clone https://github.com/ahrnbom/sorts`
- `cd sorts`
- `git submodule update --init --recursive`
- `./run_singularity.sh build`
- `./run_singularity.sh bash`
- Download and extract [this zip file](https://lunduniversityo365-my.sharepoint.com/:u:/g/personal/ma7467ah_lu_se/EVwOThgpFZpCqW9SBRjP1CYB89TseqgvLL-tdc5SJJOeIA?e=tXv6I9) and place the contents inside the `sorts` folder

Once you are inside a `bash` shell in the Singularity container, you can run e.g. 
- `./setup.sh` (this only needs to be done once)
- `python demo.py PATH/TO/VIDEO/FILE.mp4`
