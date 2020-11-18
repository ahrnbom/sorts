#!/bin/bash

SING_FILE=sing.sif
PARAM=$1
SHOULD_BUILD=false

if [ "$1" == "build" ]; then
    SHOULD_BUILD=true
    PARAM=$2
else
    if [ -f "$SING_FILE" ]; then
        echo "Singularity image $SING_FILE already exist"
        SHOULD_BUILD=false
    else
        SHOULD_BUILD=true
    fi
fi

if [ "$SHOULD_BUILD" == true ]; then
    sudo -E singularity build "$SING_FILE" sing.def
fi

singularity run --nv -B /run -B /media "$SING_FILE" $PARAM

echo "Done with Singularity!"
