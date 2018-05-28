#!/bin/bash

AURYNROOT="$HOME/auryn/build/release/"
THEME="oxford"

source ${THEME}-conf.env
echo GRID=$GRID
echo HEIGHT=$HEIGHT

$AURYNROOT/examples/sim_poisson --size $HEIGHT --simtime $GRID
mv poisson.0.ras ${THEME}-input.ras

# Clean up
rm -f poisson.0.prate sim_poisson.0.log
