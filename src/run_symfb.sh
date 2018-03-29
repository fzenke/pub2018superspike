#!/bin/bash

# Create temporary dir
TMPDIR=`mktemp -d`

EPSILON=1e-12
NHID=256
RUN="oxford"

# import GRID and HEIGHT from conf file
source ../themes/${RUN}-conf.env
echo GRID=$GRID
echo HEIGHT=$HEIGHT

# for ETA in 1e-3 5e-4 1e-4 5e-3 ;
for ETA in 5e-4 ;
do
	OUTPUTDIR=../output/$RUN/eta$ETA
	mkdir -p $OUTPUTDIR
	make && ./sim_symfb --nin 200 --nout $HEIGHT --grid $GRID --eta $ETA --input ../themes/${RUN}-input.ras --target ../themes/${RUN}-target.ras --simtime 20 --nhid $NHID --epsilon $EPSILON --delay 0.0e-3 --block 50 --w0 0.05 --layer 1 --dir $TMPDIR
	cp $TMPDIR/*.spk $OUTPUTDIR
done

# Clean up
rm -r $TMPDIR
