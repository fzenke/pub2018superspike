#!/bin/sh

THEME="oxford"
TMPDIR="/tmp/$THEME"

# Create output dir if not existing
mkdir -p $TMPDIR

# Make sure dir is empty
rm -f $TMPDIR/*.png 

# Generate frames
gnuplot plot_spike_raster.gnu

# make animated gif
convert -delay 1 -loop 0 $TMPDIR/*.png ${THEME}.gif

# make mpg
# convert -delay 10 -quality 95 $TMPDIR/*.png ${THEME}.mpg

# make mov (smallest filesize so far)
mencoder mf://$TMPDIR/*.png -mf w=640:h=480:fps=30:type=png -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -nosound -o ${THEME}.mov

LAST=`ls -tr -1 $TMPDIR | tail -n 1`
cp $TMPDIR/$LAST $THEME.png

# clean up 
rm -f $TMPDIR/*.png 
rmdir $TMPDIR
