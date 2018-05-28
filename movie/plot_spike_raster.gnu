#!/usr/bin/gnuplot

unset border
unset xtics
unset ytics
unset key

set term png size 800,600 font 'Helvetica,12'

theme = "oxford"
outputdir = sprintf("/tmp/%s", theme)
datadir = sprintf("../output/symfb/%s/eta1e-3", theme)
print datadir

gridsize = 1.29
duration = 6*gridsize

n_output = 200  
n_hidden = 255 # show only a few ?
n_input  = n_output  

framerate = 10.0

# gridsize = duration

# loop over frames
do for [i=0:600] {
	print(sprintf("Plotting frame %i",i))
	set out sprintf('%s/frame%06i.png',outputdir,i)
	load 'draw_frame.gnu'
}


f0 = 70*duration*framerate
f1 = f0+30*framerate
print(f0)
print(f1)

do for [i=f0:f1] {
	print(sprintf("Plotting frame %i",i))
	set out sprintf('%s/frame%06i.png',outputdir,i)
	load 'draw_frame.gnu'
}

