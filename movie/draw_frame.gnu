#!/usr/bin/gnuplot

# waterfall display logic
tcurr  = (1.0*i/framerate) # current time 
tstart = floor(tcurr/duration)*duration # start window time
tstop  = tcurr

set xrange [tstart:tstart+duration]

set multiplot layout 3,1

# set time bar
pos = tstart+0.85*duration
set arrow 1 from first pos, graph -0.1  to first pos+1, graph -0.1 nohead lw 4 lc -1
set label 1 at first pos+0.5, graph -0.1 "1s" offset 0,-0.7 center


# plot spike rasters
set ylabel 'Output'
set yrange [0:n_output]
plot x lc rgb "white",\
	 sprintf('< aube -f %f -t %f -i %s/output.0.spk',tstart,tstop,datadir) w d lc -1

unset label
unset arrow

set ylabel 'Hidden'
set yrange [0:n_hidden]
plot x lc rgb "white",\
	sprintf('< aube -f %f -t %f -i %s/multi_hidden.0.spk',tstart,tstop,datadir) w d lc -1

set ylabel 'Input'
set yrange [0:n_input]

set label 2 at graph 0.0, graph -0.07 sprintf("t_0=%.2fs",tstart) font 'Helvetica,10' 

plot x lc rgb "white",\
	sprintf('< aube -f %f -t %f -i %s/multi_stim.0.spk',tstart,tstop,datadir) w d lc -1

unset label 2

unset multiplot
