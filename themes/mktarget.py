#!/usr/bin/python3
from numpy  import *
import matplotlib.pyplot as plt
import scipy.misc


dt = 10e-3
filename = "oxford"

imgc = scipy.misc.imread("%s.png"%filename)
img  = imgc.mean(2)
jitter = random.rand(*img.shape)*dt

n = img.shape[0]

with open("%s-target.ras"%filename,'w') as f:
    for j in range(img.shape[1]):
        for i in range(img.shape[0]):
            if img[i,j]==0.0:
                f.write("%f %i\n"%(dt*j+jitter[i,j],n-i))

with open("%s-info.txt"%filename,'w') as f:
    f.write("# neurons %i, temporal resolution %fs, time grid %fs\n"%(n, dt, dt*img.shape[1]))
    f.write("filename=%s\n"%(filename))

with open("%s-conf.env"%filename,'w') as f:
    f.write("export HEIGHT=%i\n"%(n))
    f.write("export GRID=%f\n"%(dt*img.shape[1]))
