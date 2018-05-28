# pub2018superspike

This repository contains example code to accompany our paper 

Zenke, F., and Ganguli, S. (2018). SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks. Neural Computation 30, 1514–1541.

URL: https://www.mitpressjournals.org/doi/abs/10.1162/neco_a_01086

![SuperSpike animation](https://raw.githubusercontent.com/fzenke/pub2018superspike/master/movie/oxford-opt.gif "SuperSpike animation")



Bibtex:
```
@article{zenke_superspike:_2018,
	title = {{SuperSpike}: {Supervised} {Learning} in {Multilayer} {Spiking} {Neural} {Networks}},
	volume = {30},
	issn = {0899-7667},
	shorttitle = {{SuperSpike}},
	url = {https://doi.org/10.1162/neco_a_01086},
	doi = {10.1162/neco_a_01086},
	number = {6},
	journal = {Neural Computation},
	author = {Zenke, Friedemann and Ganguli, Surya},
	month = apr,
	year = {2018},
	pages = {1514--1541},
}
```


## Intro

SuperSpike is an approximate surrogate gradient method. The algorithm runs fully online and constitutes a three-factor rule capabable training recurrent and multi-layer spiking neural networks to perform complex spatiotemporal input output mappings. The present repository introduces the key classes extending the Auryn library to run SuperSpike and illustrates how to train a simple three layer network using either random or symmetric error feedback. 

The present code was freshly optimized and should be considered alpha stage. Please report bugs on GitHub and use the forum (https://fzenke.net/auryn/forum/index.php) for questions and discussions.   


## Requirements

You will need a working compiled version of the Auryn library (https://github.com/fzenke/auryn).
We have tested this with commit 6f88977da186e874c75ce3cdbba59d83748fe53c.


## Quick start

* Check out the repository and change into the ```sim/``` directory.
* Edit the Makefile and make sure the variable ```AURYNDIR``` points to your Auryn root directory.
* Run ```make```
* Run the script ```run_symfb.sh``` which will launch a simulation with one hidden layer and symmetric error feedback (```run_rfb.sh``` will launch the same sim with random feedback).


### Inputs

This example simulation uses the Oxford themed (a picture of the Radcliffe Camera) target spiketrain from the ```themes``` directory. Input and targe spike trains are provided in the humand readable ```ras``` format (https://fzenke.net/auryn/doku.php?id=manual:ras). 


### Outputs 
The above run scripts will generate their output in the ```.../output/``` directory. Specifically, the input spikes are written to ```multi_stim.0.spk```, the hidden layer spikes are written to ```multi_hidden.0.spk```  and the output layer spikes are saved to ```output.0.spk```. The ```stats``` file contains a learning curve over training time.


### Visualizing network activity 

To plot the spikes you will have to decode the binary output format using ```aube``` which comes with the Auryn library (see https://fzenke.net/auryn/doku.php?id=manual:aube for details). The animation shown above gives one example of how such network activity evolves over the course of training.



License & Copyright 
-------------------

Copyright 2018 Friedemann Zenke

The SuperSpike extension for Auryn is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The SuperSpike extension for Auryn is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the SuperSpike extension for Auryn.  
If not, see <http://www.gnu.org/licenses/>.
