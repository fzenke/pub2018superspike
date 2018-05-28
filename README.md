# pub2018superspike

This repository contains example code to accompany our paper 

Zenke, F., and Ganguli, S. (2018). SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks. Neural Computation 30, 1514–1541.

https://www.mitpressjournals.org/doi/abs/10.1162/neco_a_01086


## Intro

SuperSpike is an approximate surrogate gradient method. The algorithm runs fully online and constitutes a three-factor rule for training recurrent and multi-layer spiking neural networks to learn spatiotemporal input output mappings. The present repository introduces the key classes which allow SuperSpike to run using the Auryn library and illustrates how to train a simple three layer network using either random or symmetric error feedback. The code was refactored from the original classes used in the publication and should be considered alpha stage. Please report bugs on GitHub and use the forum (https://fzenke.net/auryn/forum/index.php) for questions and discussions.   


## Requirements

You will need a working compiled version of the Auryn library (https://github.com/fzenke/auryn).
We have tested this with commit 6f88977da186e874c75ce3cdbba59d83748fe53c.


## Quick start

* Check out the repository and cd into the ```sim/``` directory.
* Edit the Makefile and make sure the variable ```AURYNDIR``` points to your Auryn root directory.
* Run ```make```
* Run the script ```run_symfb.sh``` which will launch a simulation with one hidden layer with symmetric feedback (```run_rfb.sh``` will launch the same sim with random feedback).


### Inputs

This example simulation uses the Oxford themed target spiketrain from the ```themes``` directory. Input and targe spike trains are provided in the humand readable ```ras``` format (https://fzenke.net/auryn/doku.php?id=manual:ras). 


### Outputs 
The scripts will generate their output in the ```.../output/``` directory. Specifically, the input spikes are written to ```multi_stim.0.spk```, the hidden layer spikes are written to ```multi_hidden.0.spk```  and out the output layer spikes are saved to ```output.0.spk```. The ```stats``` files contains a learning curve over training time.

### Visualizing network activity 

To plot the spikes you will have to decode the binary output format using ```aube``` which comes with the Auryn library (see https://fzenke.net/auryn/doku.php?id=manual:aube for details).

Here is a visualization of the network activity during training in example simulation

![SuperSpike animation](https://raw.githubusercontent.com/fzenke/pub2018superspike/master/movie/oxford-opt.gif "SuperSpike animation")
