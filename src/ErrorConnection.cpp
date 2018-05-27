/* 
* Copyright 2014-2018 Friedemann Zenke
*
* This file is part of Auryn, a simulation package for plastic
* spiking neural networks.
* 
* Auryn is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* Auryn is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#include "ErrorConnection.h"

using namespace auryn;

ErrorConnection::ErrorConnection( 
		SpikingGroup * source, 
		NeuronGroup * destination, 
		AurynWeight weight, 
		AurynFloat sparseness, 
		TransmitterType transmitter, 
		std::string name) 
	: SparseConnection( 
		source, 
		destination, 
		weight, 
		sparseness, 
		transmitter, 
		name) 
{

	logger->verbose("Initialzing ErrorConnection");
	state_watcher = new StateWatcherGroup(src, "err");
	connect_states("err", "err_in");

	set_min_weight(-1e42); // whatever
	set_max_weight(1e42);
}

ErrorConnection::~ErrorConnection()
{
}

void ErrorConnection::connect_states(string pre_name, string post_name)
{
	state_watcher->watch(src, pre_name);
	set_target(post_name);
}

void ErrorConnection::propagate()
{
	if ( dst->evolve_locally() ) { // necessary 
		NeuronID * ind = w->get_row_begin(0); // first element of index array
		AurynWeight * data = w->get_data_begin(); // first element of data array

		// loop over spikes
		for (NeuronID i = 0 ; i < state_watcher->get_spikes()->size() ; ++i ) {
			// get spike at pos i in SpikeContainer
			const NeuronID spike = state_watcher->get_spikes()->at(i);

			// extract spike attribute from attribute stack;
			const NeuronID stackpos = i + (spike_attribute_offset)*src->get_spikes()->size();
			const AurynFloat attribute = state_watcher->get_attributes()->at(stackpos);

			// std::cout << spike << " " << attribute <<  std::endl;
			// return;

			// loop over postsynaptic targets
			for (NeuronID * c = w->get_row_begin(spike) ; 
					c != w->get_row_end(spike) ; 
					++c ) {
				AurynWeight value = data[c-ind] * attribute; 
				transmit( *c , value );
				// transmit( w->get_colind(c) , w->get_value(c) );
			}
		}
	}
}
