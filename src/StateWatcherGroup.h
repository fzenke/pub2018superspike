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

#ifndef STATEWATCHERGROUP_H_
#define STATEWATCHERGROUP_H_

#include "auryn/auryn_definitions.h"
#include "auryn/Connection.h"
#include "auryn/AurynVector.h"
#include "auryn/SparseConnection.h"
#include "auryn/System.h"
#include "auryn/ComplexMatrix.h"

namespace auryn {

/*! \brief StateWatcherGroup is a SpikingGroup used by ErrorConnection
 *
 *
 * This SpikingGroup detects nonzero states in the source state vector and uses Auryn spike propagation
 * mechanisms to make these values known across nodes. The group is used interally by ErrorConnection for
 * sync purposes.
 */

class StateWatcherGroup : public SpikingGroup
{
	private:
		SpikingGroup * group_to_watch_;
		AurynStateVector * state_to_watch_;

	public:
		StateWatcherGroup(SpikingGroup * group, string state_name) : SpikingGroup( group->get_size(), ROUNDROBIN ) 
		{
			sys->register_spiking_group(this); // TODO make sure this groups is evolved locally on all ranks

			watch(group, state_name);

			inc_num_spike_attributes(1);
			set_name("State watcher");
			
			// logger->warning("REMOVE ME CODE MARKER");
		}

		void watch(SpikingGroup * group, string state_name) {
			group_to_watch_ = group;
			state_to_watch_ = group_to_watch_->get_state_vector(state_name);
		}

		virtual void evolve() 
		{
			// iterate over state vector and generate an event (a "spike") when ne zero
			for (NeuronID li = 0; li < group_to_watch_->get_rank_size() ; ++li ) {
				const AurynState v = state_to_watch_->get(li);
				if ( v != 0.0 ) {
					// std::cout << " foo " << std::endl;
					spikes->push_back(group_to_watch_->rank2global(li));
					attribs->push_back(v);
					// state_to_watch_->set(li,0.0);
				}
			}
		}
};


} // namespace 

#endif /*STATEWATCHERGROUP_H_*/
