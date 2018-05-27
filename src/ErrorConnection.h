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

#ifndef ERRORCONNECTION_H_
#define ERRORCONNECTION_H_

#include "auryn/auryn_definitions.h"
#include "auryn/Connection.h"
#include "auryn/AurynVector.h"
#include "auryn/SparseConnection.h"
#include "auryn/System.h"
#include "auryn/ComplexMatrix.h"
#include "StateWatcherGroup.h"

#include <sstream>
#include <fstream>
#include <stdio.h>
#include <algorithm>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/exponential_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/lognormal_distribution.hpp>

namespace auryn {

typedef ComplexMatrix<AurynWeight> ForwardMatrix;


/*! \brief Acts like a sparseconnection, but instead of transmitting spikes
 * it transmits analog state values from a source state vector to a target 
 * state vector. 
 *
 * As opposed to SparseStateConnectino, this class is parallel
 * save, but should only be used for sparse states which are rarely different
 * from zero. Its main purpose is to transmit downstream error signals back to
 * hidden units in superspike paradigms.
 */

class ErrorConnection : public SparseConnection
{
private:

protected:
	
public:
	StateWatcherGroup * state_watcher;

	/*! \brief Default constructor which sets up a random sparse matrix with fixed weight between the source and destination group. 
	 *
	 * The constructor takes the weight and sparseness as secondary arguments. The latter allows Auryn to 
	 * allocate the approximately right amount of memory inadvance. It is good habit to specify at time of initialization also 
	 * a connection name and the transmitter type. Both can be set separately with set_transmitter and set_name if the function call gets
	 * too long and ugly. A connection name is often handy during debugging and the transmitter type is a crucial for obvious resons ...  */
	ErrorConnection(SpikingGroup * source, NeuronGroup * destination, 
			AurynWeight weight, AurynFloat sparseness=0.05, 
			TransmitterType transmitter=GLUT, string name="ErrorConnection");

	virtual ~ErrorConnection();

	/*! Sets the state name to connect between pre and post */
	void connect_states(string pre_name, string post_name);
	
	/*! \brief Internally used propagate method
	 *
	 * This method propagates spikes in the main simulation loop. Should usually not be called directly by the user.*/
	virtual void propagate();

};

} // namespace 

#endif /*ERRORCONNECTION_H_*/
