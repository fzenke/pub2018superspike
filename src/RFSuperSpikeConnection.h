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
*/

#ifndef RFSUPERSPIKECONNECTION_H_
#define RFSUPERSPIKECONNECTION_H_

#include "auryn/auryn_definitions.h"
#include "auryn/DuplexConnection.h"
#include "auryn/Trace.h"
#include "auryn/EulerTrace.h"
#include "auryn/LinearTrace.h"
#include "auryn/SpikeDelay.h"
#include "auryn/AurynDelayVector.h"

#include <set>


namespace auryn {


/*! \brief Double STDP All-to-All Connection
 *
 * This is a connection on which weight updates are triggered by the presence of a feedback signal.
 */
class RFSuperSpikeConnection : public DuplexConnection
{

private:

	AurynTime timestep_rmsprop_updates;
	AurynTime timestep_eligibility_trace;
	AurynFloat rms_mul; 
	AurynFloat tau_rms;

	SpikingGroup * trg;

	AurynFloat tau_avg_err;
	// AurynFloat eta_avg_err;
	AurynDouble mul_avgsqrerr;

	AurynTime delay_size;

	/*! \brief synaptic exp time constant 
	 *
	 * should mirror the actual synaptic dynamics */
	AurynFloat tau_syn_;

	/*! \brief membrane exp time constant 
	 *
	 * should mirror the actual neuronal dynamics */
	AurynFloat tau_mem_;

	void init(AurynFloat eta, AurynFloat feedback_delay, AurynFloat maxweight, AurynState tau_syn=5e-3, AurynState tau_mem=10e-3);
	void add_to_syntrace(const AurynLong didx, const AurynDouble input);

	void add_to_err(NeuronID spk, AurynState val);

protected:
	/*! \brief Timeconstant for activity signal smoothing */
	AurynFloat smoothing_tau;

	Trace * tr_pre;
	Trace * tr_post;
	Trace * tr_post_flt;

	Trace * tr_mem;
	Trace * tr_post_hom;
	AurynStateVector * hom4;

	void propagate_forward();

	void compute_err();
	void compute_van_rossum_err();
	void process_plasticity();


public:
	AurynDouble eta_;

	AurynDouble epsilon;
	int zid_weight, zid_batch, zid_el, zid_el_flt, zid_sum, zid_grad2;
	AurynSynStateVector *w_val, *el_val, *el_val_flt, *el_sum, *w_grad2;

	static std::set< RFSuperSpikeConnection* > m_instances;

	AurynStateVector * err_in;
	AurynStateVector * err;
	AurynStateVector * avgsqrerr;
	AurynStateVector * target_error_vector;
	AurynStateVector * temp;

	Trace * tr_err;
	Trace * tr_err_flt;

	/*! \brief Normalization constant or error. */
	AurynDouble scale_tr_err_flt;

	Trace * tr_pre_psp;
	AurynDelayVector * partial_delay;
	AurynDelayVector * pre_psp_delay;
	
	bool partial_enabled;
	bool augment_gradient;
	bool use_layer_lr;
	bool use_error_dependent_het_term;

	/*!\brief When set to true this connection will backpropagate error signals */
	bool back_propagate_error_signals;

	static AurynFloat beta; //!< Sharpness of soft nonlinearity
	AurynFloat gamma; //!< Effective zero for computation of error signal
	AurynFloat delta; //!< Effective zero for computation of psp signal

	AurynState maxact; //!< Maximum activity above which a loss is added
	AurynState regstrength; //!< Regularization strength
	AurynState regexponent; //!< Regularization exponent

	bool approximate;

	AurynWeight instantaneous_partial(NeuronID post);
	AurynWeight partial(NeuronID post);

	float plasticity_sign;
	/*! \brief Has to be set to represent the sign of the synape plus for excitatory and minus for inhib. 
	 *
	 * Is not set automatically by the Neurotransmitter yet!! TODO
	 * */
	int synapse_sign;

	/*! \brief Time constants of synaptic eligibility trace filter. */
	AurynFloat tau_el_rise;
	AurynFloat tau_el_decay;


	/*! \brief Time constants of van Rossum distance filter. */
	AurynFloat tau_vrd_rise;
	AurynFloat tau_vrd_decay;

	/*! \brief Sets timescale of RMax average */
	void set_tau_rms( AurynState tau );

	/*! \brief Sets synaptic timescale (should be matched to neuron model) */
	void set_tau_syn( AurynState tau );

	/*! \brief Sets membrane timescale (should be matched to neuron model) */
	void set_tau_mem( AurynState tau );

	/*! \brief Sets group with target spike train */
	void set_target_group( SpikingGroup * group );

	/*! \brief Sets max learning rate */
	void set_learning_rate( double eta );

	/*! \brief Returns moving average of mean square error */
	double get_mean_square_error( );

	bool stdp_active; //!< Blocks plastic updates if set false
	bool plasticity_stack_enabled; //!< Blocks all plasticity operations when set false

	RFSuperSpikeConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			TransmitterType transmitter=GLUT);

	RFSuperSpikeConnection(
			SpikingGroup * source, 
			NeuronGroup * destination, 
			AurynWeight weight, 
			AurynFloat sparseness=0.05, 
			AurynFloat eta=1e-3, 
			AurynFloat delay=0.0, 
			AurynFloat maxweight=1. , 
			TransmitterType transmitter=GLUT,
			string name = "RFSuperSpikeConnection" );


	virtual ~RFSuperSpikeConnection();
	void free();

	virtual void propagate();
	virtual void evolve();

};

}

#endif /*RFSuperSpikeConnection_H_*/
