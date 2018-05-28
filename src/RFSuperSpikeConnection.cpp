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

#include "RFSuperSpikeConnection.h"

using namespace auryn;

std::set< RFSuperSpikeConnection* > RFSuperSpikeConnection::m_instances = std::set< RFSuperSpikeConnection* >();

AurynFloat RFSuperSpikeConnection::beta = 1.0/1e-3; //!< steepness/temperature of nonlinearity

void RFSuperSpikeConnection::init(AurynFloat eta, AurynFloat feedback_delay, AurynFloat maxweight, AurynState tau_syn, AurynState tau_mem)
{
	if ( !dst->evolve_locally() ) return;
	logger->debug("RFSuperSpikeConnection init");

	eta_ = eta;
	auryn::logger->parameter("eta",eta);

	epsilon = 1e-32; //!< for division by zero cases

	approximate = true; //!< when enabled small error signals are not back-propagated 
	delta = 1e-5;    //!< quasi zero for presynaptic PSP
	gamma = 1e-7;    //!< quasi zero norm for error signal ( this value should be set to float precision )


	use_error_dependent_het_term = false;

	tau_syn_ = tau_syn;
	tau_mem_ = tau_mem;

	tau_el_rise = tau_syn_;
	tau_el_decay = tau_mem_;

	tau_vrd_rise = tau_el_rise;
	tau_vrd_decay = tau_el_decay;


	partial_enabled = true;
	augment_gradient = true;
	use_layer_lr = false; //!< set to false for parameter-wise learning rate

	maxact = 25.0;
	regstrength = 1.0;
	regexponent = 4.0;

	set_min_weight(0.0);
	set_max_weight(maxweight);

	synapse_sign = 1.0;
	plasticity_sign = 1.0;

	back_propagate_error_signals = false;

	trg = NULL; 
	err = dst->get_state_vector("err");
	err_in = dst->get_state_vector("err_in");
	target_error_vector = src->get_state_vector("err_in");

	// traces for van Rossum Distance
	tr_err = dst->get_post_state_trace(err, tau_vrd_rise);
	tr_err_flt = dst->get_post_state_trace(tr_err, tau_vrd_decay);

	tau_avg_err = 10.0;
	// eta_avg_err = auryn_timestep/tau_avg_err;
	mul_avgsqrerr = std::exp(-auryn_timestep/tau_avg_err);
	avgsqrerr = dst->get_state_vector("avgsqrerr");
	temp = dst->get_state_vector("_temp");

	// normalization factor for avgsqrerr
	const double a = tau_vrd_decay;
	const double b = tau_vrd_rise;
	scale_tr_err_flt = 1.0/(std::pow((a*b)/(a-b),2)*(a/2+b/2-2*(a*b)/(a+b)))/tau_avg_err;
    // std::cout << scale_tr_err_flt << std::endl;

	// pre trace 
	tr_pre     = src->get_pre_trace(tau_syn_); // new EulerTrace( src->get_pre_size(), tau_syn );
	tr_pre_psp = new EulerTrace( src->get_pre_size(), tau_mem_ );
	tr_pre_psp->set_target(tr_pre);

	tr_post_hom = dst->get_post_trace(100e-3);
	hom4 = dst->get_state_vector("_hom4");

	stdp_active = true; //!< Only blocks weight updates if disabled
	plasticity_stack_enabled = true;

	// sets timecourse for update dynamics
	// timestep_rmsprop_updates = 5000 + 257;
	timestep_rmsprop_updates = 5000;
	set_tau_rms(30.0);
	logger->parameter("timestep_rmsprop_updates", (int)timestep_rmsprop_updates);

	// compute delay size in AurynTime
	delay_size = feedback_delay/auryn_timestep;
	logger->parameter("delay_size", delay_size);

	partial_delay = new AurynDelayVector( dst->get_post_size(), delay_size+MINDELAY );
	pre_psp_delay = new AurynDelayVector( src->get_pre_size(), delay_size+MINDELAY ); 

	logger->debug("RFSuperSpikeConnection complex matrix init");
	// Set number of synaptic states
	w->set_num_synapse_states(5);

	zid_weight = 0;
	w_val   = w->get_state_vector(zid_weight);

	zid_el = 1;
	el_val = w->get_state_vector(zid_el);

	zid_el_flt = 2;
	el_val_flt = w->get_state_vector(zid_el_flt);

	zid_sum = 3;
	el_sum  = w->get_state_vector(zid_sum);

	zid_grad2 = 4;
	w_grad2 = w->get_state_vector(zid_grad2);
	// w_grad2->set_all(1.0);

	// Run finalize again to rebuild backward matrix
	logger->debug("RFSuperSpikeConnection complex matrix finalize");
	DuplexConnection::finalize();

	// store instance in static set
	m_instances.insert(this);
}


void RFSuperSpikeConnection::free()
{
	logger->debug("RFSuperSpikeConnection free");
	// store instance in static set
	m_instances.erase(this);

	delete partial_delay;
	delete pre_psp_delay;
}

RFSuperSpikeConnection::RFSuperSpikeConnection(
		SpikingGroup * source, 
		NeuronGroup * destination, 
		AurynWeight weight, 
		AurynFloat sparseness, 
		AurynFloat eta, 
		AurynFloat delay, 
		AurynFloat maxweight, 
		TransmitterType transmitter,
		std::string name) 
: DuplexConnection(source, 
		destination, 
		weight, 
		sparseness, 
		transmitter, 
		name)
{
	init(eta, delay, maxweight);
	if ( name.empty() )
		set_name("RFSuperSpikeConnection");
}

RFSuperSpikeConnection::~RFSuperSpikeConnection()
{
	if ( dst->get_post_size() > 0 ) 
		free();
}

AurynWeight RFSuperSpikeConnection::instantaneous_partial(NeuronID loc_post)
{
	if ( !partial_enabled ) return 1.0;

	// compute pseudo partial
	// const AurynFloat h = (dst->mem->get(loc_post)+50e-3)*beta;
	const AurynState voltage = dst->mem->get(loc_post);
	if ( voltage < -80e-3 ) return 0.0;
	const AurynFloat h = (voltage+50e-3)*beta;
	const AurynFloat outer = 1.0/pow((1.0+std::abs(h)),2);
	const AurynFloat part = outer*beta;
	return part;
}

void RFSuperSpikeConnection::add_to_syntrace(const AurynLong didx, const AurynDouble input)
{
	// computes plasticity update
	AurynWeight * elt = el_val->ptr(didx); // w->get_data_ptr(didx, zid_el);
	*elt += input; 
}


void RFSuperSpikeConnection::add_to_err(NeuronID spk, AurynState val) 
{
	if ( dst->localrank(spk) ) {
		const NeuronID trspk = dst->global2rank(spk);
		err->add_specific(trspk, val);
	}
}


void RFSuperSpikeConnection::compute_van_rossum_err()
{
	// reset
	err->set_zero();

	// Compute error signal
	SpikeContainer * sc = dst->get_spikes();
	for ( unsigned int i = 0 ; i < sc->size(); ++i ) add_to_err(sc->at(i), -1.0);
	sc = trg->get_spikes();
	for ( unsigned int i = 0 ; i < sc->size(); ++i ) add_to_err(sc->at(i), 1.0);
}

/* \brief Computes error signal */
void RFSuperSpikeConnection::compute_err()
{
	compute_van_rossum_err();
}

/*! \brief Computes local part of plasticity rule without multiplying the error signal
 *
 * Computes: epsilon*( sigma_prime(u_i) PSP_j )
 * */
void RFSuperSpikeConnection::process_plasticity()
{
	// compute partial deriviatives and store in delay
	for ( NeuronID i = 0 ; i < dst->get_post_size() ; ++i )
		partial_delay->set( i, instantaneous_partial(i) );

	// compute psp and store in delay
	for ( NeuronID j = 0 ; j < src->get_pre_size() ; ++j )
		pre_psp_delay->set( j, tr_pre_psp->get(j) );

	// loop over all pre neurons
	for (NeuronID j = 0; j < src->get_pre_size() ; ++j ) {
		const AurynState psp = pre_psp_delay->mem_get(j);
		if ( approximate && psp < delta ) { continue; } 
		// std::cout << std::scientific << psp << std::endl;
		
		// loop over all postsynaptic partners
	    for (const NeuronID * c = w->get_row_begin(j) ; 
					   c != w->get_row_end(j) ; 
					   ++c ) { // c = post index
			// compute data index for address in complex array
			const NeuronID li = dst->global2rank(*c);
			const AurynState sigma_prime = partial_delay->mem_get(li); 
			const AurynLong didx   = w->ind_ptr_to_didx(c); 

			// compute eligibility trace
			const AurynWeight syn_trace_input = plasticity_sign*psp*sigma_prime; // TODO delay inputs 
			add_to_syntrace( didx, syn_trace_input );
		}
	}

	partial_delay->advance(); // now 'get' points to the delayed version
	pre_psp_delay->advance(); // now 'get' points to the delayed version

	// # SECOND compute outer convolution of synaptic traces
	// a bit of a CPUHOG
	const AurynFloat mul_follow = auryn_timestep/tau_el_decay;
	el_val_flt->follow(el_val, mul_follow);

	const AurynFloat scale_const = std::exp(-auryn_timestep/tau_el_rise);
	el_val->scale(scale_const);

	// # THIRD compute correlation between el_val_flt and the filtered error signal
	// and store in el_sum 'the summed eligibilty trace'

	// // loop over all pre 
	// for (NeuronID j = 0; j < src->get_pre_size() ; ++j ) {
	//     // loop over all post
	//     for (const NeuronID * c = w->get_row_begin(j) ; 
	//      			   c != w->get_row_end(j) ; 
	//      			   ++c ) { // c = post index
	// 		const NeuronID i = *c;
	// 		const AurynLong didx = w->ind_ptr_to_didx(c); 
	// 		AurynWeight de = tr_err_flt->get(i) * el_val_flt->get(didx); // - std::abs(tr_err_flt->get(i))*1e-2*w_val->get(didx)*el_val_flt->get(didx);
	// 		// de = std::pow(de,3);
	// 		// regularization / homeostasis
	// 		// if ( tr_post_hom->normalized_get(i) > 30.0 ) de = -1*el_val_flt->get(didx);
	// 		el_sum->add_specific(didx, de);
	//     }
	// }


	// precompute fourth power before we go in the loop
	hom4->copy(tr_post_hom);
	hom4->pow(regexponent);

	// CPUHOG
	if ( use_error_dependent_het_term ) {
		for (NeuronID li = 0; li < dst->get_post_size() ; ++li ) {
			if ( approximate && std::abs(tr_err_flt->get(li)) <= gamma ) { continue; }
			const NeuronID gi = dst->rank2global(li); 
			for (const NeuronID * c = bkw->get_row_begin(gi) ; 
					c != bkw->get_row_end(gi) ; 
					++c ) {
				const AurynWeight * weight = bkw->get_data(c); 
				const AurynLong didx = w->data_ptr_to_didx(weight); 
				const AurynState e = tr_err_flt->get(li);
				// AurynWeight de = el_val_flt->get(didx) * ( tr_err_flt->get(li) ) - regstrength * *weight * std::pow(tr_post_hom->get(li),4);  
				const AurynWeight de = ( el_val_flt->get(didx) - regstrength * *weight * hom4->get(li) * e ) * e; 
				el_sum->add_specific(didx, de);
			}
		}
	} else { 
		for (NeuronID li = 0; li < dst->get_post_size() ; ++li ) {
			if ( approximate && std::abs(tr_err_flt->get(li)) < gamma ) { continue; }
			const NeuronID gi = dst->rank2global(li); 
			for (const NeuronID * c = bkw->get_row_begin(gi) ; 
					c != bkw->get_row_end(gi) ; 
					++c ) {
				const AurynWeight * weight = bkw->get_data(c); 
				const AurynLong didx = w->data_ptr_to_didx(weight); 
				const AurynWeight de = el_val_flt->get(didx) * ( tr_err_flt->get(li) ) - regstrength * *weight * hom4->get(li);  // BEST option
				el_sum->add_specific(didx, de);
			}
		}
	}
}

void RFSuperSpikeConnection::propagate_forward()
{
   // loop over all spikes
   for (SpikeContainer::const_iterator spike = src->get_spikes()->begin() ; // spike = pre_spike
				   spike != src->get_spikes()->end() ; ++spike ) {
	   // loop over all postsynaptic partners
	   for (const NeuronID * c = w->get_row_begin(*spike) ; 
					   c != w->get_row_end(*spike) ; 
					   ++c ) { // c = post index

			   // transmit signal to target at postsynaptic neuron
			   AurynWeight * weight = w->get_data_ptr(c); 
			   transmit( *c , *weight );
	   }
   }
}


template <typename T> int sgn(T val) {
	    return (T(0) < val) - (val < T(0));
}


void RFSuperSpikeConnection::propagate()
{
	// propagate spikes
	propagate_forward();
}

void RFSuperSpikeConnection::evolve()
{
	if ( !plasticity_stack_enabled || !dst->evolve_locally()) return;

	// compute squared error vector of this layer
	temp->copy(tr_err_flt);
	// temp->mul(scale_tr_err_flt);
	temp->sqr();
	temp->scale(auryn_timestep);
	avgsqrerr->scale(mul_avgsqrerr);
	avgsqrerr->add(temp);

	err->copy(err_in);
	err_in->set_zero();
	if ( trg != NULL ) compute_err();

	// add nonlinear Hebb (pre post correlations) to synaptic traces and filter these traces
	process_plasticity();

	if ( auryn::sys->get_clock()%timestep_rmsprop_updates == 0  ) {

		double gm = 0.0;
		if ( augment_gradient ) {
			// evolve complex synapse parameters
			for ( AurynLong k = 0; k < w->get_nonzero(); ++k ) {
				// AurynWeight * weight = w->get_data_ptr(k, zid_weight);
				AurynWeight * minibatch = w->get_data_ptr(k, zid_sum);
				AurynWeight * g2 = w->get_data_ptr(k, zid_grad2);

				// copies our low-pass value "minibatch" to grad and erases it
				const AurynFloat grad = (*minibatch)/timestep_rmsprop_updates;
				// *minibatch = 0.0f;

				// update moving averages 
				*g2 = std::max( grad*grad, rms_mul* *g2 );

				// To implement RMSprop we  need this line
				// *g2 = rms_mul* *g2 + (1.0-rms_mul)*std::pow(grad,2) ;
			}

			if ( use_layer_lr ) {
				gm =w->get_synaptic_state_vector(zid_grad2)->max();
			}
		}


		for ( AurynLong k = 0; k < w->get_nonzero(); ++k ) {
			AurynWeight * weight = w->get_data_ptr(k, zid_weight);
			AurynWeight * minibatch = w->get_data_ptr(k, zid_sum);

			// copies our low-pass value "minibatch" to grad and erases it
			const AurynDouble grad = (*minibatch)/timestep_rmsprop_updates;
			*minibatch = 0.0f;

			// carry out weight updates
			if ( stdp_active ) {

				// dynamic gradient rescaling
				// (per parameter learning rate)
				if ( !use_layer_lr ) {
					gm = w->get_synaptic_state_vector(zid_grad2)->get(k);
				}

				double rms_scale = 1.0;
				if ( augment_gradient ) rms_scale = 1.0/(std::sqrt(gm)+epsilon);
		
				// update weight
				*weight += rms_scale * grad * eta_ ;

				// clip weight
				if ( *weight < get_min_weight() ) *weight = get_min_weight();
				else if ( *weight > get_max_weight() ) *weight = get_max_weight();
			}
		}
	}

	// update follow traces which are not registered with the kernel
	tr_pre_psp->follow();
}

void RFSuperSpikeConnection::set_tau_rms( AurynState tau )
{
	tau_rms = tau;
	rms_mul = std::exp(-auryn_timestep*timestep_rmsprop_updates/tau_rms);
}

void RFSuperSpikeConnection::set_tau_syn( AurynState tau )
{
	tau_syn_ = tau;
	tr_pre->set_timeconstant(tau_syn_); 
}

void RFSuperSpikeConnection::set_tau_mem( AurynState tau )
{
	tau_mem_ = tau;
	tr_pre_psp->set_timeconstant(tau_mem_); 
}

void RFSuperSpikeConnection::set_target_group( SpikingGroup * group )
{
	trg = group;
	if ( dst->get_size() != trg->get_size() ) {
		logger->warning("Destination SpikingGroup and the group providing the target"
				" spike train have to have the same neuron numbers.");
		return;
	}
}

double RFSuperSpikeConnection::get_mean_square_error( )
{
	return scale_tr_err_flt*avgsqrerr->mean()/(1.0-std::exp(-sys->get_time()/tau_avg_err)+1e-9);
}

void RFSuperSpikeConnection::set_learning_rate( double eta )
{
	eta_   = eta;
}



