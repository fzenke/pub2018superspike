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

#include "auryn.h"
#include "SuperSpikeConnection.h"
#include <iostream>
#include <fstream>

#define NEURONMODEL IafPscExpGroup
#define PLASTICCONNECTION SuperSpikeConnection

using namespace auryn;

namespace po = boost::program_options;
namespace mpi = boost::mpi;

AurynWeight wmax = 0.1;
float hidden_sparseness  = 1.0;

bool enable_hidden_learning = true;
bool record_spikes = true;
bool record_states = true;
bool record_synapses = true;
double epsilon = 1e-12;
bool partial_enabled = true;

string connection_stats_string(SparseConnection * con)
{
	std::stringstream oss;
	oss << std::setprecision(5) << std::scientific; 
	for ( unsigned int i = 0 ; i < con->w->get_num_synapse_states(); ++i ) {
		AurynDouble mean, std;
		con->stats(mean, std, i);
		oss << mean << "  ";
	}
	oss << "err " << ((PLASTICCONNECTION*)con)->tr_err_flt->mean();
	return oss.str();
}

void print_connection_stats(SparseConnection * con)
{
	std::cout << "Stats of " << con->get_name()
		<< ": " << connection_stats_string(con);
	std::cout << std::endl;
}

class HiddenLayer
{
	private:
		string basename;

		void set_name(Connection * con, string suffix) 
		{
			std::stringstream s;
			s << basename 
				<< " connection: " << con->src->get_name() 
				<< " -> "
				<< con->dst->get_name()
				<< " " << suffix;
			con->set_name(s.str());
			logger->parameter("connection name", s.str());
		}


		PLASTICCONNECTION * get_layer_connection(
				SpikingGroup * src, 
				NeuronGroup * dst, 
				double eta=1.0e-3, 
				AurynFloat delay=0.0, 
				AurynWeight w0=0.0, 
				AurynFloat sparseness=1.0
				)
		{
			PLASTICCONNECTION * con = new PLASTICCONNECTION(
					src,
					dst,
					w0,
					sparseness,
					eta,
					delay
					);
			con->set_min_weight(-wmax);
			con->set_max_weight(wmax);
			con->random_data(0.0,w0);
			// con->sparse_set_data(0.05, 1e-3);
			con->plasticity_sign = 1.0;
			con->set_name("con_hid");
			con->partial_enabled = partial_enabled;
			con->regstrength = 0.0;
			con->epsilon = epsilon;
			con->use_layer_lr = false;
			con->augment_gradient = true;
			return con;
		}


	public:
		NEURONMODEL * neurons;

		PLASTICCONNECTION * con;
		/*! \brief The default constructor */
		HiddenLayer( 
				SpikingGroup * input, 
				NeuronID size, 
				string name = "HiddenLayer",
				AurynFloat eta=1e-3, 
				AurynFloat delay=0.0, 
				AurynWeight w0=0.1, 
				AurynFloat sparseness=1.0
				) {
			basename = name;
			logger->info("Initializing hidden layer neurons groups");

			neurons = new NEURONMODEL(size);
			neurons->set_tau_mem(10e-3);
			// neurons->delta_u = 1e-3;

			logger->info("Initializing hidden layer input connections");
			con = get_layer_connection( input, neurons, eta, delay, w0, sparseness );

		};


		void print_stats() 
		{
			print_connection_stats(con);
		};


		void set_stdp_active(bool value) 
		{
			con->stdp_active = value;
		};


		virtual ~HiddenLayer() {
		};
};


int main(int ac, char* av[]) 
{

	int errcode = 0;
	string outputdir = "out";

	double simtime = 1000.0;
	unsigned int simblocks = 10;
	unsigned int nin = 1000;
	unsigned int nout = 1;
	unsigned int nhidden = 10;
	double eta = 2e-4;
	double w0 = 1.0;
	unsigned int num_hidden_layers = 1;
	float delay_per_layer = 1e-3; 
	float loop_grid = 10.0;
	float bgrate = 0.0;

	string input_spikes =  "input.ras";
	string target_spikes = "target.ras";


	// Read command line params
    try {

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help", "produce help message")
            ("simtime", po::value<double>(), "simulation time")
            ("blocks", po::value<unsigned int>(), "simulation blocks")
            ("eta", po::value<double>(), "learning rate")
            ("w0", po::value<double>(), "initial weight")
            ("sparseness", po::value<double>(), "hidden layer sparseness")
            ("dir", po::value<string>(), "load/save directory")
            ("input", po::value<string>(), "pat file with stimuli input")
            ("target", po::value<string>(), "txt file with target")
            ("nin", po::value<unsigned int>(), "input dimensionality")
            ("nout", po::value<unsigned int>(), "output dimensionality")
            ("layers", po::value<unsigned int>(), "number of hidden layers -- currently either 0 or 1")
            ("nhidden", po::value<unsigned int>(), "hidden layer size (no of exc cells)")
            ("delay", po::value<float>(), "hidden layer propagation delay")
            ("grid", po::value<float>(), "loop grid size in seconds")
            ("bgrate", po::value<float>(), "background sporious spike rate")
            ("epsilon", po::value<float>(), "denom epsilon")
            ("nopartial", "deactivate partial derivative in hidden layers")
            ("fixedhidden", "deactivate learning in hidden layer")
            ("fast", "switches off most IO for performance")
        ;

        po::variables_map vm;        
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        if (vm.count("simtime")) {
			simtime = vm["simtime"].as<double>();
        } 

        if (vm.count("blocks")) {
			simblocks = vm["blocks"].as<unsigned int>();
        } 

        if (vm.count("eta")) {
			eta = vm["eta"].as<double>();
        } 

        if (vm.count("w0")) {
			w0 = vm["w0"].as<double>();
        } 

        if (vm.count("sparseness")) {
			hidden_sparseness = vm["sparseness"].as<double>();
        } 

        if (vm.count("dir")) {
			outputdir = vm["dir"].as<string>();
        } 

        if (vm.count("input")) {
			input_spikes = vm["input"].as<string>();
        } 

        if (vm.count("target")) {
			target_spikes = vm["target"].as<string>();
        } 

        if (vm.count("nin")) {
			nin = vm["nin"].as<unsigned int>();
        } 

        if (vm.count("nout")) {
			nout = vm["nout"].as<unsigned int>();
        } 

        if (vm.count("layers")) {
			num_hidden_layers = vm["layers"].as<unsigned int>();
        } 

        if (vm.count("nhidden")) {
			nhidden = vm["nhidden"].as<unsigned int>();
        } 

        if (vm.count("delay")) {
			delay_per_layer = vm["delay"].as<float>();
        } 

        if (vm.count("grid")) {
			loop_grid = vm["grid"].as<float>();
        } 

        if (vm.count("bgrate")) {
			bgrate = vm["bgrate"].as<float>();
        } 

        if (vm.count("epsilon")) {
			epsilon = vm["epsilon"].as<float>();
        } 

        if (vm.count("nopartial")) {
			partial_enabled = false;
        } 

        if (vm.count("fixedhidden")) {
			enable_hidden_learning = false;
        } 

        if (vm.count("fast")) {
			record_states = false;
			// record_spikes = false;
			record_synapses = false;
        } 
    }
    catch(std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    catch(...) {
        std::cerr << "Exception of unknown type!\n";
    }

	// BEGIN Global definitions
	auryn_init(ac, av);
	sys->set_simulation_name("default");
	sys->set_output_dir(outputdir);
	sys->set_master_seed(121);
	// END Global definitions
	

	logger->info("Setting up input");
	int input_size = nin;
	int output_size = nout;

	FileInputGroup * fin  = new FileInputGroup( input_size, input_spikes, true );
	fin->set_loop_grid(loop_grid);
	fin->set_name("input");
	if ( bgrate > 0.0 ) new PoissonSpikeInjector( fin, bgrate );

	FileInputGroup * ftrg = new FileInputGroup( output_size, target_spikes, true );
	ftrg->set_loop_grid(loop_grid);
	ftrg->set_name("target");


	logger->info("Setting up output");
	NEURONMODEL * neurons_out = new NEURONMODEL(nout);
	neurons_out->set_tau_mem(10e-3);
	sys->set_online_rate_monitor_id(neurons_out->get_uid());
	sys->set_online_rate_monitor_tau(1.0);


	if (num_hidden_layers > 1) {
		logger->error("This alpha version only supports up to one hidden layer.");
		auryn_free();
		errcode = 1;
		return errcode;
	}


	logger->msg("Connecting hidden layers ...", PROGRESS, true);
	double eta_hidden = eta;
	std::vector<HiddenLayer*> hidden_layers;
	SpikingGroup * upstream = fin;
	HiddenLayer * hidden_layer;
	for ( unsigned int i = 0 ; i < num_hidden_layers; ++i ) {
		std::stringstream layer_name ;
		layer_name << "h" << i+1;

		const float delay = delay_per_layer*(num_hidden_layers-i);
		std::cout << "Setting up layer " << i << " with delay "
			<< delay << "s" << std::endl;
		hidden_layer = new HiddenLayer( upstream, 
				nhidden, 
				layer_name.str(), 
				eta_hidden, 
				delay,
				w0/(sqrt(upstream->get_size())), 
				hidden_sparseness );
		hidden_layers.push_back(hidden_layer);
		upstream = hidden_layer->neurons;
	}

	logger->msg("Setting up output layer...", PROGRESS, true);
	double sparseness = 1.0;
	double w_out_init = w0/std::sqrt(upstream->get_size());
	PLASTICCONNECTION * con_out = new PLASTICCONNECTION( upstream, neurons_out, w_out_init, sparseness, eta);
	con_out->set_name("con_out");
	con_out->set_max_weight(wmax);
	con_out->set_min_weight(-wmax);
	con_out->random_data(0.0,w_out_init);
	con_out->set_target_group(ftrg);
	con_out->regstrength = 0.0;
	con_out->epsilon = epsilon;
	con_out->beta = 1.0/1.0e-3;
	con_out->use_layer_lr = false;
	con_out->augment_gradient = true;
	con_out->partial_enabled = partial_enabled;



	// If we were to use random feedback we would have to connect this manually.
	//
	// logger->msg("Setting up random feedback...", PROGRESS, true);
	// for ( NeuronID l = 0 ; l < num_hidden_layers ; ++l ) {
	// 	NeuronGroup  * etrg = hidden_layers[l]->neurons;
	// 	// SparseStateConnection * scon = new SparseStateConnection(neurons_out, etrg, 1.0, 1.0);
	// 	// scon->connect_state("err");
	// 	ErrorConnection * scon = new ErrorConnection(neurons_out, etrg, 1.0, 1.0);
	// 	scon->random_data(0.0,1.0);
	// 	logger->msg("Adding feedback connection", PROGRESS, true);
	// }
	
	// However, we are using symmetric feedback
	if ( enable_hidden_learning ) {
		logger->msg("Setting up symmetric feedback...", PROGRESS, true);
		for ( NeuronID l = 1 ; l < num_hidden_layers ; ++l ) {
				HiddenLayer * hl = hidden_layers[l];
				hl->con->back_propagate_error_signals = true;
		}
		con_out->back_propagate_error_signals = true;
	}



	logger->msg("Setting up monitors...", PROGRESS, true);
	if ( num_hidden_layers )
		sys->set_online_rate_monitor_id(hidden_layers[0]->neurons->get_uid());
	double test_time = 1.0;

	if ( hidden_layers.size() ) {
		new PopulationRateMonitor(hidden_layers[0]->neurons,sys->fn("hidden","prate"));
	}

	std::vector< VoltageMonitor* > voltagemons;
	std::vector< StateMonitor* > statemons;

	if ( record_states ) {
		for ( unsigned int i = 0; i < std::min(nout, (unsigned int)2); ++i ) {
			StateMonitor * errmon = new StateMonitor( con_out->tr_err_flt, i, sys->fn("multi",i,"err"));
			errmon->record_for(test_time);
			statemons.push_back(errmon);
			StateMonitor * parmon = new StateMonitor( con_out->partial_delay, i, sys->fn("multi",i,"par"),1e-4);
			parmon->record_for(test_time);
			statemons.push_back(parmon);
			StateMonitor * foomon = new StateMonitor( neurons_out, i, "syn_current", sys->fn("multi",i,"syn"));
			foomon->record_for(test_time);
			statemons.push_back(foomon);
			VoltageMonitor * vmon = new VoltageMonitor( neurons_out, i, sys->fn("multi",i,"mem"));
			vmon->record_for(test_time);
			voltagemons.push_back(vmon);
		}

		if ( hidden_layers.size() ) {

			for ( NeuronID i = 0; i < 8; ++i ) {
				StateMonitor * errmon = new StateMonitor( hidden_layers[0]->con->tr_err_flt, i, sys->fn("multi_hidden",i,"err"));
				errmon->record_for(test_time);
				statemons.push_back(errmon);
				StateMonitor * parmon = new StateMonitor( hidden_layers[0]->con->partial_delay, i, sys->fn("multi_hidden",i,"par"));
				parmon->record_for(test_time);
				statemons.push_back(parmon);
				VoltageMonitor * vmon = new VoltageMonitor( hidden_layers[0]->neurons, i, sys->fn("multi_hidden",i,"mem"));
				vmon->record_for(test_time);
				voltagemons.push_back(vmon);
			}

		}
	}


	std::vector< BinarySpikeMonitor * > spikemons;
	if ( record_spikes ) {
		BinarySpikeMonitor * spikemon_s = new BinarySpikeMonitor(
				fin,
				sys->fn("multi_stim","spk")
				);
		spikemons.push_back(spikemon_s);

		if ( hidden_layers.size() ) {
			BinarySpikeMonitor * spikemon_h = new BinarySpikeMonitor(
					hidden_layer->neurons,
					sys->fn("multi_hidden","spk")
					);
			spikemons.push_back(spikemon_h);
		}

		BinarySpikeMonitor * spikemon_e = new BinarySpikeMonitor(
				neurons_out,
				sys->fn("output","spk")
				);
		spikemons.push_back(spikemon_e);
	}

	if ( record_synapses ) {
		for ( NeuronID i = 0 ; i < hidden_layers.size() ; ++i ) {
			WeightMonitor * wmon = new WeightMonitor( hidden_layers[i]->con, sys->fn("hidden_con",i,"syn"), 0.1);
			wmon->add_equally_spaced(20);
		}
		WeightMonitor * wmon = new WeightMonitor( con_out, sys->fn("con","syn"), 0.1);
		wmon->add_equally_spaced(20);
	}

	// sys->load_network_state("out/multi");
	std::ofstream stats_file;
	stats_file.open(sys->fn("stats").c_str());

	if ( !enable_hidden_learning ) {
		// disable plasticity in hidden layers
		for ( NeuronID i = 0 ; i < num_hidden_layers ; ++i ) hidden_layers[i]->con->plasticity_stack_enabled = false;
	}

	logger->msg("Burn-in...", PROGRESS, true);
	for ( NeuronID i = 0 ; i < num_hidden_layers ; ++i ) hidden_layers[i]->set_stdp_active(false);
	con_out->stdp_active = false;
	sys->run(simtime); // burn-in time
	sys->flush_devices();
	for ( NeuronID i = 0 ; i < num_hidden_layers ; ++i ) hidden_layers[i]->set_stdp_active(true);
	con_out->stdp_active = true;

	// simblocks = 0;
	for ( unsigned int blk = 0 ; blk < simblocks ; ++blk ) {
		// Training recording off
		if ( blk && blk%20==0 ) {
			for ( NeuronID i = 0 ; i < num_hidden_layers ; ++i ) hidden_layers[i]->con->eta_ = eta_hidden/10;
			con_out->eta_ = eta/10;
		}

		for ( unsigned int i = 0 ; i < hidden_layers.size() ; ++i ) {
			hidden_layers[i]->print_stats();
		}
		
		print_connection_stats(con_out);
		stats_file << blk << " " 
			<< con_out->get_mean_square_error() << " "
			<< connection_stats_string(con_out) << " "
			<< std::endl;

		std::cout << "Block " << blk << ", mean square error " << std::scientific 
			<< con_out->get_mean_square_error() 
			<< std::endl;

		// Testing and recording

		for ( NeuronID i = 0; i < statemons.size(); ++i ) {
			statemons.at(i)->record_for(test_time);
		}
		for ( NeuronID i = 0; i < voltagemons.size(); ++i ) {
			voltagemons.at(i)->record_for(test_time);
		}

		sys->run(test_time);
		sys->flush_devices();


		// Train
		sys->run(simtime-test_time);
	}

	logger->msg("Running one grid marker with plasticity disabled...", PROGRESS, true);
	con_out->plasticity_stack_enabled = false;
	for ( NeuronID i = 0 ; i < num_hidden_layers ; ++i ) hidden_layers[i]->con->plasticity_stack_enabled = false;
	double tmp = loop_grid*(std::ceil(sys->get_time()/loop_grid)+1.0)-sys->get_time(); // make sure to finish one complete loop
	sys->run(tmp);

	logger->msg("Saving ...",PROGRESS,true);
	sys->save_network_state("multi");

	if (errcode) auryn_abort(errcode);
	auryn_free();

	stats_file.close();

	return errcode;
}
