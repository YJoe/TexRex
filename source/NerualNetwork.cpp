#include <cassert>
#include <cmath>
#include "..\include\NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(vector<int> *topology) {

	recent_average_smoothing_factor = 100.0;
	recent_average_error = 0;

	// work out how many neurons we want
	int total_neurons = 0;
	for (int i = 0; i < (*topology).size(); i++) {
		total_neurons += (*topology)[i];
		if (i < (*topology).size() - 1) {
			total_neurons += 1;
		}
	}

	// create the neurons
	for (int i = 0; i < total_neurons; i++) {
		Neuron n = Neuron('?', 0);
		neurons.emplace_back(n);
	}

	int current_neuron = 0;
	// assign all neurons to their layer
	for (int i = 0; i < (*topology).size(); i++) {
		layers.emplace_back();
		for (int j = 0; j < (*topology)[i]; j++) {
			layers.back().emplace_back(&neurons[current_neuron]);
			current_neuron++;
		}
		if (i < (*topology).size() - 1) {
			neurons[current_neuron].set_debug_tag('B');
			neurons[current_neuron].set_output_value(1.0);
			layers.back().emplace_back(&neurons[current_neuron]);
			current_neuron++;
		}
	}

	// TODO: correct this count, it is creating a few too many connections
	// create a list of connections
	for (int i = 0; i < layers.size() - 1; i++) {
		for (int j = 0; j < layers[i].size() * layers[i + 1].size() - (i + 1 != layers.size() - 1 ? 1 : 0); j++) {
			connections.emplace_back(Connection(rand_zero_point()));
		}
	}

	// define the map, neurons have a list of input connections and output connections, connections have a start neuron and an end neuron
	int c = 0;
	for (int i = 0; i < layers.size() - 1; i++) {
		cout << "\nlayer [" << i << "] to [" << i + 1 << "]" << endl;
		for (int j = 0; j < layers[i].size(); j++) {
			cout << "\tneuron j [" << j << "]" << endl;
			for (int k = 0; k < layers[i + 1].size(); k++) {
				cout << "\t\tneuron k [" << k << "]" << endl;
				cout << "\t\t\tplacing connection c [" << c << "]" << endl;

				// if the future neuron we are looking at is not the last one in that layer || the future layer is the output layer
				if (k != layers[i + 1].size() - 1 || i + 1 == layers.size() - 1) {
					connections[c].set_start_neuron(layers[i][j]);
					connections[c].set_end_neuron(layers[i + 1][k]);
					layers[i][j]->add_output_connection(&connections[c]);
					layers[i + 1][k]->add_input_connection(&connections[c]);
					c++;
				}
				else {
					//                    cout << "\t\t\t\tno connection! the k neuron is a bias neuron!" << endl;
				}
			}
		}
	}

	for (int i = 0; i < layers.size(); i++) {
		//        cout << "layer" << endl;
		for (int j = 0; j < layers[i].size(); j++) {
			//            cout << "\tneuron" << endl;
			for (int k = 0; k < layers[i][j]->output_connections.size(); k++) {
				layers[i][j]->output_connections[k]->set_weight(0.4);
				//                cout << "\t\tconnection [" << layers[i][j]->output_connections[k]->get_weight() << "]" << endl;
			}
		}

	}
}


double NeuralNetwork::rand_zero_point() {
	return static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
}

void NeuralNetwork::net_feed_forward(vector<double> *input_values) {

	if ((*input_values).size() != layers[0].size() - 1) {
		cout << "the training inputs do not match the input layer size" << endl;
		exit(-1);
	}

	// assign all of the input values to the neurons in the input layer
	for (int i = 0; i < (*input_values).size(); i++) {
		layers[0][i]->set_output_value((*input_values)[i]);
		//        cout << "set weight " << (*input_values)[i] << endl;
	}

	// for everything other than the input layer
	for (int i = 1; i < layers.size(); i++) {
		//        cout << "forward propagating layer [" << i << "]" << endl;

		// for all neurons inside that layer
		for (int j = 0; j < layers[i].size(); j++) {

			// if we aren't on the last layer
			if (i < layers.size() - 1) {

				// if we aren't on the last neuron of this layer
				if (j < layers[i].size() - 1) {
					layers[i][j]->neuron_feed_forward();
				}
			}
			else {
				layers[i][j]->neuron_feed_forward();
			}
		}
	}
}

void NeuralNetwork::backwards_propagation(vector<double> *target_values) {
	//    cout << "\nstarting back prop" << endl;

	// calculate error of network
	error = 0.0;

	for (int i = 0; i < layers.back().size(); i++) {
		double delta = (*target_values)[i] - layers.back()[i]->get_output_value();
		error += delta * delta;
	}
	error /= layers.back().size();
	error = sqrt(error);

	// measure recent error
	recent_average_error =
		(recent_average_error * recent_average_smoothing_factor + error)
		/ (recent_average_smoothing_factor + 1.0);

	// calculate output layer gradients
	for (int i = 0; i < layers.back().size(); i++) {
		layers.back()[i]->calculate_output_gradients((*target_values)[i]);
	}

	// calculate hidden layer gradients
	for (int i = layers.size() - 2; i > 0; i--) {
		for (int j = 0; j < layers[i].size(); j++) {
			layers[i][j]->calculate_hidden_gradients();
		}
	}

	// update connection weights
	//    cout << "updating input weights" << endl;
	for (int i = layers.size() - 1; i > 0; i--) {
		//        cout << "\tupdating layer [" << i << "]" << endl;
		for (int j = 0; j < layers[i].size(); j++) {
			if (i == layers.size() - 1) {
				//                cout << "\t\tUpdating input weight of neuron" << endl;
				layers[i][j]->update_input_weights();
			}
			else {
				if (j < layers[i].size() - 1) {
					//                    cout << "\t\tUpdating input weight of neuron" << endl;
					layers[i][j]->update_input_weights();
				}
			}
		}
	}
}

void NeuralNetwork::print_results() {

	cout << "[";
	for (int i = 0; i < layers.back().size(); i++) {
		cout << layers.back()[i]->get_output_value();
		if (i < layers.back().size() - 1) {
			cout << ", ";
		}
	}
	cout << "]" << endl;
}

double NeuralNetwork::get_recent_average_error() {
	return recent_average_error;
}