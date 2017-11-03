#ifndef TEXREX_NEURALNETWORK2_H
#define TEXREX_NEURALNETWORK2_H

#include "Neuron.h"
#include "Connection.h"
#include <vector>

using namespace std;

class NeuralNetwork {
public:
	explicit NeuralNetwork(vector<int>* topology);
	void net_feed_forward(vector<double> *input_values);
	void backwards_propagation(vector<double>* target_values);
	void print_results();
	double get_recent_average_error();
private:
	double error;
	double recent_average_error;
	double recent_average_smoothing_factor;
	std::vector<Connection> connections;
	double rand_zero_point();
	std::vector<std::vector<Neuron*>> layers;
	std::vector<Neuron> neurons;
};


#endif //TEXREX_NEURALNETWORK2_H
