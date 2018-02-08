#ifndef TEXREX_NEURALNETWORK2_H
#define TEXREX_NEURALNETWORK2_H

#include "Neuron.h"
#include "Connection.h"
#include <vector>

using namespace std;

class NeuralNetwork {
public:
	explicit NeuralNetwork(vector<int>* topology, float learning_rate);
	void net_feed_forward(vector<float> *input_values);
	void backwards_propagation(vector<float>* target_values);
	void get_results(vector<float>& output_values);
	void print_results();
	float get_recent_average_error();
	void log(string message);
	std::vector<std::vector<Neuron*>> layers;
	float learning_rate;
private:
	float error;
	float recent_average_error;
	float recent_average_smoothing_factor;
	std::vector<Connection> connections;
	float rand_zero_point();
	std::vector<Neuron> neurons;
};


#endif //TEXREX_NEURALNETWORK2_H
