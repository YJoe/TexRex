#ifndef TEXREX_NEURON_H
#define TEXREX_NEURON_H

#include <vector>
#include <iostream>

class Connection;

class Neuron {

public:
	Neuron(char debug_tag, double weight, float learning_rate);
	void set_output_value(double weight);
	void set_debug_tag(char debug_tag);
	void add_input_connection(Connection* connection);
	void add_output_connection(Connection* connection);
	void print_debug();
	double get_delta_hidden_sum();
	char get_debug_tag();
	void neuron_feed_forward();
	double activation_function(double total);
	double activation_function_derivative(double total);
	double get_output_value();
	void calculate_output_gradients(double target_value);
	void calculate_hidden_gradients();
	double get_gradient();
	void update_input_weights();
	double get_delta_weight();
	float get_learning_rate();
	std::vector<Connection*> input_connections;
	std::vector<Connection*> output_connections;

private:
	double weight;
	double hidden_sum;
	double delta_hidden_sum;
	char debug_tag;
	double eta;
	double alpha;
	double output_value;
	double gradient;
	double delta_weight;

	double sum_dow();
};


#endif //TEXREX_NEURON_H
