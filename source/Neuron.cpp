#include <cmath>
#include "..\include\Neuron.h"
#include "..\include\Connection.h"

using namespace std;

Neuron::Neuron(char debug_tag, double weight) {
	set_debug_tag(debug_tag);
	set_output_value(weight);
	delta_weight = 1;
	eta = 0.5;
	alpha = 0.1;
}

double Neuron::get_output_value() {
	return output_value;
}

void Neuron::set_output_value(double output_value) {
	this->output_value = output_value;
}

void Neuron::set_debug_tag(char debug_tag) {
	this->debug_tag = debug_tag;
}

void Neuron::add_input_connection(Connection* connection) {
	input_connections.emplace_back(connection);
}

void Neuron::add_output_connection(Connection* connection) {
	output_connections.emplace_back(connection);
}

char Neuron::get_debug_tag() {
	return debug_tag;
}

void Neuron::print_debug() {

	//    std::cout << input_connections.size() << std::endl;

	//    std::cout << "\ndebugging neuron [" << this << "] with tag [" << get_debug_tag() << "]" << std::endl;
	//    std::cout << "\tweight [" << get_output_value() << "]" << std::endl;
	//    std::cout << "\tinputs" << std::endl;
	//    for (auto &input_connection : input_connections) {
	//        std::cout << "\t\t" << input_connection << " [" << input_connection->get_output_value() << "]" << std::endl;
	//    }
	//    std::cout << "\toutputs" << std::endl;
	//    for (auto &output_connection : output_connections) {
	//        std::cout << "\t\t" << output_connection << " [" << output_connection->get_output_value() << "]" << std::endl;
	//    }
}

double Neuron::get_delta_hidden_sum() {
	return delta_hidden_sum;
}

double Neuron::activation_function(double x) {
	return std::tanh(x);
}

double Neuron::activation_function_derivative(double x) {
	return static_cast<double>(1.0 - x * x);
}

double Neuron::sum_dow() {
	double sum = 0.0;

	for (int i = 0; i < output_connections.size(); i++) {
		sum += output_connections[i]->get_weight() * output_connections[i]->get_end_neuron()->get_gradient();
	}

	return sum;
}

void Neuron::neuron_feed_forward() {
	double sum = 0.0;

	for (int i = 0; i < input_connections.size(); i++) {
		sum += input_connections[i]->get_start_neuron()->get_output_value()
			* input_connections[i]->get_weight();
	}

	output_value = activation_function(sum);
}

void Neuron::calculate_output_gradients(double target_value) {
	gradient = (target_value - output_value) * activation_function_derivative(output_value);
}

void Neuron::calculate_hidden_gradients() {
	gradient = (sum_dow()) * activation_function_derivative(output_value);
}

double Neuron::get_gradient() {
	return gradient;
}

void Neuron::update_input_weights() {
	for (int i = 0; i < input_connections.size(); i++) {
		double old_delta_weight = input_connections[i]->get_delta_weight();

		double new_delta_weight = eta * input_connections[i]->get_start_neuron()->get_output_value()
			* gradient + alpha * old_delta_weight;

		input_connections[i]->set_delta_weight(new_delta_weight);
		input_connections[i]->set_weight(input_connections[i]->get_weight() + new_delta_weight);
	}
}

double Neuron::get_delta_weight() {
	return delta_weight;
}
