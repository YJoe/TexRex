#include <cmath>
#include "..\include\Neuron.h"
#include "..\include\Connection.h"

Neuron::Neuron(char debug_tag, double weight) {
	set_debug_tag(debug_tag);
	set_output_value(weight);
	delta_weight = 0;
	eta = 0.15;
	alpha = 0.5;
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

double Neuron::sigmoid(double val) {
	return 1 / (1 + std::exp(-val));
}

double Neuron::sigmoid_prime(double val) {
	double sig = sigmoid(val);
	return sig * (1 - sig);
}

void Neuron::calculate_activation_value() {
	hidden_sum = 0;
	for (auto &input_connection : input_connections) {
		hidden_sum += input_connection->get_start_neuron()->get_output_value() * input_connection->get_weight();
	}
	//std::cout << "hidden_sum " << hidden_sum << std::endl;
	set_output_value(sigmoid(hidden_sum));
}

void Neuron::backwards_propagate(double change) {

	double output_weight_total = 0;
	for (auto &output_connection : output_connections) {
		output_weight_total += output_connection->get_weight();
	}

	//std::cout << "change [" << change << "] output weight total [" << output_weight_total << "] hidden sum [" << hidden_sum << "] sigmoid of hidden sum [" << sigmoid_prime(hidden_sum) << "]" << std::endl;
	delta_hidden_sum = (change / output_weight_total) * sigmoid_prime(hidden_sum);

	//std::cout << delta_hidden_sum << std::endl;

	// set weights of the output connections to this neuron
	for (auto &output_connection : output_connections) {
		output_connection->set_weight(output_connection->get_weight() + (change / get_output_value()));
	}

}

void Neuron::calculate_output_weights() {
	for (auto &output_connection : output_connections) {
		output_connection->set_weight(output_connection->get_weight() + (output_connection->get_end_neuron()->get_delta_hidden_sum() *
			get_output_value()));
	}
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
	//    std::cout << "\tsum dow [" << sum << "]" << std::endl;

	return sum;
}

void Neuron::neuron_feed_forward() {
	double sum = 0.0;

	for (int i = 0; i < input_connections.size(); i++) {
		//        std::cout << "\tadding connection" << std::endl;
		//        std::cout << "\t\t" << input_connections[i]->get_start_neuron()->get_output_value()  << " * " << input_connections[i]->get_weight() <<  std::endl;
		sum += input_connections[i]->get_start_neuron()->get_output_value()
			* input_connections[i]->get_weight();
	}

	//    std::cout<<"SUM " << sum<<std::endl;

	output_value = activation_function(sum);
}

void Neuron::calculate_output_gradients(double target_value) {
	double delta = target_value - output_value;
	gradient = delta * activation_function_derivative(output_value);
}

void Neuron::calculate_hidden_gradients() {
	double dow = sum_dow();
	gradient = dow * activation_function_derivative(output_value);
	//    std::cout << "\t\tgradient [" << gradient << "]" << std::endl;
}

double Neuron::get_gradient() {
	return gradient;
}

void Neuron::update_input_weights() {
	for (int i = 0; i < input_connections.size(); i++) {
		double old_delta_weight = input_connections[i]->get_delta_weight();

		//        std::cout << "\t\t\told delta weight [" << old_delta_weight << "]" << std::endl;
		double new_delta_weight = eta * input_connections[i]->get_start_neuron()->get_output_value()
			* gradient + alpha * old_delta_weight;

		//        std::cout << "\t\t\tnew delta weight [" << new_delta_weight << "]" << std::endl;

		input_connections[i]->set_delta_weight(new_delta_weight);
		input_connections[i]->set_weight(input_connections[i]->get_weight() + new_delta_weight);

		//        std::cout << "\t\t\tnew weight [" << input_connections[i]->get_weight() << "]" << std::endl;
	}
}

double Neuron::get_delta_weight() {
	return delta_weight;
}
