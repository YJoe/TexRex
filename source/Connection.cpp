#include "..\include\Connection.h"
#include <iostream>

Connection::Connection(double random_start_point) {
	set_weight(random_start_point);
}

double Connection::get_weight() {
	return weight;
}

void Connection::set_weight(double weight) {
	this->weight = weight;
}

void Connection::set_start_neuron(Neuron *start_neuron) {
	this->start_neuron = start_neuron;
}

void Connection::set_end_neuron(Neuron *end_neuron) {
	this->end_neuron = end_neuron;
}

Neuron *Connection::get_start_neuron() {
	return start_neuron;
}

Neuron *Connection::get_end_neuron() {
	return end_neuron;
}

void Connection::set_delta_weight(double delta_weight) {
	this->delta_weight = delta_weight;
}

double Connection::get_delta_weight() {
	return delta_weight;
}
