#ifndef TEXREX_CONNECTION_H
#define TEXREX_CONNECTION_H

class Neuron;

class Connection {
public:
	explicit Connection(double random_start_point);
	double get_weight();
	Neuron* get_start_neuron();
	Neuron* get_end_neuron();
	void set_weight(double weight);
	void set_start_neuron(Neuron* start_neuron);
	void set_end_neuron(Neuron* end_neuron);
	double get_delta_weight();
	void set_delta_weight(double);

private:
	double weight;
	Neuron* start_neuron;
	Neuron* end_neuron;
	double delta_weight = 0.0;
};


#endif //TEXREX_CONNECTION_H
