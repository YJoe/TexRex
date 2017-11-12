#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "..\include\OCLFunctions.h"
#include "..\include\AnalysisJob.h"
#include "..\include\NeuralNetwork.h"

void print_vector_neatly(vector<double> &vec) {
	cout << "[";
	for (int i = 0; i < vec.size(); i++) {
		cout << vec[i];
		if (i < vec.size() - 1) {
			cout << ", ";
		}
	}
	cout << "]";
}

void print_vector2_neatly(vector<vector<double>> &vec) {
	for (int i = 0; i < vec.size(); i++) {
		print_vector_neatly(vec[i]);
		cout << "\n";
	}
}

int random_int(int min, int max) {
	return rand() % (max - min + 1) + min;
}

void a() {
	AnalysisJob a = AnalysisJob("data/some_text.jpg");
	a.run();
}

void arrayCL() {
	OCLFunctions cl_funct = OCLFunctions(CL_DEVICE_TYPE_GPU);

	vector<vector<double>> a = {
		{  1, -1, -1, -1, -1},
		{ -1,  1, -1, -1, -1 },
		{ -1, -1,  1, -1, -1 },
		{ -1, -1, -1,  1, -1 },
		{ -1, -1, -1, -1,  1 }
	};
	vector<vector<double>> b = {
		{  1, -1, -1 },
		{ -1,  1, -1 },
		{ -1, -1,  1 }
	};

	vector<vector<double>> conv_map;

	cl_funct.apply_filter(a, b, conv_map);

	print_vector2_neatly(conv_map);

	cin.get();
}

void network() {
	//srand(unsigned(time(nullptr)));
	srand(0);
	
	vector<int> topology = { 2, 2, 1 };
	int training_index = 0;
	double has_learnt_threshold = 0.01;
	bool log_basic_stuff = false;
	NeuralNetwork network = NeuralNetwork(&topology);

	vector<vector<double>> inputs = {};

	vector<vector<double>> targets = {};

	for (int i = 0; i < 10; i++) {
		double a = (double)random_int(0, 50) / 100;
		double b = (double)random_int(0, 50) / 100;
		vector<double> args = { a, b };
		vector<double> targ = { a + b };
		inputs.emplace_back(args);
		targets.emplace_back(targ);
	}

	srand(0);
	for (int i = 0; i < 100000; i++) {
		if (log_basic_stuff) {
			cout << "\nTraining pass [" << i << "]" << endl;
		}

		// setting the training set
		training_index = random_int(0, (int)inputs.size() - 10);

		if (log_basic_stuff) {
			//printing the training set
			print_vector_neatly(inputs[training_index]);
			cout << " = ";
			print_vector_neatly(targets[training_index]);
			cout << endl;
		}

		//feeding forward
		network.net_feed_forward(&inputs[training_index]);

		if (log_basic_stuff) {
			//print the results
			cout << "network results ";
			network.print_results();
		}

		// back propagate to correct the network
		network.backwards_propagation(&targets[training_index]);

		if (log_basic_stuff) {
			// how're we doin'?
			cout << "net recent average error [" << network.get_recent_average_error() << "] [";
			if (network.get_recent_average_error() < has_learnt_threshold) {
				cout << "yayy]" << endl;
			}
			else {
				cout << "nope]" << endl;
			}
		}
		//cin.get();
	}

	cout << "ERROR [" << network.get_recent_average_error() << "]" << endl;

	cout << "\n-------------------------------------------------------" << endl;

	// see how well the network can mirror the training sets
	for (int i = 0; i < inputs.size() - 10; i++) {
		cout << "\nthinking about the set [" << i << "] ";
		print_vector_neatly(inputs[i]);
		cout << " = ";
		print_vector_neatly(targets[i]);
		cout << endl;

		network.net_feed_forward(&inputs[i]);
		cout << "the network came up with ";
		network.print_results();
	}

	cout << "\n-------------------------------------------------------\nTesting unseen problems" << endl;

	for (int i = (int)inputs.size() - 10; i < inputs.size(); i++) {
		cout << "\nthinking about the set [" << i << "] ";
		print_vector_neatly(inputs[i]);
		cout << " = ";
		print_vector_neatly(targets[i]);
		cout << endl;

		network.net_feed_forward(&inputs[i]);
		cout << "the network came up with ";
		network.print_results();
	}

	cin.get();
}

int main(){

	arrayCL();

}
