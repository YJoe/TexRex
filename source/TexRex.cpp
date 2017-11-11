#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "..\include\CLProgram.h"
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

int random_int(int min, int max) {
	return rand() % (max - min + 1) + min;
}

void a() {
	AnalysisJob a = AnalysisJob("data/some_text.jpg");
	a.run();
}

void x() {
	CLProgram p = CLProgram("cl/hello_numbers.cl", CL_DEVICE_TYPE_CPU);
	cl_kernel kernel = p.create_kernel("multiply");

	// Create and set argumentss
	int result;
	int a = 4;
	int b = 7;
	cl_mem out_int_buffer = p.create_buffer(CL_MEM_READ_WRITE, sizeof(int), NULL);
	p.set_kernel_arg(kernel, 0, sizeof(int), &a);
	p.set_kernel_arg(kernel, 1, sizeof(int), &b);
	p.set_kernel_arg(kernel, 2, sizeof(cl_mem), &out_int_buffer);

	// Run kernel
	p.run_kernel(kernel);

	// Reading object populated by the kernel
	p.read_buffer(out_int_buffer, CL_TRUE, sizeof(int), &result);
	cout << "[" << result << "]" << endl;

	// Cleaning up
	p.free_program();
	p.free_buffer(out_int_buffer);
	p.free_kernel(kernel);

	cin.get();
}

void y() {
	CLProgram p = CLProgram("cl/matrix_add.cl", CL_DEVICE_TYPE_GPU);
	cl_kernel kernel = p.create_kernel("add");

	int arr1[5] = {2, 3, 4, 5, 6};
	int arr2[5] = {2, 2, 2, 2, 2};
	cout << "Before" << endl;
	cout << "[" << arr1[0] << "]" << endl;
	cout << "[" << arr1[1] << "]" << endl;
	cout << "[" << arr1[2] << "]" << endl;
	cout << "[" << arr1[3] << "]" << endl;
	cout << "[" << arr1[4] << "]" << endl;
	
	cl_mem d_arr1 = p.create_buffer(CL_MEM_READ_WRITE, sizeof(int) * 5, NULL);
	cl_mem d_arr2 = p.create_buffer(CL_MEM_READ_WRITE, sizeof(int) * 5, NULL);
	p.set_kernel_arg(kernel, 0, sizeof(cl_mem), &d_arr1);
	p.set_kernel_arg(kernel, 1, sizeof(cl_mem), &d_arr2);

	p.run_kernel(kernel);

	p.read_buffer(d_arr1, CL_TRUE, sizeof(int), arr1);
	cout << "\nAfter" << endl;
	cout << "[" << arr1[0] << "]" << endl;
	cout << "[" << arr1[1] << "]" << endl;
	cout << "[" << arr1[2] << "]" << endl;
	cout << "[" << arr1[3] << "]" << endl;
	cout << "[" << arr1[4] << "]" << endl;

	// Cleaning up
	p.free_program();
	p.free_buffer(d_arr1);
	p.free_buffer(d_arr2);
	p.free_kernel(kernel);

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

	vector<vector<double>> inputs = {
		{ 0, 0 },
		{ 0, 1 },
		{ 1, 0 },
		{ 1, 1 }
	};

	vector<vector<double>> targets = {
		{ 0 },
		{ 1 },
		{ 1 },
		{ 0 }
	};

	srand(0);
	for (int i = 0; i < 1000; i++) {

		// setting the training set
		training_index = random_int(0, inputs.size() - 1);

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

	cout << "\n-------------------------------------------------------" << endl;

	// see how well the network can mirror the training sets
	for (int i = 0; i < inputs.size(); i++) {
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

	network();

}
