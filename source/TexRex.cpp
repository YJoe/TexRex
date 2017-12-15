#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <bitset>
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
	OCLFunctions cl_funct = OCLFunctions(CL_DEVICE_TYPE_CPU);

	vector<vector<double>> image = {
		{  1, -1, -1, -1, -1},
		{ -1,  1, -1, -1, -1 },
		{ -1, -1,  1, -1, -1 },
		{ -1, -1, -1,  1, -1 },
		{ -1, -1, -1, -1,  1 }
	};
	vector<vector<double>> filter_a = {
		{  1, -1, -1 },
		{ -1,  1, -1 },
		{ -1, -1,  1 }
	};
	vector<vector<double>> filter_b = {
		{ -1, -1,  1 },
		{ -1,  1, -1 },
		{  1, -1, -1 }
	};
	vector<vector<double>> filter_c = {
		{  1, -1,  1 },
		{ -1,  1, -1 },
		{  1, -1,  1 }
	};
	vector<vector<double>> conv_map;

	cl_funct.apply_filter_convolution(image, filter_a, conv_map);
	print_vector2_neatly(conv_map);
	cout << "\n";

	cl_funct.apply_filter_convolution(image, filter_b, conv_map);
	print_vector2_neatly(conv_map);
	cout << "\n";

	cl_funct.apply_filter_convolution(image, filter_c, conv_map);
	print_vector2_neatly(conv_map);
	cout << "\n";

	cin.get();
}

void network() {
	
	vector<int> topology = { 2, 2, 1 };
	int training_index = 0;
	double has_learnt_threshold = 0.01;
	bool log_basic_stuff = false;
	NeuralNetwork network = NeuralNetwork(&topology);

	vector<vector<double>> inputs = {};

	vector<vector<double>> targets = {};

	for (int i = 0; i < 100; i++) {
		double a = (double)random_int(0, 50) / 100;
		double b = (double)random_int(0, 50) / 100;
		vector<double> args = { a, b };
		vector<double> targ = { a + b };
		inputs.emplace_back(args);
		targets.emplace_back(targ);
	}

	for (int i = 0; i < 200000; i++) {
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

void image_convolution() {
	OCLFunctions cl_funct = OCLFunctions(CL_DEVICE_TYPE_CPU);
	AnalysisJob a = AnalysisJob("data/some_text.jpg");
	a.run();

	vector<vector<vector<double>>> filters;
	filters.push_back({
		{ 0, 1, 1, 1 },
		{ 1, 0, 1, 1 },
		{ 1, 1, 0, 1 },
		{ 1, 1, 1, 0 }
	});
	filters.push_back({
		{ 1, 1, 1, 0 },
		{ 1, 1, 0, 1 },
		{ 1, 0, 1, 1 },
		{ 0, 1, 1, 1 }
	});
	filters.push_back({
		{ 1, 0, 0, 1 },
		{ 1, 0, 0, 1 },
		{ 1, 0, 0, 1 },
		{ 1, 0, 0, 1 }
	});

	// use random filters, pick an amount through testing start from 1 -> go from there
	// do unused filters die off? hmmmm

	for (ImageSegment& image_segment : a.image_segments) {
		imshow("segment", image_segment.m);

		vector<vector<double>> image_vec;
		get_vector(image_segment.m, image_vec);

		for (vector<vector<double>> filter : filters) {
			vector<vector<double>> map;
			vector<vector<double>> pooled;
			cl_funct.apply_filter_convolution(image_vec, filter, map);
			cl_funct.pooling(image_vec, pooled, 3);

			Mat temp_image;
			get_image(map, temp_image);
			imshow("conv", temp_image);
			waitKey();
		}
	}

}

void image_pooling() {
	
}

void get_sample_file_names(vector<String>& file_names) {
	String names[] = { "ABCD", "FREE", "TDJH", "TFBW", "TQBF" };
	String styles[] = { "CAPITAL", "FREE" };
	int sample_size = 17;

	for (int i = 0; i < sample_size; i++) {
		for (String& s : styles) {
			for (String& n : names) {
				String file_name = string("data/SLICED/HW") + to_string(i + 1) + string("/") + string(n) + string("_") + string(s) + string(".jpg");
				file_names.emplace_back(file_name);
			}
		}
	}
}

void view_sand_graph() {
	// get all of the files that we are testing with
	vector<String> file_names;
	get_sample_file_names(file_names);

	// define a blur size
	Size s = Size(7, 7);

	// cycle through images 
	for (int i = 0; i < file_names.size(); i++) {
		cout << "[" << to_string(i) << "] trying to read [" << file_names[i] << "]" << endl;
		Mat image = cv::imread(file_names[i], IMREAD_GRAYSCALE);

		cv::Mat source = image.clone();
		GaussianBlur(source, image, s, 5);
		adaptiveThreshold(image, image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 101, 5);

		draw_frequencies_x(image);

		waitKey();
	}

	cout << "DONE" << endl;
	cin.get();
}

void get_all_combos() {
	// define a filter size and create a place to store the combinations
	int filter_size = 3;
	int bin_size = (int)pow(2, filter_size * filter_size);
	vector<vector<vector<double>>> filter_combinations;

	// for 2^total_filter size get the binary representation to give all combinations of 1s and 0s
	for (int i = 0; i < bin_size; i++) {

		// get the binary representation of whatever cycle we are on 
		String bin_rep = bitset<16>(i).to_string().substr(16 - filter_size * filter_size);
		int current = 16 - filter_size * filter_size + 1;

		// populate the current filter with the current binary string
		vector<vector<double>> filter;
		for (int j = 0; j < filter_size; j++) {
			vector<double> filter_line;
			for (int k = 0; k < filter_size; k++) {
				filter_line.emplace_back((double)bin_rep[current] - 48);
				current--;
			}
			filter.emplace_back(filter_line);
		}
		filter_combinations.emplace_back(filter);
	}

	cout << "A total number of [" << filter_combinations.size() << "] filters have been created" << endl;

}

void read_features() {

	
	

	cin.get(); 
}


int main(){

	read_features();

	//view_sand_graph();
	//network();
	//image_convolution();
}
