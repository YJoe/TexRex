#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <bitset>
#include "..\library\json.hpp"
#include "..\include\ConvolutionalNeuralNetwork.h"
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
			cv::waitKey();
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

void get_simple_file_names(vector<String>& file_names) {
	String folder_names[] = { "A", "B", "C" };
	int file_per_folder = 7;

	for (String& folder : folder_names) {
		for (int i = 0; i < file_per_folder; i++) {
			String name = string("data/SIMPLE") + string("/") + string(folder) + string("/") + string(folder) + to_string(i + 1) + string(".jpg");
			cout << name << endl;
			file_names.emplace_back(name);
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

		draw_frequencies(image);

		cv::waitKey();
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

void get_lines(String image_name, int smoothing_factor) {

	// define a blur size
	Size s = Size(7, 7);

	// load and clean image
	cout << "Trying to read [" << image_name << "]" << endl;
	Mat image = cv::imread(image_name, IMREAD_GRAYSCALE);
	cv::Mat source = image.clone();
	GaussianBlur(source, image, s, 5);
	adaptiveThreshold(image, image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 101, 5);
	//cv::imshow("ORIGINAL", image);

	// read the occurances of black pixels per row
	vector<int> row_instances;
	for (int i = 0; i < image.rows; i++) {
		int row_count = 0;
		for (int j = 0; j < image.cols; j++) {
			if (image.at<uchar>(i, j) == 0) {
				row_count += 1;
			}
		}
		row_instances.emplace_back(row_count);
	}

	// using the bounds, draw what it thinks the lines are
	cv::Mat image_copy = image.clone();
	cv::cvtColor(image, image_copy, cv::COLOR_GRAY2BGR);

	for (int i = 0; i < image_copy.rows; i++) {
		for (int j = 0; j < row_instances[i]; j++) {
			image_copy.at<cv::Vec3b>(i, j)[0] = 0;
			image_copy.at<cv::Vec3b>(i, j)[1] = 0;
		}
	}

	//cv::imshow("1", image_copy);

	vector<int> smoothed;

	// smoothing data on edges that dont fit
	int av_start = 0;
	for (int i = 0; i < smoothing_factor / 2; i++) {
		av_start += row_instances[i];
	}
	av_start /= (int)row_instances.size();
	
	for (int i = 0; i < smoothing_factor / 2; i++) {
		smoothed.emplace_back(av_start);
	}

	// smoothing values with a local neighbourhood
	for (int i = smoothing_factor / 2; i < row_instances.size() - smoothing_factor / 2; i++) {
		int total = 0;
		for (int j = -(smoothing_factor / 2); j < smoothing_factor / 2 + 1; j++) {
			total += row_instances[i + j];
		}
		smoothed.emplace_back(total / smoothing_factor);
	}

	// smoothing data on edges that dont fit
	int av_end = 0;
	for (int i = (int)row_instances.size() - smoothing_factor / 2; i < (int)row_instances.size(); i++) {
		av_end += row_instances[i];
	}
	av_end /= (int)row_instances.size();

	for (int i = (int)row_instances.size() - smoothing_factor / 2; i < (int)row_instances.size(); i++) {
		smoothed.emplace_back(av_end);
	}

	cv::Mat image_copy2 = image.clone();
	cv::cvtColor(image, image_copy2, cv::COLOR_GRAY2BGR);

	for (int i = 0; i < image_copy2.rows; i++) {
		for (int j = 0; j < smoothed[i]; j++) {
			image_copy2.at<cv::Vec3b>(i, j)[1] = 0;
			image_copy2.at<cv::Vec3b>(i, j)[2] = 0;
		}
	}

	// search for peaks in data
	bool ascending = false;
	int last = 0;
	vector<vector<int>> peaks;
	
	for (int i = 0; i < smoothed.size(); i++) {

		// we found a low point
		if (!ascending && smoothed[i] > last) {
			ascending = true;
		}

		// we found a high point
		if (ascending && smoothed[i] < last) {
			ascending = false;
			vector<int> vals;
			vals.emplace_back(i - 1);
			vals.emplace_back(smoothed[i]);
			peaks.emplace_back(vals);
		}
		last = smoothed[i];
	}


	for (int i = 0; i < peaks.size(); i++) {
		for (int j = 0; j < image_copy2.cols; j++) {
			image_copy2.at<cv::Vec3b>(peaks[i][0], j)[0] = 0;
			image_copy2.at<cv::Vec3b>(peaks[i][0], j)[1] = 0;
			image_copy2.at<cv::Vec3b>(peaks[i][0], j)[2] = 200;
		}

		//// find the peak half value negative
		//for (int j = peaks[i][0]; j > -1; j--) {
		//	if (row_instances[j] < peaks[i][1] / 2) {
		//		for (int k = 0; k < image.cols; k++) {
		//			image_copy2.at<cv::Vec3b>(j, k)[0] = 200;
		//			image_copy2.at<cv::Vec3b>(j, k)[1] = 0;
		//			image_copy2.at<cv::Vec3b>(j, k)[2] = 0;
		//		}
		//		break;
		//	}
		//}

		//// find the peak half value positive
		//for (int j = peaks[i][0]; j < image.rows; j++) {
		//	if (row_instances[j] < peaks[i][1] / 2) {
		//		for (int k = 0; k < image.cols; k++) {
		//			image_copy2.at<cv::Vec3b>(j, k)[0] = 0;
		//			image_copy2.at<cv::Vec3b>(j, k)[1] = 200;
		//			image_copy2.at<cv::Vec3b>(j, k)[2] = 0;
		//		}
		//		break;
		//	}
		//}
	}

	cv::imshow("2", image_copy2);

	cv::waitKey();
}


int main(){

	// read input image and store it into segments
	Mat image = cv::imread("data/SLICED/HW1/ABCD_CAPITAL.jpg", IMREAD_GRAYSCALE);
	cv::Mat source = image.clone();
	GaussianBlur(source, image, Size(7, 7), 5);
	adaptiveThreshold(image, image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 151, 5);
	vector<ImageSegment> image_segments;
	segment_image_islands(image, image_segments);


	// input->conv->ReLU->Pool->conv->ReLU->Pool->FC->softmax
	ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork("data/CNN_JSON/network.txt");
	cnn.feed_forward(image_segments[0]);

	cin.get();

	exit(0);

	//Mat image = cv::imread("data/SLICED/HW1/ABCD_CAPITAL.jpg", IMREAD_GRAYSCALE);

	//cv::Mat source = image.clone();
	//GaussianBlur(source, image, Size(7, 7), 5);
	//adaptiveThreshold(image, image, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 151, 5);

	//vector<ImageSegment> image_segments;

	//segment_image_islands(image, image_segments);

	//// JUST USING THE FIRST THREE SEGMENTS WHICH I KNOW ARE A B AND C	
	//Size scale_size = Size(30, 30);
	//int filter_count = 8;
	//Size filter_size = Size(5, 5);
	//int pooling_size = 4;
	//int how_many_images = 3;
	//OCLFunctions ocl_functions = OCLFunctions(CL_DEVICE_TYPE_CPU);
	//int map_width = scale_size.width - filter_size.width + 1;
	//int map_height = scale_size.height - filter_size.height + 1;
	//int input_neurons = scale_size.width * scale_size.height;
	//int filter_convolutional_neurons = map_width * map_height;
	//int total_convolution_neurons = filter_convolutional_neurons * filter_count;

	//cout << "with an input of [w" << scale_size.width << ", h" << scale_size.height << "] [" << input_neurons << "]" << endl;
	//cout << "with [" << filter_count << "] filters" << endl;
	//cout << "of size [w" << filter_size.width << ", h" << filter_size.height << "] [" << filter_size.height * filter_size.width << "]" << endl;
	//cout << "a single convolution map will be [w" << map_width << ", h" << map_height << "] [" << filter_convolutional_neurons << "]" << endl;
	//cout << "the first convolutional layer will have [" << total_convolution_neurons << "] neurons" << endl;

	//// CREATE FILTERS
	//cout << "[TEXREX] Generating random filter weights" << endl;
	//vector<vector<vector<double>>> filters;
	//for (int i = 0; i < filter_count; i++) {
	//	vector<vector<double>> filter;
	//	get_random_filter(filter, filter_size.width, filter_size.height, -1.0, 1.0);
	//	filters.emplace_back(filter);
	//	Mat filter_image;
	//	get_image(filter, filter_image);
	//	imshow("F", filter_image);
	//	cv::waitKey();
	//	cv::destroyWindow("F");
	//}

	//// START WORK ON ONE IMAGE

	//exit(0);

	//// SCALE AND THEN DOBLEIFY IMAGES
	//cout << "[TEXREX] Putting each image into the range 0.0 - 1.0" << endl;
	//for (int i = 0; i < how_many_images; i++) {
	//	resize(image_segments[i].m, image_segments[i].m, scale_size);
	//	imshow("SCALED", image_segments[i].m);
	//	cv::waitKey();
	//	image_segments[i].m.convertTo(image_segments[i].m, CV_32FC1, 1.0 / 255.0);
	//	get_vector(image_segments[i].m, image_segments[i].double_m);
	//	image_segments[i].m.release();
	//}

	//// CONVOLVE
	//cout << "[TEXREX] Convolving using filters" << endl;
	//vector<vector<vector<double>>> convolved_segments;
	//for (int i = 0; i < 1; i++) {
	//	for (int j = 0; j < filters.size(); j++) {
	//		vector<vector<double>> convolved;
	//		ocl_functions.apply_filter_convolution(image_segments[i].double_m, filters[j], convolved);
	//		convolved_segments.emplace_back(convolved);
	//	}
	//}

	//// POOLING
	//cout << "[TEXREX] Pooling convolved map" << endl;
	//vector<vector<vector<double>>> pooled_segments;
	//for (int i = 0; i < convolved_segments.size(); i++) {
	//	vector<vector<double>> pooled;
	//	ocl_functions.pooling(convolved_segments[i], pooled, pooling_size);
	//	pooled_segments.emplace_back(pooled);
	//	cout << pooled_segments.back().size() << ", " << pooled_segments.back()[0].size() << endl << endl;
	//	print_vector2_neatly(pooled_segments.back());
	//	cout << endl << endl << endl;
	//}

	//cin.get();

/*
	for (int i = 0; i < 3; i++) {
		imshow("O", image_segments[i].m);
		resize(image_segments[i].m, image_segments[i].m, scale_size);
		imshow("S", image_segments[i].m);
		cv::waitKey();
	}
*/

	
	//view_sand_graph();

	//network();
	//image_convolution();
}
