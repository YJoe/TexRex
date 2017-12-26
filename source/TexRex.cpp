#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <bitset>
#include "..\library\json.hpp"
#include "..\include\ConvolutionalNeuralNetwork.h"

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

int main(){

	// read input image and store it into segments
	cv::Mat image = cv::imread("data/SLICED/HW1/ABCD_CAPITAL.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat source = image.clone();
	GaussianBlur(source, image, cv::Size(7, 7), 5);
	adaptiveThreshold(image, image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 151, 5);
	vector<ImageSegment> image_segments;
	segment_image_islands(image, image_segments);
	
	// input->conv->ReLU->Pool->conv->ReLU->Pool->FC->softmax
	OCLFunctions ocl = OCLFunctions(CL_DEVICE_TYPE_GPU);
	ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork("data/CNN_JSON/network.txt", ocl);
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
