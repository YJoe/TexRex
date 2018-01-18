#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <bitset>
#include "..\library\json.hpp"
#include "..\include\ConvolutionalNeuralNetwork.h"

int random_int(int min, int max) {
	return rand() % (max - min + 1) + min;
}

void print_vector_neatly2(vector<float> &vec) {
	cout << "[";
	for (int i = 0; i < vec.size(); i++) {
		cout << vec[i];
		if (i < vec.size() - 1) {
			cout << ", ";
		}
	}
	cout << "]";
}

void print_vector2_neatly2(vector<vector<float>> &vec) {
	for (int i = 0; i < vec.size(); i++) {
		print_vector_neatly2(vec[i]);
		cout << "\n";
	}
}

int main() {

	// read input image and store it into segments
	cv::Mat image = cv::imread("data/SLICED/HW1/ABCD_CAPITAL_CHEATING.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat source = image.clone();
	GaussianBlur(source, image, cv::Size(7, 7), 5);
	adaptiveThreshold(image, image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 151, 5);
	vector<ImageSegment> image_segments;
	segment_image_islands(image, image_segments);

	int how_many_segments = 4;

	// input->conv->ReLU->Pool->conv->ReLU->Pool->FC->softmax
	OCLFunctions ocl = OCLFunctions(CL_DEVICE_TYPE_GPU);
	ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork("data/CNN_JSON/network.txt", ocl);

	vector<vector<float>> targets;
	for (int i = 0; i < how_many_segments; i++) {
		vector<float> temp;
		for (int j = 0; j < how_many_segments; j++) {
			temp.emplace_back(i == j ? 1.0f : 0.0f);
		}

		targets.emplace_back(temp);
	}

	for (int i = 0; i < image_segments.size(); i++) {
	
		// scale the image to meet the network input sized
		resize(image_segments[i].m, image_segments[i].m, cnn.input_size);
		bitwise_not(image_segments[i].m, image_segments[i].m);
		image_segments[i].m.convertTo(image_segments[i].m, CV_32FC1, 1.0 / 255.0);

		// store the image as a vector of floats and release the image to save space
		get_vector(image_segments[i].m, image_segments[i].float_m_mini);
		image_segments[i].m.release();
	}
	
	cnn.show_filters("before");

	for (int i = 0; i < 2000; i++) {
		int random_number = random_int(0, targets.size() - 1);
		cnn.feed_forward(image_segments[random_number].float_m_mini);
		cnn.backwards_propagate(targets[random_number]);
	}

	cnn.show_filters("after");

	cnn.json_dump_network("data/CNN_JSON/network_save.txt");
	cin.get();
	exit(0);
}
