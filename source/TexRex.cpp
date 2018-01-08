#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <bitset>
#include "..\library\json.hpp"
#include "..\include\ConvolutionalNeuralNetwork.h"

//void print_vector_neatly(vector<float> &vec) {
//	cout << "[";
//	for (int i = 0; i < vec.size(); i++) {
//		cout << vec[i];
//		if (i < vec.size() - 1) {
//			cout << ", ";
//		}
//	}
//	cout << "]";
//}
//
//void print_vector2_neatly(vector<vector<float>> &vec) {
//	for (int i = 0; i < vec.size(); i++) {
//		print_vector_neatly(vec[i]);
//		cout << "\n";
//	}
//}

int random_int(int min, int max) {
	return rand() % (max - min + 1) + min;
}

int main() {

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
	vector<float> a_targets = { 1.0, 0.0, 0.0 };
	vector<float> b_targets = { 0.0, 1.0, 0.0 };
	vector<float> c_targets = { 0.0, 0.0, 1.0 };

	for (int i = 0; i < image_segments.size(); i++) {

		// scale the image to meet the network input sized
		resize(image_segments[i].m, image_segments[i].m, cnn.input_size);
		bitwise_not(image_segments[i].m, image_segments[i].m);
		image_segments[i].m.convertTo(image_segments[i].m, CV_32FC1, 1.0 / 255.0);

		// store the image as a vector of floats and release the image to save space
		get_vector(image_segments[i].m, image_segments[i].float_m_mini);
		image_segments[i].m.release();
	}

	for (int i = 0; i < 10; i++) {
		cnn.feed_forward(image_segments[0].float_m_mini);
		cnn.backwards_propagate(a_targets);

		cnn.feed_forward(image_segments[1].float_m_mini);
		cnn.backwards_propagate(b_targets);
		
		cnn.feed_forward(image_segments[2].float_m_mini);
		cnn.backwards_propagate(c_targets);
	}

	cin.get();
	exit(0);
}
