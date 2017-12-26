#pragma once

#include <vector>
#include "../include/NeuralNetwork.h"
#include "../include/ImageFun.h"
#include "../include/OCLFunctions.h"


using namespace std;

struct ConvolutionLayer {
	vector<vector<vector<float>>> filters;
};

struct PoolingLayer {
	// todo make use of stride
	int sample_size;
};

class ConvolutionalNeuralNetwork {
public:
	explicit ConvolutionalNeuralNetwork(string conf_file, OCLFunctions ocl);
	void feed_forward(ImageSegment & input_image);

private:
	void get_random_filter(vector<vector<float>>& filter, int width, int height, float min, float max);
	float float_rand(float min, float max);

	OCLFunctions ocl_functions;
	cv::Size input_size;
	vector<ConvolutionLayer> convolution_layers;
	vector<PoolingLayer> pooling_layers;
	vector<NeuralNetwork> fully_connected_networks;
	vector<string> layer_type_stack;
	vector<vector<vector<vector<float>>>> layer_results_stack;
};