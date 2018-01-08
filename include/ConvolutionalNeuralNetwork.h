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
	cv::Size input_size;
	void feed_forward(vector<vector<float>>& input_image);
	void backwards_propagate(vector<float>& target_values);

private:
	void get_random_filter(vector<vector<float>>& filter, int width, int height, float min, float max);
	float float_rand(float min, float max);
	float error;
	int current_conv;
	int current_pool;
	int current_full;
	int layer_result_index;

	OCLFunctions ocl_functions;
	vector<ConvolutionLayer> convolution_layers;
	vector<PoolingLayer> pooling_layers;
	vector<NeuralNetwork> fully_connected_networks;
	vector<string> layer_type_stack;
	vector<vector<vector<vector<float>>>> layer_results_stack;
};