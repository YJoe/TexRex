#pragma once

#include <vector>
#include "../include/NeuralNetwork.h"
#include "../include/ImageFun.h"
#include "../include/DataSample.h"
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
	explicit ConvolutionalNeuralNetwork(string conf_file, OCLFunctions ocl, int log_level);
	cv::Size input_size;
	void feed_forward(vector<vector<float>>& input_image);
	void backwards_propagate(vector<float>& target_values);
	void print_network_results(vector<char>& res);
	void show_filters(string prefix);
	void json_dump_network(string file_name);
	void setTrainingSamples(vector<DataSample>& dataSamples);
	void setMapping(vector<char>& mapping);
	void train(ofstream& data_file);
	char evaluate(vector<vector<float>>& image);
	bool (ConvolutionalNeuralNetwork::*terminating_function)();
	bool threshold_check();
	bool iteration_check();
	int iteration_target;
	float threshold_target;
	int current_iteration = 0;


private:
	void get_random_filter(vector<vector<float>>& filter, int width, int height, float min, float max);
	float float_rand(float min, float max);
	float error;
	int log_level;
	int current_conv;
	int current_pool;
	int current_full;
	int layer_result_index;
	vector<DataSample> trainingSamples;
	vector<char> mapping;
	OCLFunctions ocl_functions;
	vector<ConvolutionLayer> convolution_layers;
	vector<PoolingLayer> pooling_layers;
	vector<NeuralNetwork> fully_connected_networks;
	vector<string> layer_type_stack;
	vector<vector<vector<vector<float>>>> layer_results_stack;
};