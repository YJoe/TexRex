#pragma once

#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>

using namespace std;

struct ConvolutionLayer {
	vector<vector<vector<double>>> filters;
};

struct PoolingLayer {
	// todo make use of stride
	int sample_size;
};

class ConvolutionalNeuralNetwork {
public:
	explicit ConvolutionalNeuralNetwork(string conf_file);
	void get_random_filter(vector<vector<double>>& filter, int width, int height, double min, double max);
	double double_rand(double min, double max);
private:
	cv::Size input_size;
	vector<ConvolutionLayer> convolution_layers;
	vector<PoolingLayer> pooling_layers;
	vector<string> layer_type_stack;
};
