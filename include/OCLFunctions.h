#pragma once
#include<iostream>
#include<vector>
#include <CL/cl.hpp>

class OCLFunctions {
public:
	OCLFunctions(int device_type);
	OCLFunctions();
	void apply_filter_convolution(std::vector<std::vector<float>> &image, std::vector<std::vector<float>> &filter, std::vector<std::vector<float>> &result_map);
	void inverse_convolution(std::vector<std::vector<float>>& im, std::vector<std::vector<float>> fi, std::vector<std::vector<float>>& res);
	void pooling(std::vector<std::vector<float>> &image, std::vector<std::vector<float>> &target, int sample_size);
	void reLu(std::vector<std::vector<float>>& input, std::vector<std::vector<float>>& output);
	std::vector<cl::Platform> all_platforms;
	cl::Platform default_platform;
	std::vector<cl::Device> all_devices;
	cl::Device default_device;
	cl::Context context;
	cl::Program::Sources sources;
	cl::Program program;
private:
};