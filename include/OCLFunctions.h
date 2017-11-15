#pragma once
#include<iostream>
#include<vector>
#include <CL/cl.hpp>

class OCLFunctions {
public:
	OCLFunctions(int device_type);
	void apply_filter_convolution(std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &filter, std::vector<std::vector<double>> &result_map);
	void pooling(std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &target, int sample_size);
	std::vector<cl::Platform> all_platforms;
	cl::Platform default_platform;
	std::vector<cl::Device> all_devices;
	cl::Device default_device;
	cl::Context context;
	cl::Program::Sources sources;
	cl::Program program;
private:
};
