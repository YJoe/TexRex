#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <bitset>
#include <filesystem>
#include <regex>
#include "..\include\DataSample.h"
#include "..\library\json.hpp"
#include "..\include\ConvolutionalNeuralNetwork.h"
#include <cstdio>
#include <ctime>

class SimpleCommandInterface {
private:
	ConvolutionalNeuralNetwork cnn;
	vector<string> regex_split(const string& s, string regex);
	char get_type_code(string e);
	void handle_input(vector<string>& input, string pattern_code);
	void handle_s(vector<string>& input);
	void handle_sd(vector<string>& input);
	void handle_ss(vector<string>& input);
	void handle_sdii(vector<string>& input);
	void handle_sdis(vector<string>& input);
	void handle_sddii(vector<string>& input);
	boolean file_exists(const std::string & name);
	void create_template(string file_name);
	void load_mnist(vector<DataSample>& data_samples, string folder, string num_string, int sample_count, cv::Size network_input_size);
public:
	SimpleCommandInterface();
	void begin();
};