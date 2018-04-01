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
	vector<string> regex_split(const string& s, string regex);
	vector<tuple<string, string, string, void (SimpleCommandInterface::*)(vector<string>& s)>> function_help;
	char get_type_code(string e);
	void handle_input(vector<string>& input);
	boolean file_exists(const std::string & name);
	boolean is_number(const std::string & s);
	void create_template(string file_name);
	void load_nist(vector<DataSample>& data_samples, string folder, string num_string, int sample_count, bool random, cv::Size network_input_size);
	void load_nist_binary(vector<DataSample>& data_samples, string folder, string target, string all_in_folder, int sample_count, cv::Size network_input_size);
	void set_evaluation(vector<string>& input);
	void loadnet(vector<string>& input);
	void createnet(vector<string>& input);
	void formatset(vector<string>& input);
	void loadset(vector<string>& input);
	void trainnet(vector<string>& input);
	void testnet(vector<string>& input);
	void setiteration(vector<string>& input);
	void viewgraph(vector<string>& input);
	void setseed(vector<string>& input);
	void savenet(vector<string>& intput);
	void help(vector<string>& input);
	void view_evaluations(vector<string>& input);
	void group_net_test(vector<string>& input);
	void error_message(string function);
public:
	SimpleCommandInterface();
	boolean evaluate_command(string input);
	void begin();
	ConvolutionalNeuralNetwork cnn;
	void test();
};