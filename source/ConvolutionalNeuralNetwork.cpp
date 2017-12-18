#include <iostream>
#include <fstream>
#include "..\library\json.hpp"
#include "../include/ConvolutionalNeuralNetwork.h"

using namespace std;
using json = nlohmann::json;


ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(string conf_file) {

	// read the config file 
	ifstream t(conf_file);
	stringstream buffer;
	buffer << t.rdbuf();
	
	// parse the file into a json object
	json cnn_json = json::parse(buffer.str());
	cout << "Loading config [" << cnn_json.dump() << "]" << endl;
	
	// read and create layers
	json layer_array = cnn_json["cnn"]["layers"];
	cout << "Reading network json" << endl;
	for (json::iterator l = layer_array.begin(); l != layer_array.end(); ++l) {
		
		// parse an input layer
		if ((*l)["type"] == "I") {
			cout << "\tCreating an input layer" << endl;
			if (layer_type_stack.size() == 0) {
				layer_type_stack.emplace_back("I");
				input_size = cv::Size((*l)["details"]["input_width"], (*l)["details"]["input_height"]);
			}
			else {
				cout << "\t\tYou can't have an input layer here :( inputs must be at the start of the structure" << endl;
				cout << "\t\tPress any key to quit" << endl;
				cin.get();
				exit(-1);
			}
		}

		// parse a convolution layer
		else if ((*l)["type"] == "C") {
			if (layer_type_stack.size() > 0) {
				cout << "\tCreating a convolution layer" << endl;
				layer_type_stack.emplace_back("C");
				ConvolutionLayer conv_layer;

				// cycle through each filter defined in the layer
				json filter_array = (*l)["details"]["filters"];
				for (json::iterator f = filter_array.begin(); f != filter_array.end(); ++f) {
					
					// if there is already a filter defined in the json
					if ((*f)["filter_obj"].size() != 0) {
						cout << "\t\tOops, under construction :( we can't load filters just yet" << endl;
						cin.get();
						exit(-1);
					}

					// there is no filter defined so we will just create a new one
					else {
						vector<vector<double>> new_filter;
						get_random_filter(new_filter, (*f)["filter_width"], (*f)["filter_height"], -1.0, 1.0);
						conv_layer.filters.emplace_back(new_filter);
					}
				}

				// store the convolution layer we just made
				convolution_layers.emplace_back(conv_layer);
			}
		}

		// parse a pooling layer
		else if ((*l)["type"] == "P"){
			cout << "\tCreating a pooling layer" << endl;
			layer_type_stack.emplace_back("P");
			PoolingLayer pooling_layer;
			pooling_layer.sample_size = (*l)["details"]["sample_size"];
			pooling_layers.emplace_back(pooling_layer);
		}

		// create a fully connected network
		else if ((*l)["type"] == "F") {
			cout << "\tCreating a fully connected layer" << endl;
			layer_type_stack.emplace_back("F");
			
			// get the topology for this fully connected network
			vector<int> network_topology;
			json full_layer_array = (*l)["details"]["sub_layers"];
			for (json::iterator f = full_layer_array.begin(); f != full_layer_array.end(); ++f) {
				network_topology.emplace_back((*f)["neurons"]);
			}

			// create and store the neural network
			fully_connected_networks.emplace_back(NeuralNetwork(&network_topology));
		}
		else {
			cout << "\tI don't know what layer type [" << (*l)["type"] << "] is" << endl;
			cin.get();
			exit(-1);
		}
	}

	cout << "Finished reading config file, network has a structure of [";
	for (int i = 0; i < layer_type_stack.size(); i++) {
		cout << layer_type_stack[i];
	}
	cout << "]" << endl;

	cin.get();
	exit(0);
}

void ConvolutionalNeuralNetwork::get_random_filter(vector<vector<double>>& filter, int width, int height, double min, double max) {

	for (int i = 0; i < width; i++) {
		vector<double> filter_line;
		for (int j = 0; j < height; j++) {
			filter_line.emplace_back(double_rand(min, max));
		}
		filter.emplace_back(filter_line);
	}
}

double ConvolutionalNeuralNetwork::double_rand(double min, double max) {
	double f = (double)rand() / RAND_MAX;
	return min + f * (max - min);
}