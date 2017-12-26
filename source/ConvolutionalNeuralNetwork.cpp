#include <iostream>
#include <fstream>
#include "..\library\json.hpp"
#include "../include/ConvolutionalNeuralNetwork.h"

using namespace std;
using json = nlohmann::json;


ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(string conf_file, OCLFunctions ocl) {

	ocl_functions = ocl;

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
						vector<vector<float>> new_filter;
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
}

void ConvolutionalNeuralNetwork::feed_forward(ImageSegment & input_image_segment){
	cout << "\nFeeding forward " << endl;

	// scale the image to meet the network input size
	resize(input_image_segment.m, input_image_segment.m, input_size);
	input_image_segment.m.convertTo(input_image_segment.m, CV_32FC1, 1.0 / 255.0);

	// store the image as a vector of floats and release the image to save space
	get_vector(input_image_segment.m, input_image_segment.float_m);
	input_image_segment.m.release();

	// keep track of which convolution pooling and fully connected section we are on
	int current_conv = 0;
	int current_pool = 0;
	int current_full = 0;
	int layer_result_index = 0;

	// cycle through the layers, the input layer can be ignored
	for (int i = 1; i < layer_type_stack.size(); i++) {

		if (layer_type_stack[i] == "C") {
			cout << "[C] Performing convolution using convolution filter set at index [" << current_conv << "]" << endl;
			cout << "\tThe last network layer was [" << layer_type_stack[i - 1] << "]" << endl;

			// if we need to use the input image as the convolution input
			if (layer_type_stack[i - 1] == "I") {
				cout << "\tUsing the input image as inputs to convolution" << endl;
				for (int i = 0; i < convolution_layers[current_conv].filters.size(); i++) {
					vector<vector<float>> conv_result;
					cout << "\t\tConvolving the image with filter [" << i << "]" << endl;
					ocl_functions.apply_filter_convolution(input_image_segment.float_m, convolution_layers[current_conv].filters[i], conv_result);
				}
			}

			// if we need to use the last result in the stack as the convolution input
			else {
				cout << "\tUsing the results stack index [" << layer_result_index << "]" << endl;
				for (int i = 0; i < convolution_layers[current_conv].filters.size(); i++) {
					cout << "\t\tConvolving the results map with filter [" << i << "]" << endl;
				}
			}
			cout << "\tresults of this convolution will be stored in [" << layer_result_index + 1 << "]" << endl;

			current_conv++;
			layer_result_index++;
		}
		else if (layer_type_stack[i] == "P") {
			cout << "[P] Performing pooling with pooling sample index [" << current_pool << "]" << endl;
			cout << "\tThe last network layer was [" << layer_type_stack[i - 1] << "]" << endl;

			if (layer_type_stack[i - 1] == "I") {
				cout << "\tUsing the input image as inputs to pool" << endl;
			}
			else {
				cout << "\tUsing the results stack index [" << layer_result_index << "]" << endl;
			}
			cout << "\tresults of this pool will be stored in [" << layer_result_index + 1 << "]" << endl;


			current_pool++;
			layer_result_index++;
		}
		else if (layer_type_stack[i] == "F") {
			cout << "[F] Performing fully connected [" << current_full << "]" << endl;
			current_full++;
			layer_result_index++;
		}
	}
}

void ConvolutionalNeuralNetwork::get_random_filter(vector<vector<float>>& filter, int width, int height, float min, float max) {

	for (int i = 0; i < width; i++) {
		vector<float> filter_line;
		for (int j = 0; j < height; j++) {
			filter_line.emplace_back(float_rand(min, max));
		}
		filter.emplace_back(filter_line);
	}
}

float ConvolutionalNeuralNetwork::float_rand(float min, float max) {
	float f = (float)rand() / RAND_MAX;
	return min + f * (max - min);
}