#include <iostream>
#include <fstream>
#include "..\library\json.hpp"
#include "../include/ConvolutionalNeuralNetwork.h"

using namespace std;
using json = nlohmann::json;

void print_vector_neatly(vector<float> &vec) {
	cout << "[";
	for (int i = 0; i < vec.size(); i++) {
		cout << vec[i];
		if (i < vec.size() - 1) {
			cout << ", ";
		}
	}
	cout << "]";
}

void print_vector2_neatly(vector<vector<float>> &vec) {
	for (int i = 0; i < vec.size(); i++) {
		print_vector_neatly(vec[i]);
		cout << "\n";
	}
}

ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork(string conf_file, OCLFunctions ocl) {

	ocl_functions = ocl;

	// read the config file 
	ifstream t(conf_file);
	stringstream buffer;
	buffer << t.rdbuf();
	
	// parse the file into a json object
	json cnn_json = json::parse(buffer.str());
	cout << "Loading config [" << cnn_json.dump() << "]" << endl;
	
	// used to work out the size of each element at the last layer before the fully connected network
	int current_element_width = 0;
	int current_element_height = 0;
	int current_layer_element_count = 1;

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
				current_element_width = input_size.width;
				current_element_height = input_size.height;
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
						cout << "Created filter" << endl;
						print_vector2_neatly(new_filter);
						conv_layer.filters.emplace_back(new_filter);
					}
				}

				// keep track of the 2d element width and height
				current_element_width = current_element_width - (int)conv_layer.filters.back().back().size() + 1;
				current_element_height = current_element_height - (int)conv_layer.filters.back().size() + 1;
				current_layer_element_count *= (int)conv_layer.filters.size();
				
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

			// keep track of the 2d element width and height
			current_element_width = current_element_width / pooling_layer.sample_size + (current_element_width % pooling_layer.sample_size == 0 ? 0 : 1);
			current_element_height = current_element_height / pooling_layer.sample_size + (current_element_height % pooling_layer.sample_size == 0 ? 0 : 1);
			
			pooling_layers.emplace_back(pooling_layer);
		}

		// create a fully connected network
		else if ((*l)["type"] == "F") {
			cout << "\tCreating a fully connected layer" << endl;
			layer_type_stack.emplace_back("F");
			
			// get the topology for this fully connected network
			vector<int> network_topology;
			
			// work out the fully connected network input size from conv and pooling combinations
			int fully_connected_input_size = current_element_width * current_element_height * current_layer_element_count;
			cout << "The input to the fully connected network needs to be [" << fully_connected_input_size << "]" << endl;
			network_topology.emplace_back(fully_connected_input_size);

			// append the rest of the topology as defined in the json file
			json full_layer_array = (*l)["details"]["sub_layers"];
			for (json::iterator f = full_layer_array.begin(); f != full_layer_array.end(); ++f) {
				network_topology.emplace_back((*f)["neurons"]);
			}

			// create and store the neural network
			fully_connected_networks.emplace_back(NeuralNetwork(&network_topology));
		}
		
		// Create a ReLu layer
		else if ((*l)["type"] == "R") {
			layer_type_stack.emplace_back("R");
		}

		else {
			cout << "\tI don't know what layer type [" << (*l)["type"] << "] is, quitting :(" << endl;
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

bool super_serious_logging =true;

void print_layer(vector<vector<vector<float>>>& layer) {	

	if (super_serious_logging) {
		cout << "The operation created [" << layer.size() << "] 2D values" << endl;
		for (int i = 0; i < layer.size(); i++) {
			for (int j = 0; j < layer[i].size(); j++) {
				if (j == 0) {
					cout << "[" << i << "]\t";
				}
				else {
					cout << "\t";
				}
				for (int k = 0; k < layer[i][j].size(); k++) {
					cout << layer[i][j][k] << ", ";
				}
				cout << endl;
			}
		}
	}
}

void print_gradients(vector<float>& gradients) {
	cout << "Gradients [";
	for (int j = 0; j < gradients.size(); j++) {
		cout << gradients[j];
		if (j < gradients.size() - 1) {
			cout << ", ";
		}
	}
	cout << "]" << endl;

}

void ConvolutionalNeuralNetwork::feed_forward(vector<vector<float>>& input_image_segment){

	cout << "\n__________________________________________" << endl;
	cout << "STARTING FORWARD PROPAGATION" << endl;

	// keep track of which convolution pooling and fully connected section we are on
	current_conv = 0;
	current_pool = 0;
	current_full = 0;
	layer_result_index = 0;

	// store the input image as the first element in the results stack
	//cout << "INPUT" << endl;
	//print_vector2_neatly(input_image_segment);
	vector<vector<vector<float>>> temp;
	temp.emplace_back(input_image_segment);
	layer_results_stack.emplace_back(temp);
	layer_result_index += 1;

	// cycle through the layers, the input layer can be ignored
	for (int i = 1; i < layer_type_stack.size(); i++) {

		if (layer_type_stack[i] == "C") {
			//cout << "\n[C] Performing convolution using convolution filter set at index [" << current_conv << "]" << endl;
			//cout << "\tThe last network layer was [" << layer_type_stack[i - 1] << "]" << endl;
			vector<vector<vector<float>>> conv_results;

			// if we need to use the input image as the convolution input
			if (layer_type_stack[i - 1] == "I") {
				//cout << "\tUsing the input image as inputs to convolution" << endl;
				for (int j = 0; j < convolution_layers[current_conv].filters.size(); j++) {
					vector<vector<float>> conv_result;
					//cout << "\t\tConvolving the image with filter [" << i << "]" << endl;
					ocl_functions.apply_filter_convolution(input_image_segment, convolution_layers[current_conv].filters[j], conv_result);
					conv_results.emplace_back(conv_result);
				}
				layer_results_stack.emplace_back(conv_results);
			}

			// if we need to use the last result in the stack as the convolution input
			else {
				vector<vector<vector<float>>> conv_results;
				//cout << "\tUsing the results stack index [" << layer_result_index - 1<< "]" << endl;
				
				// for the amount of filters we want to use on this layer
				for (int j = 0; j < convolution_layers[current_conv].filters.size(); j++) {
					
					// for the amount of elements in the previous layer
					for (int k = 0; k < layer_results_stack[layer_result_index - 1].size(); k++) {
						vector<vector<float>> conv_result;
						//cout << "\t\tConvolving the results layer [" << i << "][" << k << "] with filter [" << j << "]" << endl;
						ocl_functions.apply_filter_convolution(layer_results_stack[layer_result_index - 1][k], convolution_layers[current_conv].filters[j], conv_result);
						conv_results.emplace_back(conv_result);
					}
				}
				layer_results_stack.emplace_back(conv_results);
			}
			//print_layer(layer_results_stack.back());

			current_conv++;
			layer_result_index++;
		}
		else if (layer_type_stack[i] == "P") {
			//cout << "\n[P] Performing pooling with pooling sample index [" << current_pool << "]" << endl;
			//cout << "\tThe last network layer was [" << layer_type_stack[i - 1] << "]" << endl;

			vector<vector<vector<float>>> pool_results;
			if (layer_type_stack[i - 1] == "I") {
				vector<vector<float>> pool_result;
				//cout << "\tUsing the input image as inputs to pool" << endl;
				ocl_functions.pooling(input_image_segment, pool_result, pooling_layers[current_pool].sample_size);
				
				// these two seem silly but the layer results stack needs to have a 3d vec pushed back
				pool_results.emplace_back(pool_result);
				layer_results_stack.emplace_back(pool_results);
			}
			else {
				vector<vector<vector<float>>> pool_results;
				//cout << "\tUsing the results stack index [" << layer_result_index - 1 << "]" << endl;
				for (int j = 0; j < layer_results_stack[layer_result_index - 1].size(); j++) {
					vector<vector<float>> pool_result;
					ocl_functions.pooling(layer_results_stack[layer_result_index - 1][j], pool_result, pooling_layers[current_pool].sample_size);
					pool_results.emplace_back(pool_result);
				}
				layer_results_stack.emplace_back(pool_results);
			}
			//print_layer(layer_results_stack.back());

			current_pool++;
			layer_result_index++;
		}
		else if (layer_type_stack[i] == "R") {
			//cout << "\n[R] Performing ReLu step" << endl;
			//cout << "\tUsing the results stack index [" << layer_result_index - 1 << "]" << endl;
			vector<vector<vector<float>>> reluResults;

			for (int j = 0; j < layer_results_stack[layer_result_index - 1].size(); j++) {
				vector<vector<float>> result;
				ocl_functions.reLu(layer_results_stack[layer_result_index - 1][j], result);
				reluResults.emplace_back(result);
			}	

			layer_results_stack.emplace_back(reluResults);
			//print_layer(layer_results_stack.back());
			layer_result_index++;
		}
		else if (layer_type_stack[i] == "F") {
			//cout << "\n[F] Performing fully connected [" << current_full << "]" << endl;

			// convert the input data into an array for the fully connected network to use as its input
			vector<float> input_vec;
			for (int j = 0; j < layer_results_stack.back().size(); j++) {
				for (int k = 0; k < layer_results_stack.back()[j].size(); k++) {
					for (int l = 0; l < layer_results_stack.back()[j][k].size(); l++) {
						input_vec.emplace_back(layer_results_stack.back()[j][k][l]);
					}
				}
			}

			// forward propagate the fully conencted network
			fully_connected_networks[current_full].net_feed_forward(&input_vec);

			// increment counters
			current_full++;
			layer_result_index++;
		}
	}
}

void ConvolutionalNeuralNetwork::backwards_propagate(vector<float>& target_values) {

	cout << "\n__________________________________________" << endl;
	cout << "STARTING BACK PROPAGATION" << endl;

	// print the output of the network
	vector<float> output_results;
	fully_connected_networks.back().get_results(output_results);
	cout << "Network results [";
	for (int i = 0; i < output_results.size(); i++) {
		cout << output_results[i] << ", ";
	}
	cout << "]" << endl;

	// print the targets we want to aim for
	cout << "Target values [";
	for (int i = 0; i < target_values.size(); i++) {
		cout << target_values[i] << ", ";
	}
	cout << "]" << endl;

	// calculate error of network
	error = 0.0;
	for (int i = 0; i < output_results.size(); i++) {
		float delta = target_values[i] - output_results[i];
		error += delta * delta;
	}
	error /= output_results.size();
	error = sqrt(error);
	cout << "Network error [" << error << "]" << endl;

	vector<float> next_neuron_gradients;

	// cycle through the layer types, using the result stack solve the gradients for the layer - 1 from the current iteration
	for (int i = (int)layer_type_stack.size() - 1; i > -1; i--) {

		if (layer_type_stack[i] == "F") {
			//cout << "\n[F] Back propagating fully connected network at [" << current_full - 1<< "]" << endl;
			fully_connected_networks[current_full - 1].backwards_propagation(&target_values);
			
			// get the gradients of the fully connected network's input layer
			for (int j = 0; j < fully_connected_networks[current_full - 1].layers[0].size() - 1; j++) {
				next_neuron_gradients.emplace_back(fully_connected_networks[current_full - 1].layers[0][j]->get_gradient());
			}

			//print_gradients(next_neuron_gradients);
			current_full--;
		}
		if (layer_type_stack[i] == "P") {
			//cout << "\n[P] Back propagating pooling layer at [" << current_pool - 1 << "]" << endl;
			layer_result_index -= 1; 
			current_pool -= 1;
			int next_grad_count = 0;
			
			// there should be a 1-1 result relation for a pooling layer and the previous
			// cycle through the previous layer size
			vector<float> temp_next_neuron_gradients;
			for (int j = 0; j < layer_results_stack[layer_result_index - 2].size(); j++) {
				//cout << "\tWorking on prev layer element [" << j << "]" << endl;
				for (int k = 0; k < layer_results_stack[layer_result_index - 2][j].size(); k += pooling_layers[current_pool].sample_size) {
					for (int l = 0; l < layer_results_stack[layer_result_index - 2][j][k].size(); l += pooling_layers[current_pool].sample_size) {
						//cout << "\t\tNew pooling position" << endl;
						int x = 0;
						int y = 0;
						for (int m = 0; m < pooling_layers[current_pool].sample_size; m++) {
							for (int n = 0; n < pooling_layers[current_pool].sample_size; n++) {
								if (k + m < layer_results_stack[layer_result_index - 2][j].size() && l + n < layer_results_stack[layer_result_index - 2][j][0].size()) {
									//cout << "\t\t" << layer_results_stack[layer_result_index - 2][j][k + m][l + n] << endl;
									if (layer_results_stack[layer_result_index - 2][j][k + m][l + n] > layer_results_stack[layer_result_index - 2][j][k + y][l + x]) {
										x = n;
										y = m;
									}
								}
							}
						}
						//cout << "\t\tThis segments highest is [" << layer_results_stack[layer_result_index - 2][j][k + y][l + x] << "]" << endl;
						for (int m = 0; m < pooling_layers[current_pool].sample_size; m++) {
							for (int n = 0; n < pooling_layers[current_pool].sample_size; n++) {
								if (k + m < layer_results_stack[layer_result_index - 2][j].size() && l + n < layer_results_stack[layer_result_index - 2][j][0].size()) {
									if (m == y && n == x) {
										temp_next_neuron_gradients.emplace_back(next_neuron_gradients[next_grad_count]);
										next_grad_count += 1;
									}
									else {
										temp_next_neuron_gradients.emplace_back(0);
									}
								}
							}
						}
					}
				}
			}
			next_neuron_gradients.clear();
			next_neuron_gradients = temp_next_neuron_gradients;
			//print_gradients(next_neuron_gradients);
			current_pool--;
		}
		if (layer_type_stack[i] == "R") {
			//cout << "\n[R] Back propagating ReLu layer" << endl;
			layer_result_index -= 1;

			// kill gradients that are less than 0
			for (int j = 0; j < next_neuron_gradients.size(); j++) {
				next_neuron_gradients[j] = next_neuron_gradients[j] > 0 ? next_neuron_gradients[j] : next_neuron_gradients[j] * 0.01;
			}

			//print_gradients(next_neuron_gradients);
		}

		if (layer_type_stack[i] == "C") {
			//cout << "\n[C]Back propagating convolutional layer at [" << current_conv - 1 << "]" << endl;
			layer_result_index -= 1;
			//cout << "Layer results index [" << layer_result_index - 1 << "]" << endl;
			//print_layer(layer_results_stack[layer_result_index - 1]);
			
			//cout << "layer type stack is [" << i << "]" << endl;

			// the previous layer layer_results_index - 2
			//cout << "I think this is the input image" << endl;
			//print_layer(layer_results_stack[layer_result_index - 2]);	

			// work out the filter deltas
			int delta_index = 0;
			for (int j = 0; j < convolution_layers[current_conv - 1].filters.size(); j++) {
				//cout << "Working out filter weights for conv filter [" << j << "]" << endl;
				
				// start a running total vector for this filter
				vector<vector<float>> filter_deltas;
				for (int k = 0; k < convolution_layers[current_conv - 1].filters[j].size(); k++) {
					vector<float> temp;
					for (int l = 0; l < convolution_layers[current_conv - 1].filters[j].size(); l++) {
						temp.emplace_back(0);
					}
					filter_deltas.emplace_back(temp);
				}

				// what inputs does this weight effect
				//cout << "This filter runs on the following inputs" << endl;
				for (int k = 0; k < layer_results_stack[layer_result_index - 2].size(); k++) {
					//cout << "Input [" << k << "]" << endl;
					
					// for this input and filter, what is the delta array we want to use
					int delta_width = layer_results_stack[layer_result_index - 1][k][0].size();
					int delta_height = layer_results_stack[layer_result_index - 1][k].size();
					//cout << "The deltas for this operation in the format [" << delta_width << " by " << delta_height << "]" << endl;
					vector<vector<float>> structured_delta;
					for (int l = 0; l < delta_height; l++) {
						vector<float> temp_row;
						for (int m = 0; m < delta_width; m++) {
							temp_row.emplace_back(next_neuron_gradients[delta_index]);
							delta_index++;
						}
						structured_delta.emplace_back(temp_row);
					}
					//cout << "Found a structured delta" << endl;
					//print_vector2_neatly(structured_delta);

					// TODO: WORK OUT IF THIS RESULT NEEDS TO BE FLIPPED OR NOT, RIGHT NOW THAT FUNCTION WILL FLIP IT I THINK
					vector<vector<float>> result;
					ocl_functions.apply_filter_convolution(layer_results_stack[layer_result_index - 2][k], structured_delta, result);
					//cout << "\nResults" << endl;
					//print_vector2_neatly(result);

					//cout << "\nAddition" << endl;
					// add these results to a running total for the modification to this filter
					for (int l = 0; l < convolution_layers[current_conv - 1].filters[j].size(); l++) {
						for (int m = 0; m < convolution_layers[current_conv - 1].filters[j].size(); m++) {
							filter_deltas[l][m] += result[l][m];
						}
					}
					//print_vector2_neatly(filter_deltas);
				}

				//cout << "This filter will be modified by this matrix" << endl;
				//print_vector2_neatly(filter_deltas);

				// adjust the filter weights by adding the sum of the convolution of the deltas
				for (int k = 0; k < convolution_layers[current_conv - 1].filters[j].size(); k++) {
					for (int l = 0; l < convolution_layers[current_conv - 1].filters[j].size(); l++) {
						convolution_layers[current_conv - 1].filters[j][k][l] += filter_deltas[k][l];
					}
				}
			}

			/*cout << "Current filters" << endl;
			print_layer(convolution_layers[current_conv - 1].filters);
			*/
			// work out the input deltas by doing inverse convolution I think

			current_conv--;
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