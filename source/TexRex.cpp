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
#include "..\include\SimpleCommandInterface.h"
#include <cstdio>
#include <ctime>


void print_vector_neatly2(vector<float> &vec) {
	cout << "[";
	for (int i = 0; i < vec.size(); i++) {
		cout << vec[i];
		if (i < vec.size() - 1) {
			cout << ", ";
		}
	}
	cout << "]";
}

void print_vector2_neatly2(vector<vector<float>> &vec) {
	for (int i = 0; i < vec.size(); i++) {
		print_vector_neatly2(vec[i]);
		cout << "\n";
	}
}

void generateBasicSamples(vector<DataSample>& data_samples, cv::Size input_size) {

	int letter_count = 26;
	int sample_count = 10;

	// read input image and store it into segments
	cv::Mat image = cv::imread("data/alphabet2.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat source = image.clone();
	GaussianBlur(source, image, cv::Size(7, 7), 5);
	adaptiveThreshold(image, image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 151, 5);
	vector<ImageSegment> image_segments;
	segment_image_islands(image, image_segments);

	for (int i = 0; i < image_segments.size(); i++) {

		// scale the image to meet the network input sized
		resize(image_segments[i].m, image_segments[i].m, input_size);
		bitwise_not(image_segments[i].m, image_segments[i].m);
		image_segments[i].m.convertTo(image_segments[i].m, CV_32FC1, 1.0 / 255.0);

		// store the image as a vector of floats and release the image to save space
		get_vector(image_segments[i].m, image_segments[i].float_m_mini);
		//image_segments[i].m.release();

		// create the data set
		DataSample data_temp;
		data_temp.image_segment = image_segments[i];

		// create the answer
		for (int j = 0; j < letter_count; j++) {
			if (j == i / sample_count) {
				data_temp.answer.emplace_back(1.0f);
			}
			else {
				data_temp.answer.emplace_back(0.0f);
			}
		}
		// store the data sample
		data_samples.emplace_back(data_temp);

		print_vector_neatly2(data_temp.answer);
		cout << endl;

		imshow("", image_segments[i].m);
		cv::waitKey();
	}

	exit(0);
}

void split_save_image(string file_name, string folder_path) {

	// create the alpha bet folder if it isn't there already
	if (!CreateDirectory(folder_path.c_str(), NULL) && !ERROR_ALREADY_EXISTS == GetLastError()){
		cout << "Creating the directory failed :(" << endl;
		cin.get();
		exit(-1);
	}

	int alpha_size = 26;
	int sample_count = 15;

	// read input image
	cv::Mat image = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
	cv::Mat source = image.clone();
	
	// blur and threshold the data
	GaussianBlur(source, image, cv::Size(7, 7), 5);
	adaptiveThreshold(image, image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 151, 5);
	
	// segment the image
	vector<ImageSegment> image_segments;
	segment_image_islands(image, image_segments);

	// save each segment
	for (int i = 0; i < image_segments.size(); i++) {

		// invert so that letters are white and blank space is black
		bitwise_not(image_segments[i].m, image_segments[i].m);

		// construct the image name
		string image_name = folder_path + "image_" + to_string(i / sample_count) + "_" + to_string(i % sample_count) + ".jpg";
	
		// save the file
		imwrite(image_name, image_segments[i].m);
	}
}

void load_data_split(vector<DataSample>& data_samples, string folder_path, cv::Size network_image_size) {

	cout << "Loading training data" << endl;
	int alpha_size = 26;
	int sample_count = 15;

	for (int i = 0; i < alpha_size; i++) {
		for (int j = 0; j < sample_count; j++) {
			DataSample temp_data;
			
			// create the answer for this image
			for (int k = 0; k < alpha_size; k++) {
				if (k == i) {
					temp_data.answer.emplace_back(1);
				}
				else {
					temp_data.answer.emplace_back(0);
				}
			}

			// construct the file name and load the image
			string image_name = folder_path + "image_" + to_string(i) + "_" + to_string(j) + ".jpg";
			temp_data.image_segment.m = cv::imread(image_name, cv::IMREAD_GRAYSCALE);
			
			// scale the image to meet the network input sized and convert to float
			resize(temp_data.image_segment.m, temp_data.image_segment.m, network_image_size);
			temp_data.image_segment.m.convertTo(temp_data.image_segment.m, CV_32FC1, 1.0 / 255.0);
			
			// store the image as a vector of floats and release the image to save space
			get_vector(temp_data.image_segment.m, temp_data.image_segment.float_m_mini);
			data_samples.emplace_back(temp_data);
		}
	}
	cout << "Folder data loaded :D" << endl;
}

void generateTrainingWord(vector<DataSample>& data_samples, cv::Size input_size, string file_name) {
	
	// read input image
	cv::Mat image = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
	imshow("INPUT", image);

	// clean input image
	cv::Mat source = image.clone();
	GaussianBlur(source, image, cv::Size(7, 7), 5);
	adaptiveThreshold(image, image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 151, 5);
	imshow("CLEANED", image);

	// segment image
	vector<ImageSegment> image_segments;
	segment_image_islands(image, image_segments);

	// manipulate image for use within this network
	for (int i = 0; i < image_segments.size(); i++) {
		DataSample data_temp;

		// scale the image to meet the network input sized
		resize(image_segments[i].m, image_segments[i].m, input_size);

		// invert
		bitwise_not(image_segments[i].m, image_segments[i].m);

		// convert to float
		image_segments[i].m.convertTo(image_segments[i].m, CV_32FC1, 1.0 / 255.0);

		// get the image as a vector
		get_vector(image_segments[i].m, image_segments[i].float_m_mini);

		// store the temp sample into the vector
		data_temp.image_segment = image_segments[i];
		data_samples.emplace_back(data_temp);
	}
}

void evaluate_single_word(ConvolutionalNeuralNetwork cnn, string file_name) {
	vector<DataSample> input_samples;
	generateTrainingWord(input_samples, cnn.input_size, file_name);

	for (int i = 0; i < input_samples.size(); i++) {
		cout << cnn.evaluate(input_samples[i].image_segment.float_m_mini) << endl;
	}
}

int evaluate_single_word(ConvolutionalNeuralNetwork cnn, DataSample input) {
	char network_out = cnn.evaluate(input.image_segment.float_m_mini);
	
	int max_index = 0;
	for (int i = 0; i < input.answer.size(); i++) {
		if (input.answer[i] > input.answer[max_index]) {
			max_index = i;
		}
	}

	cout << "Network evaluation [" << network_out << "] it should be [" << cnn.get_mapping()[max_index] << "]" << endl;

	return cnn.get_mapping()[max_index] == network_out ? 1 : 0;
}

void load_mnist(vector<DataSample>& data_samples, string folder, string num_string, int sample_count, cv::Size network_input_size) {

	// variables that we will use to read directories
	HANDLE hFind;
	WIN32_FIND_DATA data;

	// for how many numbers we should train on
	for (int i = 0; i < num_string.size(); i++) {
	
		cout << "loading [" << sample_count << "] letter [" << num_string[i] << "] files" << endl;

		// form a folder to look at
		string f = folder + num_string[i] + "/" + "*.png";
		int count = 0;

		// search for all files in this dir that are png
		hFind = FindFirstFile(f.c_str(), &data);
		if (hFind != INVALID_HANDLE_VALUE) {
			
			// while there is another file
			do {
				cout << "\t" << data.cFileName << endl;
				
				DataSample temp_data;

				// store which index is correct
				temp_data.correct_index = i;

				// create the answer for this image
				for (int k = 0; k < num_string.size(); k++) {
					if (k == i) {
						temp_data.answer.emplace_back(1);
					}
					else {
						temp_data.answer.emplace_back(0);
					}
				}

				// construct the file name and load the image
				temp_data.image_segment.m = cv::imread(folder + num_string[i] + "/" + data.cFileName, cv::IMREAD_GRAYSCALE);

				// scale the image to meet the network input sized and convert to float
				resize(temp_data.image_segment.m, temp_data.image_segment.m, network_input_size);
				temp_data.image_segment.m.convertTo(temp_data.image_segment.m, CV_32FC1, 1.0 / 255.0);

				// store the image as a vector of floats and release the image to save space
				get_vector(temp_data.image_segment.m, temp_data.image_segment.float_m_mini);
				data_samples.emplace_back(temp_data);

				count++;
			} while (FindNextFile(hFind, &data) && count < sample_count);
			
			// close the handle on the file we found
			FindClose(hFind);
		}
	}
}

void train_example(string training_folder, string result_mapping, int sample_count, string data_file_name, int log_level) {

	//srand(time(NULL));
	srand(0);

	// start a timer
	double duration;
	clock_t start = clock();

	////////////////////////////////////////////////////////

	ofstream data_file_clear(data_file_name);
	data_file_clear.close();
	ofstream data_file(data_file_name, ios::app);
	if (data_file.is_open()){
		data_file << "# NETWORK LEARNING DATA\n";
		cout << "Data logging enabled" << endl;
	}
	else {
		cout << "Failed to create data file" << endl;
		exit(-1);
	}

	// define a handle on a device and load a network structure
	OCLFunctions ocl = OCLFunctions(CL_DEVICE_TYPE_GPU);
	ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork("data/CNN_JSON/network.json", ocl, log_level);

	// generate the training set
	vector<DataSample> data_samples;
	load_mnist(data_samples, training_folder, result_mapping, sample_count, cnn.input_size);

	// define the results to compare maximums to
	string str = result_mapping;
	vector<char> mapping;
	for (int i = 0; i < str.size(); i++) {
		mapping.emplace_back(str[i]);
	}

	cout << "Setting training samples" << endl;
	cnn.setTrainingSamples(data_samples);
	
	cout << "Setting training mapping" << endl;
	cnn.setMapping(mapping);

	cout << "Starting traing" << endl;
	cnn.train(data_file);
	//cnn.json_dump_network("data/CNN_JSON/network_save.json");
	data_file.close();

	////////////////////////////////////////////////////////////////

	// stop the timer
	duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Duration [" << duration << "]\n";

	return;

	// show a plot of the learning rate
	//ShellExecute(0, 0, "data/plot/test_plot.plt", 0, 0, SW_SHOW);
	
	int correct_count = 0;
	for (int i = 0; i < data_samples.size(); i++) {
		correct_count += evaluate_single_word(cnn, data_samples[i]);
	}

	cout << "Network success rate [" << (float)correct_count / (float)data_samples.size() * 100.0f << "]" << endl;
}

void load_example() {
	// define a handle on a device and load a network structure
	OCLFunctions ocl = OCLFunctions(CL_DEVICE_TYPE_GPU);
	ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork("data/cnn_json/network_save.json", ocl, 1);

	// generate the training set
	vector<DataSample> data_samples;
	load_mnist(data_samples, "data/MNIST/training/", "0123456789", 200, cnn.input_size);

	// define the results to compare maximums to
	string str = "0123456789";
	vector<char> mapping;
	for (int i = 0; i < str.size(); i++) {
		mapping.emplace_back(str[i]);
	}

	cnn.setMapping(mapping);

	int correct_count = 0;
	for (int i = 0; i < data_samples.size(); i++) {
		correct_count += evaluate_single_word(cnn, data_samples[i]);
	}

	cout << "Network success rate [" << (float)correct_count / (float)data_samples.size() * 100.0f << "]" << endl;

	// define the results to compare maximums to
	/*string str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	vector<char> mapping;
	for (int i = 0; i < str.size(); i++) {
		mapping.emplace_back(str[i]);
	}
	cnn.setMapping(mapping);

*/

	//evaluate_single_word(cnn, "data/TestWords/BAD.jpg");
	//evaluate_single_word(cnn, "data/TestWords/CAB.jpg");
	//evaluate_single_word(cnn, "data/TestWords/HELLO_WORLD.jpg");
	//evaluate_single_word(cnn, "data/TestWords/QBF.jpg");
	//evaluate_single_word(cnn, "data/TestWords/TEST.jpg");
}

int main() {

	SimpleCommandInterface sci;
	sci.begin();

	//train_example("data/MNIST/training/", "0123456789", 10, "data1.dat", 1);
	//load_example();
	
}
