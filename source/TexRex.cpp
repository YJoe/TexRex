#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <bitset>
#include <filesystem>
#include "..\include\DataSample.h"
#include "..\library\json.hpp"
#include "..\include\ConvolutionalNeuralNetwork.h"
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

void split_save_image(string file_name, cv::Size network_image_size) {

	if (CreateDirectory("data/alpha", NULL) || ERROR_ALREADY_EXISTS == GetLastError()){
	}
	else{
		cout << "Creating the directory failed" << endl;
		exit(-1);
	}

	string str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	vector<char> mapping;
	for (int i = 0; i < str.size(); i++) {
		mapping.emplace_back(str[i]);
	}
	int sample_count = 15;

	// read input image and store it into segments
	cv::Mat image = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
	cv::Mat source = image.clone();
	GaussianBlur(source, image, cv::Size(7, 7), 5);
	adaptiveThreshold(image, image, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 151, 5);
	vector<ImageSegment> image_segments;
	segment_image_islands(image, image_segments);

	for (int i = 0; i < image_segments.size(); i++) {

		// scale the image to meet the network input sized
		resize(image_segments[i].m, image_segments[i].m, network_image_size);
		bitwise_not(image_segments[i].m, image_segments[i].m);

		string image_name = "data/alpha/image_" + to_string(i / sample_count) + "_" + to_string(i % sample_count) + ".jpg";
		cout << "Savinig [" + image_name << "]" << endl;
		cout << "\tLocation [" + image_segments[i].x << ", " << image_segments[i].y << "]" << endl << endl;

		imshow("", image_segments[i].m);
		imwrite(image_name, image_segments[i].m);
		cv::waitKey();
	}

	exit(0);
}

void load_data_split(vector<DataSample>& data_samples, string folder_path) {

	cout << "Loading training data" << endl;
	string str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	vector<char> mapping;
	for (int i = 0; i < str.size(); i++) {
		mapping.emplace_back(str[i]);
	}
	int sample_count = 15;

	for (int i = 0; i < mapping.size(); i++) {
		for (int j = 0; j < sample_count; j++) {
			DataSample temp_data;
			
			// create the answer for this image
			for (int k = 0; k < mapping.size(); k++) {
				if (k == i) {
					temp_data.answer.emplace_back(1);
				}
				else {
					temp_data.answer.emplace_back(0);
				}
			}

			// load the image
			string image_name = folder_path + "image_" + to_string(i) + "_" + to_string(j) + ".jpg";
			temp_data.image_segment.m = cv::imread(image_name, cv::IMREAD_GRAYSCALE);
			temp_data.image_segment.m.convertTo(temp_data.image_segment.m, CV_32FC1, 1.0 / 255.0);
			
			// store the image as a vector of floats and release the image to save space
			get_vector(temp_data.image_segment.m, temp_data.image_segment.float_m_mini);
			data_samples.emplace_back(temp_data);
		}
	}
	cout << "Folder data loaded :D" << endl;
	cin.get();
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
	cv::waitKey();
}

void evaluate_single_word(ConvolutionalNeuralNetwork cnn, DataSample input) {
	cout << cnn.evaluate(input.image_segment.float_m_mini) << endl;
	cv::waitKey();
}

void train_example(string data_file_name) {

	//srand(time(NULL));
	srand(0);

	// start a timer
	std::clock_t start;
	double duration;
	start = std::clock();

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
	ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork("data/CNN_JSON/network.json", ocl, 1);

	// generate the training set
	vector<DataSample> samples;
	load_data_split(samples, "data/alpha/");

	// define the results to compare maximums to
	string str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	vector<char> mapping;
	for (int i = 0; i < str.size(); i++) {
		mapping.emplace_back(str[i]);
	}

	cout << "Setting training samples" << endl;
	cnn.setTrainingSamples(samples);
	
	cout << "Setting training mapping" << endl;
	cnn.setMapping(mapping);

	cout << "Starting traing" << endl;
	cnn.train(data_file);
	cnn.json_dump_network("data/CNN_JSON/network_save.json");

	////////////////////////////////////////////////////////////////

	// stop the timer
	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "Duration [" << duration << "]\n";
	cin.get();

	// show a plot of the learning rate
	ShellExecute(0, 0, "test_plot.plt", 0, 0, SW_SHOW);
	
	evaluate_single_word(cnn, samples[0]);
	cin.get();
}

void load_example() {
	// define a handle on a device and load a network structure
	OCLFunctions ocl = OCLFunctions(CL_DEVICE_TYPE_GPU);
	ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork("data/CNN_JSON/network_save.json", ocl, 1);

	// define the results to compare maximums to
	string str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	vector<char> mapping;
	for (int i = 0; i < str.size(); i++) {
		mapping.emplace_back(str[i]);
	}
	cnn.setMapping(mapping);

	evaluate_single_word(cnn, "data/TestWords/BAD.jpg");
	evaluate_single_word(cnn, "data/TestWords/CAB.jpg");
	evaluate_single_word(cnn, "data/TestWords/HELLO_WORLD.jpg");
	evaluate_single_word(cnn, "data/TestWords/QBF.jpg");
	evaluate_single_word(cnn, "data/TestWords/TEST.jpg");
}

int main() {

	// run the code
	train_example("data0.dat");
	//load_example();
}
