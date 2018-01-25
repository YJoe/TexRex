#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <bitset>
#include "..\include\DataSample.h"
#include "..\library\json.hpp"
#include "..\include\ConvolutionalNeuralNetwork.h"


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

	int letter_count = 5;
	int sample_count = 11;

	// read input image and store it into segments
	cv::Mat image = cv::imread("data/ABCDE.jpg", cv::IMREAD_GRAYSCALE);
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

		/*print_vector_neatly2(data_temp.answer);
		cout << endl;

		imshow("", image_segments[i].m);
		cv::waitKey();*/
	}
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

void train_example() {
	// define a handle on a device and load a network structure
	OCLFunctions ocl = OCLFunctions(CL_DEVICE_TYPE_GPU);
	ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork("data/CNN_JSON/network.json", ocl);

	// generate the training set
	vector<DataSample> samples;
	generateBasicSamples(samples, cnn.input_size);

	// define the results to compare maximums to
	vector<char> mapping = { 'A', 'B', 'C', 'D', 'E' };

	cnn.setTrainingSamples(samples);
	cnn.setMapping(mapping);

	cnn.train();
	cnn.json_dump_network("data/CNN_JSON/network_save.json");
	cin.get();
}

void evaluate_single_word(ConvolutionalNeuralNetwork cnn, string file_name) {
	vector<DataSample> input_samples;
	generateTrainingWord(input_samples, cnn.input_size, file_name);

	for (int i = 0; i < input_samples.size(); i++) {
		cout << cnn.evaluate(input_samples[i].image_segment.float_m_mini) << endl;
	}
	cv::waitKey();
}

void load_example() {
	// define a handle on a device and load a network structure
	OCLFunctions ocl = OCLFunctions(CL_DEVICE_TYPE_GPU);
	ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork("data/CNN_JSON/network_save.json", ocl);

	// define the results to compare maximums to
	vector<char> mapping = { 'A', 'B', 'C', 'D', 'E' };
	cnn.setMapping(mapping);

	evaluate_single_word(cnn, "data/TestWords/BAD.jpg");
	evaluate_single_word(cnn, "data/TestWords/CAB.jpg");
}

int main() {
	//train_example();
	load_example();
}
