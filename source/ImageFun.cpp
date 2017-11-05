#include "../include/ImageFun.h"

using namespace std;


// cleaning images

void segment_image(cv::Mat& source_image, vector<cv::Mat>& destination_vector, int divide_x, int divide_y) {
	// get the size of the image
	int source_x = source_image.cols;
	int source_y = source_image.rows;

	// get the size of each new image
	int segment_x = source_x / divide_x;
	int segment_y = source_y / divide_y;

	// for each segment we want to create
	for (int i = 0; i < divide_x; i++) {
		for (int j = 0; j < divide_y; j++) {

			// cut a segment of the image out and store it in the vector
			destination_vector.push_back(source_image(cv::Rect(i * segment_x, j * segment_y, segment_x, segment_y)));
		}
	}
}

double distance(double x1, double y1, double x2, double y2) {
	return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

void binary_threshold(cv::Mat& source_image, cv::Mat& target_image, float threshold) {
	cout << "\nBinary threshold" << endl;

	cout << threshold << endl;

	// for all pixels within the image
	for (int i = 0; i < source_image.rows; i++) {
		for (int j = 0; j < source_image.cols; j++) {

			// get the image value
			float val = source_image.at<float>(cv::Point(j, i));

			// if the value is greater than the threshold, set the pixel value to the max
			if (val > threshold) {
				target_image.at<float>(cv::Point(j, i)) = 1.0;
			}

			// if the value is less than or equal to the threshold, set the pixel value to the lowest
			else {
				target_image.at<float>(cv::Point(j, i)) = 0.0;
			}
		}
	}
}

float get_image_grey_avg(cv::Mat& source_image) {
	float avg = 0;

	// for all pixels in the image
	for (int i = 0; i < source_image.rows; i++) {
		for (int j = 0; j < source_image.cols; j++) {

			// add the current pixels value to the total
			avg += source_image.at<float>(cv::Point(j, i));
		}
	}

	cout << "avg " << avg / (float)(source_image.rows * source_image.cols) << endl;
	// return the total divided by the pixel count
	return avg / (float)(source_image.rows * source_image.cols);
}

void binary_threshold_auto(cv::Mat& source_image, cv::Mat& target_image) {

	// get the average image gray value
	float avg = get_image_grey_avg(source_image);

	// operate on the target image using the average threshold worked out by the other function
	binary_threshold(source_image, target_image, avg);
}


// frequency domain

void take_dft(cv::Mat& source_image, cv::Mat& destination) {
	// create a cv::Mat object that can hold real and complex values
	cv::Mat original_complex[2] = { source_image, cv::Mat::zeros(source_image.size(), CV_32F) };

	// merge the two channels into the one dft_ready cv::Mat object
	cv::Mat dft_ready;
	merge(original_complex, 2, dft_ready);

	// perform the dft calculation of the dft_ready image and store the output in the destination image
	dft(dft_ready, destination, cv::DFT_COMPLEX_OUTPUT);
}

void take_inverse_dft(cv::Mat& source_image, cv::Mat& destination) {

	// call the dft function with flags to invert the image
	dft(source_image, destination, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
}

void get_visual_of_dft(cv::Mat& source_dft, cv::Mat& destination_dft) {
	// split the two channels up into a new split array
	cv::Mat split_arr[2] = { cv::Mat::zeros(source_dft.size(), CV_32F), cv::Mat::zeros(source_dft.size(), CV_32F) };
	split(source_dft, split_arr);

	// take the magnitude
	magnitude(split_arr[0], split_arr[1], destination_dft);

	// add one to all elements
	destination_dft += cv::Scalar::all(1);

	// take log of magnitude to scale things down
	log(destination_dft, destination_dft);

	// normalise the image within the values 0.0 - 1.0
	normalize(destination_dft, destination_dft, 0, 1, CV_MINMAX);
}
 
void shift(cv::Mat& magnitude) {

	// crop if it has an odd number of rows or columns
	magnitude = magnitude(cv::Rect(0, 0, magnitude.cols & -2, magnitude.rows & -2));

	// find the center x and center y of the image
	int cx = magnitude.cols / 2;
	int cy = magnitude.rows / 2;

	// get quadrants of the image
	cv::Mat q0(magnitude, cv::Rect(0, 0, cx, cy));   // Top-Left
	cv::Mat q1(magnitude, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(magnitude, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(magnitude, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat temp;

	// swap quadrant top left with bottom right
	q0.copyTo(temp);
	q3.copyTo(q0);
	temp.copyTo(q3);

	// swap quadrant top right with bottom left
	q1.copyTo(temp);
	q2.copyTo(q1);
	temp.copyTo(q2);
}
