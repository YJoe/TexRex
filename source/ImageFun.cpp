#include "../include/ImageFun.h"

using namespace std;


// cleaning images

void segment_image(Mat& source_image, vector<Mat>& destination_vector, int divide_x, int divide_y) {
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
			destination_vector.push_back(source_image(Rect(i * segment_x, j * segment_y, segment_x, segment_y)));
		}
	}
}


// frequency domain

void take_dft(Mat& source_image, Mat& destination) {
	// create a mat object that can hold real and complex values
	Mat original_complex[2] = { source_image, Mat::zeros(source_image.size(), CV_32F) };

	// merge the two channels into the one dft_ready mat object
	Mat dft_ready;
	merge(original_complex, 2, dft_ready);

	// perform the dft calculation of the dft_ready image and store the output in the destination image
	dft(dft_ready, destination, DFT_COMPLEX_OUTPUT);
}

void take_inverse_dft(Mat& source_image, Mat& destination) {

	// call the dft function with flags to invert the image
	dft(source_image, destination, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
}

void get_visual_of_dft(Mat& source_dft, Mat& destination_dft) {
	// split the two channels up into a new split array
	Mat split_arr[2] = { Mat::zeros(source_dft.size(), CV_32F), Mat::zeros(source_dft.size(), CV_32F) };
	split(source_dft, split_arr);

	// take the magnitude
	magnitude(split_arr[0], split_arr[1], destination_dft);

	// add one to all elements
	destination_dft += Scalar::all(1);

	// take log of magnitude to scale things down
	log(destination_dft, destination_dft);

	// normalise the image within the values 0.0 - 1.0
	normalize(destination_dft, destination_dft, 0, 1, CV_MINMAX);
}
 
void shift(Mat& magnitude) {

	// crop if it has an odd number of rows or columns
	magnitude = magnitude(Rect(0, 0, magnitude.cols & -2, magnitude.rows & -2));

	// find the center x and center y of the image
	int cx = magnitude.cols / 2;
	int cy = magnitude.rows / 2;

	// get quadrants of the image
	Mat q0(magnitude, Rect(0, 0, cx, cy));   // Top-Left
	Mat q1(magnitude, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magnitude, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magnitude, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat temp;

	// swap quadrant top left with bottom right
	q0.copyTo(temp);
	q3.copyTo(q0);
	temp.copyTo(q3);

	// swap quadrant top right with bottom left
	q1.copyTo(temp);
	q2.copyTo(q1);
	temp.copyTo(q2);
}
