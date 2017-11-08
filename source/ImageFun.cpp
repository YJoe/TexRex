#include "../include/ImageFun.h"

using namespace std;


enum FEATURE_FLAG{X_MIN, X_MAX, Y_MIN, Y_MAX};

// cleaning images

float distance(float x1, float y1, float x2, float y2) {
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

void gaussian_blur(cv::Mat & source_image, cv::Mat & target_image, int neighbourhood_size){

	// define the maximum distance a pixel can be from another by using the neighbourhood size
	float max = distance((float)0, (float)0, (float)(neighbourhood_size / 2 + 1), (float)(neighbourhood_size / 2 + 1));

	// for all pixels in the image
	for (int i = 0; i < source_image.rows; i++) {
		for (int j = 0; j < source_image.cols; j++) {
			float total = 0;
			int sample_count = 0;
			float total_distance_weight = 0;

			// for all pixels within the neighbourhood
			for (int k = -(neighbourhood_size / 2); k < (neighbourhood_size / 2) + 1; k++) {
				for (int l = -(neighbourhood_size / 2); l < (neighbourhood_size / 2) + 1; l++) {

					// check that the pixel is within the correct range
					if (j + l > -1 && j + l < source_image.cols && i + k > -1 && i + k < source_image.rows) {

						// note that the pixel is valid and that we are using it within the total
						sample_count += 1;

						// calculate the distance between the current pixel and the current neighbourhood pixel
						// a greater distance will be lower because we are subtracting it from the max distance
						float distance_weight = max - distance((float)j, (float)i, (float)(j + l), (float)(i + k));

						// add the calculated distance to the total distance
						total_distance_weight += distance_weight;

						// add to the total the pixel's value * the distance weight
						total += source_image.at<float>(cv::Point(j + l, i + k)) * distance_weight;
					}
				}
			}

			// divide the total by the total distance weight of the neighbour hood to get the weighted average
			total /= total_distance_weight;

			// set the current pixel to the neighbourhood's weighted average
			target_image.at<float>(cv::Point(j, i)) = total;
		}
	}
}

// 

void segment_image_squares(cv::Mat& source_image, vector<cv::Mat>& destination_vector, int divide_x, int divide_y) {
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

cv::Point adjacent_map[] = {cv::Point(-1, -1), cv::Point(0, -1), cv::Point(1, -1),
							cv::Point(-1, 0),                    cv::Point(1, 0),
							cv::Point(-1, 1), cv::Point(0, 1), cv::Point(1, 1) };

void f(cv::Mat& s, cv::Mat& c, int x, int y) {
	
	s.at<float>(cv::Point(x, y)) = 0.5;
	c.at<float>(cv::Point(x, y)) = 0.0;

	for (cv::Point p: adjacent_map) {
		if (x + p.x > -1 && y + p.y > -1 && x + p.x < s.cols && y + p.y < s.rows) {
			if (s.at<float>(cv::Point(x + p.x, y + p.y)) == 0.0) {
				f(s, c, x + p.x, y + p.y);
			}
		}
	}
}

void segment_image_islands(cv::Mat& source_image, vector<ImageSegment>& destination){
	// assumes that the image is in black(1.0) and white(0.0);
	cout << "[ImageFun] Island Segments" << endl;

	for (int i = 0; i < source_image.cols; i++) {
		for (int j = 0; j < source_image.rows; j++) {
			if (source_image.at<float>(j, i) == 0.0) {
				ImageSegment is = { cv::Mat::ones(source_image.size(), CV_32F), 0, 0 };
				f(source_image, is.m, i, j);
				destination.emplace_back(is);
			}
		}
	}
	
	cout << "[ImageFun] created [" << destination.size() << "] segments" << endl;
}

int find_feature(cv::Mat& source_segment, int flag) {
	switch (flag) {
	case X_MIN:
		for (int i = 0; i < source_segment.size().width; i++) {
			for (int j = 0; j < source_segment.size().height; j++) {
				if (source_segment.at<float>(j, i) == 0.0) {
					return i;
				}
			}
		}
		break;
	case X_MAX:
		for (int i = source_segment.size().width - 1; i > -1; i--) {
			for (int j = 0; j < source_segment.size().height; j++) {
				if (source_segment.at<float>(j, i) == 0.0) {
					return i;
				}
			}
		}
		break;
	case Y_MIN:
		for (int j = 0; j < source_segment.size().height; j++) {
			for (int i = 0; i < source_segment.size().width; i++) {
				if (source_segment.at<float>(j, i) == 0.0) {
					return j;
				}
			}
		}
		break;
	case Y_MAX:
		for (int j = source_segment.size().height - 1; j > -1; j--) {
			for (int i = 0; i < source_segment.size().width; i++) {
				if (source_segment.at<float>(j, i) == 0.0) {
					return j;
				}
			}
		}
	}
	return -1;
}

void crop_segment(ImageSegment& source_segment, int padding) {
	
	// find the minimum and maximums of the island
	int min_x = find_feature(source_segment.m, X_MIN);
	int max_x = find_feature(source_segment.m, X_MAX) + 1;
	int min_y = find_feature(source_segment.m, Y_MIN);
	int max_y = find_feature(source_segment.m, Y_MAX) + 1;

	// remember where the image was taken from the source image
	source_segment.x = min_x;
	source_segment.y = min_y;

	// crop the segment to remove white space
	source_segment.m = source_segment.m(cv::Rect(min_x - padding, min_y - padding, max_x - min_x + padding * 2, max_y - min_y + padding * 2));
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
