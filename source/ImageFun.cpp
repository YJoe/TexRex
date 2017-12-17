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

void get_vector(cv::Mat& source_image, vector<vector<double>>& target_vector){
	for (int i = 0; i < source_image.size().height; i++) {
		vector<double> temp;
		for (int j = 0; j < source_image.size().width; j++) {
			temp.emplace_back(source_image.at<float>(i, j));
		}
		target_vector.emplace_back(temp);
	}
}

void get_image(vector<vector<double>>& source_vector, cv::Mat& target_image) {
	target_image = cv::Mat::zeros(cv::Size((int)source_vector[0].size(), (int)source_vector.size()), CV_32F);
	normalize(target_image, target_image, 0, 1, CV_MINMAX);

	for (int i = 0; i < source_vector.size(); i++) {
		for (int j = 0; j < source_vector[0].size(); j++) {
			target_image.at<float>(i, j) = (float)source_vector[i][j];
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

	s.at<uchar>(y, x) = 200;
	c.at<uchar>(y, x) = 0;

	for (cv::Point p : adjacent_map) {
		if (x + p.x > -1 && x + p.x < c.cols && y + p.y > -1 && y + p.y < c.rows) {
			if (s.at<uchar>(y + p.y, x + p.x) == 0) {
				f(s, c, x + p.x, y + p.y);
			}
		}
	}
}

void find_gravity_center(ImageSegment& segment) {
	int total_x = 0;
	int total_y = 0;
	int count = 0;
	for (int i = 0; i < segment.m.rows; i++) {
		for (int j = 0; j < segment.m.cols; j++) {
			if (segment.m.at<uchar>(i, j) == 0) {
				count += 1;
				total_x += j;
				total_y += i;
			}
		}
	}
	segment.gravity_x = total_x / count;
	segment.gravity_y = total_y / count;
}

void segment_image_islands(cv::Mat& source_image, vector<ImageSegment>& destination){
	// assumes that the image is in black(1.0) and white(0.0);
	cout << "[ImageFun] Island Segments" << endl;


	for (int x = 0; x < source_image.cols; x++) {
		for (int y = 0; y < source_image.rows; y++) {
			if (source_image.at<uchar>(y, x) == 0) {
				ImageSegment is = { cv::Mat(source_image.size[0], source_image.size[1], CV_8UC1, cv::Scalar(255)), 0, 0, 0, 0};
				f(source_image, is.m, x, y);
				crop_segment(is, 5);
				find_gravity_center(is);
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
				if (source_segment.at<uchar>(j, i) == 0) {
					return i;
				}
			}
		}
		break;
	case X_MAX:
		for (int i = source_segment.size().width - 1; i > -1; i--) {
			for (int j = 0; j < source_segment.size().height; j++) {
				if (source_segment.at<uchar>(j, i) == 0) {
					return i;
				}
			}
		}
		break;
	case Y_MIN:
		for (int j = 0; j < source_segment.size().height; j++) {
			for (int i = 0; i < source_segment.size().width; i++) {
				if (source_segment.at<uchar>(j, i) == 0) {
					return j;
				}
			}
		}
		break;
	case Y_MAX:
		for (int j = source_segment.size().height - 1; j > -1; j--) {
			for (int i = 0; i < source_segment.size().width; i++) {
				if (source_segment.at<uchar>(j, i) == 0) {
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
	cv::Mat temp = source_segment.m(cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y));
	source_segment.m.release();
	copyMakeBorder(temp, source_segment.m, padding, padding, padding, padding, cv::BORDER_CONSTANT, 255);
}


int truncate(int val, int min, int max) {
	return val > 255 ? 255 : val < 0 ? 255 - val : val;
}

void draw_frequencies(cv::Mat& source) {
	cv::Mat source_copy = source.clone();
	cv::cvtColor(source, source_copy, cv::COLOR_GRAY2BGR);

	imshow("F", source_copy);

	int streak = 0;
	int b = rand() % (255 + 1);
	int g = rand() % (255 + 1);
	int r = rand() % (255 + 1);
	for (int i = 0; i < source_copy.cols; i++) {
		int total = 0;
		for (int j = 0; j < source_copy.rows; j++) {
			if (source.at<uchar>(j, i) == 0) {
				int temp_b = truncate(source_copy.at<cv::Vec3b>(total, i)[0] - b, 0, 255);
				int temp_g = truncate(source_copy.at<cv::Vec3b>(total, i)[1] - g, 0, 255);
				int temp_r = truncate(source_copy.at<cv::Vec3b>(total, i)[2] - r, 0, 255);
				source_copy.at<cv::Vec3b>(total, i)[0] = temp_b;
				source_copy.at<cv::Vec3b>(total, i)[1] = temp_g;
				source_copy.at<cv::Vec3b>(total, i)[2] = temp_r;
				total += 1;
			}
		}
		if (total == 0) {
			streak = 0;
			b = rand() % (255 + 1);
			g = rand() % (255 + 1);
			r = rand() % (255 + 1);
		}
		else {
			streak += 1;
		}
	}

	streak = 0;
	b = rand() % (255 + 1);
	g = rand() % (255 + 1);
	r = rand() % (255 + 1);
	for (int i = 0; i < source_copy.rows; i++) {
		int total = 0;
		for (int j = 0; j < source_copy.cols; j++) {
			if (source.at<uchar>(i, j) == 0) {
				int temp_b = truncate(source_copy.at<cv::Vec3b>(i, total)[0] - b, 0, 255);
				int temp_g = truncate(source_copy.at<cv::Vec3b>(i, total)[1] - g, 0, 255);
				int temp_r = truncate(source_copy.at<cv::Vec3b>(i, total)[2] - r, 0, 255);
				source_copy.at<cv::Vec3b>(i, total)[0] = temp_b;
				source_copy.at<cv::Vec3b>(i, total)[1] = temp_g;
				source_copy.at<cv::Vec3b>(i, total)[2] = temp_r;
				total += 1;
			}
		}
		if (total == 0) {
			streak = 0;
			b = rand() % (255 + 1);
			g = rand() % (255 + 1);
			r = rand() % (255 + 1);
		}
		else {
			streak += 1;
		}
	}

	imshow("F", source_copy);
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
