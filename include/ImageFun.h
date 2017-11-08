#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;

// 
struct ImageSegment {
	cv::Mat m;
	int x;
	int y;
};

// image clean up
void segment_image_squares(cv::Mat& source_image, vector<cv::Mat>& destination_vector, int divide_x, int divide_y);
void segment_image_islands(cv::Mat& source_image, vector<ImageSegment>& destination);
void crop_segment(ImageSegment& source_segment, int padding);
void binary_threshold(cv::Mat& source_image, cv::Mat& target_image, float threshold);
float get_image_grey_avg(cv::Mat& source_image);
void binary_threshold_auto(cv::Mat& source_image, cv::Mat& target_image);
void gaussian_blur(cv::Mat& source_image, cv::Mat& target_image, int neighbour_hood);


// frequency domain
void take_dft(cv::Mat& source_image, cv::Mat& destination);
void take_inverse_dft(cv::Mat& source_image, cv::Mat& destination);
void get_visual_of_dft(cv::Mat& source_dft, cv::Mat& destination_dft);
void shift(cv::Mat& magnitude);