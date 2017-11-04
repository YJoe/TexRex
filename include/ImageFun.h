#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

// image clean up
void segment_image(Mat& source_image, vector<Mat>& destination_vector, int divide_x, int divide_y);

// frequency domain
void take_dft(Mat& source_image, Mat& destination);
void take_inverse_dft(Mat& source_image, Mat& destination);
void get_visual_of_dft(Mat& source_dft, Mat& destination_dft);
void shift(Mat& magnitude);