#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

void segment_image(Mat& source_image, vector<Mat>& destination_vector, int divide_x, int divide_y);
