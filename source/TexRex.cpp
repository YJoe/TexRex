#include <iostream>
#include <ctime>
#include "..\include\NeuralNetwork.h"
#include "..\include\ImageFun.h"

using namespace std;
using namespace cv;

void print_vector_neatly(vector<double> &vec) {
	cout << "[";
	for (int i = 0; i < vec.size(); i++) {
		cout << vec[i];
		if (i < vec.size() - 1) {
			cout << ", ";
		}
	}
	cout << "]";
}

int random_int(int min, int max) {
	return rand() % (max - min + 1) + min;
}

int main() {
	Mat test_image_int = imread("data/ABXY/ABXY.jpg", IMREAD_GRAYSCALE);
	Mat test_image;
	test_image_int.convertTo(test_image, CV_32FC1, 1.0 / 255.0);

	vector<Mat> split_list;
	segment_image(test_image, split_list, 4, 6);

	for (Mat &image : split_list) {

		imshow("image segment", image);

		Mat image_dft;
		take_dft(image, image_dft);

		Mat image_dft_visual;
		get_visual_of_dft(image_dft, image_dft_visual);
		shift(image_dft_visual);

		imshow("dft of source", image_dft_visual);
		waitKey();
	}
	
    return 0;
}