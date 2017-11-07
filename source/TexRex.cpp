#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "..\include\CLProgram.h"
#include "..\include\AnalysisJob.h"

int main(){

	//cv::Mat some_text_int = cv::imread("data/some_text.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat some_text_int = cv::imread("data/some_text.jpg", cv::IMREAD_GRAYSCALE);
	cv::Mat test_image;
	some_text_int.convertTo(test_image, CV_32FC1, 1.0 / 255.0);
	cv::imshow("source", test_image);
	waitKey();

	cv::Mat source = test_image.clone();
	gaussian_blur(source, test_image, 5);
	cv::imshow("source_blur", test_image);
	waitKey();

	cv::threshold(test_image, test_image, 0.5, 1.0, 0);
	cv::imshow("source_blur_binary", test_image);
	waitKey();

	AnalysisJob a = AnalysisJob(test_image);
	a.run();

	/*
	cv::imshow("a window", test_image);
	cv::waitKey();
*/



	/*CLProgram cl = CLProgram("cl/hello_world.cl", 1);
	
	char string[128];
	cl_mem memory_obj = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, sizeof(string), NULL, NULL);

	clSetKernelArg(cl.kernels[0], 0, sizeof(memory_obj), &memory_obj);
	clEnqueueTask(cl.command_queue, cl.kernels[0], 0, NULL, NULL);

	clEnqueueReadBuffer(cl.command_queue, memory_obj, CL_TRUE, 0, sizeof(string), string, 0, NULL, NULL);
	std::cout << "Result [" << string << "]" << std::endl;
	std::cin.get();

	cl.~CLProgram();*/
}
