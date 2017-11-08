#include "..\include\AnalysisJob.h"

AnalysisJob::AnalysisJob(string file_name){

	cout << "[AnalysisJob] Loading" << endl;
	cv::Mat some_text_int = cv::imread(file_name, cv::IMREAD_GRAYSCALE);
	some_text_int.convertTo(analysis_source, CV_32FC1, 1.0 / 255.0);
	cv::imshow("Source", analysis_source);

	clean();

	cv::imshow("Cleaned", analysis_source);
	waitKey();
}

void AnalysisJob::run() {
	segment_image_islands(analysis_source, image_segments);
	for (ImageSegment is : image_segments) {
		is.m = crop_segment(is.m, 10);
		cv::imshow("i", is.m);
		cv::waitKey();
	}
}

void AnalysisJob::clean(){
	// automate this clean up a bit
	cout << "[AnalysisJob] Cleaning" << endl;

	cv::Mat source = analysis_source.clone();
	gaussian_blur(source, analysis_source, 5);

	cv::threshold(analysis_source, analysis_source, 0.5, 1.0, 0);
}
