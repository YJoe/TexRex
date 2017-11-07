#pragma once
#include "ImageFun.h"

using namespace cv;

class AnalysisJob {
public:
	AnalysisJob(Mat source);

	void run();

	Mat analysis_source;
	vector<ImageSegment> image_segments;

private:
};
