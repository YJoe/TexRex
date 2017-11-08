#pragma once
#include "ImageFun.h"

using namespace cv;

class AnalysisJob {
public:
	AnalysisJob(string file_name);

	void run();
	void clean();

	Mat analysis_source;
	vector<ImageSegment> image_segments;

private:
};
