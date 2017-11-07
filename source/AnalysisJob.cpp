#include "..\include\AnalysisJob.h"

AnalysisJob::AnalysisJob(Mat source){
	analysis_source = source;
}

void AnalysisJob::run() {
	segment_image_islands(analysis_source, image_segments);
}
