#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "..\include\CLProgram.h"
#include "..\include\AnalysisJob.h"

int main(){

	AnalysisJob a = AnalysisJob("data/some_text.jpg");
	a.run();

}
