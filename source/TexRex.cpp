#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "..\include\CLProgram.h"

int main(){

	CLProgram cl = CLProgram("cl/hello_world.cl", 1);
	
	char string[128];
	cl_mem memory_obj = clCreateBuffer(cl.context, CL_MEM_READ_WRITE, sizeof(string), NULL, NULL);

	clSetKernelArg(cl.kernels[0], 0, sizeof(memory_obj), &memory_obj);
	clEnqueueTask(cl.command_queue, cl.kernels[0], 0, NULL, NULL);

	clEnqueueReadBuffer(cl.command_queue, memory_obj, CL_TRUE, 0, sizeof(string), string, 0, NULL, NULL);
	std::cout << "Result [" << string << "]" << std::endl;
	std::cin.get();

	cl.~CLProgram();
}
