#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <CL/cl.h>

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

int main(){

	std::cout << "Setting up OpenCL" << std::endl;

	// get a platform
	cl_platform_id platform_id;
	cl_uint num_platforms;
	clGetPlatformIDs(1, &platform_id, &num_platforms);
	std::cout << "Number of platforms available [" << num_platforms << "]" << std::endl;

	// get a device
	cl_device_id device_id;
	cl_uint num_devices;
	clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_devices);
	std::cout << "Number of GPU devices available [" << num_devices << "]" << std::endl;

	// create a context with the device
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
	
	// create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, NULL);

	// create a place for the string to show up from the GPU
	char string[MEM_SIZE];

	// create a memory object
	cl_mem memory_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(string), NULL, NULL);

	// read the cl program that we want to run 
	FILE* fp = fopen("cl/hello_world.cl", "r");
	_ASSERT(fp);
	char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
	size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// create and then build the program for the device
	cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, NULL);
	clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	// create a kernal and then give it some arguments
	cl_kernel kernel = clCreateKernel(program, "hello", NULL);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &memory_obj);

	// excecute the kernel
	clEnqueueTask(command_queue, kernel, 0, NULL, NULL);

	// read the result from the kernel
	clEnqueueReadBuffer(command_queue, memory_obj, CL_TRUE, 0, MEM_SIZE * sizeof(char), string, 0, NULL, NULL);
	std::cout << "Result from GPU [" << string << "]" << std::endl;

	// free up memory
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(memory_obj);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(source_str);

	std::cin.get();

	return 0;
}
