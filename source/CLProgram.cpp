#include "..\include\CLProgram.h"

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

using namespace std;

CLProgram::CLProgram(char* source_file, int kernel_count){
	build_platform_device(CL_DEVICE_TYPE_GPU);
	build_context();
	build_command_queue();
	read_source_cl(source_file);
	build_program();

	for (int i = 0; i < kernel_count; i++) {
		kernels.emplace_back(create_kernel());
	}
}

CLProgram::~CLProgram(){
	// free up memory
	/*for (cl_kernel k : kernels) {
		clReleaseKernel(k);
	}
	clReleaseProgram(program);
	clReleaseMemObject(memory_obj);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(source_str);*/
}

void CLProgram::build_platform_device(int device_type){
	// get a platform
	clGetPlatformIDs(1, &platform_id, &num_platforms);
	std::cout << "Number of platforms available [" << num_platforms << "]" << std::endl;

	// get a device
	clGetDeviceIDs(platform_id, device_type, 1, &device_id, &num_devices);
	std::cout << "Number of GPU devices available [" << num_devices << "]" << std::endl;
}

void CLProgram::build_context() {
	// create a context with the device
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
}

void CLProgram::build_command_queue(){
	// create a command queue
	command_queue = clCreateCommandQueue(context, device_id, 0, NULL);
}

void CLProgram::read_source_cl(char* source_file) {
	FILE* fp = fopen(source_file, "r");
	_ASSERT(fp);
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);
}

void CLProgram::build_program() {
	// create and then build the program for the device
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, NULL);
	clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
}

cl_kernel CLProgram::create_kernel() {
	// create a kernal
	return clCreateKernel(program, "hello", NULL);
}