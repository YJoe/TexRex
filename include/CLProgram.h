#include<iostream>
#include<vector>
#include <CL/cl.h>

class CLProgram{
public:
	CLProgram(char* source_file, int kernel_count);
	~CLProgram();
	void build_platform_device(int device_type);
	void build_context();
	void build_command_queue();
	void read_source_cl(char* source_file);
	void build_program();
	cl_kernel create_kernel();

	std::vector<cl_kernel> kernels;
	cl_context context;
	cl_command_queue command_queue;

private:
	cl_platform_id platform_id;
	cl_uint num_platforms;
	cl_device_id device_id;
	cl_uint num_devices;
	cl_mem memory_obj;
	char* source_str;
	cl_program program;
	cl_kernel kernel;
	size_t source_size;
};