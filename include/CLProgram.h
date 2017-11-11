#include<iostream>
#include<vector>
#include<typeinfo>
#include<string>
#include <CL/cl.h>

class CLProgram{
public:
	CLProgram(char* source_file, cl_int device_type);
	~CLProgram();

	void load_and_build_source(char* source_file);
	void default_to_single_device(cl_int device_type);
	void read_work_buffer_size(cl_kernel kernel);
	cl_kernel create_kernel(char* function_name);
	cl_mem create_buffer(cl_int flag, int size, void* ptr);
	void read_buffer(cl_mem cl_buffer, cl_int read_flag, unsigned int flag, void* output);
	void set_kernel_arg(cl_kernel kernel, int arg, unsigned int size, void* ptr);
	void run_kernel(cl_kernel kernel);
	void run_kernel(cl_kernel kernel, cl_uint work_dimenion, size_t* global_work_size, size_t* local_work_size, cl_uint num_events, const cl_event* event_wait_list, cl_event* event);
	void free_program();
	void free_kernel(cl_kernel kernel);
	void free_buffer(cl_mem buffer);
	
private:
	// Error handling
	void error_message(std::string string, int error_code);
	char* get_error_string(cl_int error_code);

	// Vars
	cl_int error;
	char *source_str;
	size_t source_size;
	cl_platform_id platform_id;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_device_id device_id;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;
};