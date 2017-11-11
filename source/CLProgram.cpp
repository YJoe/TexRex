#include "..\include\CLProgram.h"

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

using namespace std;

CLProgram::CLProgram(char* source_file, cl_int device_type){

	default_to_single_device(CL_DEVICE_TYPE_CPU);
	load_and_build_source(source_file);
}

CLProgram::~CLProgram(){
	free(source_str);
}

void CLProgram::load_and_build_source(char * source_file){
	FILE *fp;

	/* Load the source code containing the kernel*/
	fp = fopen(source_file, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Building program
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &error);
	error_message("Creating program", error);
	error = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	error_message("Building program", error);
}

void CLProgram::default_to_single_device(cl_int device_type){
	error = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	error_message("Getting platform", error);

	error = clGetDeviceIDs(platform_id, device_type, 1, &device_id, &ret_num_devices);
	error_message("Getting id", error);

	// Context and command queue
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &error);
	error_message("Creating context", error);

	command_queue = clCreateCommandQueue(context, device_id, 0, &error);
	error_message("Creating command queue", error);
}

void CLProgram::read_work_buffer_size(cl_kernel kernel){
	size_t local;
	error = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
	error_message("Getting work group info", error);
	cout << "Work group size [" << local << "]" << endl;
}

cl_kernel CLProgram::create_kernel(char * function_name){
	// creating kernel
	cl_kernel k = clCreateKernel(program, function_name, &error);
	error_message("Creating kernel", error);
	return k;
}

cl_mem CLProgram::create_buffer(cl_int flag, int size, void* ptr){
	cl_mem b = clCreateBuffer(context, CL_MEM_READ_WRITE, size, ptr, &error);
	error_message("Creating buffer", error);
	return b;
}

void CLProgram::set_kernel_arg(cl_kernel kernel, int arg, unsigned int size, void* ptr){
	error = clSetKernelArg(kernel, arg, size, ptr);
	string s = "Setting argument number [" + to_string(arg) + "]";
	error_message(s, error);
}

void CLProgram::run_kernel(cl_kernel kernel) {
	error = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);
	error_message("Enqueue task", error);
}

void CLProgram::run_kernel(cl_kernel kernel, cl_uint work_dimenion, size_t* global_work_size, 
							size_t* local_work_size, cl_uint num_events, const cl_event* event_wait_list,
							cl_event* event){
	error = clEnqueueNDRangeKernel(command_queue, kernel, work_dimenion, NULL, global_work_size,
							local_work_size, num_events, event_wait_list, event);
	error = clEnqueueNDRangeKernel(command_queue, kernel, work_dimenion, NULL, global_work_size,
		local_work_size, 0, NULL, NULL);
	//error = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);
	error_message("Enqueue task", error);
}

void CLProgram::free_program(){
	error = clFlush(command_queue);
	error = clFinish(command_queue);
	error = clReleaseProgram(program);
	error = clReleaseCommandQueue(command_queue);
	error = clReleaseContext(context);
}

void CLProgram::read_buffer(cl_mem cl_buffer, cl_int read_flag, unsigned int flag, void* output) {
	error = clEnqueueReadBuffer(command_queue, cl_buffer, CL_TRUE, 0, sizeof(int), output, 0, NULL, NULL);
	error_message("Reading buffer", error);
}

void CLProgram::free_buffer(cl_mem buffer) {
	error = clReleaseMemObject(buffer);
	error_message("Freeing buffer", error);
}

void CLProgram::free_kernel(cl_kernel kernel) {
	error = clReleaseKernel(kernel);
	error_message("Freeing kernel", error);
}

void CLProgram::error_message(string string, int error_code){
	if (error_code < 0) {
		cout << "CL ERROR FROM [" << string << "] CODE [(" << error_code << ")" << get_error_string(error_code) << "]" << endl;
		cin.get();
		exit(0);
	}
}

char* CLProgram::get_error_string(cl_int error) {
	switch (error) {
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}
