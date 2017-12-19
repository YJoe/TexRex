#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#include "../include/OCLFunctions.h"

using namespace std;

OCLFunctions::OCLFunctions(int device_type) {
	
	//get all platforms (drivers)
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		std::cout << "[OCL] No platforms found. Check OpenCL installation!\n";
		std::cin.get();
		exit(1);
	}

	default_platform = all_platforms[0];
	std::cout << "[OCL] Using platform [" << default_platform.getInfo<CL_PLATFORM_NAME>() << "]\n";

	//get default device of the default platform
	default_platform.getDevices(device_type, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << "[OCL] No devices found. Check OpenCL installation!\n";
		std::cin.get();
		exit(1);
	}
	default_device = all_devices[0];
	std::cout << "[OCL] Using device [" << default_device.getInfo<CL_DEVICE_NAME>() << "]\n";

	context = cl::Context({ default_device });

	// define source code to be compiled by opencl on runtime
	std::string kernel_code =
		"	void __kernel apply_filter(__global const double* image, int image_w, __global const double* filter, int filter_w, int filter_h, __global double* map, int map_w){	"
		"       int map_space = get_global_id(0);																																"
		"		int offset = map_space % map_w + ((map_space / map_w) * image_w);																								"
		"		double temp = 0;																																				"
		"		for (int i = 0; i < filter_w; i++) {																															"
		"			for (int j = 0; j < filter_h; j++) {																														"
		"				temp += image[offset + (i + (j * image_w))] * filter[i + (j * filter_w)];																				"
		"			}																																							"
		"		}																																								"
		"		map[map_space] = temp / (filter_w * filter_h);																													"
		"	}																																									"
		"																																										"
		"	void __kernel pool(__global const double* image, int image_w, int image_h, int sample_size, __global double* map, int pooled_width){		"
		"		int g_id = get_global_id(0);																					"
		"		int fake_x = g_id % pooled_width;																			"
		"		int fake_y = g_id / pooled_width;																			"
		"		double max = 0;																										"
		"		for(int i = 0; i < sample_size; i++){																				"
		"			for(int j = 0; j < sample_size; j++){																			"
		"				int fake_pooled_y = fake_y * sample_size + i;																			"
		"				int fake_pooled_x = fake_x * sample_size + j;																			"
		"				if(fake_pooled_x < image_w && fake_pooled_y < image_h){																	"
		"					int index = fake_pooled_x + fake_pooled_y * image_w;																		"
		"					if(image[index] > max){																				"
		"						max = image[index];											"
		"					}												"
		"				}"																	
		"			}																												"
		"		}																													"
		"		map[g_id] = max;																								"
		"	}																														"
		"																										"
	;

	/*int g_id = g;
	cout << "gid [" << g_id << "]" << endl;
	int fake_x = g_id / pooled_width;
	int fake_y = g_id % pooled_width;
	cout << "fake indexes [" << fake_x << ", " << fake_y << "]" << endl;
	int max = 0;

	for (int i = 0; i < sample_s; i++) {
		for (int j = 0; j < sample_s; j++) {
			int fake_pooled_y = fake_x * sample_s + i;
			int fake_pooled_x = fake_y * sample_s + j;
			int index = fake_pooled_x + fake_pooled_y * image_width;
			if (fake_pooled_y < image_height && fake_pooled_x < image_width) {
				cout << "\tfake pooled indexes [" << fake_pooled_x << ", " << fake_pooled_y << "] [" << index << "]" << endl;
				if (image_arr[index] > max) {
					max = image_arr[index];
				}
			}
		}
	}

	cout << "\t\tMax is [" << max << "]" << endl;
	cout << endl;*/

	sources.push_back({ kernel_code.c_str(),kernel_code.length() });

	program = cl::Program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		cin.get();
		exit(1);
	}
}

void vector_to_arr(std::vector<std::vector<double>> &vec, double** arr_ptr) {
	
	// allocate space to the pointer
	(*arr_ptr) = new double[(int)vec[0].size() * (int)vec.size()];
	int x = 0;

	// cycle through the vector values, assigning them to the array
	for (int i = 0; i < vec.size(); i++) {
		for (int j = 0; j < vec[0].size(); j++) {
			(*arr_ptr)[x] = vec[i][j];
			x++;
		}
	}
}

void OCLFunctions::apply_filter_convolution(std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &filter, std::vector<std::vector<double>> &result_map) {
	
	// store the true dimensions of the vector
	int a_height = (int)image.size();
	int a_width = (int)image[0].size();
	int b_height = (int)filter.size();
	int b_width = (int)filter[0].size();
	
	// work out the single dimension array size
	int a_size = a_width * a_height;
	int b_size = b_width * b_height;

	// create a 1d array from vector a
	double* a_arr = NULL;
	vector_to_arr(image, &a_arr);

	// create a 1d array from vector b
	double* b_arr = NULL;
	vector_to_arr(filter, &b_arr);

	// working out the map size
	int map_width = a_width - b_width + 1;
	int map_height = a_height - b_height + 1;
	int map_size = map_width * map_height;
	double* map = new double[map_size];

	// create buffers on the device to store image filter and map
	cl::Buffer buffer_A(context, CL_MEM_READ_WRITE, sizeof(double) * a_size);
	cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, sizeof(double) * b_size);
	cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, sizeof(double) * map_size);

	//create queue to which we will push commands for the device.
	cl::CommandQueue queue(context, default_device);

	//write arrays A and B to the device
	queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, sizeof(double) * a_size, a_arr);
	queue.enqueueWriteBuffer(buffer_B, CL_TRUE, 0, sizeof(double) * b_size, b_arr);

	//run the kernel
	cl::Kernel kernel_add = cl::Kernel(program, "apply_filter");
	kernel_add.setArg(0, buffer_A);
	kernel_add.setArg(1, a_width);
	kernel_add.setArg(2, buffer_B);
	kernel_add.setArg(3, b_width);
	kernel_add.setArg(4, b_height);
	kernel_add.setArg(5, buffer_C);
	kernel_add.setArg(6, map_width);
	queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(map_size), cl::NullRange);
	queue.finish();

	//read result map from the device to array map
	queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(double) * map_size, map);

	// make sure this is empty before reusing
	result_map.clear();
	
	// fill up the map array with the results of the kernel
	vector<double> temp;
	for (int i = 0; i < map_size; i++) {
		if (i % (map_width) == 0) {
			result_map.emplace_back(temp);
		}
		result_map.back().emplace_back(map[i]);
	}

	// free allocated arrays
	free(a_arr);
	free(b_arr);
	free(map);
}

void OCLFunctions::pooling(std::vector<std::vector<double>>& image, std::vector<std::vector<double>>& target, int sample_size) {
	target.clear();

	// convert the vector into a 1d array
	double* image_arr = NULL;
	vector_to_arr(image, &image_arr);

	// work out the size of the array we just created
	int image_height = (int)image.size();
	int image_width = (int)image[0].size();
	int image_size = image_width * image_height;

	// what is the new map size
	int pooled_width = image_width / sample_size + (image_width % sample_size == 0 ? 0 : 1);
	int pooled_height = image_height / sample_size + (image_height % sample_size == 0 ? 0 : 1);
	int pooled_size = pooled_width * pooled_height;

	double* pooled_map = new double[pooled_size];

	// <insert magic opencl stuff here>
	// create buffers on the device to store image filter and map
	cl::Buffer image_buffer(context, CL_MEM_READ_WRITE, sizeof(double) * image_size);
	cl::Buffer pooled_buffer(context, CL_MEM_READ_WRITE, sizeof(double) * pooled_size);

	//create queue to which we will push commands for the device.
	cl::CommandQueue queue(context, default_device);

	//write arrays A and B to the device
	queue.enqueueWriteBuffer(image_buffer, CL_TRUE, 0, sizeof(double) * image_size, image_arr);

	//run the kernel
	cl::Kernel kernel_add = cl::Kernel(program, "pool");
	kernel_add.setArg(0, image_buffer);
	kernel_add.setArg(1, image_width);
	kernel_add.setArg(2, image_height);
	kernel_add.setArg(3, sample_size);
	kernel_add.setArg(4, pooled_buffer);
	kernel_add.setArg(5, pooled_width);
	queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, cl::NDRange(pooled_size), cl::NullRange);
	queue.finish();

	//read result map from the device to array map
	queue.enqueueReadBuffer(pooled_buffer, CL_TRUE, 0, sizeof(double) * pooled_size, pooled_map);

	// fill up the map array with the results of the kernel
	vector<double> temp;
	for (int i = 0; i < pooled_size; i++) {
		if (i % (pooled_width) == 0) {
			target.emplace_back(temp);
		}
		target.back().emplace_back(pooled_map[i]);
	}

	free(pooled_map);
}
