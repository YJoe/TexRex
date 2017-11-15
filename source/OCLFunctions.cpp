#include "../include/OCLFunctions.h"

using namespace std;

OCLFunctions::OCLFunctions(int device_type) {
	
	//get all platforms (drivers)
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	default_platform = all_platforms[0];
	std::cout << "Using platform [" << default_platform.getInfo<CL_PLATFORM_NAME>() << "]\n";

	//get default device of the default platform
	default_platform.getDevices(device_type, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	default_device = all_devices[0];
	std::cout << "Using device [" << default_device.getInfo<CL_DEVICE_NAME>() << "]\n";

	context = cl::Context({ default_device });

	// kernel calculates for each element C=A+B
	std::string kernel_code =
		"   void __kernel simple_add(global const int* A, __global const int* B, __global int* C){												"
		"       C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];																"
		"   }																																	"
		"    																																	"
		"	void __kernel simple_mul(__global const int* A, __global const int* B, __global int* C){											"
		"		C[get_global_id(0)] = A[get_global_id(0)] * B[get_global_id(0)];																"
		"	};																																	"
		"    																																	"
		"	void __kernel apply_filter(__global const double* I, int Iw, __global const double* F, int Fw, int Fh, __global double* M, int Mw){	"
		"       int m = get_global_id(0);																										"
		"		int s = m % Mw + ((m / Mw) * Iw);																								"
		"		double temp = 0;																												"
		"		for (int i = 0; i < Fw; i++) {																									"
		"			for (int j = 0; j < Fh; j++) {																								"
		"				temp += I[s + (i + (j * Iw))] * F[i + (j * Fw)];																		"
		"			}																															"
		"		}																																"
		"		M[m] = temp / (Fw * Fh);																										"
		"	}																																	"
	;

	sources.push_back({ kernel_code.c_str(),kernel_code.length() });

	program = cl::Program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		cin.get();
		exit(1);
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
	double* a_arr = new double[a_size];
	int x = 0;
	for (int i = 0; i < image.size(); i++) {
		for (int j = 0; j < image[0].size(); j++) {
			a_arr[x] = image[i][j];
			x++;
		}
	}

	// create a 1d array from vector b
	double* b_arr = new double[b_size];
	int y = 0;
	for (int i = 0; i < filter.size(); i++) {
		for (int j = 0; j < filter[0].size(); j++) {
			b_arr[y] = filter[i][j];
			y++;
		}
	}

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