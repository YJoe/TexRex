__kernel void find_corners(__global int* bounds){
	bounds[0] = 1;
	bounds[1] = 0;
	bounds[2] = 0;
	bounds[3] = 0;
}