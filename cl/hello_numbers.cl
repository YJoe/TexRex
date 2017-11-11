__kernel void multiply(int a, int b, __global int* res){
	(*res) = a * b;
}