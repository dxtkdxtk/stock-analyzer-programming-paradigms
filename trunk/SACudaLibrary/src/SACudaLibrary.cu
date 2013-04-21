#include "SACudaLibrary.h"
#include "SACudaKernel.h"

extern "C" int TestCudaAdd(int a, int b)
{
	int * d_result;
	int result;

	cudaMalloc((void **)&d_result, sizeof(int));

	TestCudaAddKernel<<<1,1>>>(a, b, d_result);

	cudaMemcpy((void *)&result, (void *)d_result, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_result);

	return result;
}

