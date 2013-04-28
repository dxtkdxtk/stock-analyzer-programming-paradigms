#include <cuda_runtime.h>
#include <stdlib.h>

#include "SACudaLibrary.h"
#include "SACudaKernel.h"

extern "C" IntArray FindInverseTrends(double * h_data, int length)
{
	IntArray returnvalue;
	double * d_data;

	REQUIRE_SUCCESS(cudaMalloc((void **)&d_data, sizeof(double) * length));
	REQUIRE_SUCCESS(cudaMemcpy((void *)d_data, h_data, sizeof(double) * length, cudaMemcpyDeviceToHost));

	FindInverseTrendsKernel<<<1,1>>>(d_data, length);

	cudaFree((void *)d_data);

	return returnvalue;
}

extern "C" DoubleArray CalculateMarketAverage(double * h_data, int entries, int timesteps)
{
	DoubleArray returnvalue;
	double * d_data;

	REQUIRE_SUCCESS(cudaMalloc((void **)&d_data, sizeof(double) * entries * timesteps));
	REQUIRE_SUCCESS(cudaMemcpy((void *)d_data, h_data, sizeof(double) * entries * timesteps, cudaMemcpyHostToDevice));

	CalculateMarketAverageKernel<<<1,1>>>(d_data, entries, timesteps);



	cudaFree((void *)d_data);

	return returnvalue;
}

