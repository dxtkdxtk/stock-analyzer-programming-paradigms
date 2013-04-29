#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "SACudaLibrary.h"
#include "SACudaKernel.h"

extern "C" IntArray FindInverseTrends(double * h_data, int length)
{
	// CUDA SECTION
	double * d_data;

	REQUIRE_SUCCESS(cudaMalloc((void **)&d_data, sizeof(double) * length));
	REQUIRE_SUCCESS(cudaMemcpy((void *)d_data, h_data, sizeof(double) * length, cudaMemcpyDeviceToHost));

	FindInverseTrendsKernel<<<1,1>>>(d_data, length);

	cudaFree((void *)d_data);

	// LIBRARY SECTION
	IntArray returnvalue;

	returnvalue.length = 0;

	return returnvalue;
}

extern "C" DoubleArray CalculateMarketAverage(double * h_data, int entries, int timesteps)
{
	// CUDA SECTION
	double * d_data;

	REQUIRE_SUCCESS(cudaMalloc((void **)&d_data, sizeof(double) * entries * timesteps));
	REQUIRE_SUCCESS(cudaMemcpy((void *)d_data, (void *)h_data, sizeof(double) * entries * timesteps, cudaMemcpyHostToDevice));

	CalculateMarketAverageKernel<<<(timesteps-1)/512 + 1,512>>>(d_data, entries, timesteps);
	
	REQUIRE_SUCCESS(cudaMemcpy((void *)h_data, (void *)d_data, sizeof(double) * timesteps, cudaMemcpyDeviceToHost));

	cudaFree((void *)d_data);

	// DEVICE SECTION
	DoubleArray returnvalue;

	returnvalue.values = h_data;
	returnvalue.length = timesteps;

	return returnvalue;
}

