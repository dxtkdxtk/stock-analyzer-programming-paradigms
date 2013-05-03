//#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "SACudaLibrary.h"
#include "SACudaKernel.h"

extern "C" IntArray FindInverseTrends(double * h_data, int length)
{
	IntArray returnvalue;

	if(length == 0)
	{
		returnvalue.length = 0;
		return returnvalue;
	}

	//*************************** CUDA ***************************
	double * d_data;

	REQUIRE_SUCCESS(cudaMalloc((void **)&d_data, sizeof(double) * length));
	REQUIRE_SUCCESS(cudaMemcpy((void *)d_data, h_data, sizeof(double) * length, cudaMemcpyDeviceToHost));

	FindInverseTrendsKernel<<<1,1>>>(d_data, length);

	cudaFree((void *)d_data);
	//************************* END CUDA *************************

	
	returnvalue.length = 0;

	return returnvalue;
}

extern "C" DoubleArray CalculateMarketAverage(double * h_data, int entries, int timesteps)
{
	DoubleArray returnvalue;
	
	if(entries == 0 || timesteps == 0)
	{
		returnvalue.length = 0;
		return returnvalue;
	}

	//*************************** CUDA ***************************
	double * d_data;

	REQUIRE_SUCCESS(cudaMalloc((void **)&d_data, sizeof(double) * entries * timesteps));
	REQUIRE_SUCCESS(cudaMemcpy((void *)d_data, (void *)h_data, sizeof(double) * entries * timesteps, cudaMemcpyHostToDevice));

	for(int i = entries; i > 1; i = ((i-1)/2)+1)
	{
		CalculateMarketAverageKernelReduce<<<dim3(((i/2 + i%2)-1)/32 + 1, (timesteps-1/32 + 1)), dim3(32, 32)>>>(d_data, i, timesteps);
		cudaDeviceSynchronize();
	}

	CalculateMarketAverageKernelFinalize<<<(entries-1)/512 + 1,512>>>(d_data, entries, timesteps);
	
	REQUIRE_SUCCESS(cudaMemcpy((void *)h_data, (void *)d_data, sizeof(double) * timesteps*entries, cudaMemcpyDeviceToHost));

	cudaFree((void *)d_data);
	//************************* END CUDA *************************
	
	returnvalue.values = h_data;
	returnvalue.length = timesteps;

	return returnvalue;
}

