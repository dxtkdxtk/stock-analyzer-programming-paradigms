//#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "SACudaLibrary.h"
#include "SACudaKernel.h"

extern "C" IntArray FindInverseTrends(double * h_data, int h_length)
{
	IntArray returnvalue;
	int oldlength = h_length;
	if(h_length == 0)
	{
		returnvalue.length = 0;
		return returnvalue;
	}

	//*************************** CUDA ***************************
	double * d_data;
	int * d_length;

	REQUIRE_SUCCESS(cudaMalloc((void **)&d_data, sizeof(double) * h_length));
	REQUIRE_SUCCESS(cudaMalloc((void **)&d_length, sizeof(int)));
	REQUIRE_SUCCESS(cudaMemcpy((void *)d_data, (void *)h_data, sizeof(double) * h_length, cudaMemcpyHostToDevice));
	REQUIRE_SUCCESS(cudaMemcpy((void *)d_length, (void *)&h_length, sizeof(int), cudaMemcpyHostToDevice));

	FindInverseTrendsKernel<<<(h_length-1)/1024 + 1,1024>>>(d_data, d_length);

	REQUIRE_SUCCESS(cudaMemcpy((void *)&h_length, (void *)d_length, sizeof(int), cudaMemcpyDeviceToHost));
	if(h_length != 0)
		REQUIRE_SUCCESS(cudaMemcpy((void *)h_data, (void *)d_data, sizeof(double) * oldlength, cudaMemcpyDeviceToHost));

	cudaFree((void *)d_data);
	cudaFree((void *)d_length);
	//************************* END CUDA *************************

	int found = 0;
	int i = 0;
	returnvalue.values = (int *)malloc(sizeof(int) * h_length);
	while(found < h_length)
	{
		if(h_data[i] > 0)
			returnvalue.values[found++] = i+2;
		++i;
	}
	
	returnvalue.length = h_length;

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

