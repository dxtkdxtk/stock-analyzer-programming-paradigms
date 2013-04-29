#include "SACudaKernel.h"

__global__ void FindInverseTrendsKernel(double * data, int length)
{
	
}

__global__ void CalculateMarketAverageKernel(double * data, int entries, int timesteps)
{
	int ordinal = threadIdx.x + blockIdx.x*blockDim.x - 1;
	double sum = 0;

	if(ordinal >= timesteps)
		return;

	for(int i = 0; i < entries; ++i)
	{
		sum += data[i*timesteps + ordinal];
	}

	data[ordinal] = sum/entries;
}

