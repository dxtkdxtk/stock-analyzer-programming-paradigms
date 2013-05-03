#include "SACudaKernel.h"

__global__ void FindInverseTrendsKernel(double * data, int length)
{
	
}

#define DATAINDEX(e, t) ((e<entries)?(e*timesteps + t):-1)
#define GETDATA(e, t) ((e<entries)?data[DATAINDEX(e,t)]:0)

__global__ void CalculateMarketAverageKernelReduce(double * data, int entries, int timesteps)
{
	int entryid = (threadIdx.x + blockIdx.x*blockDim.x);
	int timestepid = (threadIdx.y + blockIdx.y*blockDim.y);

	if(entryid*2 >= entries || timestepid >= timesteps)
		return;

	data[DATAINDEX(entryid, timestepid)] += GETDATA(((entries-1)/2 + 1 + entryid), timestepid);
}

__global__ void CalculateMarketAverageKernelFinalize(double * data, int entries, int timesteps)
{
	int timestepid = (threadIdx.x + blockIdx.x*blockDim.x);

	if(timestepid >= timesteps)
		return;

	data[timestepid] /= entries;
}

