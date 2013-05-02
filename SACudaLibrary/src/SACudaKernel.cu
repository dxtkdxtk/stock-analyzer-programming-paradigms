#include "SACudaKernel.h"

__global__ void FindInverseTrendsKernel(double * data, int length)
{
	
}

#define GETDATA(e, t) ((e<entries)?(e*timesteps + t):0)

__global__ void CalculateMarketAverageKernel(double * data, int entries, int timesteps)
{
	int entryid = (threadIdx.x + blockIdx.x*blockDim.x);
	int timestepid = (threadIdx.y + blockIdx.y*blockDim.y);

	if(entryid*2 >= entries || timestepid >= timesteps)
		return;

	data[GETDATA(entryid, timestepid)] += data[GETDATA(((entries-1)/2 + 1 + entryid), timestepid)];
	data[GETDATA(entryid, timestepid)] /= 2;
}

