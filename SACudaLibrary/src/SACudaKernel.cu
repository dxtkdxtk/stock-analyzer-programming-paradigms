#include "SACudaKernel.h"

__global__ void FindInverseTrendsKernel(double * data, int * length)
{
	int timestepid = threadIdx.x + blockIdx.x*blockDim.x;
	float sum = 0;
	int count = 0;
	int oldlength = *length;
	bool interest = false;

	if(timestepid >= *length)
		return;

	for(int i = -5; i < 6; ++i) //computes moving average
		if(!((timestepid + i*2) >= *length || (timestepid + i*2) < 0))
		{
			sum += data[(timestepid + i*2)];
			++count;
		}

	__syncthreads(); // we want each thread to write their data simultaneously

	data[timestepid] = sum / count; // now both stocks are moving averages

	__syncthreads();

	if(timestepid % 2 == 0)
		data[timestepid] /=  data[timestepid+1]; // compute ratio

	if(timestepid == 0)
		*length = 0;

	__syncthreads();

	if(timestepid % 2 == 0)
		if(timestepid+6 < oldlength)
		{
			float difference = data[timestepid+6] - data[timestepid];
			float threshhold = data[timestepid]*.025;
			if(difference < 0)
				difference *= -1;

			if(threshhold < 0)
				threshhold *= -1;

			if(difference > threshhold)
			{
				atomicAdd(length, 1);
				interest = true;
			}
		}
	
	__syncthreads();

	if(timestepid % 2 == 0)
	{
		if(!interest)
			data[timestepid/2] = -1;
		else
			data[timestepid/2] = 1;
	}
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

