#ifndef SACUDAKERNEL_H
#define SACUDAKERNEL_H

__global__ void FindInverseTrendsKernel(double * data, int length);
__global__ void CalculateMarketAverageKernel(double * data, int entries, int timesteps);

#endif

