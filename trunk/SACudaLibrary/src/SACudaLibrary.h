#ifndef SACUDALIBRARY_H
#define SACUDALIBRARY_H

//#include <cuda_runtime.h>

#define REQUIRE_SUCCESS(arg) if(arg != cudaSuccess) exit(-1);

struct IntArray
{
	int length;
	int * values;
};

struct DoubleArray
{
	int length;
	double * values;
};

extern "C" IntArray FindInverseTrends(double * h_data, int length);
extern "C" DoubleArray CalculateMarketAverage(double * h_data, int entries, int timesteps);

#endif

