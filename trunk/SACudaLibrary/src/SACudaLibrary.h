#ifndef SACUDALIBRARY_H
#define SACUDALIBRARY_H

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

extern "C" IntArray FindInverseTrends(double * data, int length);
extern "C" DoubleArray CalculateMarketAverage(double * data, int entries, int timesteps);

#endif

