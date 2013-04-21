__global__ void TestCudaAddKernel(int a, int b, int * c)
{
	*c = a + b;
}

