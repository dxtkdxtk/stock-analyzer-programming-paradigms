# Provides a Python interface and call stubs for operating the Cuda part of the project

from ctypes import *

class IntArray(Structure):
# Structure used to pass information
	_fields_ = [
		("length", c_int), 
		("values", POINTER(c_int))
	]

class DoubleArray(Structure):
# Structure used to pass information
	_fields_ = [
		("length", c_int), 
		("values", POINTER(c_double))
	]

class SACudaProxy(object):
# Proxies between Python and library
	def __init__(self):
		# Load library
		self.LibraryHandle = cdll.LoadLibrary("./SACudaLibrary.so")
		
		# Bind dll function FindInverseTrends + signature
		self.CudaFindInverseTrends = self.LibraryHandle.FindInverseTrends
		self.CudaFindInverseTrends.argtypes = [POINTER(c_double), c_int]
		self.CudaFindInverseTrends.restype = IntArray
		
		# Bind dll function CalculateMarketAverage + signature
		self.CudaCalculateMarketAverage = self.LibraryHandle.CalculateMarketAverage
		self.CudaCalculateMarketAverage.argtypes = [POINTER(c_double), c_int, c_int]
		self.CudaCalculateMarketAverage.restype = DoubleArray

	def FindInverseTrends(self, data):
	# Format for data variable:
	# (
	#    (stock1, stock2)
	#    (stock1, stock2)
	#    ...
	#    (stock1, stock2)
	# )
	# where stock1 and stock2 are values at one distinct point in time, and each pair is ordered sequentially in time
	#
	# Returns a zero-indexed variable-length list of integers (i.e. (1, 2, 4)) listing points of interest in the data
		
		# Initialize cstruct
		length = len(data) * 2
		values = (c_double * length)(*[c_double(element) for item in data for element in item])
		
		# Check data validity
		for pair in data:
			if(len(pair) != 2):
				raise TypeError("FindInverseTrends() passed incorrect data")
		
		# Call
		result = self.CudaFindInverseTrends(values, length)
		
		# Unpack data
		resultdata = list()
		for idx in range(result.length):
			resultdata.append(result.values[idx])
		
		return resultdata
	
	def CalculateMarketAverage(self, data):
	# Format for data variable:
	# (
	#    (value1, value2, ..., valuen)
	#    (value1, value2, ..., valuen)
	#    ...
	#    (value1, value2, ..., valuen)
	# )
	# where each list *must* be of length n (number of timesteps)
	#
	# Returns a list of floats (i.e. (1.4, 2.2, 4.8)) describing the overall trend calculated from the data.
		
		# Initialize cstruct
		numentries = len(data)
		numtimesteps = len(data[0])
		values = (c_double * (numentries * numtimesteps))(*[c_double(element) for item in data for element in item])
		
		# Check data validity
		for item in data:
			if(len(item) != numtimesteps):
				raise TypeError("CalculateMarketAverage() passed incorrect data")
		
		# Call
		result = self.CudaCalculateMarketAverage(values, numentries, numtimesteps)
		
		# Unpack data
		resultdata = list()
		for idx in range(result.length):
			resultdata.append(result.values[idx])
		
		return resultdata

