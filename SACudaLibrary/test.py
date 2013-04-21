from ctypes import *

libraryhandle = cdll.LoadLibrary("./SACudaLibrary.so")
print libraryhandle.TestCudaAdd(1, 6)

