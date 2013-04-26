# Y u no set up cuda correctly...
COMPILE_FLAGS = -O3 -I /afs/crc.nd.edu/x86_64_linux/cuda/4.0/cuda/include
LINK_FLAGS =

# Define objects included in the compilation here
OBJECTS = ./src/SACudaKernel.o ./src/SACudaLibrary.o

all: SACudaLibrary

SACudaLibrary: $(OBJECTS)
	g++ $(OBJECTS) -shared -lcudart -o SACudaLibrary.so

%.o: %.cpp
	g++ -c -fPIC $(COMPILE_FLAGS) $< -o $@

%.o: %.cu
	nvcc -c $(COMPILE_FLAGS) --generate-code arch=compute_20,code=sm_20 -Xcompiler -fPIC $< -o $@

.PHONY: clean
clean:
	rm ./src/*.o SACudaLibrary.so
