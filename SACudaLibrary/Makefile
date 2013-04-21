COMPILE_FLAGS = -O3
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
