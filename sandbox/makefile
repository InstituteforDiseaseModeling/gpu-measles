# GPU implementation

NVCC_FLAGS=-gencode arch=compute_60,code=sm_60 -lineinfo -rdc=true --maxrregcount=63 --ptxas-options=-v
CXX_FLAGS=-I/usr/local/include -std=c++11

histogram: histogram.cu kernel.cu
	nvcc \
		$(NVCC_FLAGS) \
		$(CXX_FLAGS) \
		-o bin/$@ $^

prng: prng-main.cu prng-kernel.cu
	nvcc \
		$(NVCC_FLAGS) \
		$(CXX_FLAGS) \
		-o bin/$@ $^
